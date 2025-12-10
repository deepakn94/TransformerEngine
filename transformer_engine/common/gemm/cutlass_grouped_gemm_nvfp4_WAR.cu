/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "../common.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "common/util/cuda_runtime.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

  constexpr float factor_inv = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);
  __global__ void compute_nvfp4_grouped_alpha_kernel(int num_gemms,
                                                   const float **amaxA_list,
                                                   const float **amaxB_list,
                                                   float *alpha_array,
                                                   float **alpha_ptr_list) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gemms) {
    return;
  }

  const float *amaxA_ptr = amaxA_list[idx];
  const float *amaxB_ptr = amaxB_list[idx];

  float alpha = (*amaxA_ptr) * (*amaxB_ptr) * factor_inv;
  alpha_array[idx] = alpha;
  alpha_ptr_list[idx] = alpha_array + idx;
}

template <typename T>
__global__ void Print_tensor(T* tensor, int m, int n) {
  printf("Print_tensor shape: %d x %d\n", m, n);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        printf("%f ", static_cast<float>(tensor[i * n + j]));
      }
      printf("\n");
    }
  }
}

template <typename ElementInput, typename ElementSF, typename ElementC, bool TransA, bool TransB, bool TransD>
void generic_moe_gemm_kernelLauncher(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                                     int num_gemms, char* host_workspace, char* device_workspace,
                                     size_t workspaceSize, bool accumulate, int device,
                                     int math_sm_count, cudaStream_t stream) {

  static_assert(cute::is_same_v<ElementInput, cutlass::float_e2m1_t>, "Unsupported input type. Expected e2m1.");
  static_assert(cute::is_same_v<ElementSF, cutlass::float_ue4m3_t>, "Unsupported SF type. Expected ue4m3.");

  static_assert(cute::is_same_v<ElementC, cutlass::bfloat16_t> ||
                    cute::is_same_v<ElementC, cutlass::half_t> || cute::is_same_v<ElementC, float>,
                "Unsupported output type. Expected bf16/fp16/fp32.");

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = cutlass::nv_float4_t<ElementInput>;  // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;            // Layout type for A matrix operand
  constexpr int AlignmentA = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<ElementInput>;  // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = ElementC;
  using LayoutC = conditional_t<TransD, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;

  // Core kernel configurations
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  struct MMA2SMConfig {
    using MmaTileShape =
        Shape<_256, _256, _256>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      Shape<_128,_64>, ElementAccumulator, ElementAccumulator, ElementC,
      LayoutC *, AlignmentC, ElementD, LayoutC *, AlignmentD,
      typename MMA2SMConfig::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *,
      AlignmentB, ElementAccumulator, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
      typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
  using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                                             CollectiveEpilogue2SM>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

  using StrideA = typename GemmGrouped::GemmKernel::InternalStrideA;
  using StrideB = typename GemmGrouped::GemmKernel::InternalStrideB;
  using StrideC = typename GemmGrouped::GemmKernel::InternalStrideC;
  using StrideD = typename GemmGrouped::GemmKernel::InternalStrideD;

  using LayoutSFA = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename GemmGrouped::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  auto get_aligned_offset = [](size_t current_offset, size_t alignment) -> size_t {
    return (current_offset + alignment - 1) & ~(alignment - 1);
  };

  if (host_workspace == nullptr || device_workspace == nullptr) {
    throw std::runtime_error("TE CUTLASS grouped gemm nvfp4_WAR workspace is null");
  }

  size_t offset = 0;
  auto problem_sizes_host =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(host_workspace);
  auto problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(device_workspace);
  offset = get_aligned_offset(offset + num_gemms * sizeof(ProblemShape::UnderlyingProblemShape), 128);

  auto ptr_A_host =
      reinterpret_cast<typename GemmGrouped::ElementA **>(host_workspace + offset);
  auto ptr_A =
      reinterpret_cast<typename GemmGrouped::ElementA **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(typename GemmGrouped::ElementA *), 128);
  auto ptr_B_host =
      reinterpret_cast<typename GemmGrouped::ElementB **>(host_workspace + offset);
  auto ptr_B =
      reinterpret_cast<typename GemmGrouped::ElementB **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(typename GemmGrouped::ElementB *), 128);
  auto ptr_D_host =
      reinterpret_cast<typename GemmGrouped::ElementD **>(host_workspace + offset);
  auto ptr_D =
      reinterpret_cast<typename GemmGrouped::ElementD **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(typename GemmGrouped::ElementD *), 128);

  auto ptr_SFA_host =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(host_workspace + offset);
  auto ptr_SFA =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);
  auto ptr_SFB_host =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(host_workspace + offset);
  auto ptr_SFB =
      reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  auto stride_A_host = reinterpret_cast<StrideA *>(host_workspace + offset);
  auto stride_A = reinterpret_cast<StrideA *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(StrideA), 128);
  auto stride_B_host = reinterpret_cast<StrideB *>(host_workspace + offset);
  auto stride_B = reinterpret_cast<StrideB *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(StrideB), 128);
  auto stride_D_host = reinterpret_cast<StrideD *>(host_workspace + offset);
  auto stride_D = reinterpret_cast<StrideD *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(StrideD), 128);
  auto layout_SFA_host = reinterpret_cast<LayoutSFA *>(host_workspace + offset);
  auto layout_SFA = reinterpret_cast<LayoutSFA *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(LayoutSFA), 128);
  auto layout_SFB_host = reinterpret_cast<LayoutSFB *>(host_workspace + offset);
  auto layout_SFB = reinterpret_cast<LayoutSFB *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(LayoutSFB), 128);

  // Device-side lists of amax pointers per group (A and B).
  auto amaxA_list_host = reinterpret_cast<const float **>(host_workspace + offset);
  auto amaxA_list = reinterpret_cast<const float **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(const float *), 128);
  auto amaxB_list_host = reinterpret_cast<const float **>(host_workspace + offset);
  auto amaxB_list = reinterpret_cast<const float **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(const float *), 128);

  for (int i = 0; i < num_gemms; i++) {
    const transformer_engine::Tensor* inputA = transformer_engine::convertNVTETensor(A[i]);
    const transformer_engine::Tensor* inputB = transformer_engine::convertNVTETensor(B[i]);
    transformer_engine::Tensor* outputD = transformer_engine::convertNVTETensor(D[i]);

    if constexpr (TransA) {
      NVTE_CHECK(inputA->has_columnwise_data(), "Input A is missing column-wise usage");
    } else {
      NVTE_CHECK(inputA->has_data(), "Input A is missing row-wise usage");
    }
    if constexpr (TransB) {
      NVTE_CHECK(inputB->has_data(), "Input B is missing row-wise usage");
    } else {
      NVTE_CHECK(inputB->has_columnwise_data(), "Input B is missing column-wise usage");
    } 
    const int gemm_m = TransA ? inputA->flat_last_dim() : inputA->flat_first_dim();
    const int gemm_n = TransB ? inputB->flat_first_dim() : inputB->flat_last_dim();
    const int gemm_k = TransA ? inputA->flat_first_dim() : inputA->flat_last_dim();
    // printf("cutlass_grouped_gemm_nvfp4_WAR expert %d gemm_m: %d, gemm_n: %d, gemm_k: %d (TransA: %d, TransB: %d)\n", i, gemm_m, gemm_n, gemm_k, TransA, TransB);
    problem_sizes_host[i] = cute::make_shape(gemm_m, gemm_n, gemm_k);

    ptr_A_host[i] = reinterpret_cast<typename GemmGrouped::ElementA *>(TransA ? inputA->columnwise_data.dptr : inputA->data.dptr);
    ptr_B_host[i] = reinterpret_cast<typename GemmGrouped::ElementB *>(TransB ? inputB->data.dptr : inputB->columnwise_data.dptr);
    ptr_D_host[i] = reinterpret_cast<typename GemmGrouped::ElementD *>(outputD->data.dptr); // outputD only has rowwise data pointer
    ptr_SFA_host[i] = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(TransA ? inputA->columnwise_scale_inv.dptr : inputA->scale_inv.dptr);
    ptr_SFB_host[i] = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(TransB ? inputB->scale_inv.dptr : inputB->columnwise_scale_inv.dptr);

    stride_A_host[i] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
    stride_B_host[i] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(gemm_n, gemm_k, 1));
    stride_D_host[i] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(gemm_m, gemm_n, 1));
    layout_SFA_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));
    layout_SFB_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));
    // printf("stride_A_host:");print(stride_A_host[i]);printf("\n");
    // printf("stride_B_host:");print(stride_B_host[i]);printf("\n");
    // printf("stride_D_host:");print(stride_D_host[i]);printf("\n");
    // printf("layout_SFA_host:");print(layout_SFA_host[i]);printf("\n");
    // printf("layout_SFB_host:");print(layout_SFB_host[i]);printf("\n");

    // 仅在 host 侧收集每个 group 的 amax device 指针，真正的 alpha 计算放到 device kernel 里。
    const float *amaxA_ptr = reinterpret_cast<const float *>(
        TransA ? inputA->columnwise_amax.dptr : inputA->amax.dptr);
    const float *amaxB_ptr = reinterpret_cast<const float *>(
        TransB ? inputB->amax.dptr : inputB->columnwise_amax.dptr);
    NVTE_CHECK(amaxA_ptr != nullptr && amaxB_ptr != nullptr);

    amaxA_list_host[i] = amaxA_ptr;
    amaxB_list_host[i] = amaxB_ptr;

  }

//   NVTE_CHECK(offset <= kCPUWorkSpaceSize,
//     "Insufficient host workspace size: required=", (long long)offset,
//     ", available=", (long long)kCPUWorkSpaceSize);
// NVTE_CHECK(workspaceSize >= offset,
//     "Insufficient device workspace size for params: required=", (long long)offset,
//     ", available=", (long long)workspaceSize);
  cudaMemcpyAsync(device_workspace, host_workspace, offset, cudaMemcpyHostToDevice, stream);

  auto alpha_array = reinterpret_cast<float *>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(float), 128);
  auto alpha_ptr_list = reinterpret_cast<float **>(device_workspace + offset);
  offset = get_aligned_offset(offset + num_gemms * sizeof(float *), 128);
  compute_nvfp4_grouped_alpha_kernel<<<1, 128, 0, stream>>>(num_gemms, amaxA_list, amaxB_list, alpha_array, alpha_ptr_list);


  constexpr bool Dgrad = !TransA && !TransB;
  constexpr bool Wgrad = TransA && !TransB;
  typename GemmGrouped::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha = 0;
  fusion_args.alpha_ptr_array = alpha_ptr_list;
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = Wgrad && accumulate ? 1 : 0;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = device;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  
  if (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = (Dgrad || Wgrad) ? dim3(2, 2, 1) : dim3(4, 4, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  typename GemmGrouped::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongN;

  args = typename GemmGrouped::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_gemms, problem_sizes, problem_sizes_host},
      {const_cast<const typename GemmGrouped::ElementA **>(ptr_A), stride_A,
       const_cast<const typename GemmGrouped::ElementB **>(ptr_B), stride_B,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFA),
       layout_SFA,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFB),
       layout_SFB},
      {fusion_args, const_cast<const typename GemmGrouped::ElementC **>(ptr_D), stride_D, ptr_D, stride_D},
      hw_info,
      scheduler};

  GemmGrouped gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspaceSize < offset + workspace_size) {  // 16MB limit
    throw std::runtime_error("TE CUTLASS device grouped gemm calculated workspace size (" +
                             std::to_string(offset + workspace_size) + ") exceeds buffer size (" +
                             std::to_string(workspaceSize) + ")\n");
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "TE CUTLASS device grouped gemm will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto init_status = gemm.initialize(args, device_workspace + offset);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass device grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass device grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }
}

// cpu workspace size is 4MB
static constexpr size_t kCPUWorkSpaceSize = 4 * 1024 * 1024;

static char* getHostWorkspace() {
  static std::once_flag flag;
  static std::shared_ptr<char> workspace;

  std::call_once(flag, [&]() {
    workspace =
        std::shared_ptr<char>(reinterpret_cast<char*>(std::malloc(kCPUWorkSpaceSize)), [](char* p) {
          if (p) std::free(p);
        });

    if (!workspace) {
      throw std::bad_alloc();
    }
  });

  return workspace.get();
}


void cutlass_grouped_gemm_nvfp4_WAR(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms,
  bool transa, bool transb, bool grad, NVTETensor* workspace,
  bool accumulate, int device, int math_sm_count, cudaStream_t stream) {
  NVTE_API_CALL(cutlass_grouped_gemm_nvfp4_WAR);
  using namespace transformer_engine;


  // Dispatch
  using transformer_engine::DType;
  const DType ab_dtype = DType::kFloat4E2M1;
  const DType d_dtype = convertNVTETensor(D[0])->data.dtype;

  char* host_workspace = getHostWorkspace();
  auto* wspace_tensor = convertNVTETensor(workspace[0]);
  char* device_workspace = reinterpret_cast<char*>(wspace_tensor->data.dptr);
  const size_t workspaceSize =
      get_buffer_size_bytes(wspace_tensor->data.numel(), wspace_tensor->data.dtype);
  // printf("workspaceSize: %zu\n", workspaceSize);

  auto dispatch_layout = [&](auto ab_dtype, auto d_dtype) {
    // dispatch based on input layout and dgrad flag
    using ABType = decltype(ab_dtype);
    using ABSFType = cutlass::float_ue4m3_t;
    using DType = decltype(d_dtype);

    // Swap A and B
    if (!transb && transa) {  // fprop case
      generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, false, true, false>(
          B, A, D, num_gemms, host_workspace, device_workspace, workspaceSize, accumulate, device,
          math_sm_count, stream);
    } else if (!transb && !transa) { // dgrad case
      generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, false, false, false>(
          B, A, D, num_gemms, host_workspace, device_workspace, workspaceSize, accumulate, device,
          math_sm_count, stream);
    } else if (transb && !transa) { // wgrad case
      generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, true, false, true>(
          A, B, D, num_gemms, host_workspace, device_workspace, workspaceSize, accumulate, device,
          math_sm_count, stream);
    } else {
      throw std::runtime_error("Unsupported layout TT.");
    }
  };

  auto dispatch_output_dtype = [&](auto ab_dtype) {
    // dispatch based on D dtype
    switch (d_dtype) {
      case DType::kBFloat16:
        dispatch_layout(ab_dtype, cutlass::bfloat16_t{});
        break;
      case DType::kFloat16:
        dispatch_layout(ab_dtype, cutlass::half_t{});
        break;
      case DType::kFloat32:
        dispatch_layout(ab_dtype, float{});
        break;
      default:
        throw std::runtime_error("Unsupported output dtype. Expected BF16/FP16/FP32.");
    }
  };

  auto dispatch = [&]() {
    // dispatch based on A/B dtype
    switch (ab_dtype) {
      case DType::kFloat4E2M1:
        dispatch_output_dtype(cutlass::float_e2m1_t{});
        break;
      default:
        throw std::runtime_error("Unsupported input dtype. A/B must be FP4 e2m1.");
    }
  };

  dispatch();
}