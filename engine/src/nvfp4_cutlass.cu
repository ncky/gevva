#include "g4/gpu.hpp"
#include "g4/runtime.hpp"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g4_cutlass_detail {

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void cutlass_check(cutlass::Status status, const char* operation) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count) cuda_check(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }
  void copy(const T* source, std::size_t count) {
    if (count > count_) throw std::runtime_error("device metadata copy overflow");
    cuda_check(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy metadata");
  }
 private:
  T* data_{};
  std::size_t count_{};
};

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
using ElementType = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementA = cutlass::nv_float4_t<ElementType>;
using ElementB = cutlass::nv_float4_t<ElementType>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ClusterShape = Shape<_1, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinearCombination<
    ElementD, Accumulator, ElementC, Accumulator>;
template <int TileN, bool Cooperative = false>
struct ExpertGemmTraits {
using TileShape = Shape<_128, Int<TileN>, Int<(TileN == 256 || Cooperative ? 128 : 256)>>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator, Accumulator, ElementC, LayoutC*, 8, ElementD, LayoutC*, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA*, 32, ElementB, LayoutB*, 32, Accumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    std::conditional_t<Cooperative,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using DefaultGemmTraits = ExpertGemmTraits<128>;
using TileShape = DefaultGemmTraits::TileShape;
using Epilogue = DefaultGemmTraits::Epilogue;
using Mainloop = DefaultGemmTraits::Mainloop;
using Kernel = DefaultGemmTraits::Kernel;
using Gemm = DefaultGemmTraits::Gemm;
using StrideA = typename Kernel::InternalStrideA;
using StrideB = typename Kernel::InternalStrideB;
using StrideC = typename Kernel::InternalStrideC;
using LayoutSFA = typename Mainloop::InternalLayoutSFA;
using LayoutSFB = typename Mainloop::InternalLayoutSFB;
using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
using Shape3 = typename ProblemShape::UnderlyingProblemShape;

constexpr int kRoutes = 8;
constexpr int kExperts = 128;
constexpr int kM = 1;
constexpr int kN = 1408;
constexpr int kK = 2816;
constexpr std::size_t kPackedInputPerRoute = kK / 2;
// Scale-factor layouts pad M to 128 rows and interleave each four K groups.
constexpr std::size_t kScaleInputPerRoute = 128 * (kK / 16);

__global__ void quantize_bf16_routes(const __nv_bfloat16* input,
                                     const float* input_scale_quant,
                                     const int* expert_ids,
                                     std::uint8_t* packed,
                                     std::uint8_t* scale_factors,
                                     int width,
                                     const std::byte* weight_base = nullptr,
                                     const std::byte* weight_scale_base = nullptr,
                                     const float* alpha_base = nullptr,
                                     const ElementType** weight_ptrs = nullptr,
                                     const ElementSF** weight_scale_ptrs = nullptr,
                                     float** alpha_ptrs = nullptr,
                                     int output_width = 0,
                                     int routes = kRoutes,
                                     const int* route_indices = nullptr) {
  const int route = blockIdx.x;
  const int group = threadIdx.x;
  if (route >= routes || group >= width / 16) return;
  const int expert = expert_ids[route];
  const int source_route = route_indices ? route_indices[route] : route;
  if (group == 0 && weight_ptrs) {
    weight_ptrs[route] = reinterpret_cast<const ElementType*>(
        weight_base + static_cast<std::size_t>(expert) * output_width * (width / 2));
    weight_scale_ptrs[route] = reinterpret_cast<const ElementSF*>(
        weight_scale_base + static_cast<std::size_t>(expert) * output_width * (width / 16));
    alpha_ptrs[route] = const_cast<float*>(alpha_base) + expert;
  }
  const float multiplier = input_scale_quant[expert];
  float values[16];
  float maximum = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = __bfloat162float(
        input[(source_route / 8) * width + i + group * 16]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const float unrounded_scale = multiplier * maximum / 6.0F;
  const __nv_fp8_e4m3 encoded_scale(unrounded_scale);
  const float rounded_scale = static_cast<float>(encoded_scale);
  const float inverse = rounded_scale == 0.0F ? 0.0F : multiplier / rounded_scale;
  auto* route_packed = packed + route * (width / 2) + group * 8;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float2 value{values[2 * pair] * inverse, values[2 * pair + 1] * inverse};
    route_packed[pair] = __nv_fp4x2_e2m1(value).__x;
  }
  // SM120 SFA: [Ktile][outer-M][inner-M][inner-K]. Here M is one.
  const int scale_offset = (group / 4) * 512 + group % 4;
  scale_factors[route * kScaleInputPerRoute + scale_offset] = encoded_scale.__x;
}

__device__ float gelu(float value) {
  return 0.5F * value * (1.0F + erff(value * 0.7071067811865475F));
}

__global__ void geglu_quantize_routes(const __nv_bfloat16* fused,
                                      const float* input_scale_quant,
                                      const int* expert_ids,
                                      std::uint8_t* packed,
                                      std::uint8_t* scale_factors,
                                      const std::byte* weight_base = nullptr,
                                      const std::byte* weight_scale_base = nullptr,
                                      const float* alpha_base = nullptr,
                                      const ElementType** weight_ptrs = nullptr,
                                      const ElementSF** weight_scale_ptrs = nullptr,
                                      float** alpha_ptrs = nullptr,
                                      int routes = kRoutes) {
  constexpr int width = 704;
  const int route = blockIdx.x;
  const int group = threadIdx.x;
  if (route >= routes || group >= width / 16) return;
  const auto* row = fused + route * 2 * width;
  const int expert = expert_ids[route];
  if (group == 0 && weight_ptrs) {
    weight_ptrs[route] = reinterpret_cast<const ElementType*>(
        weight_base + static_cast<std::size_t>(expert) * 2816 * (width / 2));
    weight_scale_ptrs[route] = reinterpret_cast<const ElementSF*>(
        weight_scale_base + static_cast<std::size_t>(expert) * 2816 * (width / 16));
    alpha_ptrs[route] = const_cast<float*>(alpha_base) + expert;
  }
  const float multiplier = input_scale_quant[expert];
  float values[16];
  float maximum = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int col = group * 16 + i;
    const float up = __bfloat162float(row[col]);
    const float gate = __bfloat162float(row[width + col]);
    values[i] = gelu(gate) * up;
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const __nv_fp8_e4m3 encoded_scale(multiplier * maximum / 6.0F);
  const float rounded_scale = static_cast<float>(encoded_scale);
  const float inverse = rounded_scale == 0.0F ? 0.0F : multiplier / rounded_scale;
  auto* route_packed = packed + route * (width / 2) + group * 8;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float2 value{values[2 * pair] * inverse, values[2 * pair + 1] * inverse};
    route_packed[pair] = __nv_fp4x2_e2m1(value).__x;
  }
  constexpr int scale_stride = 128 * (width / 16);
  const int scale_offset = (group / 4) * 512 + group % 4;
  scale_factors[route * scale_stride + scale_offset] = encoded_scale.__x;
}

__global__ void reduce_expert_routes(const __nv_bfloat16* routes,
                                     const float* route_weights,
                                     __nv_bfloat16* output, int tokens = 1,
                                     const int* inverse_routes = nullptr) {
  constexpr int width = 2816;
  const int token = blockIdx.x;
  if (token >= tokens) return;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    float sum = 0.0F;
#pragma unroll
    for (int route = 0; route < 8; ++route) {
      const int index = token * 8 + route;
      const int source = inverse_routes ? inverse_routes[index] : index;
      sum = fmaf(__bfloat162float(routes[source * width + col]),
                 route_weights[index], sum);
    }
    output[token * width + col] = __float2bfloat16_rn(sum);
  }
}

__global__ void initialize_route_indices(int* indices, int routes) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route < routes) indices[route] = route;
}

__device__ int sorted_lower_bound(const int* keys, int size, int value) {
  int first = 0, count = size;
  while (count > 0) {
    const int step = count / 2;
    const int probe = first + step;
    if (keys[probe] < value) {
      first = probe + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__global__ void configure_moe_groups(
    const int* sorted_experts, int routes, int input_width, int output_width,
    int scale_stride, int* counts, int* offsets,
    const ElementType** input_ptrs, const ElementSF** input_scale_ptrs,
    ElementD** output_ptrs, float** alpha_ptrs,
    const std::uint8_t* input, const std::uint8_t* input_scales,
    ElementD* output, const float* alphas) {
  const int expert = threadIdx.x;
  if (expert >= kExperts) return;
  const int begin = sorted_lower_bound(sorted_experts, routes, expert);
  const int end = sorted_lower_bound(sorted_experts, routes, expert + 1);
  counts[expert] = end - begin;
  offsets[expert] = begin;
  input_ptrs[expert] = reinterpret_cast<const ElementType*>(
      input + static_cast<std::size_t>(begin) * (input_width / 2));
  input_scale_ptrs[expert] = reinterpret_cast<const ElementSF*>(
      input_scales + static_cast<std::size_t>(expert) * scale_stride);
  output_ptrs[expert] = output + static_cast<std::size_t>(begin) * output_width;
  alpha_ptrs[expert] = const_cast<float*>(alphas) + expert;
}

__global__ void quantize_bf16_sorted_moe(
    const __nv_bfloat16* input, const float* input_scale_quant,
    const int* sorted_experts, const int* sorted_routes, const int* offsets,
    std::uint8_t* packed, std::uint8_t* scale_factors,
    const int* scale_row_offsets,
    int width, int routes, int* inverse_routes = nullptr) {
  const int route = blockIdx.x;
  const int group = threadIdx.x;
  if (route >= routes || group >= width / 16) return;
  const int expert = sorted_experts[route];
  const int row = route - offsets[expert];
  const int source_route = sorted_routes[route];
  if (group == 0 && inverse_routes) inverse_routes[source_route] = route;
  const int source_token = source_route / 8;
  const float multiplier = input_scale_quant[expert];
  float values[16], maximum = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = __bfloat162float(
        input[static_cast<std::size_t>(source_token) * width + group * 16 + i]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const __nv_fp8_e4m3 encoded_scale(multiplier * maximum / 6.0F);
  const float rounded_scale = static_cast<float>(encoded_scale);
  const float inverse = rounded_scale == 0.0F ? 0.0F : multiplier / rounded_scale;
  auto* destination = packed +
      static_cast<std::size_t>(route) * (width / 2) + group * 8;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair)
    destination[pair] = __nv_fp4x2_e2m1(
        float2{values[2 * pair] * inverse, values[2 * pair + 1] * inverse}).__x;
  const int offset = (row / 128) * (128 * (width / 16)) +
      (group / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 +
      group % 4;
  scale_factors[static_cast<std::size_t>(scale_row_offsets[expert]) *
                    (width / 16) + offset] =
      encoded_scale.__x;
}

__global__ void configure_dynamic_grouped(
    const int* sorted_experts, int routes, int output_width, int input_width,
    Shape3* shapes, int* offsets, const ElementType** input_ptrs,
    const ElementSF** input_scale_ptrs, ElementD** output_ptrs,
    const std::uint8_t* input, const std::uint8_t* input_scales,
    ElementD* output, int* scale_row_offsets) {
  __shared__ int scale_rows[kExperts];
  const int expert = threadIdx.x;
  if (expert >= kExperts) return;
  const int begin = sorted_lower_bound(sorted_experts, routes, expert);
  const int end = sorted_lower_bound(sorted_experts, routes, expert + 1);
  scale_rows[expert] = ((end - begin + 127) / 128) * 128;
  __syncthreads();
  if (expert == 0) {
    int sum = 0;
    for (int index = 0; index < kExperts; ++index) {
      const int rows = scale_rows[index];
      scale_rows[index] = sum;
      sum += rows;
    }
  }
  __syncthreads();
  const int scale_row = scale_rows[expert];
  offsets[expert] = begin;
  scale_row_offsets[expert] = scale_row;
  shapes[expert] = Shape3{end - begin, output_width, input_width};
  input_ptrs[expert] = reinterpret_cast<const ElementType*>(
      input + static_cast<std::size_t>(begin) * (input_width / 2));
  input_scale_ptrs[expert] = reinterpret_cast<const ElementSF*>(
      input_scales + static_cast<std::size_t>(scale_row) * (input_width / 16));
  output_ptrs[expert] = output + static_cast<std::size_t>(begin) * output_width;
}

// The verifier always has at most 40 routes. Sorting a padded 128-element
// network in one resident block is cheaper than a D2D copy, index-init kernel,
// general-purpose radix sort, grouped configuration, and inverse-map kernel.
// Ties use the original route index, matching stable radix-sort ordering.
__global__ void sort_and_configure_dynamic_grouped(
    const int* expert_ids, int routes, int output_width, int input_width,
    Shape3* shapes, int* offsets, const ElementType** input_ptrs,
    const ElementSF** input_scale_ptrs, ElementD** output_ptrs,
    const std::uint8_t* input, const std::uint8_t* input_scales,
    ElementD* output, int* scale_row_offsets,
    int* sorted_experts, int* sorted_routes, int* inverse_routes) {
  constexpr int kSortSize = 128;
  __shared__ int keys[kSortSize];
  __shared__ int values[kSortSize];
  __shared__ int scale_rows[kExperts];
  const int lane = threadIdx.x;
  keys[lane] = lane < routes ? expert_ids[lane] : 0x7fffffff;
  values[lane] = lane < routes ? lane : 0x7fffffff;
  __syncthreads();
  for (int span = 2; span <= kSortSize; span <<= 1) {
    for (int stride = span >> 1; stride; stride >>= 1) {
      const int other = lane ^ stride;
      if (other > lane) {
        const int left_key = keys[lane], right_key = keys[other];
        const int left_value = values[lane], right_value = values[other];
        const bool greater = left_key > right_key ||
            (left_key == right_key && left_value > right_value);
        const bool ascending = (lane & span) == 0;
        if (greater == ascending) {
          keys[lane] = right_key;
          values[lane] = right_value;
          keys[other] = left_key;
          values[other] = left_value;
        }
      }
      __syncthreads();
    }
  }
  if (lane < routes) {
    sorted_experts[lane] = keys[lane];
    sorted_routes[lane] = values[lane];
    inverse_routes[values[lane]] = lane;
  }
  __syncthreads();
  const int begin = sorted_lower_bound(keys, routes, lane);
  const int end = sorted_lower_bound(keys, routes, lane + 1);
  scale_rows[lane] = ((end - begin + 127) / 128) * 128;
  __syncthreads();
  if (lane == 0) {
    int sum = 0;
    for (int index = 0; index < kExperts; ++index) {
      const int rows = scale_rows[index];
      scale_rows[index] = sum;
      sum += rows;
    }
  }
  __syncthreads();
  const int scale_row = scale_rows[lane];
  offsets[lane] = begin;
  scale_row_offsets[lane] = scale_row;
  shapes[lane] = Shape3{end - begin, output_width, input_width};
  input_ptrs[lane] = reinterpret_cast<const ElementType*>(
      input + static_cast<std::size_t>(begin) * (input_width / 2));
  input_scale_ptrs[lane] = reinterpret_cast<const ElementSF*>(
      input_scales + static_cast<std::size_t>(scale_row) * (input_width / 16));
  output_ptrs[lane] = output + static_cast<std::size_t>(begin) * output_width;
}

// Stable counting sort specialized for the verifier's 128 experts and at
// most 320 routes. It replaces five separately launched copy/init/radix-sort/
// configure/invert operations. Each route computes its stable rank among
// earlier equal keys; at this tiny fixed geometry that is cheaper than a
// second global synchronization boundary.
__global__ void count_sort_and_configure_dynamic_grouped(
    const int* expert_ids, int routes, int output_width, int input_width,
    Shape3* shapes, int* offsets, const ElementType** input_ptrs,
    const ElementSF** input_scale_ptrs, ElementD** output_ptrs,
    const std::uint8_t* input, const std::uint8_t* input_scales,
    ElementD* output, int* scale_row_offsets,
    int* sorted_experts, int* sorted_routes, int* inverse_routes) {
  __shared__ int counts[kExperts];
  __shared__ int begins[kExperts + 1];
  __shared__ int scale_rows[kExperts];
  __shared__ int ids[320];
  const int lane = threadIdx.x;
  if (lane < kExperts) counts[lane] = 0;
  if (lane < routes) ids[lane] = expert_ids[lane];
  __syncthreads();
  if (lane < routes) atomicAdd(counts + ids[lane], 1);
  __syncthreads();
  if (lane == 0) {
    begins[0] = 0;
    int scale_row = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
      begins[expert + 1] = begins[expert] + counts[expert];
      scale_rows[expert] = scale_row;
      scale_row += ((counts[expert] + 127) / 128) * 128;
    }
  }
  __syncthreads();
  if (lane < routes) {
    const int expert = ids[lane];
    int rank = 0;
    for (int earlier = 0; earlier < lane; ++earlier)
      rank += ids[earlier] == expert;
    const int sorted = begins[expert] + rank;
    sorted_experts[sorted] = expert;
    sorted_routes[sorted] = lane;
    inverse_routes[lane] = sorted;
  }
  if (lane < kExperts) {
    const int begin = begins[lane];
    const int end = begins[lane + 1];
    const int scale_row = scale_rows[lane];
    offsets[lane] = begin;
    scale_row_offsets[lane] = scale_row;
    shapes[lane] = Shape3{end - begin, output_width, input_width};
    input_ptrs[lane] = reinterpret_cast<const ElementType*>(
        input + static_cast<std::size_t>(begin) * (input_width / 2));
    input_scale_ptrs[lane] = reinterpret_cast<const ElementSF*>(
        input_scales + static_cast<std::size_t>(scale_row) * (input_width / 16));
    output_ptrs[lane] = output + static_cast<std::size_t>(begin) * output_width;
  }
}

// Prefill can carry 4096 routes. Sort 16 routes per thread in one resident CUB
// block, then build the inverse map and grouped pointers before the block
// retires. Route index is the value, so stable radix ordering is preserved.
__global__ void block_sort_and_configure_dynamic_grouped(
    const int* expert_ids, int routes, int output_width, int input_width,
    Shape3* shapes, int* offsets, const ElementType** input_ptrs,
    const ElementSF** input_scale_ptrs, ElementD** output_ptrs,
    const std::uint8_t* input, const std::uint8_t* input_scales,
    ElementD* output, int* scale_row_offsets,
    int* sorted_experts, int* sorted_routes, int* inverse_routes) {
  constexpr int kThreads = 256;
  constexpr int kItems = 16;
  using Sort = cub::BlockRadixSort<int, kThreads, kItems, int>;
  __shared__ typename Sort::TempStorage storage;
  __shared__ int scale_rows[kExperts];
  int keys[kItems];
  int values[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int route = static_cast<int>(threadIdx.x) * kItems + item;
    keys[item] = route < routes ? expert_ids[route] : 0x7fffffff;
    values[item] = route;
  }
  Sort(storage).Sort(keys, values, 0, 7);
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int destination = static_cast<int>(threadIdx.x) * kItems + item;
    if (destination < routes) {
      sorted_experts[destination] = keys[item];
      sorted_routes[destination] = values[item];
      inverse_routes[values[item]] = destination;
    }
  }
  __syncthreads();
  const int expert = threadIdx.x;
  int begin = 0, end = 0;
  if (expert < kExperts) {
    begin = sorted_lower_bound(sorted_experts, routes, expert);
    end = sorted_lower_bound(sorted_experts, routes, expert + 1);
    scale_rows[expert] = ((end - begin + 127) / 128) * 128;
  }
  __syncthreads();
  if (expert == 0) {
    int sum = 0;
    for (int index = 0; index < kExperts; ++index) {
      const int rows = scale_rows[index];
      scale_rows[index] = sum;
      sum += rows;
    }
  }
  __syncthreads();
  if (expert < kExperts) {
    const int scale_row = scale_rows[expert];
    offsets[expert] = begin;
    scale_row_offsets[expert] = scale_row;
    shapes[expert] = Shape3{end - begin, output_width, input_width};
    input_ptrs[expert] = reinterpret_cast<const ElementType*>(
        input + static_cast<std::size_t>(begin) * (input_width / 2));
    input_scale_ptrs[expert] = reinterpret_cast<const ElementSF*>(
        input_scales + static_cast<std::size_t>(scale_row) * (input_width / 16));
    output_ptrs[expert] =
        output + static_cast<std::size_t>(begin) * output_width;
  }
}

__global__ void geglu_quantize_sorted_moe(
    const __nv_bfloat16* fused, const float* input_scale_quant,
    const int* sorted_experts, const int* offsets, std::uint8_t* packed,
    std::uint8_t* scale_factors, const int* scale_row_offsets, int routes,
    int output_width, Shape3* shapes, const ElementType** input_ptrs,
    const ElementSF** input_scale_ptrs, ElementD** output_ptrs,
    ElementD* output) {
  constexpr int width = 704;
  const int route = blockIdx.x;
  const int group = threadIdx.x;
  if (route == 0 && group < kExperts) {
    const int begin = sorted_lower_bound(sorted_experts, routes, group);
    const int end = sorted_lower_bound(sorted_experts, routes, group + 1);
    shapes[group] = Shape3{end - begin, output_width, width};
    input_ptrs[group] = reinterpret_cast<const ElementType*>(
        packed + static_cast<std::size_t>(begin) * (width / 2));
    input_scale_ptrs[group] = reinterpret_cast<const ElementSF*>(
        scale_factors + static_cast<std::size_t>(scale_row_offsets[group]) *
                            (width / 16));
    output_ptrs[group] = output + static_cast<std::size_t>(begin) * output_width;
  }
  if (route >= routes || group >= width / 16) return;
  const int expert = sorted_experts[route];
  const int local_row = route - offsets[expert];
  const auto* row = fused + static_cast<std::size_t>(route) * 2 * width;
  const float multiplier = input_scale_quant[expert];
  float values[16], maximum = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int column = group * 16 + i;
    values[i] = gelu(__bfloat162float(row[width + column])) *
                __bfloat162float(row[column]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  const __nv_fp8_e4m3 encoded_scale(multiplier * maximum / 6.0F);
  const float rounded_scale = static_cast<float>(encoded_scale);
  const float inverse = rounded_scale == 0.0F ? 0.0F : multiplier / rounded_scale;
  auto* destination = packed +
      static_cast<std::size_t>(route) * (width / 2) + group * 8;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair)
    destination[pair] = __nv_fp4x2_e2m1(
        float2{values[2 * pair] * inverse, values[2 * pair + 1] * inverse}).__x;
  const int offset = (local_row / 128) * (128 * (width / 16)) +
      (group / 4) * 512 + (local_row % 32) * 16 +
      ((local_row / 32) % 4) * 4 + group % 4;
  scale_factors[static_cast<std::size_t>(scale_row_offsets[expert]) *
                    (width / 16) + offset] =
      encoded_scale.__x;
}

float time_launch(cudaStream_t stream, int iterations, auto&& launch) {
  for (int i = 0; i < 20; ++i) launch();
  cudaEvent_t begin{}, end{};
  cuda_check(cudaEventCreate(&begin), "cudaEventCreate");
  cuda_check(cudaEventCreate(&end), "cudaEventCreate");
  cuda_check(cudaEventRecord(begin, stream), "cudaEventRecord");
  for (int i = 0; i < iterations; ++i) launch();
  cuda_check(cudaEventRecord(end, stream), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize");
  float milliseconds{};
  cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return milliseconds * 1000.0F / iterations;
}

__global__ void configure_grouped_experts(
    const int* expert_ids, const std::byte* weight_base,
    const std::byte* scale_base, const float* alpha_base,
    const ElementType** weight_ptrs, const ElementSF** scale_ptrs,
    float** alpha_ptrs, int n, int k, int routes) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) return;
  const int expert = expert_ids[route];
  weight_ptrs[route] = reinterpret_cast<const ElementType*>(
      weight_base + static_cast<std::size_t>(expert) * n * (k / 2));
  scale_ptrs[route] = reinterpret_cast<const ElementSF*>(
      scale_base + static_cast<std::size_t>(expert) * n * (k / 16));
  alpha_ptrs[route] = const_cast<float*>(alpha_base) + expert;
}

class GroupedPlan {
 public:
  GroupedPlan(int n, int k, const std::byte* weight, const std::byte* scales,
              const float* alphas, const std::uint8_t* input,
              const std::uint8_t* input_scales, ElementD* output,
              int routes = kRoutes)
      : n_(n), k_(k), routes_(routes), weight_(weight), scales_(scales),
        alphas_(alphas), shapes_(routes), shapes_device_(routes), a_(routes),
        b_(routes), sfa_(routes), sfb_(routes), d_(routes), alpha_(routes),
        stride_a_(routes), stride_b_(routes), stride_c_(routes),
        layout_sfa_(routes), layout_sfb_(routes) {
    if (routes < 1 || routes > 4096 || routes % 8 != 0)
      throw std::runtime_error("grouped expert routes must be 8..4096");
    const std::size_t input_stride = static_cast<std::size_t>(k) / 2;
    const std::size_t input_scale_stride = 128 * (static_cast<std::size_t>(k) / 16);
    std::vector<const ElementType*> a(routes);
    std::vector<const ElementType*> b(routes);
    std::vector<const ElementSF*> sfa(routes);
    std::vector<const ElementSF*> sfb(routes);
    std::vector<ElementD*> d(routes);
    std::vector<float*> alpha(routes);
    std::vector<StrideA> stride_a(routes);
    std::vector<StrideB> stride_b(routes);
    std::vector<StrideC> stride_c(routes);
    std::vector<LayoutSFA> layout_sfa(routes);
    std::vector<LayoutSFB> layout_sfb(routes);
    for (int route = 0; route < routes; ++route) {
      shapes_[route] = {1, n, k};
      a[route] = reinterpret_cast<const ElementType*>(input + route * input_stride);
      b[route] = reinterpret_cast<const ElementType*>(
          weight + static_cast<std::size_t>(route) * n * (k / 2));
      sfa[route] = reinterpret_cast<const ElementSF*>(
          input_scales + route * input_scale_stride);
      sfb[route] = reinterpret_cast<const ElementSF*>(
          scales + static_cast<std::size_t>(route) * n * (k / 16));
      d[route] = output + static_cast<std::size_t>(route) * n;
      alpha[route] = const_cast<float*>(alphas) + route;
      stride_a[route] = cutlass::make_cute_packed_stride(StrideA{}, {1, k, 1});
      stride_b[route] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
      stride_c[route] = cutlass::make_cute_packed_stride(StrideC{}, {1, n, 1});
      layout_sfa[route] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(1, n, k, 1));
      layout_sfb[route] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, n, k, 1));
    }
    shapes_device_.copy(shapes_.data(), routes);
    a_.copy(a.data(), routes); b_.copy(b.data(), routes);
    sfa_.copy(sfa.data(), routes); sfb_.copy(sfb.data(), routes);
    d_.copy(d.data(), routes); alpha_.copy(alpha.data(), routes);
    stride_a_.copy(stride_a.data(), routes);
    stride_b_.copy(stride_b.data(), routes);
    stride_c_.copy(stride_c.data(), routes);
    layout_sfa_.copy(layout_sfa.data(), routes);
    layout_sfb_.copy(layout_sfb.data(), routes);

    cutlass::KernelHardwareInfo hardware;
    cuda_check(cudaGetDevice(&hardware.device_id), "cudaGetDevice");
    hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hardware.device_id);
    typename Kernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
    typename Kernel::MainloopArguments mainloop{
        a_.get(), stride_a_.get(), b_.get(), stride_b_.get(),
        sfa_.get(), layout_sfa_.get(), sfb_.get(), layout_sfb_.get()};
    typename Kernel::EpilogueArguments epilogue{
        {}, nullptr, stride_c_.get(), d_.get(), stride_c_.get()};
    epilogue.thread.alpha_ptr_array = alpha_.get();
    epilogue.thread.dAlpha = {_0{}, _0{}, 1};
    epilogue.thread.beta = 0.0F;
    typename Kernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {routes, shapes_device_.get(), shapes_.data()},
        mainloop, epilogue, hardware, scheduler};
    workspace_bytes_ = Gemm::get_workspace_size(arguments);
    if (workspace_bytes_) cuda_check(cudaMalloc(&workspace_, workspace_bytes_), "cudaMalloc CUTLASS workspace");
    cutlass_check(gemm_.can_implement(arguments), "CUTLASS dynamic can_implement");
    cutlass_check(gemm_.initialize(arguments, workspace_), "CUTLASS dynamic initialize");
  }

  ~GroupedPlan() { if (workspace_) cudaFree(workspace_); }
  GroupedPlan(const GroupedPlan&) = delete;
  GroupedPlan& operator=(const GroupedPlan&) = delete;

  void run(const int* expert_ids, cudaStream_t stream) {
    configure_grouped_experts<<<(routes_ + 255) / 256, 256, 0, stream>>>(
        expert_ids, weight_, scales_, alphas_, b_.get(), sfb_.get(),
        alpha_.get(), n_, k_, routes_);
    cutlass_check(gemm_.run(stream), "CUTLASS dynamic run");
  }

  void run_configured(cudaStream_t stream) {
    cutlass_check(gemm_.run(stream), "CUTLASS dynamic configured run");
  }
  const std::byte* weight() const { return weight_; }
  const std::byte* scales() const { return scales_; }
  const float* alphas() const { return alphas_; }
  const ElementType** weight_ptrs() const { return b_.get(); }
  const ElementSF** scale_ptrs() const { return sfb_.get(); }
  float** alpha_ptrs() const { return alpha_.get(); }

 private:
  int n_{};
  int k_{};
  int routes_{};
  const std::byte* weight_{};
  const std::byte* scales_{};
  const float* alphas_{};
  std::vector<Shape3> shapes_;
  DeviceBuffer<Shape3> shapes_device_;
  DeviceBuffer<const ElementType*> a_, b_;
  DeviceBuffer<const ElementSF*> sfa_, sfb_;
  DeviceBuffer<ElementD*> d_;
  DeviceBuffer<float*> alpha_;
  DeviceBuffer<StrideA> stride_a_;
  DeviceBuffer<StrideB> stride_b_;
  DeviceBuffer<StrideC> stride_c_;
  DeviceBuffer<LayoutSFA> layout_sfa_;
  DeviceBuffer<LayoutSFB> layout_sfb_;
  Gemm gemm_;
  void* workspace_{};
  std::size_t workspace_bytes_{};
};

class DynamicExpertPlan {
 public:
  DynamicExpertPlan(int output_width, int input_width, int maximum_rows,
                    const std::byte* weight, const std::byte* weight_scales,
                    const float* alphas, const std::uint8_t* input,
                    const std::uint8_t* input_scales, ElementD* output,
                    int* offsets, int* scale_row_offsets)
      : output_width_(output_width), input_width_(input_width),
        maximum_rows_(maximum_rows),
        weight_(weight), weight_scales_(weight_scales), alphas_(alphas),
        input_(input), input_scales_(input_scales), output_(output),
        offsets_(offsets), scale_row_offsets_(scale_row_offsets),
        shapes_(kExperts, Shape3{maximum_rows, output_width, input_width}),
        shapes_device_(kExperts), a_(kExperts), b_(kExperts),
        sfa_(kExperts), sfb_(kExperts), d_(kExperts), alpha_(kExperts),
        stride_a_(kExperts), stride_b_(kExperts), stride_c_(kExperts),
        layout_sfa_(kExperts), layout_sfb_(kExperts) {
    std::vector<const ElementType*> a(kExperts), b(kExperts);
    std::vector<const ElementSF*> sfa(kExperts), sfb(kExperts);
    std::vector<ElementD*> d(kExperts);
    std::vector<float*> alpha(kExperts);
    std::vector<StrideA> stride_a(kExperts);
    std::vector<StrideB> stride_b(kExperts);
    std::vector<StrideC> stride_c(kExperts);
    std::vector<LayoutSFA> layout_sfa(kExperts);
    std::vector<LayoutSFB> layout_sfb(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
      a[expert] = reinterpret_cast<const ElementType*>(input);
      b[expert] = reinterpret_cast<const ElementType*>(
          weight + static_cast<std::size_t>(expert) * output_width *
                       (input_width / 2));
      sfa[expert] = reinterpret_cast<const ElementSF*>(input_scales);
      sfb[expert] = reinterpret_cast<const ElementSF*>(
          weight_scales + static_cast<std::size_t>(expert) * output_width *
                              (input_width / 16));
      d[expert] = output;
      alpha[expert] = const_cast<float*>(alphas) + expert;
      stride_a[expert] = cutlass::make_cute_packed_stride(
          StrideA{}, {maximum_rows, input_width, 1});
      stride_b[expert] = cutlass::make_cute_packed_stride(
          StrideB{}, {output_width, input_width, 1});
      stride_c[expert] = cutlass::make_cute_packed_stride(
          StrideC{}, {maximum_rows, output_width, 1});
      layout_sfa[expert] = ScaleConfig::tile_atom_to_shape_SFA(
          make_shape(maximum_rows, output_width, input_width, 1));
      layout_sfb[expert] = ScaleConfig::tile_atom_to_shape_SFB(
          make_shape(maximum_rows, output_width, input_width, 1));
    }
    shapes_device_.copy(shapes_.data(), kExperts);
    a_.copy(a.data(), kExperts); b_.copy(b.data(), kExperts);
    sfa_.copy(sfa.data(), kExperts); sfb_.copy(sfb.data(), kExperts);
    d_.copy(d.data(), kExperts); alpha_.copy(alpha.data(), kExperts);
    stride_a_.copy(stride_a.data(), kExperts);
    stride_b_.copy(stride_b.data(), kExperts);
    stride_c_.copy(stride_c.data(), kExperts);
    layout_sfa_.copy(layout_sfa.data(), kExperts);
    layout_sfb_.copy(layout_sfb.data(), kExperts);
    cutlass::KernelHardwareInfo hardware;
    cuda_check(cudaGetDevice(&hardware.device_id), "cudaGetDevice");
    hardware.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hardware.device_id);
    auto initialize = [&]<class Traits>(typename Traits::Gemm& gemm,
                                       void*& workspace) {
    using Kernel = typename Traits::Kernel;
    using Mainloop = typename Traits::Mainloop;
    using Epilogue = typename Traits::Epilogue;
    using Gemm = typename Traits::Gemm;
    static_assert(std::is_same_v<LayoutSFA, typename Mainloop::InternalLayoutSFA>);
    static_assert(std::is_same_v<LayoutSFB, typename Mainloop::InternalLayoutSFB>);
    typename Kernel::TileSchedulerArguments scheduler;
    scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
    typename Mainloop::Arguments mainloop{
        a_.get(), stride_a_.get(), b_.get(), stride_b_.get(),
        sfa_.get(), layout_sfa_.get(), sfb_.get(), layout_sfb_.get()};
    typename Epilogue::Arguments epilogue{
        {}, nullptr, stride_c_.get(), d_.get(), stride_c_.get()};
    epilogue.thread.alpha_ptr_array = alpha_.get();
    epilogue.thread.dAlpha = {_0{}, _0{}, 1};
    epilogue.thread.beta = 0.0F;
    typename Kernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        // With no host mirror, CUTLASS uses its device-dynamic grouped
        // scheduler. configure_dynamic_grouped updates shapes and A/D pointers
        // on the inference stream before every launch.
        {kExperts, shapes_device_.get(), nullptr}, mainloop, epilogue,
        hardware, scheduler};
    const auto workspace_bytes = Gemm::get_workspace_size(arguments);
    if (workspace_bytes)
      cuda_check(cudaMalloc(&workspace, workspace_bytes),
                 "cudaMalloc dynamic expert workspace");
    cutlass_check(gemm.can_implement(arguments),
                  "CUTLASS dynamic expert can_implement");
    cutlass_check(gemm.initialize(arguments, workspace),
                  "CUTLASS dynamic expert initialize");
    };
    initialize.template operator()<DefaultGemmTraits>(gemm_, workspace_);
    if (std::getenv("G4_PREPARE_EXPERT_TILES")) {
      narrow_ = std::make_unique<ExpertGemmTraits<64>::Gemm>();
      wide_ = std::make_unique<ExpertGemmTraits<256>::Gemm>();
      cooperative_ = std::make_unique<ExpertGemmTraits<128, true>::Gemm>();
      initialize.template operator()<ExpertGemmTraits<64>>(*narrow_, narrow_workspace_);
      initialize.template operator()<ExpertGemmTraits<256>>(*wide_, wide_workspace_);
      initialize.template operator()<ExpertGemmTraits<128, true>>(
          *cooperative_, cooperative_workspace_);
    }
  }

  ~DynamicExpertPlan() {
    if (workspace_) cudaFree(workspace_);
    if (narrow_workspace_) cudaFree(narrow_workspace_);
    if (wide_workspace_) cudaFree(wide_workspace_);
    if (cooperative_workspace_) cudaFree(cooperative_workspace_);
  }
  void configure(const int* sorted_experts, int routes, cudaStream_t stream) {
    configure_dynamic_grouped<<<1, kExperts, 0, stream>>>(
        sorted_experts, routes, output_width_, input_width_,
        shapes_device_.get(), offsets_, a_.get(), sfa_.get(), d_.get(), input_,
        input_scales_, output_, scale_row_offsets_);
    // The device-dynamic scheduler consumes all 128 device shapes directly,
    // including zero-row experts. Avoid the old D2H synchronization, host
    // compaction, ten metadata uploads, and per-launch GEMM reinitialization.
  }
  void sort_and_configure(const int* expert_ids, int routes,
                          int* sorted_experts, int* sorted_routes,
                          int* inverse_routes, cudaStream_t stream,
                          bool counting_sort = false) {
    if (counting_sort) {
      if (routes <= 320)
        count_sort_and_configure_dynamic_grouped<<<1, 512, 0, stream>>>(
            expert_ids, routes, output_width_, input_width_,
            shapes_device_.get(), offsets_, a_.get(), sfa_.get(), d_.get(),
            input_, input_scales_, output_, scale_row_offsets_, sorted_experts,
            sorted_routes, inverse_routes);
      else
        block_sort_and_configure_dynamic_grouped<<<1, 256, 0, stream>>>(
                expert_ids, routes, output_width_, input_width_,
                shapes_device_.get(), offsets_, a_.get(), sfa_.get(), d_.get(),
                input_, input_scales_, output_, scale_row_offsets_,
                sorted_experts, sorted_routes, inverse_routes);
    }
    else
      sort_and_configure_dynamic_grouped<<<1, kExperts, 0, stream>>>(
          expert_ids, routes, output_width_, input_width_,
          shapes_device_.get(), offsets_, a_.get(), sfa_.get(), d_.get(),
          input_, input_scales_, output_, scale_row_offsets_, sorted_experts,
          sorted_routes, inverse_routes);
  }
  void configure_and_quantize_geglu(
      const __nv_bfloat16* fused, const float* input_scale_quant,
      const int* sorted_experts, const int* offsets, std::uint8_t* packed,
      std::uint8_t* scale_factors, int routes, cudaStream_t stream) {
    geglu_quantize_sorted_moe<<<routes, kExperts, 0, stream>>>(
        fused, input_scale_quant, sorted_experts, offsets, packed,
        scale_factors, scale_row_offsets_, routes, output_width_,
        shapes_device_.get(), a_.get(), sfa_.get(), d_.get(), output_);
  }
  void run(cudaStream_t stream) {
    // All plans share device-resident shapes, routing and weight pointers.
    // Alternative plans are initialized at load, never inside graph capture.
    // Environment polling exists only in explicitly prepared experiments.
    if (cooperative_ && std::getenv("G4_EXPERT_COOPERATIVE")) {
      cutlass_check(cooperative_->run(stream), "CUTLASS cooperative grouped experts run");
      return;
    }
    if (wide_ && std::getenv("G4_EXPERT_WIDE_TILES")) {
      cutlass_check(wide_->run(stream), "CUTLASS wide grouped experts run");
      return;
    }
    if (narrow_ && std::getenv("G4_EXPERT_NARROW_TILES")) {
      cutlass_check(narrow_->run(stream), "CUTLASS narrow grouped experts run");
      return;
    }
    cutlass_check(gemm_.run(stream), "CUTLASS grouped experts run");
  }

 private:
  int output_width_{}, input_width_{}, maximum_rows_{};
  const std::byte* weight_{};
  const std::byte* weight_scales_{};
  const float* alphas_{};
  const std::uint8_t* input_{};
  const std::uint8_t* input_scales_{};
  ElementD* output_{};
  int* offsets_{};
  int* scale_row_offsets_{};
  std::vector<Shape3> shapes_;
  DeviceBuffer<Shape3> shapes_device_;
  DeviceBuffer<const ElementType*> a_, b_;
  DeviceBuffer<const ElementSF*> sfa_, sfb_;
  DeviceBuffer<ElementD*> d_;
  DeviceBuffer<float*> alpha_;
  DeviceBuffer<StrideA> stride_a_;
  DeviceBuffer<StrideB> stride_b_;
  DeviceBuffer<StrideC> stride_c_;
  DeviceBuffer<LayoutSFA> layout_sfa_;
  DeviceBuffer<LayoutSFB> layout_sfb_;
  Gemm gemm_;
  void* workspace_{};
  std::size_t workspace_bytes_{};
  std::unique_ptr<ExpertGemmTraits<64>::Gemm> narrow_;
  std::unique_ptr<ExpertGemmTraits<256>::Gemm> wide_;
  std::unique_ptr<ExpertGemmTraits<128, true>::Gemm> cooperative_;
  void* narrow_workspace_{};
  void* wide_workspace_{};
  void* cooperative_workspace_{};
};

constexpr int kVocabRows = 262144;
constexpr int kVocabMaximumCandidates = 1024;

__global__ void quantize_vocab_rows(
    const __nv_bfloat16* source, std::uint8_t* packed,
    std::uint8_t* scale_factors, int active_rows, int physical_rows,
    int columns) {
  const int groups = columns / 16;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= physical_rows * groups) return;
  const int row = index / groups;
  const int group = index % groups;
  float values[16]{};
  float maximum = 0.0F;
  if (row < active_rows) {
#pragma unroll
    for (int element = 0; element < 16; ++element) {
      values[element] = __bfloat162float(
          source[static_cast<std::size_t>(row) * columns +
                 group * 16 + element]);
      maximum = fmaxf(maximum, fabsf(values[element]));
    }
  }
  const __nv_fp8_e4m3 encoded_scale(maximum / 6.0F);
  const float rounded_scale = static_cast<float>(encoded_scale);
  const float inverse = rounded_scale == 0.0F ? 0.0F : 1.0F / rounded_scale;
  auto* destination = packed +
      static_cast<std::size_t>(row) * (columns / 2) + group * 8;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float2 value{values[2 * pair] * inverse,
                       values[2 * pair + 1] * inverse};
    destination[pair] = __nv_fp4x2_e2m1(value).__x;
  }
  // SM120 SFA layout for M <= 128:
  // [K tile][outer M][inner M][inner K].
  const int scale_offset = (group / 4) * 512 + (row % 32) * 16 +
                           ((row / 32) % 4) * 4 + group % 4;
  scale_factors[scale_offset] = encoded_scale.__x;
}

template <int kKeep, int kTileRows = 1024>
__global__ void local_top_vocab_rows(
    const __nv_bfloat16* logits, int* candidate_ids, int rows,
    int physical_rows) {
  static_assert(kTileRows % 256 == 0);
  constexpr int kItems = kTileRows / 256;
  constexpr int kTiles = kVocabRows / kTileRows;
  constexpr int kGroups = 128;
  constexpr int kRowsPerGroup = kVocabRows / kGroups;
  using Sort = cub::BlockRadixSort<float, 256, kItems, int>;
  __shared__ typename Sort::TempStorage storage;
  const int row = blockIdx.x / kTiles;
  const int tile = blockIdx.x % kTiles;
  if (row >= rows) return;
  float keys[kItems];
  int values[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int token = tile * kTileRows + threadIdx.x * kItems + item;
    const int group = token / kRowsPerGroup;
    const int within_group = token % kRowsPerGroup;
    keys[item] = __bfloat162float(
        logits[(static_cast<std::size_t>(group) * physical_rows + row) *
                   kRowsPerGroup + within_group]);
    values[item] = token;
  }
  Sort(storage).SortDescending(keys, values);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int item = 0; item < kItems; ++item)
      if (item < kKeep)
        candidate_ids[(static_cast<std::size_t>(row) * kTiles + tile) *
                          kKeep + item] = values[item];
  }
}

__device__ __forceinline__ bool vocab_candidate_better(
    float left_value, int left_id, float right_value, int right_id) {
  return left_value > right_value ||
         (left_value == right_value && left_id < right_id);
}

// Four finalists do not require a complete radix sort of all 2048 logits.
// Each lane keeps its local four, then four warp/block argmax rounds select
// the tile winners. This retains the exact ordering contract while avoiding
// the radix-sort exchange traffic and register footprint.
template <int kKeep>
__global__ void local_top_vocab_rows_reduce_2048(
    const __nv_bfloat16* logits, int* candidate_ids, int rows,
    int physical_rows, bool row_major, int vocab_rows = kVocabRows) {
  static_assert(kKeep == 2 || kKeep == 4);
  constexpr int kItems = 8;
  constexpr int kRowsPerGroup = 2048;
  const int kTiles = vocab_rows / kRowsPerGroup;
  __shared__ float warp_values[8];
  __shared__ int warp_ids[8];
  __shared__ int winner_id;
  const int row = blockIdx.x / kTiles;
  const int tile = blockIdx.x % kTiles;
  if (row >= rows) return;
  float local_values[kKeep];
  int local_ids[kKeep];
#pragma unroll
  for (int rank = 0; rank < kKeep; ++rank) {
    local_values[rank] = -INFINITY;
    local_ids[rank] = INT_MAX;
  }
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int token = tile * kRowsPerGroup + threadIdx.x * kItems + item;
    const float value = __bfloat162float(
        logits[row_major
            ? static_cast<std::size_t>(row) * vocab_rows + token
            : (static_cast<std::size_t>(tile) * physical_rows + row) *
                  kRowsPerGroup + threadIdx.x * kItems + item]);
    int insert = kKeep;
#pragma unroll
    for (int rank = 0; rank < kKeep; ++rank)
      if (insert == kKeep && vocab_candidate_better(
              value, token, local_values[rank], local_ids[rank]))
        insert = rank;
#pragma unroll
    for (int rank = kKeep - 1; rank > 0; --rank)
      if (insert <= rank - 1) {
        local_values[rank] = local_values[rank - 1];
        local_ids[rank] = local_ids[rank - 1];
      }
    if (insert < kKeep) {
      local_values[insert] = value;
      local_ids[insert] = token;
    }
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int rank = 0; rank < kKeep; ++rank) {
    float value = local_values[0];
    int token = local_ids[0];
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffff, value, offset);
      const int other_token = __shfl_down_sync(0xffffffff, token, offset);
      if (vocab_candidate_better(other_value, other_token, value, token)) {
        value = other_value;
        token = other_token;
      }
    }
    if (lane == 0) {
      warp_values[warp] = value;
      warp_ids[warp] = token;
    }
    __syncthreads();
    if (warp == 0) {
      value = lane < 8 ? warp_values[lane] : -INFINITY;
      token = lane < 8 ? warp_ids[lane] : INT_MAX;
#pragma unroll
      for (int offset = 16; offset; offset >>= 1) {
        const float other_value = __shfl_down_sync(0xffffffff, value, offset);
        const int other_token = __shfl_down_sync(0xffffffff, token, offset);
        if (vocab_candidate_better(other_value, other_token, value, token)) {
          value = other_value;
          token = other_token;
        }
      }
      if (lane == 0) winner_id = token;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      candidate_ids[(static_cast<std::size_t>(row) * kTiles + tile) *
                        kKeep + rank] = winner_id;
    if (local_ids[0] == winner_id) {
#pragma unroll
      for (int item = 0; item < kKeep - 1; ++item) {
        local_values[item] = local_values[item + 1];
        local_ids[item] = local_ids[item + 1];
      }
      local_values[kKeep - 1] = -INFINITY;
      local_ids[kKeep - 1] = INT_MAX;
    }
  }
}

__global__ void correct_nvfp4_vocab_candidates(
    const __nv_bfloat16* exact, const __nv_bfloat16* activations,
    int* candidate_ids, float* corrected, int rows, int candidates,
    int columns, const int* token_map = nullptr) {
  __shared__ float warp_sums[8];
  const int row = blockIdx.x / candidates;
  const int candidate = blockIdx.x % candidates;
  if (row >= rows) return;
  const int candidate_row =
      candidate_ids[static_cast<std::size_t>(row) * candidates +
                    candidate];
  const int vocabulary_row = token_map ? token_map[candidate_row]
                                       : candidate_row;
  const auto* activation = activations +
      static_cast<std::size_t>(row) * columns;
  float sum = 0.0F;
  for (int column = threadIdx.x; column < columns;
       column += blockDim.x)
    sum = fmaf(__bfloat162float(
                   exact[static_cast<std::size_t>(vocabulary_row) *
                             columns + column]),
               __bfloat162float(activation[column]), sum);
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) {
      corrected[static_cast<std::size_t>(row) * candidates +
                  candidate] = __bfloat162float(__float2bfloat16_rn(sum));
      if (token_map)
        candidate_ids[
            static_cast<std::size_t>(row) * candidates + candidate] =
            vocabulary_row;
    }
  }
}

__global__ void gather_hot_vocab_sidecar(
    const std::uint8_t* full_weight, const std::uint8_t* full_scales,
    const int* token_map, std::uint8_t* hot_weight,
    std::uint8_t* hot_scales, int rows, int columns) {
  const int weight_stride = columns / 2;
  const int scale_stride = columns / 16;
  const std::size_t weight_bytes =
      static_cast<std::size_t>(rows) * weight_stride;
  const std::size_t total = weight_bytes +
      static_cast<std::size_t>(rows) * scale_stride;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (index < weight_bytes) {
      const int row = index / weight_stride;
      const int column = index % weight_stride;
      hot_weight[index] = full_weight[
          static_cast<std::size_t>(token_map[row]) * weight_stride + column];
    } else {
      const std::size_t local = index - weight_bytes;
      const int row = local / scale_stride;
      const int column = local % scale_stride;
      hot_scales[local] = full_scales[
          static_cast<std::size_t>(token_map[row]) * scale_stride + column];
    }
  }
}

__global__ void argmax_nvfp4_vocab_candidates(
    const float* corrected, const int* candidate_ids, int* selected,
    int rows, int candidates) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float maximum = -INFINITY;
  int token = 0;
  for (int candidate = threadIdx.x; candidate < candidates;
       candidate += blockDim.x) {
    const float value =
        corrected[static_cast<std::size_t>(row) * candidates +
                  candidate];
    const int other =
        candidate_ids[static_cast<std::size_t>(row) * candidates +
                      candidate];
    if (value > maximum || (value == maximum && other < token)) {
      maximum = value;
      token = other;
    }
  }
  maxima[threadIdx.x] = maximum;
  indices[threadIdx.x] = token;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_value = maxima[threadIdx.x + stride];
      const int other_token = indices[threadIdx.x + stride];
      if (other_value > maxima[threadIdx.x] ||
          (other_value == maxima[threadIdx.x] &&
           other_token < indices[threadIdx.x])) {
        maxima[threadIdx.x] = other_value;
        indices[threadIdx.x] = other_token;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) selected[row] = indices[0];
}

class VocabPlan {
 public:
  VocabPlan(int rows, int columns, int vocab_rows,
            const std::byte* weight, const std::byte* scales,
            const float* alpha, const std::uint8_t* input,
            const std::uint8_t* input_scales, ElementD* output,
            bool single_problem)
      : rows_(rows), groups_(single_problem ? 1 : vocab_rows / 2048),
        shapes_(groups_),
        shapes_device_(groups_), a_(groups_), b_(groups_), sfa_(groups_),
        sfb_(groups_), d_(groups_), alpha_(groups_), stride_a_(groups_),
        stride_b_(groups_), stride_c_(groups_), layout_sfa_(groups_),
        layout_sfb_(groups_) {
    const int rows_per_group = vocab_rows / groups_;
    const ElementType* input_pointer =
        reinterpret_cast<const ElementType*>(input);
    const ElementSF* input_scale_pointer =
        reinterpret_cast<const ElementSF*>(input_scales);
    auto* alpha_pointer = const_cast<float*>(alpha);
    std::vector<const ElementType*> input_pointers(groups_), weight_pointers(groups_);
    std::vector<const ElementSF*> input_scale_pointers(groups_),
        weight_scale_pointers(groups_);
    std::vector<ElementD*> output_pointers(groups_);
    std::vector<float*> alpha_pointers(groups_);
    std::vector<StrideA> strides_a(groups_);
    std::vector<StrideB> strides_b(groups_);
    std::vector<StrideC> strides_c(groups_);
    std::vector<LayoutSFA> layouts_sfa(groups_);
    std::vector<LayoutSFB> layouts_sfb(groups_);
    for (int group = 0; group < groups_; ++group) {
      shapes_[group] = {rows, rows_per_group, columns};
      input_pointers[group] = input_pointer;
      weight_pointers[group] = reinterpret_cast<const ElementType*>(
          weight + static_cast<std::size_t>(group) * rows_per_group *
                       (columns / 2));
      input_scale_pointers[group] = input_scale_pointer;
      weight_scale_pointers[group] = reinterpret_cast<const ElementSF*>(
          scales + static_cast<std::size_t>(group) * rows_per_group *
                       (columns / 16));
      output_pointers[group] = output +
          static_cast<std::size_t>(group) * rows * rows_per_group;
      alpha_pointers[group] = alpha_pointer;
      strides_a[group] = cutlass::make_cute_packed_stride(
          StrideA{}, {rows, columns, 1});
      strides_b[group] = cutlass::make_cute_packed_stride(
          StrideB{}, {rows_per_group, columns, 1});
      strides_c[group] = cutlass::make_cute_packed_stride(
          StrideC{}, {rows, rows_per_group, 1});
      layouts_sfa[group] = ScaleConfig::tile_atom_to_shape_SFA(
          make_shape(rows, rows_per_group, columns, 1));
      layouts_sfb[group] = ScaleConfig::tile_atom_to_shape_SFB(
          make_shape(rows, rows_per_group, columns, 1));
    }
    shapes_device_.copy(shapes_.data(), groups_);
    a_.copy(input_pointers.data(), groups_);
    b_.copy(weight_pointers.data(), groups_);
    sfa_.copy(input_scale_pointers.data(), groups_);
    sfb_.copy(weight_scale_pointers.data(), groups_);
    d_.copy(output_pointers.data(), groups_);
    alpha_.copy(alpha_pointers.data(), groups_);
    stride_a_.copy(strides_a.data(), groups_);
    stride_b_.copy(strides_b.data(), groups_);
    stride_c_.copy(strides_c.data(), groups_);
    layout_sfa_.copy(layouts_sfa.data(), groups_);
    layout_sfb_.copy(layouts_sfb.data(), groups_);

    cutlass::KernelHardwareInfo hardware;
    cuda_check(cudaGetDevice(&hardware.device_id), "cudaGetDevice");
    hardware.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hardware.device_id);
    typename Kernel::TileSchedulerArguments scheduler;
    scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
    typename Kernel::MainloopArguments mainloop{
        a_.get(), stride_a_.get(), b_.get(), stride_b_.get(),
        sfa_.get(), layout_sfa_.get(), sfb_.get(), layout_sfb_.get()};
    typename Kernel::EpilogueArguments epilogue{
        {}, nullptr, stride_c_.get(), d_.get(), stride_c_.get()};
    epilogue.thread.alpha_ptr_array = alpha_.get();
    epilogue.thread.dAlpha = {_0{}, _0{}, 1};
    epilogue.thread.beta = 0.0F;
    typename Kernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups_, shapes_device_.get(), shapes_.data()}, mainloop, epilogue,
        hardware, scheduler};
    workspace_bytes_ = Gemm::get_workspace_size(arguments);
    if (workspace_bytes_)
      cuda_check(cudaMalloc(&workspace_, workspace_bytes_),
                 "cudaMalloc vocabulary CUTLASS workspace");
    cutlass_check(gemm_.can_implement(arguments),
                  "CUTLASS vocabulary can_implement");
    cutlass_check(gemm_.initialize(arguments, workspace_),
                  "CUTLASS vocabulary initialize");
  }

  ~VocabPlan() { if (workspace_) cudaFree(workspace_); }
  void run(cudaStream_t stream) {
    cutlass_check(gemm_.run(stream), "CUTLASS vocabulary run");
  }

 private:
  int rows_{};
  int groups_{};
  std::vector<Shape3> shapes_;
  DeviceBuffer<Shape3> shapes_device_;
  DeviceBuffer<const ElementType*> a_, b_;
  DeviceBuffer<const ElementSF*> sfa_, sfb_;
  DeviceBuffer<ElementD*> d_;
  DeviceBuffer<float*> alpha_;
  DeviceBuffer<StrideA> stride_a_;
  DeviceBuffer<StrideB> stride_b_;
  DeviceBuffer<StrideC> stride_c_;
  DeviceBuffer<LayoutSFA> layout_sfa_;
  DeviceBuffer<LayoutSFB> layout_sfb_;
  Gemm gemm_;
  void* workspace_{};
  std::size_t workspace_bytes_{};
};

}  // namespace g4_cutlass_detail

namespace g4 {

using namespace g4_cutlass_detail;

struct Nvfp4VocabRunner::Impl {
  int maximum_tokens{};
  int columns{};
  int vocab_rows{};
  bool single_problem{};
  // Two exact finalists per 1024-row tile preserved every tested text and
  // multimodal trajectory. One is measurably faster, but changed a real B8
  // text continuation and is therefore diagnostic-only.
  int candidates_per_tile{2};
  const __nv_bfloat16* exact{};
  DeviceBuffer<std::uint8_t> input;
  DeviceBuffer<std::uint8_t> input_scales;
  DeviceBuffer<ElementD> logits;
  DeviceBuffer<int> candidate_ids;
  DeviceBuffer<float> corrected;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> compact_weight;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> compact_scales;
  std::unique_ptr<DeviceBuffer<int>> token_map;
  std::unique_ptr<VocabPlan> plan;
  std::unique_ptr<VocabPlan> alternate_plan;
  int alternate_vocab_rows{};

  Impl(const DeviceModel& exact_model, const std::string& exact_tensor,
       const DeviceModel& nvfp4_model, int tokens,
       std::span<const int> host_token_map, int alternate_rows)
      : maximum_tokens(tokens),
        columns(static_cast<int>(exact_model.tensor(exact_tensor).info->shape.at(1))),
        vocab_rows(host_token_map.empty()
                       ? kVocabRows
                       : static_cast<int>(host_token_map.size())),
        single_problem(
            std::getenv("G4_NVFP4_VOCAB_SINGLE_PROBLEM") != nullptr ||
            (!host_token_map.empty() &&
             std::getenv("G4_DISABLE_HOT_NVFP4_VOCAB_SINGLE_PROBLEM") ==
                 nullptr)),
        input(static_cast<std::size_t>(tokens) * (columns / 2)),
        input_scales(128 * (columns / 16)),
        logits(static_cast<std::size_t>(tokens) * vocab_rows),
        candidate_ids(static_cast<std::size_t>(tokens) *
                      kVocabMaximumCandidates),
        corrected(static_cast<std::size_t>(tokens) *
                  kVocabMaximumCandidates) {
    if (tokens < 1 || tokens > 64)
      throw std::runtime_error("NVFP4 vocabulary rows must be in [1, 64]");
    if (vocab_rows < 2048 || vocab_rows > kVocabRows ||
        vocab_rows % 2048 != 0)
      throw std::runtime_error(
          "NVFP4 hot vocabulary must contain a multiple of 2048 rows");
    if (alternate_rows != 0 &&
        (host_token_map.empty() || alternate_rows < 2048 ||
         alternate_rows >= vocab_rows || alternate_rows % 2048 != 0))
      throw std::runtime_error(
          "alternate NVFP4 vocabulary rows must be a mapped proper prefix");
    alternate_vocab_rows = alternate_rows;
    if (!host_token_map.empty()) {
      std::vector<unsigned char> seen(kVocabRows);
      for (const int token : host_token_map) {
        if (token < 0 || token >= kVocabRows || seen[token])
          throw std::runtime_error("invalid NVFP4 hot-vocabulary token map");
        seen[token] = 1;
      }
    }
    const auto exact_view = exact_model.tensor(exact_tensor);
    const auto weight = nvfp4_model.tensor("weight");
    const auto scales = nvfp4_model.tensor("weight_scale");
    const auto alpha = nvfp4_model.tensor("alpha");
    if (exact_view.info->dtype != "BF16" ||
        (exact_view.info->shape.size() != 2 ||
         exact_view.info->shape[0] != kVocabRows ||
         exact_view.info->shape[1] != static_cast<std::uint64_t>(columns)) ||
        columns % 128 != 0 ||
        weight.info->dtype != "U8" ||
        weight.bytes != static_cast<std::size_t>(kVocabRows) *
                            (columns / 2) ||
        scales.info->dtype != "U8" ||
        scales.bytes != static_cast<std::size_t>(kVocabRows) *
                            (columns / 16) ||
        alpha.info->dtype != "F32" || alpha.bytes != sizeof(float))
      throw std::runtime_error("invalid NVFP4 vocabulary sidecar");
    exact = reinterpret_cast<const __nv_bfloat16*>(exact_view.data);
    const char* candidate_env = exact_tensor == "model.embed_tokens.weight"
        ? std::getenv("G4_ASSISTANT_NVFP4_VOCAB_CANDIDATES_PER_TILE")
        : nullptr;
    if (!candidate_env)
      candidate_env = std::getenv("G4_NVFP4_VOCAB_CANDIDATES_PER_TILE");
    if (const char* value = candidate_env) {
      candidates_per_tile = std::atoi(value);
      if (candidates_per_tile != 1 && candidates_per_tile != 2 &&
          candidates_per_tile != 4)
        throw std::runtime_error(
            "G4_NVFP4_VOCAB_CANDIDATES_PER_TILE must be 1, 2, or 4");
    }
    if (!host_token_map.empty() && candidates_per_tile == 4)
      throw std::runtime_error(
          "hot NVFP4 vocabulary supports 1 or 2 candidates per tile");
    cuda_check(cudaMemset(input_scales.get(), 0,
                          128 * (columns / 16)),
               "clear vocabulary input scales");
    const std::byte* selected_weight = weight.data;
    const std::byte* selected_scales = scales.data;
    if (!host_token_map.empty()) {
      compact_weight = std::make_unique<DeviceBuffer<std::uint8_t>>(
          static_cast<std::size_t>(vocab_rows) * (columns / 2));
      compact_scales = std::make_unique<DeviceBuffer<std::uint8_t>>(
          static_cast<std::size_t>(vocab_rows) * (columns / 16));
      token_map = std::make_unique<DeviceBuffer<int>>(vocab_rows);
      token_map->copy(host_token_map.data(), host_token_map.size());
      const std::size_t gather_bytes =
          static_cast<std::size_t>(vocab_rows) *
          (columns / 2 + columns / 16);
      gather_hot_vocab_sidecar<<<
          static_cast<unsigned int>(std::min<std::size_t>(
              (gather_bytes + 255) / 256, 65535)), 256>>>(
          reinterpret_cast<const std::uint8_t*>(weight.data),
          reinterpret_cast<const std::uint8_t*>(scales.data), token_map->get(),
          compact_weight->get(), compact_scales->get(), vocab_rows, columns);
      cuda_check(cudaGetLastError(), "gather NVFP4 hot vocabulary");
      selected_weight = reinterpret_cast<const std::byte*>(compact_weight->get());
      selected_scales = reinterpret_cast<const std::byte*>(compact_scales->get());
    }
    plan = std::make_unique<VocabPlan>(
        tokens, columns, vocab_rows, selected_weight, selected_scales,
        reinterpret_cast<const float*>(alpha.data), input.get(),
        input_scales.get(), logits.get(), single_problem);
    if (alternate_vocab_rows)
      alternate_plan = std::make_unique<VocabPlan>(
          tokens, columns, alternate_vocab_rows, selected_weight,
          selected_scales, reinterpret_cast<const float*>(alpha.data),
          input.get(), input_scales.get(), logits.get(), single_problem);
  }

  void launch(const __nv_bfloat16* activations, int* selected, int tokens,
              cudaStream_t stream, int requested_vocab_rows) const {
    if (tokens < 1 || tokens > maximum_tokens)
      throw std::runtime_error("NVFP4 vocabulary active rows exceed plan");
    const int active_vocab_rows = requested_vocab_rows == 0
        ? vocab_rows
        : requested_vocab_rows;
    if (active_vocab_rows != vocab_rows &&
        active_vocab_rows != alternate_vocab_rows)
      throw std::runtime_error("unsupported active NVFP4 vocabulary rows");
    const int groups = columns / 16;
    quantize_vocab_rows<<<
        (maximum_tokens * groups + 255) / 256, 256, 0, stream>>>(
        activations, input.get(), input_scales.get(), tokens,
        maximum_tokens, columns);
    (active_vocab_rows == vocab_rows ? plan.get() : alternate_plan.get())
        ->run(stream);
    const bool hot_vocab = token_map != nullptr;
    const bool grouped_top = candidates_per_tile <= 2 &&
        std::getenv("G4_DISABLE_NVFP4_VOCAB_GROUPED_TOP") == nullptr;
    if (hot_vocab) {
      if (candidates_per_tile == 1)
        local_top_vocab_rows_reduce_2048<2>
          <<<tokens * (active_vocab_rows / 2048), 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.get()),
                candidate_ids.get(), tokens, maximum_tokens, single_problem,
                active_vocab_rows);
      else
        local_top_vocab_rows_reduce_2048<4>
          <<<tokens * (active_vocab_rows / 2048), 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.get()),
                candidate_ids.get(), tokens, maximum_tokens, single_problem,
                active_vocab_rows);
    } else if (grouped_top) {
      if (std::getenv("G4_DISABLE_NVFP4_VOCAB_REDUCTION_TOP") == nullptr) {
        if (candidates_per_tile == 1)
          local_top_vocab_rows_reduce_2048<2>
              <<<tokens * 128, 256, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(logits.get()),
                  candidate_ids.get(), tokens, maximum_tokens,
                  single_problem);
        else
          local_top_vocab_rows_reduce_2048<4>
              <<<tokens * 128, 256, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(logits.get()),
                  candidate_ids.get(), tokens, maximum_tokens,
                  single_problem);
      }
      else
        if (candidates_per_tile == 1)
          local_top_vocab_rows<2, 2048><<<tokens * 128, 256, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(logits.get()),
              candidate_ids.get(), tokens, maximum_tokens);
        else
          local_top_vocab_rows<4, 2048><<<tokens * 128, 256, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(logits.get()),
              candidate_ids.get(), tokens, maximum_tokens);
    }
    else if (candidates_per_tile == 1)
      local_top_vocab_rows<1><<<tokens * 256, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(logits.get()),
          candidate_ids.get(), tokens, maximum_tokens);
    else if (candidates_per_tile == 2)
      local_top_vocab_rows<2><<<tokens * 256, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(logits.get()),
          candidate_ids.get(), tokens, maximum_tokens);
    else
      local_top_vocab_rows<4><<<tokens * 256, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(logits.get()),
          candidate_ids.get(), tokens, maximum_tokens);
    const int candidates = candidates_per_tile * (active_vocab_rows / 1024);
    correct_nvfp4_vocab_candidates<<<tokens * candidates, 256, 0, stream>>>(
        exact, activations, candidate_ids.get(), corrected.get(), tokens,
        candidates, columns, token_map ? token_map->get() : nullptr);
    argmax_nvfp4_vocab_candidates<<<tokens, 256, 0, stream>>>(
        corrected.get(), candidate_ids.get(), selected, tokens, candidates);
    cuda_check(cudaGetLastError(), "NVFP4 vocabulary launch");
  }
};

Nvfp4VocabRunner::Nvfp4VocabRunner(
    const DeviceModel& exact_model, const std::string& exact_tensor,
    const DeviceModel& nvfp4_model, int maximum_tokens,
    std::span<const int> token_map, int alternate_vocab_rows)
    : impl_(std::make_unique<Impl>(exact_model, exact_tensor, nvfp4_model,
                                  maximum_tokens, token_map,
                                  alternate_vocab_rows)) {}
Nvfp4VocabRunner::~Nvfp4VocabRunner() = default;
void Nvfp4VocabRunner::launch_batch(const void* activations,
                                    int* selected_tokens, int tokens,
                                    cudaStream_t stream,
                                    int active_vocab_rows) const {
  impl_->launch(static_cast<const __nv_bfloat16*>(activations),
                selected_tokens, tokens, stream, active_vocab_rows);
}

Nvfp4VocabBenchmark benchmark_nvfp4_vocab(
    const DeviceModel& exact_model, const DeviceModel& int8_model,
    const DeviceModel& nvfp4_model, int rows,
    const std::string& exact_tensor, std::span<const int> hot_token_map) {
  if (rows < 1 || rows > 64)
    throw std::runtime_error("NVFP4 vocabulary benchmark rows must be 1..64");
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "create vocabulary benchmark stream");
  const auto exact_view = exact_model.tensor(exact_tensor);
  if (exact_view.info->shape.size() != 2)
    throw std::runtime_error("NVFP4 vocabulary benchmark requires a matrix");
  const int columns = static_cast<int>(exact_view.info->shape[1]);
  CompressedVocabRunner int8(exact_model, exact_tensor, int8_model, stream);
  Nvfp4VocabRunner nvfp4(exact_model, exact_tensor, nvfp4_model, rows);
  std::unique_ptr<Nvfp4VocabRunner> hot_nvfp4;
  if (!hot_token_map.empty())
    hot_nvfp4 = std::make_unique<Nvfp4VocabRunner>(
        exact_model, exact_tensor, nvfp4_model, rows, hot_token_map);
  DeviceBuffer<__nv_bfloat16> activations(
      static_cast<std::size_t>(rows) * columns);
  DeviceBuffer<int> int8_tokens(rows), nvfp4_tokens(rows), hot_tokens(rows);
  std::vector<__nv_bfloat16> host(
      static_cast<std::size_t>(rows) * columns);
  for (std::size_t index = 0; index < host.size(); ++index)
    host[index] = __float2bfloat16_rn(
        std::sin(index * 0.013F) * 0.7F + std::cos(index * 0.007F) * 0.2F);
  activations.copy(host.data(), host.size());
  int8.launch_batch(activations.get(), int8_tokens.get(), rows, stream);
  nvfp4.launch_batch(activations.get(), nvfp4_tokens.get(), rows, stream);
  if (hot_nvfp4)
    hot_nvfp4->launch_batch(activations.get(), hot_tokens.get(), rows, stream);
  cuda_check(cudaStreamSynchronize(stream), "warm vocabulary benchmark");
  const float int8_us = time_launch(
      stream, 20,
      [&] { int8.launch_batch(activations.get(), int8_tokens.get(), rows,
                              stream); });
  const float nvfp4_us = time_launch(
      stream, 20,
      [&] { nvfp4.launch_batch(activations.get(), nvfp4_tokens.get(), rows,
                               stream); });
  const float hot_nvfp4_us = hot_nvfp4
      ? time_launch(stream, 20, [&] {
          hot_nvfp4->launch_batch(activations.get(), hot_tokens.get(), rows,
                                  stream);
        })
      : 0.0F;
  std::vector<int> host_int8(rows), host_nvfp4(rows), host_hot(rows);
  cuda_check(cudaMemcpy(host_int8.data(), int8_tokens.get(), rows * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "copy INT8 vocabulary tokens");
  cuda_check(cudaMemcpy(host_nvfp4.data(), nvfp4_tokens.get(),
                        rows * sizeof(int), cudaMemcpyDeviceToHost),
             "copy NVFP4 vocabulary tokens");
  if (hot_nvfp4)
    cuda_check(cudaMemcpy(host_hot.data(), hot_tokens.get(),
                          rows * sizeof(int), cudaMemcpyDeviceToHost),
               "copy hot NVFP4 vocabulary tokens");
  int mismatches = 0;
  int hot_mismatches = 0;
  for (int row = 0; row < rows; ++row) {
    mismatches += host_int8[row] != host_nvfp4[row];
    hot_mismatches += hot_nvfp4 && host_int8[row] != host_hot[row];
  }
  cudaStreamDestroy(stream);
  return {int8_us, nvfp4_us, hot_nvfp4_us, rows, mismatches,
          hot_mismatches};
}

struct Nvfp4ExpertRunner::Impl {
  int maximum_tokens{};
  int routes{};
  DeviceBuffer<std::uint8_t> input;
  DeviceBuffer<std::uint8_t> input_scales;
  DeviceBuffer<ElementD> w13_output;
  DeviceBuffer<std::uint8_t> intermediate;
  DeviceBuffer<std::uint8_t> intermediate_scales;
  DeviceBuffer<ElementD> w2_output;
  DeviceBuffer<int> sorted_expert_ids;
  DeviceBuffer<int> route_indices;
  DeviceBuffer<int> sorted_route_indices;
  DeviceBuffer<int> inverse_routes;
  DeviceBuffer<int> expert_offsets;
  DeviceBuffer<int> scale_row_offsets;
  std::unique_ptr<DeviceBuffer<std::uint8_t>> sort_workspace;
  std::size_t sort_workspace_bytes{};
  const float* g1_input_scale{};
  const float* g2_input_scale{};
  std::unique_ptr<GroupedPlan> w13;
  std::unique_ptr<GroupedPlan> w2;
  std::unique_ptr<DynamicExpertPlan> dynamic_w13;
  std::unique_ptr<DynamicExpertPlan> dynamic_w2;
  bool fused_route_sort{};
  bool route_indices_initialized{};

  Impl(const DeviceModel& experts, int layer, int tokens)
      : maximum_tokens(tokens), routes(tokens * 8),
        input(routes * kPackedInputPerRoute),
        input_scales(static_cast<std::size_t>(routes + kExperts * 127) *
                     (2816 / 16)),
        w13_output(routes * 1408), intermediate(routes * (704 / 2)),
        intermediate_scales(static_cast<std::size_t>(routes + kExperts * 127) *
                            (704 / 16)),
        w2_output(routes * 2816), sorted_expert_ids(routes),
        route_indices(routes), sorted_route_indices(routes),
        inverse_routes(routes), expert_offsets(kExperts),
        scale_row_offsets(kExperts),
        // Shared-memory sorters cover both decode and 4096-route prefill sets.
        // Five-token B1 verification has only 40 routes and is faster on the
        // 128-element bitonic network. B8 (40 rows) and prefill use the fused
        // counting/block-radix paths.
        fused_route_sort(tokens >= 16 &&
                         std::getenv("G4_DISABLE_FUSED_ROUTE_SORT") == nullptr) {
    if (layer < 0 || layer >= 30) throw std::runtime_error("layer must be in [0, 29]");
    if (tokens < 1 || tokens > 4608)
      throw std::runtime_error("expert runner tokens must be in [1, 4608]");
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    const auto w13_weight = experts.tensor(prefix + "w13.weight");
    const auto w13_scale = experts.tensor(prefix + "w13.weight_scale");
    const auto g1_alpha = experts.tensor(prefix + "g1.alpha");
    const auto g1_quant = experts.tensor(prefix + "g1.input_scale_quant");
    const auto w2_weight = experts.tensor(prefix + "w2.weight");
    const auto w2_scale = experts.tensor(prefix + "w2.weight_scale");
    const auto g2_alpha = experts.tensor(prefix + "g2.alpha");
    const auto g2_quant = experts.tensor(prefix + "g2.input_scale_quant");
    g1_input_scale = reinterpret_cast<const float*>(g1_quant.data);
    g2_input_scale = reinterpret_cast<const float*>(g2_quant.data);
    cuda_check(cudaMemset(input.get(), 0,
                          static_cast<std::size_t>(routes) *
                              kPackedInputPerRoute),
               "cudaMemset expert input");
    cuda_check(cudaMemset(intermediate.get(), 0,
                          static_cast<std::size_t>(routes) * (704 / 2)),
               "cudaMemset expert intermediate");
    const std::size_t input_scale_bytes =
        static_cast<std::size_t>(routes + kExperts * 127) * (2816 / 16);
    const std::size_t intermediate_scale_bytes =
        static_cast<std::size_t>(routes + kExperts * 127) * (704 / 16);
    cuda_check(cudaMemset(input_scales.get(), 0, input_scale_bytes),
               "cudaMemset dynamic input scales");
    cuda_check(cudaMemset(intermediate_scales.get(), 0,
                          intermediate_scale_bytes),
               "cudaMemset dynamic intermediate scales");
    if (maximum_tokens == 1) {
      w13 = std::make_unique<GroupedPlan>(
          1408, 2816, w13_weight.data, w13_scale.data,
          reinterpret_cast<const float*>(g1_alpha.data), input.get(),
          input_scales.get(), w13_output.get(), routes);
      w2 = std::make_unique<GroupedPlan>(
          2816, 704, w2_weight.data, w2_scale.data,
          reinterpret_cast<const float*>(g2_alpha.data), intermediate.get(),
          intermediate_scales.get(), w2_output.get(), routes);
    } else {
      dynamic_w13 = std::make_unique<DynamicExpertPlan>(
          1408, 2816, routes, w13_weight.data, w13_scale.data,
          reinterpret_cast<const float*>(g1_alpha.data), input.get(),
          input_scales.get(), w13_output.get(), expert_offsets.get(),
          scale_row_offsets.get());
      dynamic_w2 = std::make_unique<DynamicExpertPlan>(
          2816, 704, routes, w2_weight.data, w2_scale.data,
          reinterpret_cast<const float*>(g2_alpha.data), intermediate.get(),
          intermediate_scales.get(), w2_output.get(), expert_offsets.get(),
          scale_row_offsets.get());
      cuda_check(cub::DeviceRadixSort::SortPairs(
                     nullptr, sort_workspace_bytes,
                     static_cast<const int*>(nullptr), sorted_expert_ids.get(),
                     route_indices.get(),
                     sorted_route_indices.get(), routes, 0, 7),
                 "query route sort workspace");
      sort_workspace =
          std::make_unique<DeviceBuffer<std::uint8_t>>(sort_workspace_bytes);
    }
  }

  void launch(const void* activation, const int* expert_ids,
              const float* route_weights, void* output, int tokens,
              cudaStream_t stream, bool reduce_routes = true) {
    if (tokens < 1 || tokens > maximum_tokens ||
        (maximum_tokens == 1 && tokens != maximum_tokens))
      throw std::runtime_error("expert runner token count exceeds its plan");
    const int active_routes = tokens * 8;
    if (maximum_tokens > 1 && !route_indices_initialized) {
      initialize_route_indices<<<(routes + 255) / 256, 256, 0, stream>>>(
          route_indices.get(), routes);
      cuda_check(cudaGetLastError(), "initialize static route indices");
      route_indices_initialized = true;
    }
    const int* ordered_experts = expert_ids;
    const int* ordered_routes = nullptr;
    const int* inverse = nullptr;
    int* quantize_inverse = nullptr;
    if (maximum_tokens > 1) {
      ordered_experts = sorted_expert_ids.get();
      ordered_routes = sorted_route_indices.get();
      inverse = inverse_routes.get();
      // The single-block 4096-item radix network wins through roughly 320
      // tokens on this SM120; above that, the device-wide CUB path is slightly
      // faster despite its extra launches.
      const bool use_fused_route_sort =
          fused_route_sort && active_routes <= 320 * 8;
      if (active_routes <= kExperts || use_fused_route_sort) {
        dynamic_w13->sort_and_configure(
            expert_ids, active_routes, sorted_expert_ids.get(),
            sorted_route_indices.get(), inverse_routes.get(), stream,
            use_fused_route_sort);
      } else {
        cuda_check(cub::DeviceRadixSort::SortPairs(
                       sort_workspace->get(), sort_workspace_bytes,
                       expert_ids, sorted_expert_ids.get(),
                       route_indices.get(),
                       sorted_route_indices.get(),
                       active_routes, 0, 7, stream),
                   "sort expert routes");
        dynamic_w13->configure(ordered_experts, active_routes, stream);
        quantize_inverse = inverse_routes.get();
      }
      quantize_bf16_sorted_moe<<<active_routes, 2816 / 16, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation), g1_input_scale,
          ordered_experts, ordered_routes, expert_offsets.get(), input.get(),
          input_scales.get(), scale_row_offsets.get(), 2816, active_routes,
          quantize_inverse);
      dynamic_w13->run(stream);
      dynamic_w2->configure_and_quantize_geglu(
          reinterpret_cast<const __nv_bfloat16*>(w13_output.get()),
          g2_input_scale, ordered_experts, expert_offsets.get(),
          intermediate.get(), intermediate_scales.get(),
          active_routes, stream);
      dynamic_w2->run(stream);
      if (!reduce_routes) return;
      reduce_expert_routes<<<tokens, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(w2_output.get()), route_weights,
          static_cast<__nv_bfloat16*>(output), tokens, inverse);
      return;
    }
    quantize_bf16_routes<<<routes, 2816 / 16, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation), g1_input_scale,
        ordered_experts, input.get(), input_scales.get(), 2816,
        w13->weight(), w13->scales(), w13->alphas(), w13->weight_ptrs(),
        w13->scale_ptrs(), w13->alpha_ptrs(), 1408, routes, ordered_routes);
    w13->run_configured(stream);
    geglu_quantize_routes<<<routes, 704 / 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w13_output.get()), g2_input_scale,
        ordered_experts, intermediate.get(), intermediate_scales.get(),
        w2->weight(), w2->scales(), w2->alphas(), w2->weight_ptrs(),
        w2->scale_ptrs(), w2->alpha_ptrs(), routes);
    w2->run_configured(stream);
    reduce_expert_routes<<<tokens, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w2_output.get()), route_weights,
        static_cast<__nv_bfloat16*>(output), tokens, inverse);
  }

};

Nvfp4ExpertRunner::Nvfp4ExpertRunner(const DeviceModel& experts, int layer,
                                     int maximum_tokens)
    : impl_(std::make_unique<Impl>(experts, layer, maximum_tokens)) {}
Nvfp4ExpertRunner::~Nvfp4ExpertRunner() = default;
Nvfp4ExpertRunner::Nvfp4ExpertRunner(Nvfp4ExpertRunner&&) noexcept = default;
Nvfp4ExpertRunner& Nvfp4ExpertRunner::operator=(Nvfp4ExpertRunner&&) noexcept = default;

void Nvfp4ExpertRunner::launch(const void* input, const int* expert_ids,
                               const float* route_weights, void* output,
                               cudaStream_t stream) const {
  impl_->launch(input, expert_ids, route_weights, output, 1, stream);
}

void Nvfp4ExpertRunner::launch_batch(const void* input, const int* expert_ids,
                                     const float* route_weights, void* output,
                                     int tokens, cudaStream_t stream) const {
  impl_->launch(input, expert_ids, route_weights, output, tokens, stream);
}

void Nvfp4ExpertRunner::launch_batch_unreduced(
    const void* input, const int* expert_ids, int tokens,
    cudaStream_t stream) const {
  impl_->launch(input, expert_ids, nullptr, nullptr, tokens, stream, false);
}

const void* Nvfp4ExpertRunner::routed_output() const {
  return impl_->w2_output.get();
}

const int* Nvfp4ExpertRunner::inverse_routes() const {
  return impl_->inverse_routes.get();
}

ExpertRepeatTestResult test_nvfp4_expert_repeatability(
    const DeviceModel& experts, int layer, int tokens, int repetitions) {
  if (layer < 0 || layer >= 30 || tokens < 1 || tokens > 512 ||
      repetitions < 2 || repetitions > 100)
    throw std::runtime_error("invalid expert repeatability test geometry");
  constexpr int kHidden = 2816;
  constexpr int kTop = 8;
  const int routes = tokens * kTop;
  const bool identical_rows =
      std::getenv("G4_EXPERT_IDENTICAL_ROWS") != nullptr;
  std::vector<__nv_bfloat16> host_activation(
      static_cast<std::size_t>(tokens) * kHidden);
  for (std::size_t index = 0; index < host_activation.size(); ++index) {
    const std::size_t source = identical_rows ? index % kHidden : index;
    const float value = std::sin(static_cast<float>(source) * 0.0017F) * 0.7F +
                        std::cos(static_cast<float>(source) * 0.00031F) * 0.2F;
    host_activation[index] = __float2bfloat16_rn(value);
  }
  std::vector<int> host_ids(routes);
  std::vector<float> host_weights(routes);
  constexpr std::array<float, kTop> kWeights{
      0.24F, 0.19F, 0.15F, 0.12F, 0.10F, 0.08F, 0.07F, 0.05F};
  for (int token = 0; token < tokens; ++token) {
    for (int route = 0; route < kTop; ++route) {
      // Exercise the compacted dynamic plan: a deliberately skewed subset
      // leaves most experts inactive and gives active experts unequal M sizes.
      const unsigned mixed = static_cast<unsigned>(identical_rows ? 0 : token) *
                                 2654435761U +
                             static_cast<unsigned>(route) * 2246822519U;
      host_ids[token * kTop + route] = route == 0
          ? 0
          : 1 + static_cast<int>(((mixed >> 24) + route * route) % 36);
      host_weights[token * kTop + route] = kWeights[route];
    }
  }
  DeviceBuffer<__nv_bfloat16> activation(host_activation.size());
  DeviceBuffer<int> ids(host_ids.size());
  DeviceBuffer<float> weights(host_weights.size());
  DeviceBuffer<__nv_bfloat16> output(host_activation.size());
  activation.copy(host_activation.data(), host_activation.size());
  ids.copy(host_ids.data(), host_ids.size());
  weights.copy(host_weights.data(), host_weights.size());
  Nvfp4ExpertRunner runner(experts, layer, tokens > 5 ? 512 : tokens);
  cudaStream_t stream{};
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "cudaStreamCreate expert repeatability");
  auto launch = [&] {
    runner.launch_batch(activation.get(), ids.get(), weights.get(), output.get(),
                        tokens, stream);
  };
  const bool compare_baseline = std::getenv("G4_EXPERT_COMPARE_BASELINE") != nullptr;
  const bool was_wide = std::getenv("G4_EXPERT_WIDE_TILES") != nullptr;
  const bool was_narrow = std::getenv("G4_EXPERT_NARROW_TILES") != nullptr;
  const bool was_cooperative = std::getenv("G4_EXPERT_COOPERATIVE") != nullptr;
  if (compare_baseline) {
    unsetenv("G4_EXPERT_WIDE_TILES");
    unsetenv("G4_EXPERT_NARROW_TILES");
    unsetenv("G4_EXPERT_COOPERATIVE");
  }
  launch();
  cuda_check(cudaStreamSynchronize(stream), "warm expert repeatability");
  std::vector<__nv_bfloat16> reference(host_activation.size());
  cuda_check(cudaMemcpy(reference.data(), output.get(),
                        reference.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost), "copy expert reference");
  if (compare_baseline) {
    if (was_wide) setenv("G4_EXPERT_WIDE_TILES", "1", 1);
    if (was_narrow) setenv("G4_EXPERT_NARROW_TILES", "1", 1);
    if (was_cooperative) setenv("G4_EXPERT_COOPERATIVE", "1", 1);
  }
  float maximum_difference = 0.0F;
  std::size_t different_values = 0;
  if (identical_rows) {
    for (int token = 1; token < tokens; ++token)
      for (int col = 0; col < kHidden; ++col) {
        const auto& first = reference[col];
        const auto& other = reference[static_cast<std::size_t>(token) *
                                      kHidden + col];
        if (std::memcmp(&first, &other, sizeof(__nv_bfloat16)) != 0) {
          ++different_values;
          maximum_difference = std::max(
              maximum_difference,
              std::abs(__bfloat162float(first) - __bfloat162float(other)));
        }
      }
  }
  for (int repetition = 1; repetition < repetitions; ++repetition) {
    launch();
    cuda_check(cudaStreamSynchronize(stream), "run expert repeatability");
    std::vector<__nv_bfloat16> observed(reference.size());
    cuda_check(cudaMemcpy(observed.data(), output.get(),
                          observed.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost), "copy repeated expert output");
    for (std::size_t index = 0; index < observed.size(); ++index) {
      if (std::memcmp(&reference[index], &observed[index],
                      sizeof(__nv_bfloat16)) != 0) {
        ++different_values;
        maximum_difference = std::max(
            maximum_difference,
            std::abs(__bfloat162float(reference[index]) -
                     __bfloat162float(observed[index])));
      }
    }
  }
  const float microseconds = time_launch(stream, 10, launch);
  cuda_check(cudaStreamSynchronize(stream), "time expert repeatability");
  cudaStreamDestroy(stream);
  double checksum = 0.0;
  for (const auto value : reference)
    checksum += std::abs(__bfloat162float(value));
  return {microseconds, static_cast<float>(checksum), maximum_difference,
          different_values, repetitions, different_values == 0};
}

CutlassGroupedTestResult benchmark_cutlass_grouped_w13(
    const DeviceModel& experts, int layer) {
  if (layer < 0 || layer >= 30) throw std::runtime_error("layer must be in [0, 29]");
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const auto weight = experts.tensor(prefix + "w13.weight");
  const auto scales = experts.tensor(prefix + "w13.weight_scale");
  const auto alphas = experts.tensor(prefix + "g1.alpha");
  const auto input_scale_quant = experts.tensor(prefix + "g1.input_scale_quant");
  const auto w2_weight = experts.tensor(prefix + "w2.weight");
  const auto w2_scales = experts.tensor(prefix + "w2.weight_scale");
  const auto w2_alphas = experts.tensor(prefix + "g2.alpha");
  const auto w2_input_scale_quant = experts.tensor(prefix + "g2.input_scale_quant");
  if (weight.bytes != static_cast<std::size_t>(kExperts) * kN * kK / 2 ||
      scales.bytes != static_cast<std::size_t>(kExperts) * kN * (kK / 16) ||
      alphas.bytes != kExperts * sizeof(float) ||
      input_scale_quant.bytes != kExperts * sizeof(float) ||
      w2_weight.bytes != static_cast<std::size_t>(kExperts) * 2816 * (704 / 2) ||
      w2_scales.bytes != static_cast<std::size_t>(kExperts) * 2816 * (704 / 16) ||
      w2_alphas.bytes != kExperts * sizeof(float) ||
      w2_input_scale_quant.bytes != kExperts * sizeof(float))
    throw std::runtime_error("unexpected W13 sidecar geometry");

  DeviceBuffer<std::uint8_t> input(kRoutes * kPackedInputPerRoute);
  DeviceBuffer<std::uint8_t> input_scales(kRoutes * kScaleInputPerRoute);
  DeviceBuffer<__nv_bfloat16> activation(kK);
  DeviceBuffer<ElementD> output(kRoutes * kN);
  constexpr std::size_t kW2ScalePerRoute = 128 * (704 / 16);
  DeviceBuffer<std::uint8_t> intermediate(kRoutes * (704 / 2));
  DeviceBuffer<std::uint8_t> intermediate_scales(kRoutes * kW2ScalePerRoute);
  DeviceBuffer<ElementD> down_output(kRoutes * 2816);
  DeviceBuffer<__nv_bfloat16> reduced_output(2816);
  cuda_check(cudaMemset(intermediate_scales.get(), 0,
                        kRoutes * kW2ScalePerRoute), "cudaMemset W2 FP8 scales");
  cuda_check(cudaMemset(input_scales.get(), 0, kRoutes * kScaleInputPerRoute),
             "cudaMemset FP8 scales");
  std::vector<__nv_bfloat16> host_activation(kK);
  for (int i = 0; i < kK; ++i)
    host_activation[i] = __float2bfloat16_rn(
        std::sin(i * 0.019F) + 0.31F * std::cos(i * 0.007F));
  activation.copy(host_activation.data(), host_activation.size());

  const std::array<int, kRoutes> expert_ids{3, 17, 31, 52, 73, 91, 105, 127};
  DeviceBuffer<int> d_expert_ids(kRoutes);
  d_expert_ids.copy(expert_ids.data(), expert_ids.size());
  std::array<Shape3, kRoutes> shapes;
  std::array<const ElementType*, kRoutes> ptr_a;
  std::array<const ElementType*, kRoutes> ptr_b;
  std::array<const ElementSF*, kRoutes> ptr_sfa;
  std::array<const ElementSF*, kRoutes> ptr_sfb;
  std::array<ElementD*, kRoutes> ptr_d;
  std::array<float*, kRoutes> ptr_alpha;
  std::array<StrideA, kRoutes> stride_a;
  std::array<StrideB, kRoutes> stride_b;
  std::array<StrideC, kRoutes> stride_c;
  std::array<LayoutSFA, kRoutes> layout_sfa;
  std::array<LayoutSFB, kRoutes> layout_sfb;

  for (int route = 0; route < kRoutes; ++route) {
    const int expert = expert_ids[route];
    shapes[route] = {kM, kN, kK};
    ptr_a[route] = reinterpret_cast<const ElementType*>(
        input.get() + route * kPackedInputPerRoute);
    ptr_b[route] = reinterpret_cast<const ElementType*>(
        weight.data + static_cast<std::size_t>(expert) * kN * kK / 2);
    ptr_sfa[route] = reinterpret_cast<const ElementSF*>(
        input_scales.get() + route * kScaleInputPerRoute);
    ptr_sfb[route] = reinterpret_cast<const ElementSF*>(
        scales.data + static_cast<std::size_t>(expert) * kN * (kK / 16));
    ptr_d[route] = output.get() + route * kN;
    ptr_alpha[route] = reinterpret_cast<float*>(const_cast<std::byte*>(alphas.data)) + expert;
    stride_a[route] = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
    stride_b[route] = cutlass::make_cute_packed_stride(StrideB{}, {kN, kK, 1});
    stride_c[route] = cutlass::make_cute_packed_stride(StrideC{}, {kM, kN, 1});
    layout_sfa[route] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(kM, kN, kK, 1));
    layout_sfb[route] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(kM, kN, kK, 1));
  }

  DeviceBuffer<Shape3> d_shapes(kRoutes); d_shapes.copy(shapes.data(), kRoutes);
  DeviceBuffer<const ElementType*> d_a(kRoutes); d_a.copy(ptr_a.data(), kRoutes);
  DeviceBuffer<const ElementType*> d_b(kRoutes); d_b.copy(ptr_b.data(), kRoutes);
  DeviceBuffer<const ElementSF*> d_sfa(kRoutes); d_sfa.copy(ptr_sfa.data(), kRoutes);
  DeviceBuffer<const ElementSF*> d_sfb(kRoutes); d_sfb.copy(ptr_sfb.data(), kRoutes);
  DeviceBuffer<ElementD*> d_d(kRoutes); d_d.copy(ptr_d.data(), kRoutes);
  DeviceBuffer<float*> d_alpha(kRoutes); d_alpha.copy(ptr_alpha.data(), kRoutes);
  DeviceBuffer<StrideA> d_stride_a(kRoutes); d_stride_a.copy(stride_a.data(), kRoutes);
  DeviceBuffer<StrideB> d_stride_b(kRoutes); d_stride_b.copy(stride_b.data(), kRoutes);
  DeviceBuffer<StrideC> d_stride_c(kRoutes); d_stride_c.copy(stride_c.data(), kRoutes);
  DeviceBuffer<LayoutSFA> d_layout_sfa(kRoutes); d_layout_sfa.copy(layout_sfa.data(), kRoutes);
  DeviceBuffer<LayoutSFB> d_layout_sfb(kRoutes); d_layout_sfb.copy(layout_sfb.data(), kRoutes);

  std::array<Shape3, kRoutes> shapes2;
  std::array<const ElementType*, kRoutes> ptr_a2;
  std::array<const ElementType*, kRoutes> ptr_b2;
  std::array<const ElementSF*, kRoutes> ptr_sfa2;
  std::array<const ElementSF*, kRoutes> ptr_sfb2;
  std::array<ElementD*, kRoutes> ptr_d2;
  std::array<float*, kRoutes> ptr_alpha2;
  std::array<StrideA, kRoutes> stride_a2;
  std::array<StrideB, kRoutes> stride_b2;
  std::array<StrideC, kRoutes> stride_c2;
  std::array<LayoutSFA, kRoutes> layout_sfa2;
  std::array<LayoutSFB, kRoutes> layout_sfb2;
  for (int route = 0; route < kRoutes; ++route) {
    const int expert = expert_ids[route];
    shapes2[route] = {1, 2816, 704};
    ptr_a2[route] = reinterpret_cast<const ElementType*>(intermediate.get() + route * 352);
    ptr_b2[route] = reinterpret_cast<const ElementType*>(
        w2_weight.data + static_cast<std::size_t>(expert) * 2816 * 352);
    ptr_sfa2[route] = reinterpret_cast<const ElementSF*>(
        intermediate_scales.get() + route * kW2ScalePerRoute);
    ptr_sfb2[route] = reinterpret_cast<const ElementSF*>(
        w2_scales.data + static_cast<std::size_t>(expert) * 2816 * 44);
    ptr_d2[route] = down_output.get() + route * 2816;
    ptr_alpha2[route] = reinterpret_cast<float*>(const_cast<std::byte*>(w2_alphas.data)) + expert;
    stride_a2[route] = cutlass::make_cute_packed_stride(StrideA{}, {1, 704, 1});
    stride_b2[route] = cutlass::make_cute_packed_stride(StrideB{}, {2816, 704, 1});
    stride_c2[route] = cutlass::make_cute_packed_stride(StrideC{}, {1, 2816, 1});
    layout_sfa2[route] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(1, 2816, 704, 1));
    layout_sfb2[route] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, 2816, 704, 1));
  }
  DeviceBuffer<Shape3> d_shapes2(kRoutes); d_shapes2.copy(shapes2.data(), kRoutes);
  DeviceBuffer<const ElementType*> d_a2(kRoutes); d_a2.copy(ptr_a2.data(), kRoutes);
  DeviceBuffer<const ElementType*> d_b2(kRoutes); d_b2.copy(ptr_b2.data(), kRoutes);
  DeviceBuffer<const ElementSF*> d_sfa2(kRoutes); d_sfa2.copy(ptr_sfa2.data(), kRoutes);
  DeviceBuffer<const ElementSF*> d_sfb2(kRoutes); d_sfb2.copy(ptr_sfb2.data(), kRoutes);
  DeviceBuffer<ElementD*> d_d2(kRoutes); d_d2.copy(ptr_d2.data(), kRoutes);
  DeviceBuffer<float*> d_alpha2(kRoutes); d_alpha2.copy(ptr_alpha2.data(), kRoutes);
  DeviceBuffer<StrideA> d_stride_a2(kRoutes); d_stride_a2.copy(stride_a2.data(), kRoutes);
  DeviceBuffer<StrideB> d_stride_b2(kRoutes); d_stride_b2.copy(stride_b2.data(), kRoutes);
  DeviceBuffer<StrideC> d_stride_c2(kRoutes); d_stride_c2.copy(stride_c2.data(), kRoutes);
  DeviceBuffer<LayoutSFA> d_layout_sfa2(kRoutes); d_layout_sfa2.copy(layout_sfa2.data(), kRoutes);
  DeviceBuffer<LayoutSFB> d_layout_sfb2(kRoutes); d_layout_sfb2.copy(layout_sfb2.data(), kRoutes);

  cutlass::KernelHardwareInfo hardware;
  cuda_check(cudaGetDevice(&hardware.device_id), "cudaGetDevice");
  hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
      hardware.device_id);
  typename Kernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  typename Kernel::MainloopArguments mainloop{
      d_a.get(), d_stride_a.get(), d_b.get(), d_stride_b.get(),
      d_sfa.get(), d_layout_sfa.get(), d_sfb.get(), d_layout_sfb.get()};
  typename Kernel::EpilogueArguments epilogue{
      {}, nullptr, d_stride_c.get(), d_d.get(), d_stride_c.get()};
  epilogue.thread.alpha_ptr_array = d_alpha.get();
  epilogue.thread.dAlpha = {_0{}, _0{}, 1};
  epilogue.thread.beta = 0.0F;
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kRoutes, d_shapes.get(), shapes.data()}, mainloop, epilogue, hardware, scheduler};

  typename Kernel::MainloopArguments mainloop2{
      d_a2.get(), d_stride_a2.get(), d_b2.get(), d_stride_b2.get(),
      d_sfa2.get(), d_layout_sfa2.get(), d_sfb2.get(), d_layout_sfb2.get()};
  typename Kernel::EpilogueArguments epilogue2{
      {}, nullptr, d_stride_c2.get(), d_d2.get(), d_stride_c2.get()};
  epilogue2.thread.alpha_ptr_array = d_alpha2.get();
  epilogue2.thread.dAlpha = {_0{}, _0{}, 1};
  epilogue2.thread.beta = 0.0F;
  typename Kernel::Arguments arguments2{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {kRoutes, d_shapes2.get(), shapes2.data()}, mainloop2, epilogue2, hardware, scheduler};

  Gemm gemm;
  DeviceBuffer<std::byte> workspace(Gemm::get_workspace_size(arguments));
  cutlass_check(gemm.can_implement(arguments), "CUTLASS can_implement");
  cutlass_check(gemm.initialize(arguments, workspace.get()), "CUTLASS initialize");
  Gemm gemm2;
  DeviceBuffer<std::byte> workspace2(Gemm::get_workspace_size(arguments2));
  cutlass_check(gemm2.can_implement(arguments2), "CUTLASS W2 can_implement");
  cutlass_check(gemm2.initialize(arguments2, workspace2.get()), "CUTLASS W2 initialize");
  cudaStream_t stream{};
  auto quantize = [&] {
    quantize_bf16_routes<<<kRoutes, kK / 16, 0, stream>>>(
        activation.get(), reinterpret_cast<const float*>(input_scale_quant.data),
        d_expert_ids.get(), input.get(), input_scales.get(), kK);
  };
  auto run_gemm = [&] { cutlass_check(gemm.run(stream), "CUTLASS run"); };
  auto run_gemm2 = [&] { cutlass_check(gemm2.run(stream), "CUTLASS W2 run"); };
  constexpr int iterations = 300;
  quantize();
  const float gemm_microseconds = time_launch(stream, iterations, run_gemm);
  const float quantized_microseconds = time_launch(stream, iterations, [&] {
    quantize();
    run_gemm();
  });
  const std::array<float, kRoutes> host_route_weights{
      0.24F, 0.19F, 0.15F, 0.12F, 0.10F, 0.08F, 0.07F, 0.05F};
  DeviceBuffer<float> route_weights(kRoutes);
  route_weights.copy(host_route_weights.data(), host_route_weights.size());
  auto run_block = [&] {
    quantize();
    run_gemm();
    geglu_quantize_routes<<<kRoutes, 704 / 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(output.get()),
        reinterpret_cast<const float*>(w2_input_scale_quant.data),
        d_expert_ids.get(), intermediate.get(), intermediate_scales.get());
    run_gemm2();
    reduce_expert_routes<<<1, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(down_output.get()),
        route_weights.get(), reduced_output.get());
  };
  const float expert_block_microseconds = time_launch(stream, iterations, run_block);

  Nvfp4ExpertRunner dynamic_runner(experts, layer);
  DeviceBuffer<__nv_bfloat16> dynamic_output(2816);
  const float dynamic_runner_microseconds = time_launch(stream, iterations, [&] {
    dynamic_runner.launch(activation.get(), d_expert_ids.get(), route_weights.get(),
                          dynamic_output.get(), stream);
  });

  dynamic_runner.launch(activation.get(), d_expert_ids.get(), route_weights.get(),
                        dynamic_output.get(), stream);
  std::vector<__nv_bfloat16> host_output(2816);
  cuda_check(cudaMemcpy(host_output.data(), dynamic_output.get(),
                        host_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
             "cudaMemcpy expert block output");
  double checksum = 0.0;
  int finite_values = 0;
  std::vector<float> decoded_output;
  decoded_output.reserve(host_output.size());
  for (const auto value : host_output) {
    const float decoded = __bfloat162float(value);
    decoded_output.push_back(decoded);
    if (std::isfinite(decoded)) {
      checksum += std::abs(decoded);
      ++finite_values;
    }
  }
  if (!std::isfinite(checksum) || checksum == 0.0 || finite_values != 2816)
    throw std::runtime_error("CUTLASS expert block produced invalid output: finite=" +
                             std::to_string(finite_values) + " checksum=" +
                             std::to_string(checksum));
  return {gemm_microseconds, quantized_microseconds, expert_block_microseconds,
          dynamic_runner_microseconds, static_cast<float>(checksum),
          std::move(decoded_output)};
}

}  // namespace g4
