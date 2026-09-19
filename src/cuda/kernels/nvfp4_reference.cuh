// Private implementation fragment; included only by src/gpu.cu.
__device__ __host__ float decode_e2m1(unsigned value) {
  const unsigned exponent = (value >> 1U) & 3U;
  const unsigned mantissa = value & 1U;
  const float magnitude = exponent == 0
      ? static_cast<float>(mantissa) * 0.5F
      : ldexpf(1.0F + static_cast<float>(mantissa) * 0.5F,
               static_cast<int>(exponent) - 1);
  return value & 8U ? -magnitude : magnitude;
}

__device__ __host__ float decode_e4m3(unsigned value) {
  __nv_fp8_e4m3 encoded;
  encoded.__x = static_cast<__nv_fp8_storage_t>(value);
  return static_cast<float>(encoded);
}

template <int kThreads>
__global__ void nvfp4_matvec_kernel(const std::uint8_t* packed_weight,
                                    const std::uint8_t* block_scale,
                                    float global_scale,
                                    const __nv_bfloat16* input,
                                    __nv_bfloat16* output,
                                    int outputs, int inputs) {
  __shared__ float warp_sums[kThreads / 32];
  const int row = blockIdx.x;
  if (row >= outputs) return;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < inputs; col += blockDim.x) {
    const std::uint8_t packed = packed_weight[row * (inputs / 2) + col / 2];
    const unsigned nibble = (col & 1) ? packed >> 4 : packed & 15U;
    const float scale = decode_e4m3(block_scale[row * (inputs / 16) + col / 16]);
    sum = fmaf(__bfloat162float(input[col]), decode_e2m1(nibble) * scale * global_scale, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) output[row] = __float2bfloat16_rn(sum);
  }
}

__global__ void nvfp4_routed_matvec_kernel(
    const std::uint8_t* __restrict__ packed_weights,
    const std::uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    const int* __restrict__ expert_ids, int route,
    const __nv_bfloat16* __restrict__ inputs, __nv_bfloat16* __restrict__ outputs,
    int output_features, int input_features) {
  __shared__ float warp_sums[8];
  const int row = blockIdx.x;
  const int expert = expert_ids[route];
  const auto* weight = packed_weights +
      static_cast<std::size_t>(expert) * output_features * (input_features / 2);
  const auto* scales = block_scales +
      static_cast<std::size_t>(expert) * output_features * (input_features / 16);
  const auto* input = inputs + route * input_features;
  const float global_scale = global_scales[expert];
  float sum = 0.0F;
  for (int col = threadIdx.x; col < input_features; col += blockDim.x) {
    const unsigned packed = weight[row * (input_features / 2) + col / 2];
    const unsigned nibble = (col & 1) ? packed >> 4 : packed & 15U;
    const float scale = decode_e4m3(scales[row * (input_features / 16) + col / 16]);
    sum = fmaf(__bfloat162float(input[col]),
               decode_e2m1(nibble) * scale * global_scale, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      outputs[route * output_features + row] = __float2bfloat16_rn(sum);
  }
}

