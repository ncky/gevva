// Private implementation fragment; included only by src/gpu.cu.
__global__ void vision_pool_standardize_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* bias,
    const __nv_bfloat16* scale, __nv_bfloat16* output, int soft_tokens,
    int patch_width, int pool_width) {
  constexpr int kHidden = 1152;
  const int token = blockIdx.x;
  if (token >= soft_tokens) return;
  const int group_x = token % pool_width;
  const int group_y = token / pool_width;
  for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
    float sum = 0.0F;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        const int patch = (group_y * 3 + y) * patch_width + group_x * 3 + x;
        sum += __bfloat162float(hidden[patch * kHidden + col]);
      }
    const float pooled = (sum / 9.0F) * sqrtf(static_cast<float>(kHidden));
    output[token * kHidden + col] = __float2bfloat16_rn(
        (pooled - __bfloat162float(bias[col])) * __bfloat162float(scale[col]));
  }
}

__global__ void vision_pool_standardize_batch_kernel(
    const __nv_bfloat16* hidden, const __nv_bfloat16* bias,
    const __nv_bfloat16* scale, __nv_bfloat16* output,
    int soft_tokens_per_image, int patches_per_image, int patch_width,
    int pool_width, int images) {
  constexpr int kHidden = 1152;
  const int global_token = blockIdx.x;
  if (global_token >= soft_tokens_per_image * images) return;
  const int image = global_token / soft_tokens_per_image;
  const int token = global_token % soft_tokens_per_image;
  const int group_x = token % pool_width;
  const int group_y = token / pool_width;
  const auto* image_hidden =
      hidden + static_cast<std::size_t>(image) * patches_per_image * kHidden;
  for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
    float sum = 0.0F;
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 3; ++x) {
        const int patch = (group_y * 3 + y) * patch_width + group_x * 3 + x;
        sum += __bfloat162float(image_hidden[patch * kHidden + col]);
      }
    const float pooled = (sum / 9.0F) * sqrtf(static_cast<float>(kHidden));
    output[static_cast<std::size_t>(global_token) * kHidden + col] =
        __float2bfloat16_rn((pooled - __bfloat162float(bias[col])) *
                           __bfloat162float(scale[col]));
  }
}

__global__ void vision_rmsnorm_noscale_1152_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows) {
  constexpr int kWidth = 1152;
  __shared__ float warp_sums[8];
  __shared__ float inverse;
  const int row = blockIdx.x;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const float value = __bfloat162float(input[row * kWidth + col]);
    sum = fmaf(value, value, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const int index = row * kWidth + col;
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse);
  }
}

