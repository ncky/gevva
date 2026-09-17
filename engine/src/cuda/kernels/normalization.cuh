// Private implementation fragment; included only by src/gpu.cu.
__global__ void rmsnorm_2816_kernel(const __nv_bfloat16* input,
                                    const __nv_bfloat16* weight,
                                    __nv_bfloat16* output, int rows, float eps) {
  constexpr int kWidth = 2816;
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const float x = __bfloat162float(input[row * kWidth + col]);
    sum = fmaf(x, x, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + eps);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const float y = __bfloat162float(input[row * kWidth + col]) * inverse_rms *
                    __bfloat162float(weight[col]);
    output[row * kWidth + col] = __float2bfloat16_rn(y);
  }
}

__global__ void rmsnorm_add_2816_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* residual, __nv_bfloat16* output, int rows,
    const float* activation_scales = nullptr,
    const float* projection_scale = nullptr) {
  constexpr int kWidth = 2816;
  __shared__ float warp_sums[16];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int reduction_threads = min(static_cast<int>(blockDim.x), 256);
  float sum = 0.0F;
  for (int col = threadIdx.x; threadIdx.x < reduction_threads && col < kWidth;
       col += reduction_threads) {
    float x = __bfloat162float(input[row * kWidth + col]);
    if (activation_scales) {
      const float scale = activation_scales[row] * projection_scale[0];
      x = __bfloat162float(__float2bfloat16_rn(
          x * scale));
    }
    sum = fmaf(x, x, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < reduction_threads / 32
        ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const int index = row * kWidth + col;
    float x = __bfloat162float(input[index]);
    if (activation_scales) {
      const float scale = activation_scales[row] * projection_scale[0];
      x = __bfloat162float(__float2bfloat16_rn(
          x * scale));
    }
    const auto normalized = __float2bfloat16_rn(
        x * inverse_rms *
        __bfloat162float(weight[col]));
    output[index] = __float2bfloat16_rn(
        __bfloat162float(residual[index]) + __bfloat162float(normalized));
  }
}

__global__ void dual_rmsnorm_router_2816_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* dense_weight,
    const __nv_bfloat16* expert_weight, const __nv_bfloat16* router_scale,
    __nv_bfloat16* dense_output, __nv_bfloat16* expert_output,
    __nv_bfloat16* router_output, int rows) {
  constexpr int kWidth = 2816;
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const float x = __bfloat162float(input[row * kWidth + col]);
    sum = fmaf(x, x, sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kWidth + 1e-6F);
  }
  __syncthreads();
  const float root = rsqrtf(static_cast<float>(kWidth));
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const int index = row * kWidth + col;
    const float normalized = __bfloat162float(input[index]) * inverse_rms;
    dense_output[index] = __float2bfloat16_rn(
        normalized * __bfloat162float(dense_weight[col]));
    expert_output[index] = __float2bfloat16_rn(
        normalized * __bfloat162float(expert_weight[col]));
    auto routed = __float2bfloat16_rn(normalized);
    routed = __float2bfloat16_rn(
        __bfloat162float(routed) * __bfloat162float(router_scale[col]));
    router_output[index] = __float2bfloat16_rn(
        __bfloat162float(routed) * root);
  }
}

__global__ void finalize_feedforward_2816_kernel(
    const __nv_bfloat16* dense, const __nv_bfloat16* dense_weight,
    const __nv_bfloat16* expert, const __nv_bfloat16* expert_weight,
    const __nv_bfloat16* combined_weight, const __nv_bfloat16* residual,
    const __nv_bfloat16* layer_scalar, __nv_bfloat16* output, int rows,
    const float* dense_activation_scales = nullptr,
    const float* dense_projection_scale = nullptr) {
  constexpr int kWidth = 2816;
  __shared__ float dense_warp_sums[16];
  __shared__ float expert_warp_sums[16];
  __shared__ float combined_warp_sums[16];
  __shared__ float dense_inverse;
  __shared__ float expert_inverse;
  __shared__ float combined_inverse;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int reduction_threads = min(static_cast<int>(blockDim.x), 256);
  float dense_sum = 0.0F;
  float expert_sum = 0.0F;
  for (int col = threadIdx.x; threadIdx.x < reduction_threads && col < kWidth;
       col += reduction_threads) {
    const int index = row * kWidth + col;
    float d = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      d = __bfloat162float(__float2bfloat16_rn(
          d * scale));
    }
    const float e = __bfloat162float(expert[index]);
    dense_sum = fmaf(d, d, dense_sum);
    expert_sum = fmaf(e, e, expert_sum);
  }
  for (int offset = 16; offset; offset >>= 1) {
    dense_sum += __shfl_down_sync(0xffffffff, dense_sum, offset);
    expert_sum += __shfl_down_sync(0xffffffff, expert_sum, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    dense_warp_sums[threadIdx.x >> 5] = dense_sum;
    expert_warp_sums[threadIdx.x >> 5] = expert_sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    dense_sum = threadIdx.x < reduction_threads / 32
        ? dense_warp_sums[threadIdx.x] : 0.0F;
    expert_sum = threadIdx.x < reduction_threads / 32
        ? expert_warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1) {
      dense_sum += __shfl_down_sync(0xffffffff, dense_sum, offset);
      expert_sum += __shfl_down_sync(0xffffffff, expert_sum, offset);
    }
    if (threadIdx.x == 0) {
      dense_inverse = rsqrtf(dense_sum / kWidth + 1e-6F);
      expert_inverse = rsqrtf(expert_sum / kWidth + 1e-6F);
    }
  }
  __syncthreads();
  float combined_sum = 0.0F;
  for (int col = threadIdx.x; threadIdx.x < reduction_threads && col < kWidth;
       col += reduction_threads) {
    const int index = row * kWidth + col;
    float dense_value = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      dense_value = __bfloat162float(__float2bfloat16_rn(
          dense_value * scale));
    }
    const auto dense_norm = __float2bfloat16_rn(
        dense_value * dense_inverse *
        __bfloat162float(dense_weight[col]));
    const auto expert_norm = __float2bfloat16_rn(
        __bfloat162float(expert[index]) * expert_inverse *
        __bfloat162float(expert_weight[col]));
    const auto combined = __float2bfloat16_rn(
        __bfloat162float(dense_norm) + __bfloat162float(expert_norm));
    const float x = __bfloat162float(combined);
    combined_sum = fmaf(x, x, combined_sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    combined_sum += __shfl_down_sync(0xffffffff, combined_sum, offset);
  if ((threadIdx.x & 31) == 0)
    combined_warp_sums[threadIdx.x >> 5] = combined_sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    combined_sum = threadIdx.x < reduction_threads / 32
        ? combined_warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      combined_sum += __shfl_down_sync(0xffffffff, combined_sum, offset);
    if (threadIdx.x == 0)
      combined_inverse = rsqrtf(combined_sum / kWidth + 1e-6F);
  }
  __syncthreads();
  const float scalar = __bfloat162float(*layer_scalar);
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const int index = row * kWidth + col;
    float dense_value = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      dense_value = __bfloat162float(__float2bfloat16_rn(
          dense_value * scale));
    }
    const auto dense_norm = __float2bfloat16_rn(
        dense_value * dense_inverse *
        __bfloat162float(dense_weight[col]));
    const auto expert_norm = __float2bfloat16_rn(
        __bfloat162float(expert[index]) * expert_inverse *
        __bfloat162float(expert_weight[col]));
    const auto combined = __float2bfloat16_rn(
        __bfloat162float(dense_norm) + __bfloat162float(expert_norm));
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(combined) * combined_inverse *
        __bfloat162float(combined_weight[col]));
    const auto added = __float2bfloat16_rn(
        __bfloat162float(residual[index]) + __bfloat162float(normalized));
    output[index] = __float2bfloat16_rn(__bfloat162float(added) * scalar);
  }
}

// Large prefill already has one CTA per token in the feed-forward finalizer.
// Keep the exact eight-route FP32 accumulation and BF16 rounding boundary, but
// retain that reduced row in shared memory instead of materializing and then
// rereading a full expert-output tensor in global memory.
__global__ void finalize_feedforward_routes_2816_kernel(
    const __nv_bfloat16* dense, const __nv_bfloat16* dense_weight,
    const __nv_bfloat16* expert_routes, const float* route_weights,
    const int* inverse_routes, const __nv_bfloat16* expert_weight,
    const __nv_bfloat16* combined_weight, const __nv_bfloat16* residual,
    const __nv_bfloat16* layer_scalar, __nv_bfloat16* output, int rows,
    const float* dense_activation_scales = nullptr,
    const float* dense_projection_scale = nullptr) {
  constexpr int kWidth = 2816;
  __shared__ __nv_bfloat16 reduced_expert[kWidth];
  __shared__ float dense_warp_sums[16];
  __shared__ float expert_warp_sums[16];
  __shared__ float combined_warp_sums[16];
  __shared__ float dense_inverse;
  __shared__ float expert_inverse;
  __shared__ float combined_inverse;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int reduction_threads = min(static_cast<int>(blockDim.x), 256);
  float dense_sum = 0.0F;
  float expert_sum = 0.0F;
  for (int col = threadIdx.x; threadIdx.x < reduction_threads && col < kWidth;
       col += reduction_threads) {
    const int index = row * kWidth + col;
    float d = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      d = __bfloat162float(__float2bfloat16_rn(d * scale));
    }
    float expert_value = 0.0F;
#pragma unroll
    for (int route = 0; route < 8; ++route) {
      const int route_index = row * 8 + route;
      const int source = inverse_routes ? inverse_routes[route_index]
                                        : route_index;
      expert_value = fmaf(
          __bfloat162float(expert_routes[source * kWidth + col]),
          route_weights[route_index], expert_value);
    }
    const auto rounded_expert = __float2bfloat16_rn(expert_value);
    reduced_expert[col] = rounded_expert;
    const float e = __bfloat162float(rounded_expert);
    dense_sum = fmaf(d, d, dense_sum);
    expert_sum = fmaf(e, e, expert_sum);
  }
  for (int offset = 16; offset; offset >>= 1) {
    dense_sum += __shfl_down_sync(0xffffffff, dense_sum, offset);
    expert_sum += __shfl_down_sync(0xffffffff, expert_sum, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    dense_warp_sums[threadIdx.x >> 5] = dense_sum;
    expert_warp_sums[threadIdx.x >> 5] = expert_sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    dense_sum = threadIdx.x < reduction_threads / 32
        ? dense_warp_sums[threadIdx.x] : 0.0F;
    expert_sum = threadIdx.x < reduction_threads / 32
        ? expert_warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1) {
      dense_sum += __shfl_down_sync(0xffffffff, dense_sum, offset);
      expert_sum += __shfl_down_sync(0xffffffff, expert_sum, offset);
    }
    if (threadIdx.x == 0) {
      dense_inverse = rsqrtf(dense_sum / kWidth + 1e-6F);
      expert_inverse = rsqrtf(expert_sum / kWidth + 1e-6F);
    }
  }
  __syncthreads();
  float combined_sum = 0.0F;
  for (int col = threadIdx.x; threadIdx.x < reduction_threads && col < kWidth;
       col += reduction_threads) {
    const int index = row * kWidth + col;
    float dense_value = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      dense_value = __bfloat162float(__float2bfloat16_rn(dense_value * scale));
    }
    const auto dense_norm = __float2bfloat16_rn(
        dense_value * dense_inverse * __bfloat162float(dense_weight[col]));
    const auto expert_norm = __float2bfloat16_rn(
        __bfloat162float(reduced_expert[col]) * expert_inverse *
        __bfloat162float(expert_weight[col]));
    const auto combined = __float2bfloat16_rn(
        __bfloat162float(dense_norm) + __bfloat162float(expert_norm));
    const float x = __bfloat162float(combined);
    combined_sum = fmaf(x, x, combined_sum);
  }
  for (int offset = 16; offset; offset >>= 1)
    combined_sum += __shfl_down_sync(0xffffffff, combined_sum, offset);
  if ((threadIdx.x & 31) == 0)
    combined_warp_sums[threadIdx.x >> 5] = combined_sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    combined_sum = threadIdx.x < reduction_threads / 32
        ? combined_warp_sums[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1)
      combined_sum += __shfl_down_sync(0xffffffff, combined_sum, offset);
    if (threadIdx.x == 0)
      combined_inverse = rsqrtf(combined_sum / kWidth + 1e-6F);
  }
  __syncthreads();
  const float scalar = __bfloat162float(*layer_scalar);
  for (int col = threadIdx.x; col < kWidth; col += blockDim.x) {
    const int index = row * kWidth + col;
    float dense_value = __bfloat162float(dense[index]);
    if (dense_activation_scales) {
      const float scale =
          dense_activation_scales[row] * dense_projection_scale[0];
      dense_value = __bfloat162float(__float2bfloat16_rn(dense_value * scale));
    }
    const auto dense_norm = __float2bfloat16_rn(
        dense_value * dense_inverse * __bfloat162float(dense_weight[col]));
    const auto expert_norm = __float2bfloat16_rn(
        __bfloat162float(reduced_expert[col]) * expert_inverse *
        __bfloat162float(expert_weight[col]));
    const auto combined = __float2bfloat16_rn(
        __bfloat162float(dense_norm) + __bfloat162float(expert_norm));
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(combined) * combined_inverse *
        __bfloat162float(combined_weight[col]));
    const auto added = __float2bfloat16_rn(
        __bfloat162float(residual[index]) + __bfloat162float(normalized));
    output[index] =
        __float2bfloat16_rn(__bfloat162float(added) * scalar);
  }
}

