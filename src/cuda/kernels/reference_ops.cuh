// Private implementation fragment; included only by src/gpu.cu.
__global__ void norm_rope_256_kernel(__nv_bfloat16* vectors,
                                     const __nv_bfloat16* weight, int count,
                                     int tokens, int heads, bool apply_weight,
                                     int position_base = 0) {
  constexpr int kDim = 256;
  __shared__ float reductions[256];
  __shared__ __nv_bfloat16 normed[256];
  const int vector = blockIdx.x;
  if (vector >= count) return;
  const int col = threadIdx.x;
  float x = __bfloat162float(vectors[vector * kDim + col]);
  reductions[col] = x * x;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (col < stride) reductions[col] += reductions[col + stride];
    __syncthreads();
  }
  x *= rsqrtf(reductions[0] / kDim + 1e-6F);
  if (apply_weight) x *= __bfloat162float(weight[col]);
  normed[col] = __float2bfloat16_rn(x);
  __syncthreads();

  const int token = position_base + (vector / heads) % tokens;
  const int half_col = col & 127;
  const float frequency = powf(10000.0F, -2.0F * half_col / kDim);
  const auto cosine = __float2bfloat16_rn(cosf(token * frequency));
  const auto sine = __float2bfloat16_rn(sinf(token * frequency));
  const int paired_col = col < 128 ? col + 128 : col - 128;
  const auto left = __float2bfloat16_rn(__bfloat162float(normed[col]) * __bfloat162float(cosine));
  const float rotated = col < 128 ? -__bfloat162float(normed[paired_col])
                                  : __bfloat162float(normed[paired_col]);
  const auto right = __float2bfloat16_rn(rotated * __bfloat162float(sine));
  vectors[vector * kDim + col] =
      __float2bfloat16_rn(__bfloat162float(left) + __bfloat162float(right));
}

__global__ void attention_3x256_kernel(const __nv_bfloat16* query,
                                       const __nv_bfloat16* key,
                                       const __nv_bfloat16* value,
                                       __nv_bfloat16* output,
                                       __nv_bfloat16* probabilities) {
  constexpr int kTokens = 3;
  constexpr int kDim = 256;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int col = threadIdx.x;
  __shared__ float partial[kTokens][kDim];
  __shared__ float probability[kTokens];
  const int kv_head = head / 2;
  for (int other = 0; other < kTokens; ++other) {
    partial[other][col] = __bfloat162float(query[(token * 16 + head) * kDim + col]) *
                          __bfloat162float(key[(other * 8 + kv_head) * kDim + col]);
  }
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (col < stride) {
      for (int other = 0; other < kTokens; ++other)
        partial[other][col] += partial[other][col + stride];
    }
    __syncthreads();
  }
  if (col == 0) {
    float maximum = -INFINITY;
    for (int other = 0; other <= token; ++other) {
      // BF16 matmul produces a BF16 score before the fp32 softmax.
      partial[other][0] = __bfloat162float(__float2bfloat16_rn(partial[other][0]));
      maximum = fmaxf(maximum, partial[other][0]);
    }
    float total = 0.0F;
    for (int other = 0; other <= token; ++other)
      total += expf(partial[other][0] - maximum);
    for (int other = 0; other < kTokens; ++other) {
      probability[other] = other <= token ? expf(partial[other][0] - maximum) / total : 0.0F;
      probabilities[(head * kTokens + token) * kTokens + other] =
          __float2bfloat16_rn(probability[other]);
    }
  }
  __syncthreads();
  float accumulated = 0.0F;
  for (int other = 0; other <= token; ++other) {
    accumulated += __bfloat162float(__float2bfloat16_rn(probability[other])) *
                   __bfloat162float(value[(other * 8 + kv_head) * kDim + col]);
  }
  output[(token * 16 + head) * kDim + col] = __float2bfloat16_rn(accumulated);
}

__global__ void norm_rope_512_global_kernel(__nv_bfloat16* vectors,
                                             const __nv_bfloat16* weight,
                                             int count, int tokens, int heads,
                                             bool apply_weight,
                                             int position_base = 0) {
  constexpr int kDim = 512;
  __shared__ float reductions[kDim];
  __shared__ __nv_bfloat16 normed[kDim];
  const int vector = blockIdx.x;
  if (vector >= count) return;
  const int col = threadIdx.x;
  float x = __bfloat162float(vectors[vector * kDim + col]);
  reductions[col] = x * x;
  __syncthreads();
  for (int stride = 256; stride; stride >>= 1) {
    if (col < stride) reductions[col] += reductions[col + stride];
    __syncthreads();
  }
  x *= rsqrtf(reductions[0] / kDim + 1e-6F);
  if (apply_weight) x *= __bfloat162float(weight[col]);
  normed[col] = __float2bfloat16_rn(x);
  __syncthreads();

  const int token = position_base + (vector / heads) % tokens;
  const int half_col = col & 255;
  const float frequency = half_col < 64
      ? powf(1000000.0F, -2.0F * half_col / kDim)
      : 0.0F;
  const auto cosine = __float2bfloat16_rn(cosf(token * frequency));
  const auto sine = __float2bfloat16_rn(sinf(token * frequency));
  const int paired_col = col < 256 ? col + 256 : col - 256;
  const auto left = __float2bfloat16_rn(
      __bfloat162float(normed[col]) * __bfloat162float(cosine));
  const float rotated = col < 256 ? -__bfloat162float(normed[paired_col])
                                  : __bfloat162float(normed[paired_col]);
  const auto right = __float2bfloat16_rn(rotated * __bfloat162float(sine));
  vectors[vector * kDim + col] = __float2bfloat16_rn(
      __bfloat162float(left) + __bfloat162float(right));
}

template <int kHeadDim>
__global__ void head_rmsnorm_kernel(
    __nv_bfloat16* vectors, int count,
    const float* activation_scales = nullptr,
    const float* weight_scale = nullptr, int heads_per_row = 1) {
  __shared__ float reductions[512];
  const int vector = blockIdx.x;
  if (vector >= count) return;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float value = __bfloat162float(vectors[vector * kHeadDim + col]);
    if (activation_scales) {
      const float scale = activation_scales[vector / heads_per_row] *
                          weight_scale[0];
      value = __bfloat162float(__float2bfloat16_rn(
          value * scale));
    }
    sum = fmaf(value, value, sum);
  }
  reductions[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(reductions[0] / kHeadDim + 1e-6F);
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x)
    {
      float value = __bfloat162float(vectors[vector * kHeadDim + col]);
      if (activation_scales) {
        const float scale = activation_scales[vector / heads_per_row] *
                            weight_scale[0];
        value = __bfloat162float(__float2bfloat16_rn(
            value * scale));
      }
      vectors[vector * kHeadDim + col] =
          __float2bfloat16_rn(value * inv_rms);
    }
}

__global__ void attention_3x512_kernel(const __nv_bfloat16* query,
                                       const __nv_bfloat16* key,
                                       const __nv_bfloat16* value,
                                       __nv_bfloat16* output,
                                       __nv_bfloat16* probabilities) {
  constexpr int kTokens = 3;
  constexpr int kDim = 512;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int col = threadIdx.x;
  __shared__ float partial[kTokens][kDim];
  __shared__ float probability[kTokens];
  const int kv_head = head / 8;
  for (int other = 0; other < kTokens; ++other)
    partial[other][col] =
        __bfloat162float(query[(token * 16 + head) * kDim + col]) *
        __bfloat162float(key[(other * 2 + kv_head) * kDim + col]);
  __syncthreads();
  for (int stride = 256; stride; stride >>= 1) {
    if (col < stride)
      for (int other = 0; other < kTokens; ++other)
        partial[other][col] += partial[other][col + stride];
    __syncthreads();
  }
  if (col == 0) {
    float maximum = -INFINITY;
    for (int other = 0; other <= token; ++other) {
      partial[other][0] = __bfloat162float(__float2bfloat16_rn(partial[other][0]));
      maximum = fmaxf(maximum, partial[other][0]);
    }
    float total = 0.0F;
    for (int other = 0; other <= token; ++other)
      total += expf(partial[other][0] - maximum);
    for (int other = 0; other < kTokens; ++other) {
      probability[other] = other <= token ? expf(partial[other][0] - maximum) / total : 0.0F;
      probabilities[(head * kTokens + token) * kTokens + other] =
          __float2bfloat16_rn(probability[other]);
    }
  }
  __syncthreads();
  float accumulated = 0.0F;
  for (int other = 0; other <= token; ++other)
    accumulated += __bfloat162float(__float2bfloat16_rn(probability[other])) *
                   __bfloat162float(value[(other * 2 + kv_head) * kDim + col]);
  output[(token * 16 + head) * kDim + col] = __float2bfloat16_rn(accumulated);
}

__global__ void gelu_tanh_multiply_kernel(const __nv_bfloat16* gate,
                                          const __nv_bfloat16* up,
                                          __nv_bfloat16* output, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float x = __bfloat162float(gate[index]);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  const float gelu = 0.5F * x * (1.0F + tanhf(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
  output[index] = __float2bfloat16_rn(gelu * __bfloat162float(up[index]));
}

__global__ void assistant_gelu_tanh_multiply_packed_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, int sessions) {
  constexpr int kIntermediate = 8192;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = sessions * kIntermediate;
  if (index >= elements) return;
  const int session = index / kIntermediate;
  const int column = index % kIntermediate;
  const auto* row = gate_up +
      static_cast<std::size_t>(session) * (2 * kIntermediate);
  const float x = __bfloat162float(row[column]);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  const float gelu = 0.5F * x *
      (1.0F + tanhf(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
  output[index] =
      __float2bfloat16_rn(gelu * __bfloat162float(row[kIntermediate + column]));
}

__global__ void vision_gelu_tanh_multiply_packed_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, int tokens) {
  constexpr int kIntermediate = 4304;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = tokens * kIntermediate;
  if (index >= elements) return;
  const int token = index / kIntermediate;
  const int column = index % kIntermediate;
  const auto* row = gate_up + static_cast<std::size_t>(token) *
                                  (2 * kIntermediate);
  const float x = __bfloat162float(row[column]);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  const float gelu = 0.5F * x *
      (1.0F + tanhf(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
  output[index] =
      __float2bfloat16_rn(gelu * __bfloat162float(row[kIntermediate + column]));
}

__global__ void gelu_split_kernel(const __nv_bfloat16* gate_up,
                                  __nv_bfloat16* product) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 704) return;
  const float x = __bfloat162float(gate_up[index]);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  const float gelu = 0.5F * x * (1.0F + tanhf(kSqrtTwoOverPi * (x + 0.044715F * x * x * x)));
  product[index] = __float2bfloat16_rn(gelu * __bfloat162float(gate_up[704 + index]));
}

__global__ void weighted_bf16_add_kernel(__nv_bfloat16* destination,
                                         const __nv_bfloat16* source,
                                         float weight) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 2816) return;
  const auto contribution = __float2bfloat16_rn(__bfloat162float(source[index]) * weight);
  destination[index] = __float2bfloat16_rn(
      __bfloat162float(destination[index]) + __bfloat162float(contribution));
}

__global__ void bf16_add_kernel(const __nv_bfloat16* left,
                                const __nv_bfloat16* right,
                                __nv_bfloat16* output, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __float2bfloat16_rn(__bfloat162float(left[index]) +
                                        __bfloat162float(right[index]));
  }
}

__global__ void bf16_scale_kernel(__nv_bfloat16* values, int elements,
                                  __nv_bfloat16 scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements)
    values[index] = __float2bfloat16_rn(__bfloat162float(values[index]) *
                                        __bfloat162float(scale));
}

__global__ void bf16_scale_pointer_kernel(__nv_bfloat16* values, int elements,
                                          const __nv_bfloat16* scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements)
    values[index] = __float2bfloat16_rn(__bfloat162float(values[index]) *
                                        __bfloat162float(*scale));
}

__global__ void router_input_kernel(const __nv_bfloat16* input,
                                    const __nv_bfloat16* scale,
                                    __nv_bfloat16* output, int rows) {
  constexpr int kHidden = 2816;
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  const int row = blockIdx.x;
  float sum = 0.0F;
  for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
    const float x = __bfloat162float(input[row * kHidden + col]);
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
    if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / kHidden + 1e-6F);
  }
  __syncthreads();
  const float root = rsqrtf(static_cast<float>(kHidden));
  for (int col = threadIdx.x; col < kHidden; col += blockDim.x) {
    auto value = __float2bfloat16_rn(
        __bfloat162float(input[row * kHidden + col]) * inverse_rms);
    value = __float2bfloat16_rn(__bfloat162float(value) * __bfloat162float(scale[col]));
    output[row * kHidden + col] =
        __float2bfloat16_rn(__bfloat162float(value) * root);
  }
}

__global__ void softmax_128_kernel(const __nv_bfloat16* logits,
                                   float* probabilities, int rows) {
  __shared__ float values[128];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  values[lane] = __bfloat162float(logits[row * 128 + lane]);
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (lane < stride) values[lane] = fmaxf(values[lane], values[lane + stride]);
    __syncthreads();
  }
  const float exponential = expf(__bfloat162float(logits[row * 128 + lane]) - values[0]);
  values[lane] = exponential;
  __syncthreads();
  for (int stride = 64; stride; stride >>= 1) {
    if (lane < stride) values[lane] += values[lane + stride];
    __syncthreads();
  }
  probabilities[row * 128 + lane] = exponential / values[0];
}

__global__ void reduce_routed_experts_kernel(const __nv_bfloat16* routed,
                                             const float* weights,
                                             __nv_bfloat16* output) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= 2816) return;
  __nv_bfloat16 sum = __float2bfloat16_rn(0.0F);
  for (int route = 0; route < 8; ++route) {
    const auto contribution = __float2bfloat16_rn(
        __bfloat162float(routed[route * 2816 + col]) * weights[route]);
    sum = __float2bfloat16_rn(__bfloat162float(sum) + __bfloat162float(contribution));
  }
  output[col] = sum;
}

