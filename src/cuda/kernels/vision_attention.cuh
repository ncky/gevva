// Private implementation fragment; included only by src/gpu.cu.
__global__ void add_vision_positions_kernel(
    __nv_bfloat16* hidden, const __nv_bfloat16* position_table,
    const std::int32_t* positions, int patches) {
  constexpr int kHidden = 1152;
  constexpr int kPositionCount = 10240;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= patches * kHidden) return;
  const int patch = index / kHidden;
  const int column = index % kHidden;
  const int x = positions[patch * 2];
  const int y = positions[patch * 2 + 1];
  if (x < 0 || y < 0) return;
  const auto position = __float2bfloat16_rn(
      __bfloat162float(position_table[x * kHidden + column]) +
      __bfloat162float(position_table[(kPositionCount + y) * kHidden + column]));
  hidden[index] = __float2bfloat16_rn(
      __bfloat162float(hidden[index]) + __bfloat162float(position));
}

__global__ void vision_rmsnorm_1152_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows) {
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
        __bfloat162float(input[index]) * inverse *
        __bfloat162float(weight[col]));
  }
}

__global__ void vision_rmsnorm_add_1152_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* residual, __nv_bfloat16* output, int rows) {
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
    const auto normalized = __float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse *
        __bfloat162float(weight[col]));
    output[index] = __float2bfloat16_rn(
        __bfloat162float(residual[index]) + __bfloat162float(normalized));
  }
}

__global__ void vision_qkv_norm_rope_kernel(
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    const __nv_bfloat16* query_weight, const __nv_bfloat16* key_weight,
    const std::int32_t* positions, int tokens, int token_stride = 1152) {
  constexpr int kHeads = 16;
  constexpr int kDim = 72;
  __shared__ float q_warp[4], k_warp[4], v_warp[4];
  __shared__ float q_inverse, k_inverse, v_inverse;
  __shared__ __nv_bfloat16 q_values[kDim], k_values[kDim];
  const int token = blockIdx.x / kHeads;
  const int head = blockIdx.x % kHeads;
  if (token >= tokens) return;
  const int base = token * token_stride + head * kDim;
  float q_sum = 0.0F, k_sum = 0.0F, v_sum = 0.0F;
  for (int col = threadIdx.x; col < kDim; col += blockDim.x) {
    const float q = __bfloat162float(query[base + col]);
    const float k = __bfloat162float(key[base + col]);
    const float v = __bfloat162float(value[base + col]);
    q_sum = fmaf(q, q, q_sum);
    k_sum = fmaf(k, k, k_sum);
    v_sum = fmaf(v, v, v_sum);
  }
  for (int offset = 16; offset; offset >>= 1) {
    q_sum += __shfl_down_sync(0xffffffff, q_sum, offset);
    k_sum += __shfl_down_sync(0xffffffff, k_sum, offset);
    v_sum += __shfl_down_sync(0xffffffff, v_sum, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    q_warp[threadIdx.x >> 5] = q_sum;
    k_warp[threadIdx.x >> 5] = k_sum;
    v_warp[threadIdx.x >> 5] = v_sum;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    q_sum = threadIdx.x < 4 ? q_warp[threadIdx.x] : 0.0F;
    k_sum = threadIdx.x < 4 ? k_warp[threadIdx.x] : 0.0F;
    v_sum = threadIdx.x < 4 ? v_warp[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset >>= 1) {
      q_sum += __shfl_down_sync(0xffffffff, q_sum, offset);
      k_sum += __shfl_down_sync(0xffffffff, k_sum, offset);
      v_sum += __shfl_down_sync(0xffffffff, v_sum, offset);
    }
    if (threadIdx.x == 0) {
      q_inverse = rsqrtf(q_sum / kDim + 1e-6F);
      k_inverse = rsqrtf(k_sum / kDim + 1e-6F);
      v_inverse = rsqrtf(v_sum / kDim + 1e-6F);
    }
  }
  __syncthreads();
  if (threadIdx.x < kDim) {
    q_values[threadIdx.x] = __float2bfloat16_rn(
        __bfloat162float(query[base + threadIdx.x]) * q_inverse *
        __bfloat162float(query_weight[threadIdx.x]));
    k_values[threadIdx.x] = __float2bfloat16_rn(
        __bfloat162float(key[base + threadIdx.x]) * k_inverse *
        __bfloat162float(key_weight[threadIdx.x]));
    value[base + threadIdx.x] = __float2bfloat16_rn(
        __bfloat162float(value[base + threadIdx.x]) * v_inverse);
  }
  __syncthreads();
  if (threadIdx.x < kDim) {
    const int dimension = threadIdx.x / 36;
    const int local = threadIdx.x % 36;
    const int frequency = local % 18;
    const int mate = dimension * 36 + (local < 18 ? local + 18 : local - 18);
    const float position = static_cast<float>(positions[token * 2 + dimension]);
    const float inverse_frequency = powf(100.0F, -2.0F * frequency / 36.0F);
    const auto cosine = __float2bfloat16_rn(cosf(position * inverse_frequency));
    const auto sine = __float2bfloat16_rn(sinf(position * inverse_frequency));
    const float q_rotated = local < 18 ? -__bfloat162float(q_values[mate])
                                       : __bfloat162float(q_values[mate]);
    const float k_rotated = local < 18 ? -__bfloat162float(k_values[mate])
                                       : __bfloat162float(k_values[mate]);
    const auto q_left = __float2bfloat16_rn(
        __bfloat162float(q_values[threadIdx.x]) * __bfloat162float(cosine));
    const auto q_right = __float2bfloat16_rn(q_rotated * __bfloat162float(sine));
    const auto k_left = __float2bfloat16_rn(
        __bfloat162float(k_values[threadIdx.x]) * __bfloat162float(cosine));
    const auto k_right = __float2bfloat16_rn(k_rotated * __bfloat162float(sine));
    query[base + threadIdx.x] = __float2bfloat16_rn(
        __bfloat162float(q_left) + __bfloat162float(q_right));
    key[base + threadIdx.x] = __float2bfloat16_rn(
        __bfloat162float(k_left) + __bfloat162float(k_right));
  }
}

__global__ void vision_qkv_bshd_to_bhsd_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* packed_query,
    __nv_bfloat16* packed_key, __nv_bfloat16* packed_value, int tokens,
    int sequence) {
  constexpr int kHeads = 16;
  constexpr int kDim = 72;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = tokens * kHeads * kDim;
  if (index >= count) return;
  const int dimension = index % kDim;
  const int head = (index / kDim) % kHeads;
  const int token = index / (kHeads * kDim);
  const int batch = token / sequence;
  const int position = token % sequence;
  const int packed =
      ((batch * kHeads + head) * sequence + position) * kDim + dimension;
  packed_query[packed] = query[index];
  packed_key[packed] = key[index];
  packed_value[packed] = value[index];
}

__global__ void vision_bhsd_to_bshd_kernel(const __nv_bfloat16* packed,
                                            __nv_bfloat16* output,
                                            int tokens, int sequence) {
  constexpr int kHeads = 16;
  constexpr int kDim = 72;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = tokens * kHeads * kDim;
  if (index >= count) return;
  const int dimension = index % kDim;
  const int position = (index / kDim) % sequence;
  const int head = (index / (sequence * kDim)) % kHeads;
  const int batch = index / (kHeads * sequence * kDim);
  const int token = batch * sequence + position;
  output[(token * kHeads + head) * kDim + dimension] = packed[index];
}

__global__ void vision_softmax_kernel(__nv_bfloat16* scores, int tokens,
                                      int valid_tokens) {
  __shared__ float reductions[256];
  const int row = blockIdx.x;
  auto* values = scores + static_cast<std::size_t>(row) * tokens;
  float maximum = -INFINITY;
  for (int col = threadIdx.x; col < valid_tokens; col += blockDim.x)
    maximum = fmaxf(maximum, __bfloat162float(values[col]));
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride >= 32; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                      reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float value = reductions[threadIdx.x];
    for (int offset = 16; offset; offset >>= 1)
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    if (threadIdx.x == 0) reductions[0] = value;
  }
  __syncthreads();
  maximum = reductions[0];
  // Every warp must consume the maximum before thread 0 reuses slot 0 for
  // the denominator reduction below.
  __syncthreads();
  float denominator = 0.0F;
  for (int col = threadIdx.x; col < valid_tokens; col += blockDim.x)
    denominator += expf(__bfloat162float(values[col]) - maximum);
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride >= 32; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    float value = reductions[threadIdx.x];
    for (int offset = 16; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if (threadIdx.x == 0) reductions[0] = value;
  }
  __syncthreads();
  denominator = reductions[0];
  for (int col = threadIdx.x; col < valid_tokens; col += blockDim.x)
    values[col] = __float2bfloat16_rn(
        expf(__bfloat162float(values[col]) - maximum) / denominator);
  for (int col = valid_tokens + threadIdx.x; col < tokens; col += blockDim.x)
    values[col] = __float2bfloat16_rn(0.0F);
}

__global__ void prepare_vision_batch_attention_pointers_kernel(
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    __nv_bfloat16* scores, const void** key_pointers,
    const void** query_pointers, void** score_pointers,
    const void** value_pointers, const void** probability_pointers,
    void** output_pointers, int patches_per_image, int images) {
  constexpr int kHeads = 16;
  constexpr int kHeadDim = 72;
  constexpr int kHidden = kHeads * kHeadDim;
  const int matrix = blockIdx.x * blockDim.x + threadIdx.x;
  if (matrix >= images * kHeads) return;
  const int image = matrix / kHeads;
  const int head = matrix % kHeads;
  const std::size_t activation_offset =
      static_cast<std::size_t>(image) * patches_per_image * kHidden +
      head * kHeadDim;
  const std::size_t score_offset =
      static_cast<std::size_t>(matrix) * patches_per_image * patches_per_image;
  key_pointers[matrix] = key + activation_offset;
  query_pointers[matrix] = query + activation_offset;
  score_pointers[matrix] = scores + score_offset;
  value_pointers[matrix] = value + activation_offset;
  probability_pointers[matrix] = scores + score_offset;
  output_pointers[matrix] = query + activation_offset;
}

