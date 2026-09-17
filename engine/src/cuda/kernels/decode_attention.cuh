// Private implementation fragment; included only by src/gpu.cu.
template <int kHeadDim, int kKvHeads, int kWindow>
__global__ void decode_attention_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    __nv_bfloat16* __restrict__ output, int context_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kValuesPerLane = kHeadDim / 32;
  const int query_head = blockIdx.x;
  const int lane = threadIdx.x;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int first = kWindow > 0 && context_tokens > kWindow ? context_tokens - kWindow : 0;
  float maximum = -INFINITY;
  float denominator = 0.0F;
  float accumulated[kValuesPerLane]{};
  for (int token = first; token < context_tokens; ++token) {
    const int physical_token = kWindow > 0 ? token & (kWindow - 1) : token;
    float score = 0.0F;
#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
      const int col = lane + item * 32;
      score = fmaf(__bfloat162float(query[query_head * kHeadDim + col]),
                   __bfloat162float(keys[(physical_token * kKvHeads + kv_head) * kHeadDim + col]),
                   score);
    }
    for (int offset = 16; offset; offset >>= 1)
      score += __shfl_down_sync(0xffffffff, score, offset);
    score = __shfl_sync(0xffffffff, score, 0);
    const float next_maximum = fmaxf(maximum, score);
    const float old_factor = expf(maximum - next_maximum);
    const float new_factor = expf(score - next_maximum);
    denominator = denominator * old_factor + new_factor;
#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
      const int col = lane + item * 32;
      const float value = __bfloat162float(
          values[(physical_token * kKvHeads + kv_head) * kHeadDim + col]);
      accumulated[item] = accumulated[item] * old_factor + value * new_factor;
    }
    maximum = next_maximum;
  }
#pragma unroll
  for (int item = 0; item < kValuesPerLane; ++item) {
    const int col = lane + item * 32;
    output[query_head * kHeadDim + col] =
        __float2bfloat16_rn(accumulated[item] / denominator);
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void decode_attention_scores_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ keys,
    float* __restrict__ scores, int first_token, int attended_tokens) {
  constexpr int kQueryHeads = 16;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local_token = blockIdx.x * (blockDim.x / 32) + warp;
  if (local_token >= attended_tokens) return;
  const int query_head = blockIdx.y;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int token = first_token + local_token;
  const int physical_token = kRing ? token & 1023 : token;
  float score = 0.0F;
  for (int col = lane; col < kHeadDim; col += 32)
    score = fmaf(__bfloat162float(query[query_head * kHeadDim + col]),
                 __bfloat162float(keys[(physical_token * kKvHeads + kv_head) * kHeadDim + col]), score);
  for (int offset = 16; offset; offset >>= 1)
    score += __shfl_down_sync(0xffffffff, score, offset);
  if (lane == 0) scores[query_head * attended_tokens + local_token] = score;
}

__global__ void decode_attention_softmax_kernel(float* scores, int attended_tokens) {
  __shared__ float reductions[256];
  const int head = blockIdx.x;
  float maximum = -INFINITY;
  for (int token = threadIdx.x; token < attended_tokens; token += blockDim.x)
    maximum = fmaxf(maximum, scores[head * attended_tokens + token]);
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reductions[0];
  __syncthreads();
  float denominator = 0.0F;
  for (int token = threadIdx.x; token < attended_tokens; token += blockDim.x)
    denominator += expf(scores[head * attended_tokens + token] - maximum);
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = reductions[0];
  for (int token = threadIdx.x; token < attended_tokens; token += blockDim.x) {
    float& score = scores[head * attended_tokens + token];
    score = expf(score - maximum) / denominator;
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void decode_attention_values_kernel(
    const float* __restrict__ probabilities,
    const __nv_bfloat16* __restrict__ values,
    __nv_bfloat16* __restrict__ output, int first_token, int attended_tokens) {
  constexpr int kQueryHeads = 16;
  const int query_head = blockIdx.x;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int local_token = 0; local_token < attended_tokens; ++local_token) {
      const int token = first_token + local_token;
      const int physical_token = kRing ? token & 1023 : token;
      accumulated = fmaf(probabilities[query_head * attended_tokens + local_token],
          __bfloat162float(values[(physical_token * kKvHeads + kv_head) * kHeadDim + col]), accumulated);
    }
    output[query_head * kHeadDim + col] = __float2bfloat16_rn(accumulated);
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void decode_attention_split_values_kernel(
    const float* __restrict__ probabilities,
    const __nv_bfloat16* __restrict__ values,
    float* __restrict__ partials, int first_token, int attended_tokens,
    int partitions) {
  constexpr int kQueryHeads = 16;
  const int query_head = blockIdx.x;
  const int partition = blockIdx.y;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int chunk = (attended_tokens + partitions - 1) / partitions;
  const int begin = partition * chunk;
  const int end = min(begin + chunk, attended_tokens);
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int local_token = begin; local_token < end; ++local_token) {
      const int token = first_token + local_token;
      const int physical_token = kRing ? token & 1023 : token;
      accumulated = fmaf(probabilities[query_head * attended_tokens + local_token],
          __bfloat162float(values[(physical_token * kKvHeads + kv_head) * kHeadDim + col]), accumulated);
    }
    partials[(partition * kQueryHeads + query_head) * kHeadDim + col] = accumulated;
  }
}

template <int kHeadDim>
__global__ void decode_attention_reduce_values_kernel(
    const float* __restrict__ partials, __nv_bfloat16* __restrict__ output,
    int partitions) {
  constexpr int kQueryHeads = 16;
  const int query_head = blockIdx.x;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int partition = 0; partition < partitions; ++partition)
      accumulated += partials[(partition * kQueryHeads + query_head) * kHeadDim + col];
    output[query_head * kHeadDim + col] = __float2bfloat16_rn(accumulated);
  }
}

