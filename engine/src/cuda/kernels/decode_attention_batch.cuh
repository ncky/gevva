// Private implementation fragment; included only by src/gpu.cu.
template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void decode_attention_scores_batch_kernel(
    const __nv_bfloat16* queries, const __nv_bfloat16* keys,
    const __nv_bfloat16* candidate_keys, float* scores,
    int context_base, int tokens, int score_stride) {
  constexpr int kQueryHeads = 16;
  const int query_token = blockIdx.z;
  if (query_token >= tokens) return;
  const int context = context_base + query_token;
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local_token = blockIdx.x * (blockDim.x / 32) + warp;
  if (local_token >= attended) return;
  const int query_head = blockIdx.y;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int logical_token = first + local_token;
  const int physical_token = kRing ? logical_token & 1023 : logical_token;
  const auto* query = queries +
      (static_cast<std::size_t>(query_token) * kQueryHeads + query_head) * kHeadDim;
  float score = 0.0F;
  const int candidate_start = context_base - 1;
  for (int col = lane; col < kHeadDim; col += 32) {
    const auto key = logical_token >= candidate_start
        ? candidate_keys[((static_cast<std::size_t>(logical_token - candidate_start) *
                            kKvHeads + kv_head) * kHeadDim) + col]
        : keys[((static_cast<std::size_t>(physical_token) * kKvHeads + kv_head) *
                  kHeadDim) + col];
    score = fmaf(__bfloat162float(query[col]), __bfloat162float(key), score);
  }
  for (int offset = 16; offset; offset >>= 1)
    score += __shfl_down_sync(0xffffffff, score, offset);
  if (lane == 0)
    scores[(static_cast<std::size_t>(query_token) * kQueryHeads + query_head) *
               score_stride + local_token] = score;
}

template <bool kRing>
__global__ void decode_attention_softmax_batch_kernel(
    float* scores, int context_base, int tokens, int score_stride) {
  __shared__ float reductions[256];
  constexpr int kQueryHeads = 16;
  const int query_token = blockIdx.y;
  if (query_token >= tokens) return;
  const int context = context_base + query_token;
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  const int head = blockIdx.x;
  float* row = scores +
      (static_cast<std::size_t>(query_token) * kQueryHeads + head) * score_stride;
  float maximum = -INFINITY;
  for (int token = threadIdx.x; token < attended; token += blockDim.x)
    maximum = fmaxf(maximum, row[token]);
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                      reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reductions[0];
  __syncthreads();
  float denominator = 0.0F;
  for (int token = threadIdx.x; token < attended; token += blockDim.x)
    denominator += expf(row[token] - maximum);
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = reductions[0];
  for (int token = threadIdx.x; token < attended; token += blockDim.x)
    row[token] = expf(row[token] - maximum) / denominator;
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void decode_attention_split_values_batch_kernel(
    const float* probabilities, const __nv_bfloat16* values,
    const __nv_bfloat16* candidate_values, float* partials,
    int context_base, int tokens, int score_stride, int partitions,
    __nv_bfloat16* direct_output = nullptr) {
  constexpr int kQueryHeads = 16;
  const int query_token = blockIdx.z;
  if (query_token >= tokens) return;
  const int context = context_base + query_token;
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  const int query_head = blockIdx.x;
  const int partition = blockIdx.y;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int chunk = (attended + partitions - 1) / partitions;
  const int begin = partition * chunk;
  const int end = min(begin + chunk, attended);
  const auto* row = probabilities +
      (static_cast<std::size_t>(query_token) * kQueryHeads + query_head) * score_stride;
  float* destination = partials +
      ((static_cast<std::size_t>(query_token) * partitions + partition) *
           kQueryHeads + query_head) * kHeadDim;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int local = begin; local < end; ++local) {
      const int logical = first + local;
      const int physical = kRing ? logical & 1023 : logical;
      const int candidate_start = context_base - 1;
      const auto value = logical >= candidate_start
          ? candidate_values[((static_cast<std::size_t>(logical - candidate_start) *
                                kKvHeads + kv_head) * kHeadDim) + col]
          : values[((static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
                     kHeadDim) + col];
      accumulated = fmaf(row[local], __bfloat162float(value), accumulated);
    }
    if (direct_output)
      direct_output[(static_cast<std::size_t>(query_token) * kQueryHeads +
                     query_head) * kHeadDim + col] =
          __float2bfloat16_rn(accumulated);
    else
      destination[col] = accumulated;
  }
}

template <int kHeadDim>
__global__ void decode_attention_reduce_values_batch_kernel(
    const float* partials, __nv_bfloat16* outputs, int tokens,
    int partitions) {
  constexpr int kQueryHeads = 16;
  const int query_token = blockIdx.y;
  if (query_token >= tokens) return;
  const int query_head = blockIdx.x;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int partition = 0; partition < partitions; ++partition)
      accumulated += partials[
          ((static_cast<std::size_t>(query_token) * partitions + partition) *
               kQueryHeads + query_head) * kHeadDim + col];
    outputs[(static_cast<std::size_t>(query_token) * kQueryHeads + query_head) *
                kHeadDim + col] = __float2bfloat16_rn(accumulated);
  }
}

