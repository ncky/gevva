// Private implementation fragment; included only by src/gpu.cu.
template <int kHeadDim, int kKvHeads, bool kRing, bool kCandidates>
__global__ void ragged_attention_scores_kernel(
    const __nv_bfloat16* queries, const DeviceKvView* cache_views,
    const DeviceKvView* candidate_views, const int* context_lengths,
    float* scores, int rows, int tokens_per_session, int score_stride) {
  constexpr int kQueryHeads = 16;
  const int row_index = blockIdx.z;
  if (row_index >= rows) return;
  const int session = row_index / tokens_per_session;
  const int step = row_index % tokens_per_session;
  const int cached = context_lengths[session];
  const int context = cached + (kCandidates ? step + 1 : 0);
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int local = blockIdx.x * (blockDim.x / 32) + warp;
  if (local >= attended) return;
  const int query_head = blockIdx.y;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int logical = first + local;
  const int physical = kRing ? logical & 1023 : logical;
  const auto cache = cache_views[session];
  const auto candidate = kCandidates ? candidate_views[session] : DeviceKvView{};
  const auto* query = queries +
      (static_cast<std::size_t>(row_index) * kQueryHeads + query_head) * kHeadDim;
  float score = 0.0F;
  for (int col = lane; col < kHeadDim; col += 32) {
    __nv_bfloat16 key;
    if constexpr (kCandidates) {
      key = logical >= cached
          ? reinterpret_cast<const __nv_bfloat16*>(candidate.keys)[
                ((static_cast<std::size_t>(logical - cached) * kKvHeads +
                  kv_head) * kHeadDim) + col]
          : reinterpret_cast<const __nv_bfloat16*>(cache.keys)[
                ((static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
                 kHeadDim) + col];
    } else {
      key = reinterpret_cast<const __nv_bfloat16*>(cache.keys)[
          ((static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
           kHeadDim) + col];
    }
    score = fmaf(__bfloat162float(query[col]), __bfloat162float(key), score);
  }
  for (int offset = 16; offset; offset >>= 1)
    score += __shfl_down_sync(0xffffffff, score, offset);
  if (lane == 0)
    scores[(static_cast<std::size_t>(row_index) * kQueryHeads + query_head) *
               score_stride + local] = score;
}

template <bool kRing, bool kCandidates>
__global__ void ragged_attention_softmax_kernel(
    float* scores, const int* context_lengths, int rows,
    int tokens_per_session, int score_stride) {
  __shared__ float reductions[256];
  const int row_index = blockIdx.y;
  if (row_index >= rows) return;
  const int session = row_index / tokens_per_session;
  const int step = row_index % tokens_per_session;
  const int context = context_lengths[session] + (kCandidates ? step + 1 : 0);
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  float* row = scores +
      (static_cast<std::size_t>(row_index) * 16 + blockIdx.x) * score_stride;
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
  // Finish consuming the maximum before reusing reductions for the sum.
  __syncthreads();
  float denominator = 0.0F;
  for (int token = threadIdx.x; token < attended; token += blockDim.x) {
    const float exponential = expf(row[token] - maximum);
    row[token] = exponential;
    denominator += exponential;
  }
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = reductions[0];
  for (int token = threadIdx.x; token < attended; token += blockDim.x)
    row[token] /= denominator;
}

template <int kHeadDim, int kKvHeads, bool kRing, bool kCandidates>
__global__ void ragged_attention_values_kernel(
    const float* probabilities, const DeviceKvView* cache_views,
    const DeviceKvView* candidate_views, const int* context_lengths,
    float* partials, __nv_bfloat16* output, int rows,
    int tokens_per_session, int score_stride, int partitions) {
  constexpr int kQueryHeads = 16;
  const int row_index = blockIdx.z;
  if (row_index >= rows) return;
  const int session = row_index / tokens_per_session;
  const int step = row_index % tokens_per_session;
  const int context = context_lengths[session] + (kCandidates ? step + 1 : 0);
  const int first = kRing && context > 1024 ? context - 1024 : 0;
  const int attended = context - first;
  const int head = blockIdx.x;
  const int partition = blockIdx.y;
  const int kv_head = head / (kQueryHeads / kKvHeads);
  const int chunk = (attended + partitions - 1) / partitions;
  const int begin = partition * chunk;
  const int end = min(begin + chunk, attended);
  const auto cache = cache_views[session];
  const auto candidate = kCandidates ? candidate_views[session] : DeviceKvView{};
  const float* row = probabilities +
      (static_cast<std::size_t>(row_index) * kQueryHeads + head) * score_stride;
  float* destination = partials +
      ((static_cast<std::size_t>(row_index) * partitions + partition) *
       kQueryHeads + head) * kHeadDim;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float accumulated = 0.0F;
    for (int local = begin; local < end; ++local) {
      const int logical = first + local;
      const int physical = kRing ? logical & 1023 : logical;
      __nv_bfloat16 value;
      if constexpr (kCandidates) {
        value = logical >= context_lengths[session]
            ? reinterpret_cast<const __nv_bfloat16*>(candidate.values)[
                  ((static_cast<std::size_t>(logical - context_lengths[session]) *
                    kKvHeads + kv_head) * kHeadDim) + col]
            : reinterpret_cast<const __nv_bfloat16*>(cache.values)[
                  ((static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
                   kHeadDim) + col];
      } else {
        value = reinterpret_cast<const __nv_bfloat16*>(cache.values)[
            ((static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
             kHeadDim) + col];
      }
      accumulated = fmaf(row[local], __bfloat162float(value), accumulated);
    }
    if (partitions == 1)
      output[(static_cast<std::size_t>(row_index) * kQueryHeads + head) *
                 kHeadDim + col] = __float2bfloat16_rn(accumulated);
    else
      destination[col] = accumulated;
  }
}

// The assistant's global attention has two KV heads, each shared by eight
// query heads.  The generic ragged kernels make every query head fetch the
// same 512-wide K/V rows independently.  At B4+ that is substantially more
// HBM traffic and many more CTAs than useful work.  This score kernel keeps
// the existing per-lane dot-product order but stages each key tile once for
// all eight query heads in the group.
__global__ void global_gqa_attention_scores_kernel(
    const __nv_bfloat16* queries, const DeviceKvView* cache_views,
    const int* context_lengths, float* scores, int rows,
    int tokens_per_session, int score_stride) {
  constexpr int kHeadDim = 512;
  constexpr int kKvHeads = 2;
  constexpr int kGroup = 8;
  constexpr int kTokenTile = 8;
  __shared__ __nv_bfloat16 keys[kTokenTile * kHeadDim];
  const int row_index = blockIdx.z;
  if (row_index >= rows) return;
  const int session = row_index / tokens_per_session;
  const int context = context_lengths[session];
  const int token_base = blockIdx.x * kTokenTile;
  const int valid = min(kTokenTile, context - token_base);
  if (valid <= 0) return;
  const int kv_head = blockIdx.y;
  const auto cache = cache_views[session];
  for (int index = threadIdx.x; index < valid * kHeadDim;
       index += blockDim.x) {
    const int token = token_base + index / kHeadDim;
    const int col = index % kHeadDim;
    keys[index] = reinterpret_cast<const __nv_bfloat16*>(cache.keys)[
        (static_cast<std::size_t>(token) * kKvHeads + kv_head) * kHeadDim +
        col];
  }
  __syncthreads();
  const int query_within_group = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_head = kv_head * kGroup + query_within_group;
  const auto* query = queries +
      (static_cast<std::size_t>(row_index) * 16 + query_head) * kHeadDim;
  for (int local = 0; local < valid; ++local) {
    float score = 0.0F;
    for (int col = lane; col < kHeadDim; col += 32)
      score = fmaf(__bfloat162float(query[col]),
                   __bfloat162float(keys[local * kHeadDim + col]), score);
    for (int offset = 16; offset; offset >>= 1)
      score += __shfl_down_sync(0xffffffff, score, offset);
    if (lane == 0)
      scores[(static_cast<std::size_t>(row_index) * 16 + query_head) *
                 score_stride + token_base + local] = score;
  }
}

template <int kHeadDim>
__global__ void ragged_attention_reduce_kernel(
    const float* partials, __nv_bfloat16* output, int rows, int partitions) {
  const int row = blockIdx.y;
  if (row >= rows) return;
  const int head = blockIdx.x;
  for (int col = threadIdx.x; col < kHeadDim; col += blockDim.x) {
    float sum = 0.0F;
    for (int partition = 0; partition < partitions; ++partition)
      sum += partials[((static_cast<std::size_t>(row) * partitions + partition) *
                       16 + head) * kHeadDim + col];
    output[(static_cast<std::size_t>(row) * 16 + head) * kHeadDim + col] =
        __float2bfloat16_rn(sum);
  }
}

template <bool kCandidates>
void launch_ragged_attention(
    bool global, const __nv_bfloat16* queries,
    const DeviceKvView* cache_views, const DeviceKvView* candidate_views,
    const int* context_lengths, __nv_bfloat16* output, float* scores,
    float* partials, int rows, int sessions, int tokens_per_session,
    int maximum_context, cudaStream_t stream) {
  const int score_stride = global ? maximum_context : std::min(maximum_context, 1024);
  const int partitions = std::min(16, (score_stride + 255) / 256);
  // Each warp owns one context token.  Sixteen warps halve the number of
  // score CTAs without changing a dot product's lane order or arithmetic.
  constexpr int kWarps = 16;
  const dim3 score_grid((score_stride + kWarps - 1) / kWarps, 16, rows);
  const dim3 row_grid(16, rows);
  const dim3 value_grid(16, partitions, rows);
  if (global) {
    static const bool grouped_gqa_enabled =
        std::getenv("GEVVA_DISABLE_ASSISTANT_GROUPED_GQA") == nullptr;
    const int grouped_gqa_crossover = sessions >= 8 ? 1536
                                    : sessions >= 5 ? 2048
                                                    : 3072;
    const bool grouped_gqa = !kCandidates && grouped_gqa_enabled &&
                             sessions >= 4 &&
                             maximum_context >= grouped_gqa_crossover;
    if (grouped_gqa) {
      const dim3 grouped_score_grid((score_stride + 7) / 8, 2, rows);
      global_gqa_attention_scores_kernel<<<grouped_score_grid, 256, 0, stream>>>(
          queries, cache_views, context_lengths, scores, rows,
          tokens_per_session, score_stride);
    } else {
      ragged_attention_scores_kernel<512, 2, false, kCandidates>
          <<<score_grid, kWarps * 32, 0, stream>>>(queries, cache_views,
                                          candidate_views, context_lengths,
                                          scores, rows, tokens_per_session,
                                          score_stride);
    }
    ragged_attention_softmax_kernel<false, kCandidates>
        <<<row_grid, 256, 0, stream>>>(scores, context_lengths, rows,
                                      tokens_per_session, score_stride);
    // The existing value kernel is faster here despite redundant logical KV
    // reads: those reads hit cache, while staging values for all eight query
    // heads adds more synchronization than memory-traffic savings.
    ragged_attention_values_kernel<512, 2, false, kCandidates>
        <<<value_grid, 512, 0, stream>>>(scores, cache_views, candidate_views,
                                        context_lengths, partials, output,
                                        rows, tokens_per_session, score_stride,
                                        partitions);
    if (partitions > 1)
      ragged_attention_reduce_kernel<512><<<row_grid, 512, 0, stream>>>(
          partials, output, rows, partitions);
  } else {
    ragged_attention_scores_kernel<256, 8, true, kCandidates>
        <<<score_grid, kWarps * 32, 0, stream>>>(queries, cache_views, candidate_views,
                                        context_lengths, scores, rows,
                                        tokens_per_session, score_stride);
    ragged_attention_softmax_kernel<true, kCandidates>
        <<<row_grid, 256, 0, stream>>>(scores, context_lengths, rows,
                                      tokens_per_session, score_stride);
    ragged_attention_values_kernel<256, 8, true, kCandidates>
        <<<value_grid, 256, 0, stream>>>(scores, cache_views, candidate_views,
                                        context_lengths, partials, output, rows,
                                        tokens_per_session, score_stride,
                                        partitions);
    if (partitions > 1)
      ragged_attention_reduce_kernel<256><<<row_grid, 256, 0, stream>>>(
          partials, output, rows, partitions);
  }
  (void)sessions;
  check(cudaGetLastError(), "ragged attention launch");
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void pack_ragged_gemm_attention_kernel(
    const __nv_bfloat16* queries, const DeviceKvView* cache_views,
    const DeviceKvView* candidate_views, const int* context_lengths,
    const __nv_bfloat16* new_keys, const __nv_bfloat16* new_values,
    __nv_bfloat16* packed_queries, __nv_bfloat16* packed_keys,
    __nv_bfloat16* packed_values, int sessions, int tokens_per_session,
    int key_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const int grouped_rows = tokens_per_session * kGroup;
  const int query_rows = sessions * kKvHeads * grouped_rows;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int packed_row = static_cast<int>(blockIdx.x) * 8 + warp;
  const int total_rows = query_rows + sessions * kKvHeads * key_tokens;
  if (packed_row >= total_rows) return;
  if (packed_row < query_rows) {
      int value = packed_row;
      const int grouped_row = value % grouped_rows;
      value /= grouped_rows;
      const int kv_head = value % kKvHeads;
      const int session = value / kKvHeads;
      const int step = grouped_row / kGroup;
      const int within = grouped_row % kGroup;
      const int row = session * tokens_per_session + step;
      const int query_head = kv_head * kGroup + within;
      const auto* source = reinterpret_cast<const uint4*>(
          queries + (static_cast<std::size_t>(row) * kQueryHeads +
                     query_head) * kHeadDim);
      auto* destination = reinterpret_cast<uint4*>(
          packed_queries + static_cast<std::size_t>(packed_row) * kHeadDim);
      for (int vector = lane; vector < kHeadDim / 8; vector += 32)
        destination[vector] = source[vector];
      return;
  }
  int value = packed_row - query_rows;
  const int local = value % key_tokens;
    value /= key_tokens;
    const int kv_head = value % kKvHeads;
    const int session = value / kKvHeads;
    const int cached = context_lengths[session];
    const int pool_first = kRing && cached > 1023 ? cached - 1023 : 0;
    const int logical = pool_first + local;
    const int pool_end = cached + tokens_per_session;
    const __nv_bfloat16* source_key = nullptr;
    const __nv_bfloat16* source_value = nullptr;
    if (logical < pool_end) {
      if (logical >= cached) {
        const auto candidate = candidate_views[session];
        const int step = logical - cached;
        const std::size_t source =
            ((static_cast<std::size_t>(session) * tokens_per_session + step) *
                 kKvHeads + kv_head) * kHeadDim;
        const std::size_t staged =
            (static_cast<std::size_t>(step) * kKvHeads + kv_head) * kHeadDim;
        source_key = new_keys + source;
        source_value = new_values + source;
        auto* candidate_key =
            reinterpret_cast<__nv_bfloat16*>(candidate.keys) + staged;
        auto* candidate_value =
            reinterpret_cast<__nv_bfloat16*>(candidate.values) + staged;
        const auto* vector_key = reinterpret_cast<const uint4*>(source_key);
        const auto* vector_value = reinterpret_cast<const uint4*>(source_value);
        auto* staged_key = reinterpret_cast<uint4*>(candidate_key);
        auto* staged_value = reinterpret_cast<uint4*>(candidate_value);
        for (int vector = lane; vector < kHeadDim / 8; vector += 32) {
          staged_key[vector] = vector_key[vector];
          staged_value[vector] = vector_value[vector];
        }
      } else {
        const auto cache = cache_views[session];
        const int physical = kRing ? logical & 1023 : logical;
        const std::size_t source =
            (static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
            kHeadDim;
        source_key = reinterpret_cast<const __nv_bfloat16*>(cache.keys) +
                     source;
        source_value = reinterpret_cast<const __nv_bfloat16*>(cache.values) +
                       source;
      }
    }
    const std::size_t destination_row =
        static_cast<std::size_t>(packed_row - query_rows) * kHeadDim;
    auto* destination_key = reinterpret_cast<uint4*>(
        packed_keys + destination_row);
    auto* destination_value = reinterpret_cast<uint4*>(
        packed_values + destination_row);
    const auto* vector_key = reinterpret_cast<const uint4*>(source_key);
    const auto* vector_value =
        reinterpret_cast<const uint4*>(source_value);
    for (int vector = lane; vector < kHeadDim / 8; vector += 32) {
      destination_key[vector] = source_key ? vector_key[vector] : uint4{};
      destination_value[vector] =
          source_value ? vector_value[vector] : uint4{};
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void prepare_direct_attention_kernel(
    const __nv_bfloat16* queries, const __nv_bfloat16* new_keys,
    const __nv_bfloat16* new_values, const DeviceKvView* cache_views,
    const DeviceKvView* candidate_views, const int* context_lengths,
    __nv_bfloat16* packed_queries, float* scores,
    __nv_bfloat16* probabilities, __nv_bfloat16* packed_output,
    const void** key_pointers, const void** query_pointers,
    void** score_pointers, const void** value_pointers,
    const void** probability_pointers, void** output_pointers,
    int sessions, int tokens_per_session, int key_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const int grouped_rows = tokens_per_session * kGroup;
  const int query_rows = sessions * kKvHeads * grouped_rows;
  const int candidate_rows = sessions * kKvHeads * tokens_per_session;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = static_cast<int>(blockIdx.x) * 8 + warp;
  if (row < query_rows) {
    int value = row;
    const int grouped_row = value % grouped_rows;
    value /= grouped_rows;
    const int kv_head = value % kKvHeads;
    const int session = value / kKvHeads;
    const int step = grouped_row / kGroup;
    const int within = grouped_row % kGroup;
    const int source_row = session * tokens_per_session + step;
    const int query_head = kv_head * kGroup + within;
    const auto* source = reinterpret_cast<const uint4*>(
        queries + (static_cast<std::size_t>(source_row) * kQueryHeads +
                   query_head) * kHeadDim);
    auto* destination = reinterpret_cast<uint4*>(
        packed_queries + static_cast<std::size_t>(row) * kHeadDim);
    for (int vector = lane; vector < kHeadDim / 8; vector += 32)
      destination[vector] = source[vector];
  } else if (row < query_rows + candidate_rows) {
    int value = row - query_rows;
    const int step = value % tokens_per_session;
    value /= tokens_per_session;
    const int kv_head = value % kKvHeads;
    const int session = value / kKvHeads;
    const auto live = cache_views[session];
    const auto candidate = candidate_views[session];
    const std::size_t source_row =
        (static_cast<std::size_t>(session) * tokens_per_session + step) *
            kKvHeads + kv_head;
    const int slot = kRing
        ? (context_lengths[session] + step) & 1023
        : context_lengths[session] + step;
    const int first = kRing && context_lengths[session] > 1023
        ? (context_lengths[session] - 1023) & 1023 : 0;
    // For the ring path, place speculative rows only in the contiguous tail
    // following the second mirrored ring. This preserves canonical history
    // that earlier speculative query rows may still attend to.
    const std::size_t live_token = kRing
        ? static_cast<std::size_t>(
              1024 + first + min(context_lengths[session], 1023) + step)
        : static_cast<std::size_t>(slot);
    const std::size_t live_row = live_token * kKvHeads + kv_head;
    const std::size_t candidate_row =
        static_cast<std::size_t>(step) * kKvHeads + kv_head;
    const auto* source_key = reinterpret_cast<const uint4*>(
        new_keys + source_row * kHeadDim);
    const auto* source_value = reinterpret_cast<const uint4*>(
        new_values + source_row * kHeadDim);
    auto* live_key = reinterpret_cast<uint4*>(live.keys) +
                     live_row * (kHeadDim / 8);
    auto* live_value = reinterpret_cast<uint4*>(live.values) +
                       live_row * (kHeadDim / 8);
    auto* candidate_key = reinterpret_cast<uint4*>(candidate.keys) +
                          candidate_row * (kHeadDim / 8);
    auto* candidate_value = reinterpret_cast<uint4*>(candidate.values) +
                            candidate_row * (kHeadDim / 8);
    for (int vector = lane; vector < kHeadDim / 8; vector += 32) {
      const uint4 key = source_key[vector];
      const uint4 value_bits = source_value[vector];
      live_key[vector] = key;
      live_value[vector] = value_bits;
      candidate_key[vector] = key;
      candidate_value[vector] = value_bits;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x < sessions * kKvHeads) {
    const int batch = threadIdx.x;
    const int session = batch / kKvHeads;
    const int head = batch % kKvHeads;
    const auto live = cache_views[session];
    const int first = kRing && context_lengths[session] > 1023
        ? (context_lengths[session] - 1023) & 1023 : 0;
    key_pointers[batch] = reinterpret_cast<const __nv_bfloat16*>(live.keys) +
                          static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads *
                              kHeadDim + head * kHeadDim;
    value_pointers[batch] =
        reinterpret_cast<const __nv_bfloat16*>(live.values) +
        static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads * kHeadDim +
        head * kHeadDim;
    query_pointers[batch] = packed_queries +
        static_cast<std::size_t>(batch) * grouped_rows * kHeadDim;
    score_pointers[batch] = scores +
        static_cast<std::size_t>(batch) * key_tokens * grouped_rows;
    probability_pointers[batch] = probabilities +
        static_cast<std::size_t>(batch) * key_tokens * grouped_rows;
    output_pointers[batch] = packed_output +
        static_cast<std::size_t>(batch) * grouped_rows * kHeadDim;
  }
}

// Direct B4+ attention does not consume transformed Q/K/V in their original
// layouts. Normalize/scale/rotate each projected vector and write it directly
// to packed Q plus the transactional live/candidate KV destinations. Pointer
// setup shares the first CTA, replacing the following preparation launch.
template <int kHeadDim, int kKvHeads, bool kGlobal, bool kRing>
__global__ void transform_qkv_direct_attention_kernel(
    const __nv_bfloat16* queries, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const __nv_bfloat16* query_weight,
    const __nv_bfloat16* key_weight, int rows,
    const int* context_lengths, int tokens_per_session,
    const float* query_activation_scales, const float* query_weight_scale,
    const float* key_activation_scales, const float* key_weight_scale,
    const float* value_activation_scales, const float* value_weight_scale,
    const DeviceKvView* cache_views, const DeviceKvView* candidate_views,
    __nv_bfloat16* packed_queries, float* scores,
    __nv_bfloat16* probabilities, __nv_bfloat16* packed_output,
    const void** key_pointers, const void** query_pointers,
    void** score_pointers, const void** value_pointers,
    const void** probability_pointers, void** output_pointers,
    int sessions, int key_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  __shared__ float key_reductions[512];
  __shared__ float value_reductions[512];
  __shared__ __nv_bfloat16 key_normed[512];
  const int vector = blockIdx.x;
  const int query_vectors = rows * kQueryHeads;
  const bool is_query = vector < query_vectors;
  const int local_vector = is_query ? vector : vector - query_vectors;
  const int row = is_query ? vector / kQueryHeads : local_vector / kKvHeads;
  if (row >= rows) return;
  const int col = threadIdx.x;
  const int session = row / tokens_per_session;
  const int step = row % tokens_per_session;

  const auto* source = is_query ? queries : keys;
  const int source_vector = is_query ? vector : local_vector;
  float x = __bfloat162float(
      source[static_cast<std::size_t>(source_vector) * kHeadDim + col]);
  const float* activation_scales =
      is_query ? query_activation_scales : key_activation_scales;
  const float* output_weight_scale =
      is_query ? query_weight_scale : key_weight_scale;
  if (activation_scales) {
    const float scale = activation_scales[row] * output_weight_scale[0];
    x = __bfloat162float(__float2bfloat16_rn(x * scale));
  }
  key_reductions[col] = x * x;

  float value_x = 0.0F;
  if (!is_query) {
    value_x = kGlobal
        ? __bfloat162float(
              keys[static_cast<std::size_t>(local_vector) * kHeadDim + col])
        : __bfloat162float(
              values[static_cast<std::size_t>(local_vector) * kHeadDim + col]);
    const float* scales = kGlobal ? key_activation_scales
                                  : value_activation_scales;
    const float* weight_scale = kGlobal ? key_weight_scale
                                        : value_weight_scale;
    if (scales) {
      const float scale = scales[row] * weight_scale[0];
      value_x = __bfloat162float(__float2bfloat16_rn(value_x * scale));
    }
    value_reductions[col] = value_x * value_x;
  }
  __syncthreads();
  for (int stride = kHeadDim / 2; stride; stride >>= 1) {
    if (col < stride) {
      key_reductions[col] += key_reductions[col + stride];
      if (!is_query)
        value_reductions[col] += value_reductions[col + stride];
    }
    __syncthreads();
  }
  x *= rsqrtf(key_reductions[0] / kHeadDim + 1e-6F);
  x *= __bfloat162float((is_query ? query_weight : key_weight)[col]);
  key_normed[col] = __float2bfloat16_rn(x);
  if (!is_query)
    value_x = __bfloat162float(__float2bfloat16_rn(
        value_x * rsqrtf(value_reductions[0] / kHeadDim + 1e-6F)));
  __syncthreads();

  const int position = context_lengths[session] + step;
  const int half = kHeadDim / 2;
  const int half_col = col & (half - 1);
  const float frequency = kGlobal
      ? (half_col < 64
             ? powf(1000000.0F, -2.0F * half_col / kHeadDim)
             : 0.0F)
      : powf(10000.0F, -2.0F * half_col / kHeadDim);
  const auto cosine = __float2bfloat16_rn(cosf(position * frequency));
  const auto sine = __float2bfloat16_rn(sinf(position * frequency));
  const int paired = col < half ? col + half : col - half;
  const auto left = __float2bfloat16_rn(
      __bfloat162float(key_normed[col]) * __bfloat162float(cosine));
  const float rotated = col < half ? -__bfloat162float(key_normed[paired])
                                   : __bfloat162float(key_normed[paired]);
  const auto right = __float2bfloat16_rn(rotated * __bfloat162float(sine));
  const auto transformed = __float2bfloat16_rn(
      __bfloat162float(left) + __bfloat162float(right));

  if (is_query) {
    const int query_head = vector % kQueryHeads;
    const int kv_head = query_head / kGroup;
    const int within = query_head % kGroup;
    const int grouped_rows = tokens_per_session * kGroup;
    const int packed_row =
        (session * kKvHeads + kv_head) * grouped_rows + step * kGroup + within;
    packed_queries[static_cast<std::size_t>(packed_row) * kHeadDim + col] =
        transformed;
  } else {
    const int kv_head = local_vector % kKvHeads;
    const auto live = cache_views[session];
    const auto candidate = candidate_views[session];
    const int slot = kRing ? position & 1023 : position;
    const int first = kRing && context_lengths[session] > 1023
        ? (context_lengths[session] - 1023) & 1023 : 0;
    const std::size_t live_token = kRing
        ? static_cast<std::size_t>(
              1024 + first + min(context_lengths[session], 1023) + step)
        : static_cast<std::size_t>(slot);
    const std::size_t live_index =
        (live_token * kKvHeads + kv_head) * kHeadDim + col;
    const std::size_t candidate_index =
        (static_cast<std::size_t>(step) * kKvHeads + kv_head) * kHeadDim + col;
    reinterpret_cast<__nv_bfloat16*>(live.keys)[live_index] = transformed;
    reinterpret_cast<__nv_bfloat16*>(live.values)[live_index] =
        __float2bfloat16_rn(value_x);
    reinterpret_cast<__nv_bfloat16*>(candidate.keys)[candidate_index] =
        transformed;
    reinterpret_cast<__nv_bfloat16*>(candidate.values)[candidate_index] =
        __float2bfloat16_rn(value_x);
  }

  if (vector == 0 && col < sessions * kKvHeads) {
    const int batch = col;
    const int pointer_session = batch / kKvHeads;
    const int head = batch % kKvHeads;
    const auto live = cache_views[pointer_session];
    const int first = kRing && context_lengths[pointer_session] > 1023
        ? (context_lengths[pointer_session] - 1023) & 1023 : 0;
    const int grouped_rows = tokens_per_session * kGroup;
    key_pointers[batch] = reinterpret_cast<const __nv_bfloat16*>(live.keys) +
        static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads *
            kHeadDim + head * kHeadDim;
    value_pointers[batch] = reinterpret_cast<const __nv_bfloat16*>(live.values) +
        static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads *
            kHeadDim + head * kHeadDim;
    query_pointers[batch] = packed_queries +
        static_cast<std::size_t>(batch) * grouped_rows * kHeadDim;
    score_pointers[batch] = scores +
        static_cast<std::size_t>(batch) * key_tokens * grouped_rows;
    probability_pointers[batch] = probabilities +
        static_cast<std::size_t>(batch) * key_tokens * grouped_rows;
    output_pointers[batch] = packed_output +
        static_cast<std::size_t>(batch) * grouped_rows * kHeadDim;
  }
}

template <bool kRing, bool kIncludesCandidate = true, int kSlots = 0>
__global__ void ragged_gemm_softmax_kernel(
    const float* scores, __nv_bfloat16* probabilities,
    const int* context_lengths, int tokens_per_session, int kv_heads,
    int group_size, int key_tokens, int storage_stride = 0) {
  __shared__ float reductions[256];
  const int row = blockIdx.x;
  const int grouped_rows = tokens_per_session * group_size;
  const int session = row / (kv_heads * grouped_rows);
  const int grouped_row = row % grouped_rows;
  const int step = grouped_row / group_size;
  const int cached = context_lengths[session];
  constexpr int kCandidateRows = kIncludesCandidate ? 1 : 0;
  const int end = cached + step + kCandidateRows;
  const int pool_first = kRing
      ? (kIncludesCandidate
             ? (cached > 1023 ? cached - 1023 : 0)
             : (cached > 1024 ? cached - 1024 : 0))
      : 0;
  const int first = kRing && end > 1024 ? end - 1024 : 0;
  const int row_stride = storage_stride ? storage_stride : key_tokens;
  const auto* source = scores + static_cast<std::size_t>(row) * row_stride;
  auto* destination = probabilities + static_cast<std::size_t>(row) * row_stride;
  float maximum = -INFINITY;
  for (int local = threadIdx.x; local < key_tokens; local += blockDim.x) {
    const int logical = pool_first + local;
    if (logical >= first && logical < end)
      maximum = fmaxf(maximum, source[local]);
  }
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 128; stride >= (kSlots ? 32 : 1); stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x],
                                      reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  if constexpr (kSlots != 0) {
    if (threadIdx.x < 32) {
      float value = reductions[threadIdx.x];
      for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
      if (threadIdx.x == 0) reductions[0] = value;
    }
    __syncthreads();
  }
  maximum = reductions[0];
  // Ragged/masked rows can let warp zero reach the sum store first.
  __syncthreads();
  float denominator = 0.0F;
  __nv_bfloat16 held[kSlots > 0 ? kSlots : 1];
  if constexpr (kSlots > 0) {
    #pragma unroll
    for (int i = 0; i < kSlots; ++i) {
      const int local = threadIdx.x + i * 256;
      if (local < key_tokens) {
        const int logical = pool_first + local;
        const float p = logical >= first && logical < end ? expf(source[local] - maximum) : 0.0F;
        held[i] = __float2bfloat16_rn(p);
        denominator += p;
      }
    }
  } else {
  for (int local = threadIdx.x; local < key_tokens; local += blockDim.x) {
    const int logical = pool_first + local;
    const float probability = logical >= first && logical < end
        ? expf(source[local] - maximum) : 0.0F;
    destination[local] = __float2bfloat16_rn(probability);
    denominator += probability;
  }
  }
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride >= (kSlots ? 32 : 1); stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  if constexpr (kSlots != 0) {
    if (threadIdx.x < 32) {
      float value = reductions[threadIdx.x];
      for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (threadIdx.x == 0) reductions[0] = value;
    }
    __syncthreads();
  }
  denominator = reductions[0];
  if constexpr (kSlots > 0) {
    #pragma unroll
    for (int i = 0; i < kSlots; ++i) {
      const int local = threadIdx.x + i * 256;
      if (local < key_tokens)
        destination[local] = __float2bfloat16_rn(__bfloat162float(held[i]) / denominator);
    }
  } else {
  for (int local = threadIdx.x; local < key_tokens; local += blockDim.x)
    destination[local] = __float2bfloat16_rn(
        __bfloat162float(destination[local]) / denominator);
  }
}

template<bool Ring, bool Candidates>
__global__ void warp_gemm_softmax_kernel(const float* scores, __nv_bfloat16* probabilities,
    const int* contexts, int tokens, int heads, int group, int keys, int stride) {
  const int row = blockIdx.x, lane = threadIdx.x;
  const int cached = contexts[row / (heads * tokens * group)];
  const int step = (row % (tokens * group)) / group;
  const int end = cached + step + (Candidates ? 1 : 0);
  const int pool_first = Ring ? max(0, cached - (Candidates ? 1023 : 1024)) : 0;
  const int first = Ring ? max(0, end - 1024) : 0;
  stride = stride ? stride : keys;
  scores += static_cast<std::size_t>(row) * stride;
  probabilities += static_cast<std::size_t>(row) * stride;
  float values[8], sums[8];
  __nv_bfloat16 held[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = lane + 32 * i;
    values[i] = col < keys && pool_first + col >= first && pool_first + col < end ? scores[col] : -INFINITY;
  }
  // Reproduce the original 256-thread tree: strides 128,64,32 locally,
  // then strides 16,8,4,2,1 within the warp. No sum reassociation.
  #pragma unroll
  for (int s = 4; s; s >>= 1)
    #pragma unroll
    for (int i = 0; i < s; ++i) values[i] = fmaxf(values[i], values[i + s]);
  float maximum = values[0];
  for (int s = 16; s; s >>= 1)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, s));
  maximum = __shfl_sync(0xffffffff, maximum, 0);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = lane + 32 * i;
    const float p = col < keys && pool_first + col >= first && pool_first + col < end ? expf(scores[col] - maximum) : 0.0F;
    held[i] = __float2bfloat16_rn(p);
    sums[i] = p;
  }
  #pragma unroll
  for (int s = 4; s; s >>= 1)
    #pragma unroll
    for (int i = 0; i < s; ++i) sums[i] += sums[i + s];
  float denominator = sums[0];
  for (int s = 16; s; s >>= 1) denominator += __shfl_down_sync(0xffffffff, denominator, s);
  denominator = __shfl_sync(0xffffffff, denominator, 0);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = lane + 32 * i;
    if (col < keys) probabilities[col] = __float2bfloat16_rn(__bfloat162float(held[i]) / denominator);
  }
}

template<bool Ring, bool Candidates = true>
void launch_gemm_softmax(const float* scores, __nv_bfloat16* probabilities,
    const int* contexts, int tokens, int heads, int group, int keys,
    int stride, int rows, cudaStream_t stream, bool fast) {
  if (fast && keys <= 256)
    warp_gemm_softmax_kernel<Ring, Candidates><<<rows, 32, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
  else if (fast && keys <= 512)
    ragged_gemm_softmax_kernel<Ring, Candidates, 2><<<rows, 256, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
  else if (fast && keys <= 1024)
    ragged_gemm_softmax_kernel<Ring, Candidates, 4><<<rows, 256, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
  else if (fast && keys <= 1280)
    ragged_gemm_softmax_kernel<Ring, Candidates, 5><<<rows, 256, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
  else if (fast && keys <= 4096)
    ragged_gemm_softmax_kernel<Ring, Candidates, -1><<<rows, 256, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
  else
    ragged_gemm_softmax_kernel<Ring, Candidates><<<rows, 256, 0, stream>>>(
        scores, probabilities, contexts, tokens, heads, group, keys, stride);
}

template <int kHeadDim, int kKvHeads, bool kRestoreRing = false>
__global__ void unpack_ragged_gemm_attention_kernel(
    const __nv_bfloat16* packed, __nv_bfloat16* output, int sessions,
    int tokens_per_session, const DeviceKvView* cache_views = nullptr,
    const int* context_lengths = nullptr);

template <int kHeadDim, int kKvHeads, bool kRing>
__global__ void prepare_assistant_direct_attention_kernel(
    const __nv_bfloat16* queries, const __nv_bfloat16* query_norm,
    const DeviceKvView* cache_views, const int* context_lengths,
    __nv_bfloat16* packed_queries,
    float* scores, __nv_bfloat16* probabilities,
    __nv_bfloat16* packed_output, const void** key_pointers,
    const void** query_pointers, void** score_pointers,
    const void** value_pointers, const void** probability_pointers,
    void** output_pointers, int sessions, int key_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  __shared__ float reductions[512];
  __shared__ __nv_bfloat16 normed[512];
  const int vector = blockIdx.x;
  const int session = vector / kQueryHeads;
  const int query_head = vector % kQueryHeads;
  const int col = threadIdx.x;
  float x = __bfloat162float(
      queries[static_cast<std::size_t>(vector) * kHeadDim + col]);
  reductions[col] = x * x;
  __syncthreads();
  for (int stride = kHeadDim / 2; stride; stride >>= 1) {
    if (col < stride) reductions[col] += reductions[col + stride];
    __syncthreads();
  }
  x *= rsqrtf(reductions[0] / kHeadDim + 1e-6F);
  x *= __bfloat162float(query_norm[col]);
  normed[col] = __float2bfloat16_rn(x);
  __syncthreads();
  const int half = kHeadDim / 2;
  const int half_col = col & (half - 1);
  const float frequency = !kRing
      ? (half_col < 64
             ? powf(1000000.0F, -2.0F * half_col / kHeadDim)
             : 0.0F)
      : powf(10000.0F, -2.0F * half_col / kHeadDim);
  const int position = context_lengths[session];
  const auto cosine = __float2bfloat16_rn(cosf(position * frequency));
  const auto sine = __float2bfloat16_rn(sinf(position * frequency));
  const int paired = col < half ? col + half : col - half;
  const auto left = __float2bfloat16_rn(
      __bfloat162float(normed[col]) * __bfloat162float(cosine));
  const float rotated = col < half ? -__bfloat162float(normed[paired])
                                   : __bfloat162float(normed[paired]);
  const auto right = __float2bfloat16_rn(
      rotated * __bfloat162float(sine));
  const int kv_head = query_head / kGroup;
  const int within = query_head % kGroup;
  const int packed_row = (session * kKvHeads + kv_head) * kGroup + within;
  packed_queries[static_cast<std::size_t>(packed_row) * kHeadDim + col] =
      __float2bfloat16_rn(__bfloat162float(left) +
                         __bfloat162float(right));
  if (blockIdx.x == 0 && threadIdx.x < sessions * kKvHeads) {
    const int batch = threadIdx.x;
    const int session = batch / kKvHeads;
    const int head = batch % kKvHeads;
    const auto cache = cache_views[session];
    const int cached = context_lengths[session];
    const int first = kRing && cached > 1024 ? (cached - 1024) & 1023 : 0;
    key_pointers[batch] =
        reinterpret_cast<const __nv_bfloat16*>(cache.keys) +
        static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads *
            kHeadDim +
        head * kHeadDim;
    value_pointers[batch] =
        reinterpret_cast<const __nv_bfloat16*>(cache.values) +
        static_cast<std::size_t>(first + (kRing ? 1024 : 0)) * kKvHeads *
            kHeadDim +
        head * kHeadDim;
    query_pointers[batch] =
        packed_queries + static_cast<std::size_t>(batch) * kGroup * kHeadDim;
    score_pointers[batch] =
        scores + static_cast<std::size_t>(batch) * key_tokens * kGroup;
    probability_pointers[batch] =
        probabilities + static_cast<std::size_t>(batch) * key_tokens * kGroup;
    output_pointers[batch] =
        packed_output + static_cast<std::size_t>(batch) * kGroup * kHeadDim;
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
void launch_assistant_direct_gemm_attention(
    cublasHandle_t handle, const __nv_bfloat16* queries,
    const __nv_bfloat16* query_norm,
    const DeviceKvView* cache_views, const int* context_lengths,
    __nv_bfloat16* output, __nv_bfloat16* packed_queries, float* scores,
    __nv_bfloat16* probabilities, __nv_bfloat16* packed_output,
    const void** key_pointers, const void** query_pointers,
    void** score_pointers, const void** value_pointers,
    const void** probability_pointers, void** output_pointers,
    int sessions, int maximum_context, cudaStream_t stream) {
  constexpr int kGroup = 16 / kKvHeads;
  const int key_tokens = kRing ? std::min(maximum_context, 1024)
                               : maximum_context;
  const int batches = sessions * kKvHeads;
  prepare_assistant_direct_attention_kernel<kHeadDim, kKvHeads, kRing>
      <<<sessions * 16, kHeadDim, 0, stream>>>(
          queries, query_norm, cache_views, context_lengths, packed_queries,
          scores, probabilities, packed_output, key_pointers, query_pointers,
          score_pointers, value_pointers, probability_pointers,
          output_pointers, sessions, key_tokens);
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, key_tokens, kGroup, kHeadDim,
            &alpha, key_pointers, CUDA_R_16BF, kKvHeads * kHeadDim,
            query_pointers, CUDA_R_16BF, kHeadDim, &beta, score_pointers,
            CUDA_R_32F, key_tokens, batches, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "assistant direct GEMM attention QK");
  ragged_gemm_softmax_kernel<kRing, false>
      <<<batches * kGroup, 256, 0, stream>>>(
          scores, probabilities, context_lengths, 1, kKvHeads, kGroup,
          key_tokens);
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, kGroup, key_tokens,
            &alpha, value_pointers, CUDA_R_16BF, kKvHeads * kHeadDim,
            probability_pointers, CUDA_R_16BF, key_tokens, &beta,
            output_pointers, CUDA_R_16BF, kHeadDim, batches,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "assistant direct GEMM attention PV");
  unpack_ragged_gemm_attention_kernel<kHeadDim, kKvHeads, false>
      <<<(static_cast<std::size_t>(sessions) * 16 * kHeadDim + 255) / 256,
         256, 0, stream>>>(packed_output, output, sessions, 1, nullptr,
                            nullptr);
  check(cudaGetLastError(), "assistant direct GEMM attention launch");
}

template <int kHeadDim, int kKvHeads, bool kRestoreRing>
__global__ void unpack_ragged_gemm_attention_kernel(
    const __nv_bfloat16* packed, __nv_bfloat16* output, int sessions,
    int tokens_per_session, const DeviceKvView* cache_views,
    const int* context_lengths) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const std::size_t elements = static_cast<std::size_t>(sessions) *
      kKvHeads * tokens_per_session * kGroup * kHeadDim;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index +=
           static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t value = index;
    const int col = value % kHeadDim;
    value /= kHeadDim;
    const int grouped_row = value % (tokens_per_session * kGroup);
    value /= tokens_per_session * kGroup;
    const int kv_head = value % kKvHeads;
    const int session = value / kKvHeads;
    const int step = grouped_row / kGroup;
    const int within = grouped_row % kGroup;
    const int row = session * tokens_per_session + step;
    const int query_head = kv_head * kGroup + within;
    output[(static_cast<std::size_t>(row) * kQueryHeads + query_head) *
               kHeadDim + col] = packed[index];
  }
  if constexpr (kRestoreRing) {
    const int plane = kKvHeads * kHeadDim;
    const int restore_elements = sessions * tokens_per_session * plane;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < restore_elements; index += gridDim.x * blockDim.x) {
      const int column = index % plane;
      const int value = index / plane;
      const int step = value % tokens_per_session;
      const int session = value / tokens_per_session;
      const int cached = context_lengths[session];
      const int first = cached > 1023 ? (cached - 1023) & 1023 : 0;
      const int tail_token =
          1024 + first + min(cached, 1023) + step;
      const int canonical_token = tail_token & 1023;
      const auto view = cache_views[session];
      auto* keys = reinterpret_cast<__nv_bfloat16*>(view.keys);
      auto* values = reinterpret_cast<__nv_bfloat16*>(view.values);
      keys[static_cast<std::size_t>(tail_token) * plane + column] =
          keys[static_cast<std::size_t>(canonical_token) * plane + column];
      values[static_cast<std::size_t>(tail_token) * plane + column] =
          values[static_cast<std::size_t>(canonical_token) * plane + column];
    }
  }
}

template <int kHeadDim, int kKvHeads, bool kRing>
void launch_ragged_gemm_attention(
    cublasHandle_t handle, const __nv_bfloat16* queries,
    const DeviceKvView* cache_views, const DeviceKvView* candidate_views,
    const int* context_lengths, __nv_bfloat16* output,
    const __nv_bfloat16* new_keys, const __nv_bfloat16* new_values,
    __nv_bfloat16* packed_queries, __nv_bfloat16* packed_keys,
    __nv_bfloat16* packed_values, float* scores,
    __nv_bfloat16* probabilities, __nv_bfloat16* packed_output,
    int sessions, int tokens_per_session, int maximum_context,
    cudaStream_t stream, bool skip_unpack = false, bool fast_softmax = false) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const int grouped_rows = tokens_per_session * kGroup;
  const int key_tokens = kRing
      ? std::min(maximum_context, 1024 + tokens_per_session - 1)
      : maximum_context;
  const int batches = sessions * kKvHeads;
  const std::size_t query_elements = static_cast<std::size_t>(batches) *
      grouped_rows * kHeadDim;
  const int pack_blocks =
      (batches * (grouped_rows + key_tokens) + 7) / 8;
  pack_ragged_gemm_attention_kernel<kHeadDim, kKvHeads, kRing>
      <<<pack_blocks, 256, 0, stream>>>(
          queries, cache_views, candidate_views, context_lengths,
          new_keys, new_values,
          packed_queries, packed_keys, packed_values, sessions,
          tokens_per_session, key_tokens);
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, key_tokens, grouped_rows,
            kHeadDim, &alpha, packed_keys, CUDA_R_16BF, kHeadDim,
            static_cast<long long>(key_tokens) * kHeadDim, packed_queries,
            CUDA_R_16BF, kHeadDim,
            static_cast<long long>(grouped_rows) * kHeadDim, &beta, scores,
            CUDA_R_32F, key_tokens,
            static_cast<long long>(key_tokens) * grouped_rows, batches,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "ragged GEMM attention QK");
  launch_gemm_softmax<kRing>(
          scores, probabilities, context_lengths, tokens_per_session,
          kKvHeads, kGroup, key_tokens, 0, batches * grouped_rows, stream, fast_softmax);
  check(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, grouped_rows,
            key_tokens, &alpha, packed_values, CUDA_R_16BF, kHeadDim,
            static_cast<long long>(key_tokens) * kHeadDim, probabilities,
            CUDA_R_16BF, key_tokens,
            static_cast<long long>(key_tokens) * grouped_rows, &beta,
            packed_output, CUDA_R_16BF, kHeadDim,
            static_cast<long long>(grouped_rows) * kHeadDim, batches,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "ragged GEMM attention PV");
  const int unpack_blocks = static_cast<int>(std::min<std::size_t>(
      65535, (query_elements + 255) / 256));
  if (!skip_unpack) unpack_ragged_gemm_attention_kernel<kHeadDim, kKvHeads>
      <<<unpack_blocks, 256, 0, stream>>>(packed_output, output, sessions,
                                          tokens_per_session);
  check(cudaGetLastError(), "ragged GEMM attention launch");
}

template <int kHeadDim, int kKvHeads, bool kRing>
void launch_direct_gemm_attention(
    cublasHandle_t handle, const __nv_bfloat16* queries,
    const __nv_bfloat16* new_keys, const __nv_bfloat16* new_values,
    const DeviceKvView* cache_views, const DeviceKvView* candidate_views,
    const int* context_lengths, __nv_bfloat16* output,
    __nv_bfloat16* packed_queries, float* scores,
    __nv_bfloat16* probabilities, __nv_bfloat16* packed_output,
    const void** key_pointers, const void** query_pointers,
    void** score_pointers, const void** value_pointers,
    const void** probability_pointers, void** output_pointers,
    int sessions, int tokens_per_session, int maximum_context,
    cudaStream_t stream, bool prepared = false, float* tiled_partial = nullptr,
    bool skip_unpack = false, bool fast_softmax = false) {
  constexpr int kGroup = 16 / kKvHeads;
  const int grouped_rows = tokens_per_session * kGroup;
  const int batches = sessions * kKvHeads;
  const int storage_stride = std::getenv("GEVVA_ALIGNED_DIRECT_ATTENTION")
      ? (maximum_context + 31) & ~31 : maximum_context;
  if (!prepared) {
    const int preparation_rows =
        batches * (grouped_rows + tokens_per_session);
    prepare_direct_attention_kernel<kHeadDim, kKvHeads, kRing><<<
        (preparation_rows + 7) / 8, 256, 0, stream>>>(
        queries, new_keys, new_values, cache_views, candidate_views,
        context_lengths, packed_queries, scores, probabilities, packed_output,
        key_pointers, query_pointers, score_pointers, value_pointers,
        probability_pointers, output_pointers, sessions, tokens_per_session,
        storage_stride);
  }
  if (tiled_partial) {
    launch_serving_tiled_attention(query_pointers, key_pointers, value_pointers,
        context_lengths, packed_output, tiled_partial, sessions, tokens_per_session,
        kHeadDim, maximum_context, stream);
  } else {
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, maximum_context, grouped_rows,
            kHeadDim, &alpha, key_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, query_pointers, CUDA_R_16BF, kHeadDim,
            &beta, score_pointers, CUDA_R_32F, storage_stride, batches,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "direct GEMM attention QK");
  launch_gemm_softmax<kRing>(
          scores, probabilities, context_lengths, tokens_per_session,
          kKvHeads, kGroup, maximum_context, storage_stride, batches * grouped_rows, stream, fast_softmax);
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, grouped_rows,
            maximum_context, &alpha, value_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, probability_pointers, CUDA_R_16BF,
            storage_stride, &beta, output_pointers, CUDA_R_16BF, kHeadDim,
            batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "direct GEMM attention PV");
  }
  if (const char* directory = std::getenv("GEVVA_CAPTURE_ATTENTION")) {
    capture_attention(std::string(directory) + "/d" + std::to_string(kHeadDim) +
        "-b" + std::to_string(sessions) + ".bin", query_pointers, key_pointers,
        value_pointers, context_lengths, packed_output, sessions,
        tokens_per_session, kHeadDim, kKvHeads, maximum_context, kRing, stream);
  }
  const std::size_t output_elements = static_cast<std::size_t>(sessions) *
      kKvHeads * grouped_rows * kHeadDim;
  if (!skip_unpack) unpack_ragged_gemm_attention_kernel<kHeadDim, kKvHeads, kRing>
      <<<static_cast<int>((output_elements + 255) / 256), 256, 0, stream>>>(
          packed_output, output, sessions, tokens_per_session,
          kRing ? cache_views : nullptr,
          kRing ? context_lengths : nullptr);
  check(cudaGetLastError(), "direct GEMM attention launch");
}

