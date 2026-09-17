// Private implementation fragment; included only by src/gpu.cu.
__global__ void prefill_softmax_kernel(__nv_bfloat16* scores,
                                       const std::int32_t* block_ids,
                                       int query_tokens, int key_tokens,
                                       int query_position_base,
                                       int key_position_base,
                                       int block_position_base,
                                       int block_tokens, bool sliding) {
  __shared__ float reductions[256];
  const int query = blockIdx.x % query_tokens;
  const int absolute_query = query_position_base + query;
  auto* values = scores + static_cast<std::size_t>(blockIdx.x) * key_tokens;
  const int query_group = block_ids[absolute_query - block_position_base];
  auto allowed = [&](int key) {
    const int absolute_key = key_position_base + key;
    const bool causal = absolute_key <= absolute_query &&
                        (!sliding || absolute_key > absolute_query - 1024);
    const int local_key = absolute_key - block_position_base;
    const bool same_block = query_group >= 0 && local_key >= 0 &&
                            local_key < block_tokens &&
                            block_ids[local_key] == query_group;
    return causal || same_block;
  };
  float maximum = -INFINITY;
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x)
    if (allowed(key)) maximum = fmaxf(maximum, __bfloat162float(values[key]));
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
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x)
    if (allowed(key)) denominator += expf(__bfloat162float(values[key]) - maximum);
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = reductions[0];
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x) {
    values[key] = allowed(key)
        ? __float2bfloat16_rn(
              expf(__bfloat162float(values[key]) - maximum) / denominator)
        : __float2bfloat16_rn(0.0F);
  }
}

void launch_prefill_attention_legacy(
    cublasHandle_t handle, bool global, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* scores, const std::int32_t* block_ids,
    __nv_bfloat16* output, int query_tokens, int key_tokens,
    int query_position_base, int key_position_base,
    int block_position_base, int block_tokens, cudaStream_t stream) {
  const int dim = global ? 512 : 256;
  const int kv_heads = global ? 2 : 8;
  constexpr int q_heads = 16;
  const int q_width = q_heads * dim;
  const int kv_width = kv_heads * dim;
  const int group_size = q_heads / kv_heads;
  const float alpha = 1.0F, beta = 0.0F;
  for (int head = 0; head < q_heads; ++head) {
    const int kv_head = head / group_size;
    check(cublasGemmEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, key_tokens, query_tokens, dim,
              &alpha, key + kv_head * dim, CUDA_R_16BF, kv_width,
              query + head * dim, CUDA_R_16BF, q_width, &beta,
              scores + static_cast<std::size_t>(head) * query_tokens * key_tokens,
              CUDA_R_16BF, key_tokens, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "prefill QK");
  }
  prefill_softmax_kernel<<<q_heads * query_tokens, 256, 0, stream>>>(
      scores, block_ids, query_tokens, key_tokens, query_position_base,
      key_position_base, block_position_base, block_tokens, !global);
  for (int head = 0; head < q_heads; ++head) {
    const int kv_head = head / group_size;
    check(cublasGemmEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, query_tokens, key_tokens,
              &alpha, value + kv_head * dim, CUDA_R_16BF, kv_width,
              scores + static_cast<std::size_t>(head) * query_tokens * key_tokens,
              CUDA_R_16BF, key_tokens, &beta, output + head * dim,
              CUDA_R_16BF, q_width, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "prefill PV");
  }
}

template <int kHeadDim, int kKvHeads>
__global__ void prepare_prefill_attention_pointers_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* scores,
    __nv_bfloat16* output, const void** key_pointers,
    const void** query_pointers, void** score_pointers,
    const void** value_pointers, const void** probability_pointers,
    void** output_pointers, int query_tokens, int key_tokens) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const int head = threadIdx.x;
  if (head >= kQueryHeads) return;
  const int kv_head = head / kGroup;
  key_pointers[head] = key + kv_head * kHeadDim;
  query_pointers[head] = query + head * kHeadDim;
  score_pointers[head] =
      scores + static_cast<std::size_t>(head) * query_tokens * key_tokens;
  value_pointers[head] = value + kv_head * kHeadDim;
  probability_pointers[head] = score_pointers[head];
  output_pointers[head] = output + head * kHeadDim;
}

template <int kHeadDim, int kKvHeads, bool kGlobal>
__global__ void prepare_uniform_batch_prefill_attention_pointers_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* const* cache_keys,
    __nv_bfloat16* const* cache_values,
    const __nv_bfloat16* combined_keys,
    const __nv_bfloat16* combined_values, __nv_bfloat16* scores,
    __nv_bfloat16* output, const void** key_pointers,
    const void** query_pointers, void** score_pointers,
    const void** value_pointers, const void** probability_pointers,
    void** output_pointers, int sessions, int query_tokens, int key_tokens,
    int prefix_tokens, int combined_stride) {
  constexpr int kQueryHeads = 16;
  constexpr int kGroup = kQueryHeads / kKvHeads;
  const int matrix = blockIdx.x * blockDim.x + threadIdx.x;
  if (matrix >= sessions * kQueryHeads) return;
  const int session = matrix / kQueryHeads;
  const int head = matrix % kQueryHeads;
  const int kv_head = head / kGroup;
  const int row = session * query_tokens;
  const auto* session_key = kGlobal
      ? cache_keys[session]
      : (prefix_tokens
             ? combined_keys + static_cast<std::size_t>(session) *
                                   combined_stride * kKvHeads * kHeadDim
             : key + static_cast<std::size_t>(row) * kKvHeads * kHeadDim);
  const auto* session_value = kGlobal
      ? cache_values[session]
      : (prefix_tokens
             ? combined_values + static_cast<std::size_t>(session) *
                                     combined_stride * kKvHeads * kHeadDim
             : value + static_cast<std::size_t>(row) * kKvHeads * kHeadDim);
  key_pointers[matrix] = session_key + kv_head * kHeadDim;
  query_pointers[matrix] =
      query + static_cast<std::size_t>(row) * kQueryHeads * kHeadDim +
      head * kHeadDim;
  score_pointers[matrix] =
      scores + static_cast<std::size_t>(matrix) * query_tokens * key_tokens;
  value_pointers[matrix] = session_value + kv_head * kHeadDim;
  probability_pointers[matrix] = score_pointers[matrix];
  output_pointers[matrix] =
      output + static_cast<std::size_t>(row) * kQueryHeads * kHeadDim +
      head * kHeadDim;
}

__global__ void uniform_batch_prefill_softmax_kernel(
    __nv_bfloat16* scores, const std::int32_t* block_ids,
    int query_tokens, int key_tokens, int prefix_tokens, int sessions,
    bool sliding) {
  __shared__ float reductions[256];
  constexpr int kQueryHeads = 16;
  const int matrix = blockIdx.x / query_tokens;
  const int query = blockIdx.x % query_tokens;
  const int session = matrix / kQueryHeads;
  if (session >= sessions) return;
  const int absolute_query = prefix_tokens + query;
  auto* values = scores + static_cast<std::size_t>(blockIdx.x) * key_tokens;
  const auto* session_blocks = block_ids + session * query_tokens;
  const int query_group = session_blocks[query];
  auto allowed = [&](int key) {
    const int key_position_base = sliding
        ? prefix_tokens - (key_tokens - query_tokens)
        : 0;
    const int absolute_key = key_position_base + key;
    const bool causal = absolute_key <= absolute_query &&
                        (!sliding || absolute_key > absolute_query - 1024);
    const int local_key = absolute_key - prefix_tokens;
    const bool same_block = query_group >= 0 && local_key >= 0 &&
                            local_key < query_tokens &&
                            session_blocks[local_key] == query_group;
    return causal || same_block;
  };
  float maximum = -INFINITY;
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x)
    if (allowed(key)) maximum = fmaxf(maximum, __bfloat162float(values[key]));
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
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x)
    if (allowed(key))
      denominator += expf(__bfloat162float(values[key]) - maximum);
  reductions[threadIdx.x] = denominator;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = reductions[0];
  for (int key = threadIdx.x; key < key_tokens; key += blockDim.x)
    values[key] = allowed(key)
        ? __float2bfloat16_rn(
              expf(__bfloat162float(values[key]) - maximum) / denominator)
        : __float2bfloat16_rn(0.0F);
}

template <int kHeadDim, int kKvHeads, bool kGlobal>
void launch_uniform_batch_prefill_attention(
    cublasHandle_t handle, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* const* cache_keys,
    __nv_bfloat16* const* cache_values,
    const __nv_bfloat16* combined_keys,
    const __nv_bfloat16* combined_values, __nv_bfloat16* scores,
    const std::int32_t* block_ids, __nv_bfloat16* output,
    void* pointer_workspace, int sessions, int query_tokens, int key_tokens,
    int prefix_tokens, int combined_stride, cudaStream_t stream) {
  constexpr int kQueryHeads = 16;
  const int matrices = sessions * kQueryHeads;
  auto* pointer_bytes = static_cast<std::byte*>(pointer_workspace);
  auto** key_pointers = reinterpret_cast<const void**>(pointer_bytes);
  auto** query_pointers = reinterpret_cast<const void**>(
      pointer_bytes + matrices * sizeof(void*));
  auto** score_pointers = reinterpret_cast<void**>(
      pointer_bytes + 2 * matrices * sizeof(void*));
  auto** value_pointers = reinterpret_cast<const void**>(
      pointer_bytes + 3 * matrices * sizeof(void*));
  auto** probability_pointers = reinterpret_cast<const void**>(
      pointer_bytes + 4 * matrices * sizeof(void*));
  auto** output_pointers = reinterpret_cast<void**>(
      pointer_bytes + 5 * matrices * sizeof(void*));
  prepare_uniform_batch_prefill_attention_pointers_kernel<
      kHeadDim, kKvHeads, kGlobal><<<(matrices + 127) / 128, 128, 0, stream>>>(
      query, key, value, cache_keys, cache_values, combined_keys,
      combined_values, scores, output, key_pointers, query_pointers,
      score_pointers, value_pointers, probability_pointers, output_pointers,
      sessions, query_tokens, key_tokens, prefix_tokens, combined_stride);
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, key_tokens, query_tokens,
            kHeadDim, &alpha, key_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, query_pointers, CUDA_R_16BF,
            kQueryHeads * kHeadDim, &beta, score_pointers, CUDA_R_16BF,
            key_tokens, matrices, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "uniform batch prefill QK");
  uniform_batch_prefill_softmax_kernel<<<
      matrices * query_tokens, 256, 0, stream>>>(
      scores, block_ids, query_tokens, key_tokens, prefix_tokens, sessions,
      !kGlobal);
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, query_tokens,
            key_tokens, &alpha, value_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, probability_pointers, CUDA_R_16BF,
            key_tokens, &beta, output_pointers, CUDA_R_16BF,
            kQueryHeads * kHeadDim, matrices, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "uniform batch prefill PV");
  check(cudaGetLastError(), "uniform batch prefill attention launch");
}

template <int kHeadDim, int kKvHeads>
void launch_prefill_attention_pointer_batched(
    cublasHandle_t handle, const __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* scores, const std::int32_t* block_ids,
    __nv_bfloat16* output, void* pointer_workspace,
    int query_tokens, int key_tokens, int query_position_base,
    int key_position_base, int block_position_base, int block_tokens,
    bool sliding, cudaStream_t stream) {
  constexpr int kQueryHeads = 16;
  auto* pointer_bytes = static_cast<std::byte*>(pointer_workspace);
  auto** key_pointers = reinterpret_cast<const void**>(pointer_bytes);
  auto** query_pointers = reinterpret_cast<const void**>(
      pointer_bytes + 16 * sizeof(void*));
  auto** score_pointers = reinterpret_cast<void**>(
      pointer_bytes + 32 * sizeof(void*));
  auto** value_pointers = reinterpret_cast<const void**>(
      pointer_bytes + 48 * sizeof(void*));
  auto** probability_pointers = reinterpret_cast<const void**>(
      pointer_bytes + 64 * sizeof(void*));
  auto** output_pointers = reinterpret_cast<void**>(
      pointer_bytes + 80 * sizeof(void*));
  prepare_prefill_attention_pointers_kernel<kHeadDim, kKvHeads>
      <<<1, 32, 0, stream>>>(
          query, key, value, scores, output, key_pointers, query_pointers,
          score_pointers, value_pointers, probability_pointers,
          output_pointers, query_tokens, key_tokens);
  const float alpha = 1.0F, beta = 0.0F;
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, key_tokens, query_tokens,
            kHeadDim, &alpha, key_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, query_pointers, CUDA_R_16BF,
            kQueryHeads * kHeadDim, &beta, score_pointers, CUDA_R_16BF,
            key_tokens, kQueryHeads, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "pointer-batched prefill QK");
  prefill_softmax_kernel<<<kQueryHeads * query_tokens, 256, 0, stream>>>(
      scores, block_ids, query_tokens, key_tokens, query_position_base,
      key_position_base, block_position_base, block_tokens, sliding);
  check(cublasGemmBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, query_tokens,
            key_tokens, &alpha, value_pointers, CUDA_R_16BF,
            kKvHeads * kHeadDim, probability_pointers, CUDA_R_16BF,
            key_tokens, &beta, output_pointers, CUDA_R_16BF,
            kQueryHeads * kHeadDim, kQueryHeads, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "pointer-batched prefill PV");
  check(cudaGetLastError(), "pointer-batched prefill attention launch");
}

void launch_prefill_attention_impl(
    cublasHandle_t handle, bool global, __nv_bfloat16* query,
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* scores, const std::int32_t* block_ids,
    __nv_bfloat16* output, __nv_bfloat16* packed_output,
    int query_tokens, int key_tokens,
    int query_position_base, int key_position_base,
    int block_position_base, int block_tokens, cudaStream_t stream,
    bool experimental_text = false) {
  if (experimental_text && !global) {
    const int visible_keys = std::min(key_tokens,
        query_position_base - key_position_base + query_tokens);
    g4::launch_cudnn_text_prefill(query, key, value, output, 1,
                                 query_tokens, stream, visible_keys);
    return;
  }
  if (std::getenv("G4_DISABLE_BATCHED_PREFILL_ATTENTION")) {
    launch_prefill_attention_legacy(
        handle, global, query, key, value, scores, block_ids, output,
        query_tokens, key_tokens, query_position_base, key_position_base,
        block_position_base, block_tokens, stream);
    return;
  }
  if (global)
    launch_prefill_attention_pointer_batched<512, 2>(
        handle, query, key, value, scores, block_ids, output, packed_output,
        query_tokens, key_tokens, query_position_base, key_position_base,
        block_position_base, block_tokens, false, stream);
  else
    launch_prefill_attention_pointer_batched<256, 8>(
        handle, query, key, value, scores, block_ids, output, packed_output,
        query_tokens, key_tokens, query_position_base, key_position_base,
        block_position_base, block_tokens, true, stream);
}

