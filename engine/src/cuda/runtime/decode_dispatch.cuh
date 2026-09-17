// Private implementation fragment; included only by src/gpu.cu.
void launch_decode_attention(bool global_layer, const void* query, const void* keys,
                             const void* values, void* output, int context_tokens,
                             float* score_workspace, float* partial_workspace,
                             cudaStream_t stream) {
  if (context_tokens < 1 || context_tokens > 262144)
    throw std::runtime_error("decode context must be in [1, 262144]");
  constexpr int kQueryHeads = 16;
  const auto* q = static_cast<const __nv_bfloat16*>(query);
  const auto* k = static_cast<const __nv_bfloat16*>(keys);
  const auto* v = static_cast<const __nv_bfloat16*>(values);
  auto* out = static_cast<__nv_bfloat16*>(output);
  auto launch = [&]<int kHeadDim, int kKvHeads, int kWindow>() {
    constexpr bool kUseRing = kWindow > 0;
    const int first = kWindow > 0 && context_tokens > kWindow ? context_tokens - kWindow : 0;
    const int attended_tokens = context_tokens - first;
    if (attended_tokens <= 4) {
      decode_attention_kernel<kHeadDim, kKvHeads, kWindow><<<kQueryHeads, 32, 0, stream>>>(
          q, k, v, out, context_tokens);
      return;
    }
    constexpr int kWarps = 8;
    decode_attention_scores_kernel<kHeadDim, kKvHeads, kUseRing>
        <<<dim3((attended_tokens + kWarps - 1) / kWarps, kQueryHeads),
           kWarps * 32, 0, stream>>>(q, k, score_workspace, first, attended_tokens);
    decode_attention_softmax_kernel<<<kQueryHeads, 256, 0, stream>>>(
        score_workspace, attended_tokens);
    const int partitions = std::min(16, (attended_tokens + 255) / 256);
    if (partitions == 1) {
      decode_attention_values_kernel<kHeadDim, kKvHeads, kUseRing>
          <<<kQueryHeads, 256, 0, stream>>>(
              score_workspace, v, out, first, attended_tokens);
    } else {
      decode_attention_split_values_kernel<kHeadDim, kKvHeads, kUseRing>
          <<<dim3(kQueryHeads, partitions), 256, 0, stream>>>(
              score_workspace, v, partial_workspace, first, attended_tokens, partitions);
      decode_attention_reduce_values_kernel<kHeadDim><<<kQueryHeads, 256, 0, stream>>>(
          partial_workspace, out, partitions);
    }
  };
  if (global_layer)
    launch.template operator()<512, 2, 0>();
  else
    launch.template operator()<256, 8, 1024>();
}

void launch_decode_attention_batch(bool global_layer, const void* queries,
                                   const void* keys, const void* values,
                                   const void* candidate_keys,
                                   const void* candidate_values, void* outputs,
                                   int context_tokens, int tokens,
                                   float* score_workspace,
                                   float* partial_workspace,
                                   cudaStream_t stream) {
  if (tokens < 1 || tokens > 5 || context_tokens < 1 ||
      context_tokens + tokens - 1 > 262144)
    throw std::runtime_error("batched decode attention geometry is invalid");
  const int maximum_context = context_tokens + tokens - 1;
  const int score_stride = global_layer ? maximum_context
                                        : std::min(maximum_context, 1024);
  const int partitions = std::min(16, (score_stride + 255) / 256);
  constexpr int kWarps = 8;
  const dim3 score_grid((score_stride + kWarps - 1) / kWarps, 16, tokens);
  const dim3 softmax_grid(16, tokens);
  const dim3 value_grid(16, partitions, tokens);
  const dim3 reduce_grid(16, tokens);
  if (global_layer) {
    decode_attention_scores_batch_kernel<512, 2, false>
        <<<score_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(queries),
            static_cast<const __nv_bfloat16*>(keys),
            static_cast<const __nv_bfloat16*>(candidate_keys), score_workspace,
            context_tokens, tokens, score_stride);
    decode_attention_softmax_batch_kernel<false><<<softmax_grid, 256, 0, stream>>>(
        score_workspace, context_tokens, tokens, score_stride);
    decode_attention_split_values_batch_kernel<512, 2, false>
        <<<value_grid, 256, 0, stream>>>(
            score_workspace, static_cast<const __nv_bfloat16*>(values),
            static_cast<const __nv_bfloat16*>(candidate_values),
            partial_workspace, context_tokens, tokens, score_stride, partitions,
            partitions == 1 ? static_cast<__nv_bfloat16*>(outputs) : nullptr);
    if (partitions > 1)
      decode_attention_reduce_values_batch_kernel<512>
          <<<reduce_grid, 256, 0, stream>>>(
              partial_workspace, static_cast<__nv_bfloat16*>(outputs), tokens,
              partitions);
  } else {
    decode_attention_scores_batch_kernel<256, 8, true>
        <<<score_grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(queries),
            static_cast<const __nv_bfloat16*>(keys),
            static_cast<const __nv_bfloat16*>(candidate_keys), score_workspace,
            context_tokens, tokens, score_stride);
    decode_attention_softmax_batch_kernel<true><<<softmax_grid, 256, 0, stream>>>(
        score_workspace, context_tokens, tokens, score_stride);
    decode_attention_split_values_batch_kernel<256, 8, true>
        <<<value_grid, 256, 0, stream>>>(
            score_workspace, static_cast<const __nv_bfloat16*>(values),
            static_cast<const __nv_bfloat16*>(candidate_values),
            partial_workspace, context_tokens, tokens, score_stride, partitions,
            partitions == 1 ? static_cast<__nv_bfloat16*>(outputs) : nullptr);
    if (partitions > 1)
      decode_attention_reduce_values_batch_kernel<256>
          <<<reduce_grid, 256, 0, stream>>>(
              partial_workspace, static_cast<__nv_bfloat16*>(outputs), tokens,
              partitions);
  }
  check(cudaGetLastError(), "batched decode attention launch");
}

