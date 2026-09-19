// Private implementation fragment; included only by src/gpu.cu.
void stage_candidate_kv(CandidateKvCache& candidates, int layer,
                        const void* keys, const void* values, int tokens,
                        cudaStream_t stream) {
  if (tokens < 1 || tokens > CandidateKvCache::kMaximumTokens)
    throw std::runtime_error("candidate KV token count must be in [1, 5]");
  const auto& candidate = candidates.layer(layer);
  const std::size_t bytes = static_cast<std::size_t>(tokens) *
                            candidate.kv_heads * candidate.head_dim * 2;
  check(cudaMemcpyAsync(candidate.keys, keys, bytes, cudaMemcpyDeviceToDevice,
                        stream),
        "stage candidate K");
  check(cudaMemcpyAsync(candidate.values, values, bytes, cudaMemcpyDeviceToDevice,
                        stream),
        "stage candidate V");
}

__global__ void commit_candidate_kv_kernel(
    const DeviceKvView* candidates, const DeviceKvView* destinations,
    int position, int fixed_tokens, const int* drafts = nullptr,
    const int* targets = nullptr, const __nv_bfloat16* hidden = nullptr,
    __nv_bfloat16* continuation = nullptr, DeviceMtpResult* result = nullptr,
    int draft_count = 4) {
  const int layer = blockIdx.x;
  if (layer >= 30) return;
  __shared__ int matched_drafts;
  if (threadIdx.x == 0) {
    int matched = 0;
    if (drafts)
      while (matched < draft_count && drafts[matched] == targets[matched])
        ++matched;
    matched_drafts = matched;
    if (layer == 0 && result) {
      for (int index = 0; index < 5; ++index) {
        result->target_tokens[index] = 0;
        result->output_tokens[index] = 0;
      }
      for (int index = 0; index < 4; ++index)
        result->draft_tokens[index] = 0;
      result->matched_drafts = matched;
      result->output_count = matched + 1;
      for (int index = 0; index < matched; ++index)
        result->output_tokens[index] = drafts[index];
      result->output_tokens[matched] = targets[matched];
      for (int index = 0; index < draft_count + 1; ++index)
        result->target_tokens[index] = targets[index];
      for (int index = 0; index < draft_count; ++index)
        result->draft_tokens[index] = drafts[index];
    }
  }
  __syncthreads();
  if (layer == 0 && continuation) {
    constexpr int kHidden = 2816;
    for (int column = threadIdx.x; column < kHidden; column += blockDim.x)
      continuation[column] = hidden[matched_drafts * kHidden + column];
  }
  const DeviceKvView source = candidates[layer];
  const DeviceKvView destination = destinations[layer];
  const int tokens = drafts ? matched_drafts + 1 : fixed_tokens;
  const int plane = source.kv_heads * source.head_dim;
  for (int index = threadIdx.x; index < tokens * plane;
       index += blockDim.x) {
    const int token = index / plane;
    const int column = index % plane;
    const int slot = (position + token) % destination.capacity;
    const int destination_index = slot * plane + column;
    const auto key = reinterpret_cast<const __nv_bfloat16*>(source.keys)[index];
    const auto value =
        reinterpret_cast<const __nv_bfloat16*>(source.values)[index];
    auto* destination_keys =
        reinterpret_cast<__nv_bfloat16*>(destination.keys);
    auto* destination_values =
        reinterpret_cast<__nv_bfloat16*>(destination.values);
    destination_keys[destination_index] = key;
    destination_values[destination_index] = value;
    if (destination.kv_heads == 8 && destination.capacity == 1024) {
      for (int copy = 1; copy < 3; ++copy) {
        const int mirrored = destination_index + copy * destination.capacity * plane;
        destination_keys[mirrored] = key;
        destination_values[mirrored] = value;
      }
      if (slot < 4) {
        const int mirrored = destination_index + 3 * destination.capacity * plane;
        destination_keys[mirrored] = key;
        destination_values[mirrored] = value;
      }
    }
  }
}

void commit_candidate_kv(const CandidateKvCache& candidates, KvCache& cache,
                         int position, int tokens, cudaStream_t stream) {
  if (position < 0 || tokens < 1 ||
      tokens > CandidateKvCache::kMaximumTokens ||
      position + tokens > cache.maximum_context())
    throw std::runtime_error("candidate KV commit geometry is invalid");
  commit_candidate_kv_kernel<<<30, 256, 0, stream>>>(
      candidates.device_layers(), cache.device_layers(), position, tokens);
  check(cudaGetLastError(), "commit candidate KV kernel");
}

void launch_qkv_transform(bool global_layer, void* query, void* key, void* value,
                          const void* query_norm_weight, const void* key_norm_weight,
                          int position, void* key_cache, void* value_cache,
                          cudaStream_t stream) {
  if (position < 0 || position >= 262144)
    throw std::runtime_error("QKV position must be in [0, 262143]");
  auto* q = static_cast<__nv_bfloat16*>(query);
  auto* k = static_cast<__nv_bfloat16*>(key);
  auto* v = static_cast<__nv_bfloat16*>(value);
  const auto* q_norm = static_cast<const __nv_bfloat16*>(query_norm_weight);
  const auto* k_norm = static_cast<const __nv_bfloat16*>(key_norm_weight);
  if (global_layer) {
    constexpr int kHeads = 16;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 512;
    constexpr std::size_t kKvBytes = kKvHeads * kHeadDim * sizeof(__nv_bfloat16);
    check(cudaMemcpyAsync(v, k, kKvBytes, cudaMemcpyDeviceToDevice, stream),
          "copy global K projection to V");
    norm_rope_512_global_kernel<<<kHeads, 512, 0, stream>>>(
        q, q_norm, kHeads, 1, kHeads, true, position);
    norm_rope_512_global_kernel<<<kKvHeads, 512, 0, stream>>>(
        k, k_norm, kKvHeads, 1, kKvHeads, true, position);
    head_rmsnorm_kernel<kHeadDim><<<kKvHeads, 512, 0, stream>>>(v, kKvHeads);
    check(cudaMemcpyAsync(static_cast<std::byte*>(key_cache) + position * kKvBytes,
                          k, kKvBytes, cudaMemcpyDeviceToDevice, stream),
          "append global K cache");
    check(cudaMemcpyAsync(static_cast<std::byte*>(value_cache) + position * kKvBytes,
                          v, kKvBytes, cudaMemcpyDeviceToDevice, stream),
          "append global V cache");
  } else {
    constexpr int kHeads = 16;
    constexpr int kKvHeads = 8;
    constexpr int kHeadDim = 256;
    norm_rope_256_kernel<<<kHeads, 256, 0, stream>>>(
        q, q_norm, kHeads, 1, kHeads, true, position);
    norm_rope_256_kernel<<<kKvHeads, 256, 0, stream>>>(
        k, k_norm, kKvHeads, 1, kKvHeads, true, position);
    head_rmsnorm_kernel<kHeadDim><<<kKvHeads, 256, 0, stream>>>(v, kKvHeads);
    append_qkv_batch(false, k, v, position, 1, key_cache, value_cache,
                     stream);
  }
  check(cudaGetLastError(), "QKV transform launch");
}

