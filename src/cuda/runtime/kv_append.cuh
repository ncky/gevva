// Private implementation fragment; included only by src/gpu.cu.
void transform_qkv_batch(bool global_layer, __nv_bfloat16* query,
                         __nv_bfloat16* key, __nv_bfloat16* value,
                         const __nv_bfloat16* query_norm,
                         const __nv_bfloat16* key_norm, int position,
                         int tokens, cudaStream_t stream) {
  if (global_layer) {
    constexpr int kHeads = 16;
    constexpr int kKvHeads = 2;
    constexpr int kDim = 512;
    check(cudaMemcpyAsync(value, key, tokens * kKvHeads * kDim * 2,
                          cudaMemcpyDeviceToDevice, stream),
          "copy batched global K to V");
    norm_rope_512_global_kernel<<<tokens * kHeads, 512, 0, stream>>>(
        query, query_norm, tokens * kHeads, tokens, kHeads, true, position);
    norm_rope_512_global_kernel<<<tokens * kKvHeads, 512, 0, stream>>>(
        key, key_norm, tokens * kKvHeads, tokens, kKvHeads, true, position);
    head_rmsnorm_kernel<kDim><<<tokens * kKvHeads, 512, 0, stream>>>(
        value, tokens * kKvHeads);
  } else {
    constexpr int kHeads = 16;
    constexpr int kKvHeads = 8;
    constexpr int kDim = 256;
    norm_rope_256_kernel<<<tokens * kHeads, 256, 0, stream>>>(
        query, query_norm, tokens * kHeads, tokens, kHeads, true, position);
    norm_rope_256_kernel<<<tokens * kKvHeads, 256, 0, stream>>>(
        key, key_norm, tokens * kKvHeads, tokens, kKvHeads, true, position);
    head_rmsnorm_kernel<kDim><<<tokens * kKvHeads, 256, 0, stream>>>(
        value, tokens * kKvHeads);
  }
}

__global__ void append_sliding_duplicated_kv_kernel(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    __nv_bfloat16* cache_keys, __nv_bfloat16* cache_values,
    int position, int source_begin, int tokens, int plane) {
  const int elements = (tokens - source_begin) * plane;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += gridDim.x * blockDim.x) {
    const int source_token = source_begin + index / plane;
    const int column = index % plane;
    const int slot = (position + source_token) & 1023;
    const auto key = keys[static_cast<std::size_t>(source_token) * plane +
                          column];
    const auto value = values[static_cast<std::size_t>(source_token) * plane +
                              column];
    for (int copy = 0; copy < 3; ++copy) {
      const std::size_t destination =
          (static_cast<std::size_t>(slot + copy * 1024) * plane) + column;
      cache_keys[destination] = key;
      cache_values[destination] = value;
    }
    if (slot < 4) {
      const std::size_t destination =
          (static_cast<std::size_t>(slot + 3 * 1024) * plane) + column;
      cache_keys[destination] = key;
      cache_values[destination] = value;
    }
  }
}

__global__ void append_uniform_sliding_duplicated_kv_kernel(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    __nv_bfloat16* const* cache_keys, __nv_bfloat16* const* cache_values,
    int position, int source_begin, int tokens, int plane) {
  const int session = blockIdx.y;
  const int elements = (tokens - source_begin) * plane;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += gridDim.x * blockDim.x) {
    const int source_token = source_begin + index / plane;
    const int column = index % plane;
    const int slot = (position + source_token) & 1023;
    const std::size_t source =
        (static_cast<std::size_t>(session) * tokens + source_token) * plane +
        column;
    const auto key = keys[source];
    const auto value = values[source];
    for (int copy = 0; copy < 3; ++copy) {
      const std::size_t destination =
          (static_cast<std::size_t>(slot + copy * 1024) * plane) + column;
      cache_keys[session][destination] = key;
      cache_values[session][destination] = value;
    }
    if (slot < 4) {
      const std::size_t destination =
          (static_cast<std::size_t>(slot + 3 * 1024) * plane) + column;
      cache_keys[session][destination] = key;
      cache_values[session][destination] = value;
    }
  }
}

__global__ void append_uniform_global_kv_kernel(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    __nv_bfloat16* const* cache_keys, __nv_bfloat16* const* cache_values,
    int position, int tokens, int plane) {
  const int session = blockIdx.y;
  const int elements = tokens * plane;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += gridDim.x * blockDim.x) {
    const std::size_t source =
        static_cast<std::size_t>(session) * elements + index;
    const std::size_t destination =
        static_cast<std::size_t>(position) * plane + index;
    cache_keys[session][destination] = keys[source];
    cache_values[session][destination] = values[source];
  }
}

void append_qkv_batch(bool global_layer, const __nv_bfloat16* key,
                      const __nv_bfloat16* value, int position, int tokens,
                      void* key_cache, void* value_cache,
                      cudaStream_t stream) {
  const int kv_bytes = global_layer ? 2 * 512 * 2 : 8 * 256 * 2;
  if (global_layer) {
    check(cudaMemcpyAsync(static_cast<std::byte*>(key_cache) +
                              static_cast<std::size_t>(position) * kv_bytes,
                          key, tokens * kv_bytes, cudaMemcpyDeviceToDevice, stream),
          "append batched global K");
    check(cudaMemcpyAsync(static_cast<std::byte*>(value_cache) +
                              static_cast<std::size_t>(position) * kv_bytes,
                          value, tokens * kv_bytes, cudaMemcpyDeviceToDevice, stream),
          "append batched global V");
  } else {
    // Only the newest sliding-window tokens survive a large prefill. Map
    // their absolute positions into the 1024-slot ring; copying the complete
    // prompt here would overrun the layer and retain the wrong (oldest) KV.
    const int source_begin = std::max(0, tokens - 1024);
    constexpr int kPlane = 8 * 256;
    const int elements = (tokens - source_begin) * kPlane;
    append_sliding_duplicated_kv_kernel<<<
        std::min(65535, (elements + 255) / 256), 256, 0, stream>>>(
        key, value, static_cast<__nv_bfloat16*>(key_cache),
        static_cast<__nv_bfloat16*>(value_cache), position, source_begin,
        tokens, kPlane);
    check(cudaGetLastError(), "append duplicated sliding KV");
  }
}

