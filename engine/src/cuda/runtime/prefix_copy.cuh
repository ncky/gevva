// Private implementation fragment; included only by src/gpu.cu.
void copy_sliding_prefix_arena(const void* source, void* destination,
                               int tokens, cudaStream_t stream) {
  if (!source || !destination || tokens < 1 || tokens > 1024)
    throw std::runtime_error("invalid sliding prefix copy");
  constexpr std::size_t kVectorsPerToken = (8 * 256 * 2) / sizeof(uint4);
  const std::size_t vectors = 25ULL * 2 * tokens * kVectorsPerToken;
  const int blocks = static_cast<int>(
      std::min<std::size_t>(65535, (vectors + 255) / 256));
  copy_sliding_prefix_arena_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const uint4*>(source), static_cast<uint4*>(destination),
      tokens);
  check(cudaGetLastError(), "copy sliding prefix arena launch");
}

