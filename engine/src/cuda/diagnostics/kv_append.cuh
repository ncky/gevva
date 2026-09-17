// Private implementation fragment; included only by src/gpu.cu.
SlidingKvAppendTestResult test_sliding_kv_large_append() {
  constexpr int kTokens = 1537;
  constexpr int kPosition = 317;
  constexpr int kPlane = 8 * 256;
  constexpr int kCapacity = 1024;
  const std::size_t input_elements =
      static_cast<std::size_t>(kTokens) * kPlane;
  const std::size_t cache_elements =
      static_cast<std::size_t>(3 * kCapacity + 4) * kPlane;
  std::vector<__nv_bfloat16> keys(input_elements), values(input_elements);
  for (int token = 0; token < kTokens; ++token) {
    for (int col = 0; col < kPlane; ++col) {
      keys[static_cast<std::size_t>(token) * kPlane + col] =
          __float2bfloat16_rn(static_cast<float>(token) +
                              static_cast<float>(col & 31) / 64.0F);
      values[static_cast<std::size_t>(token) * kPlane + col] =
          __float2bfloat16_rn(-static_cast<float>(token) +
                              static_cast<float>(col & 31) / 64.0F);
    }
  }
  __nv_bfloat16 *device_keys{}, *device_values{}, *cache_keys{}, *cache_values{};
  check(cudaMalloc(&device_keys, input_elements * sizeof(__nv_bfloat16)),
        "cudaMalloc(sliding append keys)");
  check(cudaMalloc(&device_values, input_elements * sizeof(__nv_bfloat16)),
        "cudaMalloc(sliding append values)");
  check(cudaMalloc(&cache_keys, cache_elements * sizeof(__nv_bfloat16)),
        "cudaMalloc(sliding append key cache)");
  check(cudaMalloc(&cache_values, cache_elements * sizeof(__nv_bfloat16)),
        "cudaMalloc(sliding append value cache)");
  check(cudaMemcpy(device_keys, keys.data(), input_elements * 2,
                   cudaMemcpyHostToDevice), "copy sliding append keys");
  check(cudaMemcpy(device_values, values.data(), input_elements * 2,
                   cudaMemcpyHostToDevice), "copy sliding append values");
  append_qkv_batch(false, device_keys, device_values, kPosition, kTokens,
                   cache_keys, cache_values, nullptr);
  check(cudaDeviceSynchronize(), "synchronize sliding append test");
  std::vector<__nv_bfloat16> actual_keys(cache_elements);
  std::vector<__nv_bfloat16> actual_values(cache_elements);
  check(cudaMemcpy(actual_keys.data(), cache_keys, cache_elements * 2,
                   cudaMemcpyDeviceToHost), "copy sliding append key cache");
  check(cudaMemcpy(actual_values.data(), cache_values, cache_elements * 2,
                   cudaMemcpyDeviceToHost), "copy sliding append value cache");
  SlidingKvAppendTestResult result;
  const int source_begin = kTokens - kCapacity;
  for (int token = source_begin; token < kTokens; ++token) {
    const int slot = (kPosition + token) & (kCapacity - 1);
    for (int col = 0; col < kPlane; ++col) {
      const auto source = static_cast<std::size_t>(token) * kPlane + col;
      const auto destination = static_cast<std::size_t>(slot) * kPlane + col;
      result.key_max_abs_error = std::max(
          result.key_max_abs_error,
          std::abs(__bfloat162float(keys[source]) -
                   __bfloat162float(actual_keys[destination])));
      result.value_max_abs_error = std::max(
          result.value_max_abs_error,
          std::abs(__bfloat162float(values[source]) -
                   __bfloat162float(actual_values[destination])));
    }
    ++result.checked_tokens;
  }
  cudaFree(cache_values);
  cudaFree(cache_keys);
  cudaFree(device_values);
  cudaFree(device_keys);
  return result;
}

