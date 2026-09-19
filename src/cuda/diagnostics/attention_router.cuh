// Private implementation fragment; included only by src/gpu.cu.
int test_attention_router_fusion() {
  constexpr int width = 2816, max_rows = 40;
  DeviceScratchPool scratch;
  std::uint32_t rng = 1234567;
  auto data = [&](int count, float offset) {
    std::vector<__nv_bfloat16> host(count);
    for (auto& value : host) {
      rng = rng * 1664525u + 1013904223u;
      value = __float2bfloat16_rn(offset +
          (static_cast<int>(rng >> 16) - 32768) / 16384.0F);
    }
    auto* device = static_cast<__nv_bfloat16*>(scratch.allocate(count * 2));
    check(cudaMemcpy(device, host.data(), count * 2, cudaMemcpyHostToDevice),
          "copy attention router test data");
    return device;
  };
  auto* input = data(max_rows * width, 0.0F);
  auto* attention = data(max_rows * width, 0.0F);
  auto* weights = data(4 * width, 1.0F);
  std::vector<float> scales(max_rows + 1);
  for (int i = 0; i <= max_rows; ++i) scales[i] = 0.1F + i * 0.013F;
  auto* device_scales = static_cast<float*>(scratch.allocate(scales.size() * 4));
  check(cudaMemcpy(device_scales, scales.data(), scales.size() * 4,
                   cudaMemcpyHostToDevice), "copy attention router test scales");
  const std::size_t capacity = max_rows * (8 * width + 8);
  auto* reference = static_cast<std::byte*>(scratch.allocate(capacity));
  auto* fused = static_cast<std::byte*>(scratch.allocate(capacity));
  int cases = 0;
  for (int rows : {1, 4, 40}) {
    for (int threads : {128, 256, 512}) {
      for (bool scaled : {false, true}) {
        auto launch = [&](std::byte* buffer, bool fusion) {
          auto* residual = reinterpret_cast<__nv_bfloat16*>(buffer);
          auto* dense = residual + rows * width;
          auto* expert = dense + rows * width;
          auto* router = reinterpret_cast<__nv_fp8_e4m3*>(expert + rows * width);
          auto* dense_fp8 = router + rows * width;
          auto* router_scales = reinterpret_cast<float*>(dense_fp8 + rows * width);
          auto* dense_scales = router_scales + rows;
          const float* activation_scale = scaled ? device_scales : nullptr;
          if (fusion) {
            dual_rmsnorm_router_quantize_fp8_2816_kernel<true><<<rows, threads>>>(
                input, weights + width, weights + 2 * width, weights + 3 * width,
                dense, expert, router, router_scales, dense_fp8, dense_scales,
                rows, attention, weights, residual, activation_scale,
                device_scales + max_rows);
          } else {
            rmsnorm_add_2816_kernel<<<rows, threads>>>(attention, weights, input,
                residual, rows, activation_scale, device_scales + max_rows);
            dual_rmsnorm_router_quantize_fp8_2816_kernel<false><<<rows, threads>>>(
                residual, weights + width, weights + 2 * width, weights + 3 * width,
                dense, expert, router, router_scales, dense_fp8, dense_scales, rows);
          }
        };
        launch(reference, false);
        launch(fused, true);
        const std::size_t bytes = rows * (8 * width + 8);
        std::vector<std::byte> expected(bytes), observed(bytes);
        check(cudaMemcpy(expected.data(), reference, bytes, cudaMemcpyDeviceToHost),
              "copy reference attention router intermediates");
        check(cudaMemcpy(observed.data(), fused, bytes, cudaMemcpyDeviceToHost),
              "copy fused attention router intermediates");
        if (expected != observed)
          throw std::runtime_error("attention router fusion is not byte-exact: rows=" +
              std::to_string(rows) + " threads=" + std::to_string(threads) +
              " scaled=" + std::to_string(scaled));
        ++cases;
      }
    }
  }
  return cases;
}

