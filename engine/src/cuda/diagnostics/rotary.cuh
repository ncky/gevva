// Private implementation fragment; included only by src/gpu.cu.
int test_precomputed_rope() {
  const auto* table = precomputed_target_rope();
  int cases = 0;
  auto run = [&]<int Dim, int KvHeads, bool Global>() {
    for (int rows : {1, 4, 40, 272}) {
      for (int base : {0, 1020, 4092, kRopePositions - rows}) {
        const std::size_t q_elements = static_cast<std::size_t>(rows) * 16 * Dim;
        const std::size_t kv_elements = static_cast<std::size_t>(rows) * KvHeads * Dim;
        const std::size_t elements = q_elements + 2 * kv_elements;
        std::vector<__nv_bfloat16> input(elements), norm(Dim);
        for (std::size_t i = 0; i < elements; ++i)
          input[i] = __float2bfloat16_rn(std::sin(i * 0.0031F) * 0.7F);
        for (int i = 0; i < Dim; ++i)
          norm[i] = __float2bfloat16_rn(0.8F + (i % 17) * 0.015F);
        std::vector<float> scales(rows + 1);
        for (int i = 0; i <= rows; ++i) scales[i] = 0.7F + (i % 7) * 0.1F;
        int contexts[8];
        for (int i = 0; i < 8; ++i) contexts[i] = base + i;
        __nv_bfloat16 *reference{}, *observed{}, *d_norm{};
        float* d_scales{};
        int* d_contexts{};
        check(cudaMalloc(&reference, elements * 2), "allocate rotary reference");
        check(cudaMalloc(&observed, elements * 2), "allocate rotary observed");
        check(cudaMalloc(&d_norm, Dim * 2), "allocate rotary norm");
        check(cudaMalloc(&d_scales, scales.size() * 4), "allocate rotary scales");
        check(cudaMalloc(&d_contexts, sizeof(contexts)), "allocate rotary contexts");
        check(cudaMemcpy(reference, input.data(), elements * 2, cudaMemcpyHostToDevice), "copy rotary reference");
        check(cudaMemcpy(observed, input.data(), elements * 2, cudaMemcpyHostToDevice), "copy rotary observed");
        check(cudaMemcpy(d_norm, norm.data(), Dim * 2, cudaMemcpyHostToDevice), "copy rotary norm");
        check(cudaMemcpy(d_scales, scales.data(), scales.size() * 4, cudaMemcpyHostToDevice), "copy rotary scales");
        check(cudaMemcpy(d_contexts, contexts, sizeof(contexts), cudaMemcpyHostToDevice), "copy rotary contexts");
        const bool ragged = rows == 40;
        const auto* activation_scales = rows == 1 ? nullptr : d_scales;
        auto launch = [&](auto* data, const __nv_bfloat162* rotary) {
          transform_qkv_ragged_kernel<Dim, KvHeads, Global>
              <<<rows * (16 + KvHeads), Dim>>>(data, data + q_elements,
                  data + q_elements + kv_elements, d_norm, d_norm,
                  rows, d_contexts, ragged ? 5 : rows,
                  activation_scales, d_scales + rows,
                  activation_scales, d_scales + rows,
                  activation_scales, d_scales + rows, ragged ? -1 : base, rotary);
        };
        launch(reference, nullptr);
        launch(observed, table);
        check(cudaGetLastError(), "launch rotary comparison");
        std::vector<__nv_bfloat16> expected(elements), actual(elements);
        check(cudaMemcpy(expected.data(), reference, elements * 2, cudaMemcpyDeviceToHost), "read rotary reference");
        check(cudaMemcpy(actual.data(), observed, elements * 2, cudaMemcpyDeviceToHost), "read rotary observed");
        const bool matches = std::memcmp(expected.data(), actual.data(), elements * 2) == 0;
        cudaFree(d_contexts); cudaFree(d_scales); cudaFree(d_norm);
        cudaFree(observed); cudaFree(reference);
        if (!matches)
          throw std::runtime_error("rotary table mismatch D=" + std::to_string(Dim) +
              " rows=" + std::to_string(rows) + " base=" + std::to_string(base));
        ++cases;
      }
    }
  };
  run.template operator()<256, 8, false>();
  run.template operator()<512, 2, true>();
  return cases;
}

