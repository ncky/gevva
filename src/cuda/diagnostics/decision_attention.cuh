// Global shared-prefix attention versus the independent combined-K/V path.
template <int kHeadDim, int kKvHeads>
double test_decision_attention_geometry() {
  constexpr int query_width = 16 * kHeadDim, kv_width = kKvHeads * kHeadDim;
  constexpr bool global = kHeadDim == 512;
  GpuExecutionContext context;
  const auto stream = context.stream();
  auto handle = static_cast<cublasHandle_t>(context.blas_handle());
  double maximum_error = 0;
  for (auto geometry : {std::array{3, 5, 0}, std::array{3, 5, 17}, std::array{3, 5, 1031},
                        std::array{1, 32, 1704}, std::array{64, 32, 1710}, std::array{1, 1033, 17}}) {
    const auto [sessions, queries, context_tokens] = geometry;
    const int prefix = global ? context_tokens : std::min(context_tokens, 1024);
    std::cerr << "attention fixture D=" << kHeadDim << " B=" << sessions << " Q=" << queries << " P=" << prefix << '\n';
    const int keys = ((prefix + 7) & ~7) + ((queries + 7) & ~7);
    DeviceScratchPool scratch;
    auto buffer = [&](std::size_t bytes) { return scratch.allocate(std::max<std::size_t>(bytes, 16)); };
    auto upload = [&](auto& values) {
      void* result = buffer(values.size() * sizeof(values[0]));
      check(cudaMemcpyAsync(result, values.data(), values.size() * sizeof(values[0]), cudaMemcpyHostToDevice, stream), "upload attention fixture");
      return result;
    };
    auto data = [](int count, int step) {
      std::vector<__nv_bfloat16> values(std::max(count, 1));
      for (int i = 0; i < count; ++i) values[i] = __float2bfloat16_rn(((i * step) % 97 - 48) * .013F);
      return values;
    };
    std::vector<int> offsets(sessions + 1), blocks(sessions * queries, -1);
    for (int i = 0; i < sessions; ++i) offsets[i + 1] = offsets[i] + queries - (i % 3);
    auto hq = data(sessions * queries * query_width, 17);
    auto hk = data(offsets.back() * kv_width, 31), hv = data(offsets.back() * kv_width, 37);
    auto hpk = data(prefix * kv_width, 41), hpv = data(prefix * kv_width, 43);
    auto* q = static_cast<__nv_bfloat16*>(upload(hq));
    auto* k = static_cast<__nv_bfloat16*>(upload(hk));
    auto* v = static_cast<__nv_bfloat16*>(upload(hv));
    auto* pk = static_cast<__nv_bfloat16*>(upload(hpk));
    auto* pv = static_cast<__nv_bfloat16*>(upload(hpv));
    auto* d_offsets = static_cast<int*>(upload(offsets));
    auto* d_blocks = static_cast<int*>(upload(blocks));
    auto* combined_k = static_cast<__nv_bfloat16*>(buffer(sessions * keys * kv_width * 2ULL));
    auto* combined_v = static_cast<__nv_bfloat16*>(buffer(sessions * keys * kv_width * 2ULL));
    auto* suffix_k = static_cast<__nv_bfloat16*>(buffer(sessions * queries * kv_width * 2ULL));
    auto* suffix_v = static_cast<__nv_bfloat16*>(buffer(sessions * queries * kv_width * 2ULL));
    auto** key_views = static_cast<__nv_bfloat16**>(buffer(sessions * sizeof(void*)));
    auto** value_views = static_cast<__nv_bfloat16**>(buffer(sessions * sizeof(void*)));
    auto* scores = static_cast<__nv_bfloat16*>(buffer(sessions * 16 * queries * keys * 2ULL));
    auto* output = static_cast<__nv_bfloat16*>(buffer(hq.size() * 2));
    auto* accumulation = static_cast<float*>(buffer(hq.size() * sizeof(float)));
    void* pointers = buffer((sessions + 1) * 96 * sizeof(void*));
    auto pack = [&](bool combined) {
      pack_decision_keys_kernel<<<128, 256, 0, stream>>>(
          reinterpret_cast<const uint4*>(k), reinterpret_cast<const uint4*>(v),
          reinterpret_cast<const uint4*>(pk), reinterpret_cast<const uint4*>(pv),
          reinterpret_cast<uint4*>(combined ? combined_k : suffix_k),
          reinterpret_cast<uint4*>(combined ? combined_v : suffix_v),
          key_views, value_views, d_offsets, sessions, queries, combined ? prefix : 0,
          global, combined ? keys : queries);
    };
    pack(true);
    if constexpr (global)
      launch_uniform_batch_prefill_attention<512, 2, true>(handle, q, combined_k, combined_v,
          key_views, value_views, combined_k, combined_v, scores, d_blocks, output, pointers,
          sessions, queries, keys, prefix, keys, stream);
    else
      for (int session = 0; session < sessions; ++session)
        launch_prefill_attention_legacy(handle, false, q + session * queries * query_width,
            combined_k + session * keys * kv_width, combined_v + session * keys * kv_width,
            scores, d_blocks, output + session * queries * query_width, queries, keys,
            prefix, 0, prefix, queries, stream);
    std::vector<__nv_bfloat16> reference(hq.size()), actual(hq.size());
    check(cudaMemcpyAsync(reference.data(), output, reference.size() * 2, cudaMemcpyDeviceToHost, stream), "read reference attention");
    pack(false);
    launch_segmented_attention<kHeadDim, kKvHeads>(handle, q, pk, pv, suffix_k, suffix_v, scores,
        accumulation, output, pointers, sessions, queries, prefix, keys, !global, stream);
    check(cudaMemcpyAsync(actual.data(), output, actual.size() * 2, cudaMemcpyDeviceToHost, stream), "read segmented attention");
    check(cudaStreamSynchronize(stream), "finish attention fixture");
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(__bfloat162float(actual[i])) || !std::isfinite(__bfloat162float(reference[i])))
        throw std::runtime_error("non-finite decision attention fixture");
      maximum_error = std::max(maximum_error, static_cast<double>(std::abs(__bfloat162float(actual[i]) - __bfloat162float(reference[i]))));
    }
  }
  if (maximum_error > .004) throw std::runtime_error("segmented global attention mismatch: " + std::to_string(maximum_error));
  return maximum_error;
}

int test_decision_attention() {
  select_target_gpu();
  const double global = test_decision_attention_geometry<512, 2>();
  const double sliding = test_decision_attention_geometry<256, 8>();
  std::cout << nlohmann::json({{"passed", true}, {"cases", 12}, {"global_max_absolute_error", global},
                              {"sliding_max_absolute_error", sliding}}).dump() << '\n';
  return 0;
}
