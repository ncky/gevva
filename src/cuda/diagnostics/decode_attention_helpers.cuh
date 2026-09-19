// Private implementation fragment; included only by src/gpu.cu.
template <int kHeadDim, int kKvHeads, int kWindow>
std::pair<float, float> test_decode_attention_case(int context_tokens) {
  constexpr int kQueryHeads = 16;
  const int cache_capacity = kWindow > 0 ? kWindow : context_tokens;
  const std::size_t cache_elements =
      static_cast<std::size_t>(cache_capacity) * kKvHeads * kHeadDim;
  std::vector<__nv_bfloat16> query(kQueryHeads * kHeadDim);
  std::vector<__nv_bfloat16> keys(cache_elements), values(cache_elements);
  for (std::size_t i = 0; i < query.size(); ++i)
    query[i] = __float2bfloat16_rn(std::sin(static_cast<float>(i) * 0.017F) * 0.04F);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    keys[i] = __float2bfloat16_rn(std::cos(static_cast<float>(i) * 0.0013F) * 0.04F);
    values[i] = __float2bfloat16_rn(
        std::sin(static_cast<float>(i) * 0.0007F) * 0.7F +
        std::cos(static_cast<float>(i) * 0.0021F) * 0.2F);
  }
  const int first = kWindow > 0 && context_tokens > kWindow ? context_tokens - kWindow : 0;
  const int attended_tokens = context_tokens - first;
  __nv_bfloat16 *d_query{}, *d_keys{}, *d_values{}, *d_output{};
  float* d_scores{};
  float* d_partials{};
  check(cudaMalloc(&d_query, query.size() * sizeof(__nv_bfloat16)), "cudaMalloc(decode query)");
  check(cudaMalloc(&d_keys, keys.size() * sizeof(__nv_bfloat16)), "cudaMalloc(decode keys)");
  check(cudaMalloc(&d_values, values.size() * sizeof(__nv_bfloat16)), "cudaMalloc(decode values)");
  check(cudaMalloc(&d_output, query.size() * sizeof(__nv_bfloat16)), "cudaMalloc(decode output)");
  check(cudaMalloc(&d_scores, static_cast<std::size_t>(kQueryHeads) * attended_tokens * sizeof(float)),
        "cudaMalloc(decode scores)");
  check(cudaMalloc(&d_partials,
        static_cast<std::size_t>(16) * kQueryHeads * kHeadDim * sizeof(float)),
        "cudaMalloc(decode value partials)");
  check(cudaMemcpy(d_query, query.data(), query.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice), "copy decode query");
  check(cudaMemcpy(d_keys, keys.data(), keys.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice), "copy decode keys");
  check(cudaMemcpy(d_values, values.data(), values.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice), "copy decode values");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate(decode)");
  auto launch = [&] {
    gevva::launch_decode_attention(kWindow == 0, d_query, d_keys, d_values, d_output,
                                context_tokens, d_scores, d_partials, stream);
  };
  launch();
  check(cudaGetLastError(), "decode attention launch");
  check(cudaStreamSynchronize(stream), "decode attention synchronize");
  std::vector<__nv_bfloat16> output(query.size());
  check(cudaMemcpy(output.data(), d_output, output.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost), "copy decode output");

  float max_error = 0.0F;
  for (int head = 0; head < kQueryHeads; ++head) {
    const int kv_head = head / (kQueryHeads / kKvHeads);
    std::vector<float> scores(context_tokens - first);
    float maximum = -INFINITY;
    for (int token = first; token < context_tokens; ++token) {
      float score = 0.0F;
      for (int col = 0; col < kHeadDim; ++col)
        score += __bfloat162float(query[head * kHeadDim + col]) *
                 __bfloat162float(keys[((token % cache_capacity) * kKvHeads + kv_head) * kHeadDim + col]);
      scores[token - first] = score;
      maximum = std::max(maximum, score);
    }
    float denominator = 0.0F;
    for (float score : scores) denominator += std::exp(score - maximum);
    for (int col = 0; col < kHeadDim; ++col) {
      float expected = 0.0F;
      for (int token = first; token < context_tokens; ++token)
        expected += std::exp(scores[token - first] - maximum) / denominator *
                    __bfloat162float(values[((token % cache_capacity) * kKvHeads + kv_head) * kHeadDim + col]);
      const float rounded = __bfloat162float(__float2bfloat16_rn(expected));
      max_error = std::max(max_error,
                           std::abs(rounded - __bfloat162float(output[head * kHeadDim + col])));
    }
  }
  const float microseconds = event_benchmark(stream, launch, 10, 200);
  cudaStreamDestroy(stream);
  cudaFree(d_partials);
  cudaFree(d_scores);
  cudaFree(d_output);
  cudaFree(d_values);
  cudaFree(d_keys);
  cudaFree(d_query);
  return {max_error, microseconds};
}

template <int kHeadDim, int kKvHeads, int kWindow>
std::pair<float, float> test_decode_attention_batch_case(int context_base,
                                                         int tokens) {
  constexpr int kQueryHeads = 16;
  const int maximum_context = context_base + tokens - 1;
  const int cache_capacity = kWindow > 0 ? kWindow : maximum_context;
  const int score_stride = kWindow > 0 ? std::min(maximum_context, kWindow)
                                       : maximum_context;
  const int partitions = std::min(16, (score_stride + 255) / 256);
  const std::size_t query_elements =
      static_cast<std::size_t>(tokens) * kQueryHeads * kHeadDim;
  const std::size_t candidate_elements =
      static_cast<std::size_t>(tokens) * kKvHeads * kHeadDim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(cache_capacity) * kKvHeads * kHeadDim;
  std::vector<__nv_bfloat16> queries(query_elements);
  std::vector<__nv_bfloat16> candidates_k(candidate_elements);
  std::vector<__nv_bfloat16> candidates_v(candidate_elements);
  std::vector<__nv_bfloat16> keys(cache_elements), values(cache_elements);
  for (std::size_t i = 0; i < queries.size(); ++i)
    queries[i] = __float2bfloat16_rn(
        std::sin(static_cast<float>(i) * 0.017F) * 0.04F);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    keys[i] = __float2bfloat16_rn(
        std::cos(static_cast<float>(i) * 0.0013F) * 0.04F);
    values[i] = __float2bfloat16_rn(
        std::sin(static_cast<float>(i) * 0.0007F) * 0.7F +
        std::cos(static_cast<float>(i) * 0.0021F) * 0.2F);
  }
  for (std::size_t i = 0; i < candidate_elements; ++i) {
    candidates_k[i] = __float2bfloat16_rn(
        std::sin(static_cast<float>(i) * 0.011F + 0.4F) * 0.04F);
    candidates_v[i] = __float2bfloat16_rn(
        std::cos(static_cast<float>(i) * 0.0037F + 0.2F) * 0.6F);
  }

  __nv_bfloat16 *d_queries{}, *d_keys{}, *d_values{};
  __nv_bfloat16 *d_candidates_k{}, *d_candidates_v{}, *d_outputs{};
  float *d_scores{}, *d_partials{};
  check(cudaMalloc(&d_queries, queries.size() * 2),
        "cudaMalloc(batch attention queries)");
  check(cudaMalloc(&d_keys, keys.size() * 2),
        "cudaMalloc(batch attention keys)");
  check(cudaMalloc(&d_values, values.size() * 2),
        "cudaMalloc(batch attention values)");
  check(cudaMalloc(&d_candidates_k, candidates_k.size() * 2),
        "cudaMalloc(batch attention candidate keys)");
  check(cudaMalloc(&d_candidates_v, candidates_v.size() * 2),
        "cudaMalloc(batch attention candidate values)");
  check(cudaMalloc(&d_outputs, queries.size() * 2),
        "cudaMalloc(batch attention outputs)");
  check(cudaMalloc(&d_scores, static_cast<std::size_t>(tokens) * kQueryHeads *
                                  score_stride * sizeof(float)),
        "cudaMalloc(batch attention scores)");
  check(cudaMalloc(&d_partials, static_cast<std::size_t>(tokens) * partitions *
                                    kQueryHeads * kHeadDim * sizeof(float)),
        "cudaMalloc(batch attention partials)");
  check(cudaMemcpy(d_queries, queries.data(), queries.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy batch attention queries");
  check(cudaMemcpy(d_keys, keys.data(), keys.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy batch attention keys");
  check(cudaMemcpy(d_values, values.data(), values.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy batch attention values");
  check(cudaMemcpy(d_candidates_k, candidates_k.data(), candidates_k.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy batch attention candidate keys");
  check(cudaMemcpy(d_candidates_v, candidates_v.data(), candidates_v.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy batch attention candidate values");

  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(batch attention)");
  auto launch = [&] {
    gevva::launch_decode_attention_batch(
        kWindow == 0, d_queries, d_keys, d_values, d_candidates_k,
        d_candidates_v, d_outputs, context_base, tokens, d_scores, d_partials,
        stream);
  };
  launch();
  check(cudaStreamSynchronize(stream), "batch attention synchronize");
  std::vector<__nv_bfloat16> outputs(queries.size());
  check(cudaMemcpy(outputs.data(), d_outputs, outputs.size() * 2,
                   cudaMemcpyDeviceToHost),
        "copy batch attention outputs");

  float max_error = 0.0F;
  const int candidate_start = context_base - 1;
  for (int query_token = 0; query_token < tokens; ++query_token) {
    const int context = context_base + query_token;
    const int first = kWindow > 0 && context > kWindow ? context - kWindow : 0;
    const int attended = context - first;
    for (int head = 0; head < kQueryHeads; ++head) {
      const int kv_head = head / (kQueryHeads / kKvHeads);
      const auto* query = queries.data() +
          (static_cast<std::size_t>(query_token) * kQueryHeads + head) * kHeadDim;
      std::vector<float> host_scores(attended);
      float maximum = -INFINITY;
      for (int local = 0; local < attended; ++local) {
        const int logical = first + local;
        const int physical = kWindow > 0 ? logical % kWindow : logical;
        const auto* key = logical >= candidate_start
            ? candidates_k.data() +
                  (static_cast<std::size_t>(logical - candidate_start) * kKvHeads +
                   kv_head) * kHeadDim
            : keys.data() +
                  (static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
                      kHeadDim;
        float score = 0.0F;
        for (int col = 0; col < kHeadDim; ++col)
          score += __bfloat162float(query[col]) * __bfloat162float(key[col]);
        host_scores[local] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (const float score : host_scores)
        denominator += std::exp(score - maximum);
      for (int col = 0; col < kHeadDim; ++col) {
        float expected = 0.0F;
        for (int local = 0; local < attended; ++local) {
          const int logical = first + local;
          const int physical = kWindow > 0 ? logical % kWindow : logical;
          const auto* value = logical >= candidate_start
              ? candidates_v.data() +
                    (static_cast<std::size_t>(logical - candidate_start) *
                         kKvHeads +
                     kv_head) * kHeadDim
              : values.data() +
                    (static_cast<std::size_t>(physical) * kKvHeads + kv_head) *
                        kHeadDim;
          expected += std::exp(host_scores[local] - maximum) / denominator *
                      __bfloat162float(value[col]);
        }
        const float rounded = __bfloat162float(__float2bfloat16_rn(expected));
        const auto actual = outputs[
            (static_cast<std::size_t>(query_token) * kQueryHeads + head) *
                kHeadDim +
            col];
        max_error = std::max(
            max_error, std::abs(rounded - __bfloat162float(actual)));
      }
    }
  }
  const float microseconds = event_benchmark(stream, launch, 10, 200);
  cudaStreamDestroy(stream);
  cudaFree(d_partials);
  cudaFree(d_scores);
  cudaFree(d_outputs);
  cudaFree(d_candidates_v);
  cudaFree(d_candidates_k);
  cudaFree(d_values);
  cudaFree(d_keys);
  cudaFree(d_queries);
  return {max_error, microseconds};
}

