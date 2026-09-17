// Private implementation fragment; included only by src/gpu.cu.
MultimodalEmbeddingBenchmark benchmark_multimodal_embedding(
    const DeviceModel& model, std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images) {
  constexpr int kHidden = 2816;
  constexpr std::uint32_t kVocabulary = 262144;
  if (input_ids.empty() || multimodal_types.size() != input_ids.size())
    throw std::runtime_error("invalid multimodal embedding layout");
  std::vector<int> image_rows(input_ids.size(), -1);
  int image_tokens = 0;
  for (std::size_t token = 0; token < input_ids.size(); ++token) {
    if (input_ids[token] >= kVocabulary)
      throw std::runtime_error("multimodal input id is outside the vocabulary");
    if (multimodal_types[token] > 1)
      throw std::runtime_error("unsupported multimodal token type");
    if (multimodal_types[token] == 1) image_rows[token] = image_tokens++;
  }
  if (projected_images.size() !=
      static_cast<std::size_t>(image_tokens) * kHidden)
    throw std::runtime_error("image feature count does not match prompt slots");
  const auto embedding = model.tensor("model.language_model.embed_tokens.weight");
  if (embedding.info->dtype != "BF16" ||
      embedding.info->shape != std::vector<std::uint64_t>{kVocabulary, kHidden})
    throw std::runtime_error("unexpected language embedding geometry");

  std::vector<__nv_bfloat16> host_images(projected_images.size());
  for (std::size_t index = 0; index < projected_images.size(); ++index)
    host_images[index] = __float2bfloat16_rn(projected_images[index]);
  std::uint32_t* d_ids{};
  int* d_rows{};
  __nv_bfloat16 *d_images{}, *d_output{};
  check(cudaMalloc(&d_ids, input_ids.size_bytes()), "cudaMalloc(prompt ids)");
  check(cudaMalloc(&d_rows, image_rows.size() * sizeof(int)),
        "cudaMalloc(prompt image rows)");
  check(cudaMalloc(&d_images, host_images.size() * 2),
        "cudaMalloc(prompt image features)");
  check(cudaMalloc(&d_output, input_ids.size() * kHidden * 2),
        "cudaMalloc(prompt embeddings)");
  check(cudaMemcpy(d_ids, input_ids.data(), input_ids.size_bytes(),
                   cudaMemcpyHostToDevice), "copy prompt ids");
  check(cudaMemcpy(d_rows, image_rows.data(), image_rows.size() * sizeof(int),
                   cudaMemcpyHostToDevice), "copy prompt image rows");
  check(cudaMemcpy(d_images, host_images.data(), host_images.size() * 2,
                   cudaMemcpyHostToDevice), "copy prompt image features");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(prompt embeddings)");
  const int elements = static_cast<int>(input_ids.size()) * kHidden;
  auto launch = [&] {
    compose_multimodal_embeddings_kernel<<<(elements + 255) / 256, 256, 0,
                                            stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding.data), d_ids, d_rows,
        d_images, d_output, static_cast<int>(input_ids.size()));
  };
  launch();
  check(cudaGetLastError(), "launch prompt embeddings");
  check(cudaStreamSynchronize(stream), "synchronize prompt embeddings");
  const float microseconds = event_benchmark(stream, launch, 10, 100);
  std::vector<__nv_bfloat16> output(static_cast<std::size_t>(elements));
  check(cudaMemcpy(output.data(), d_output, output.size() * 2,
                   cudaMemcpyDeviceToHost), "copy prompt embeddings");

  const auto* vocabulary = reinterpret_cast<const __nv_bfloat16*>(
      model.host_tensor("model.language_model.embed_tokens.weight").bytes.data());
  const auto scale = __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden)));
  MultimodalEmbeddingBenchmark result{};
  result.microseconds = microseconds;
  result.tokens = static_cast<int>(input_ids.size());
  result.image_tokens = image_tokens;
  for (int token = 0; token < result.tokens; ++token) {
    for (int column = 0; column < kHidden; ++column) {
      const std::size_t element = static_cast<std::size_t>(token) * kHidden + column;
      const auto expected = image_rows[token] >= 0
          ? host_images[static_cast<std::size_t>(image_rows[token]) * kHidden + column]
          : __float2bfloat16_rn(
                __bfloat162float(vocabulary[
                    static_cast<std::size_t>(input_ids[token]) * kHidden + column]) *
                __bfloat162float(scale));
      const float actual = __bfloat162float(output[element]);
      result.checksum += actual;
      result.max_abs_error = std::max(
          result.max_abs_error, std::abs(actual - __bfloat162float(expected)));
    }
  }
  cudaStreamDestroy(stream);
  cudaFree(d_output);
  cudaFree(d_images);
  cudaFree(d_rows);
  cudaFree(d_ids);
  return result;
}

PrefillAttentionTestResult test_prefill_attention(
    std::span<const std::int32_t> block_sequence_ids) {
  const int tokens = static_cast<int>(block_sequence_ids.size());
  if (tokens < 2 || tokens > 4608)
    throw std::runtime_error("prefill attention test expects 2..4608 tokens");
  for (const int group : block_sequence_ids)
    if (group < -1) throw std::runtime_error("invalid prefill block id");

  auto run_case = [&](bool global) {
    const int dim = global ? 512 : 256;
    const int kv_heads = global ? 2 : 8;
    constexpr int q_heads = 16;
    const int q_width = q_heads * dim;
    const int kv_width = kv_heads * dim;
    const int group_size = q_heads / kv_heads;
    std::vector<__nv_bfloat16> host_q(
        static_cast<std::size_t>(tokens) * q_width);
    std::vector<__nv_bfloat16> host_k(
        static_cast<std::size_t>(tokens) * kv_width);
    std::vector<__nv_bfloat16> host_v(host_k.size());
    for (std::size_t index = 0; index < host_q.size(); ++index)
      host_q[index] = __float2bfloat16_rn(
          std::sin(index * 0.0017F) * 0.09F + std::cos(index * 0.0003F) * 0.03F);
    for (std::size_t index = 0; index < host_k.size(); ++index) {
      host_k[index] = __float2bfloat16_rn(
          std::sin(index * 0.0021F) * 0.08F - std::cos(index * 0.0007F) * 0.02F);
      host_v[index] = __float2bfloat16_rn(
          std::cos(index * 0.0013F) * 0.2F + std::sin(index * 0.0009F) * 0.04F);
    }
    __nv_bfloat16 *q{}, *k{}, *v{}, *scores{}, *output{}, *packed_output{};
    std::int32_t* groups{};
    check(cudaMalloc(&q, host_q.size() * 2), "cudaMalloc(prefill Q)");
    check(cudaMalloc(&k, host_k.size() * 2), "cudaMalloc(prefill K)");
    check(cudaMalloc(&v, host_v.size() * 2), "cudaMalloc(prefill V)");
    check(cudaMalloc(&scores,
                     static_cast<std::size_t>(q_heads) * tokens * tokens * 2),
          "cudaMalloc(prefill scores)");
    check(cudaMalloc(&output, host_q.size() * 2), "cudaMalloc(prefill output)");
    check(cudaMalloc(&packed_output, 96 * sizeof(void*)),
          "cudaMalloc(prefill attention workspace)");
    check(cudaMalloc(&groups, block_sequence_ids.size_bytes()),
          "cudaMalloc(prefill groups)");
    check(cudaMemcpy(q, host_q.data(), host_q.size() * 2, cudaMemcpyHostToDevice),
          "copy prefill Q");
    check(cudaMemcpy(k, host_k.data(), host_k.size() * 2, cudaMemcpyHostToDevice),
          "copy prefill K");
    check(cudaMemcpy(v, host_v.data(), host_v.size() * 2, cudaMemcpyHostToDevice),
          "copy prefill V");
    check(cudaMemcpy(groups, block_sequence_ids.data(), block_sequence_ids.size_bytes(),
                     cudaMemcpyHostToDevice), "copy prefill groups");
    cudaStream_t stream{};
    cublasHandle_t handle{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate(prefill)");
    check(cublasCreate(&handle), "cublasCreate(prefill)");
    check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
          "cublasSetAtomicsMode(prefill)");
    check(cublasSetStream(handle, stream), "cublasSetStream(prefill)");
    auto launch = [&] {
      if (!global && std::getenv("G4_CUDNN_TEXT_PREFILL") &&
          std::getenv("G4_TEST_TEXT_PREFILL_CHUNKS") &&
          std::all_of(block_sequence_ids.begin(), block_sequence_ids.end(),
                      [](int id) { return id < 0; })) {
        for (int first = 0; first < tokens; first += 257) {
          const int count = std::min(257, tokens - first);
          const int begin = std::max(0, first - 1024);
          launch_cudnn_text_prefill(q + static_cast<std::size_t>(first) * q_width,
              k + static_cast<std::size_t>(begin) * kv_width,
              v + static_cast<std::size_t>(begin) * kv_width,
              output + static_cast<std::size_t>(first) * q_width,
              1, count, stream, first + count - begin);
        }
        return;
      }
      launch_prefill_attention_impl(handle, global, q, k, v, scores, groups,
                                    output, packed_output, tokens, tokens,
                                    0, 0, 0, tokens, stream,
                                    std::getenv("G4_CUDNN_TEXT_PREFILL") &&
                                    std::all_of(block_sequence_ids.begin(), block_sequence_ids.end(),
                                                [](int id) { return id < 0; }));
    };
    launch();
    check(cudaGetLastError(), "launch prefill attention");
    check(cudaStreamSynchronize(stream), "synchronize prefill attention");
    std::vector<__nv_bfloat16> repeat_reference(host_q.size());
    check(cudaMemcpy(repeat_reference.data(), output,
                     repeat_reference.size() * 2, cudaMemcpyDeviceToHost),
          "copy prefill repeatability reference");
    const float microseconds = event_benchmark(stream, launch, 10, 100);
    std::vector<__nv_bfloat16> actual(host_q.size());
    check(cudaMemcpy(actual.data(), output, actual.size() * 2,
                     cudaMemcpyDeviceToHost), "copy prefill output");
    if (std::memcmp(repeat_reference.data(), actual.data(), actual.size() * 2) != 0)
      throw std::runtime_error("prefill attention is not bit-repeatable");

    float maximum_error = 0.0F;
    std::vector<float> logits(tokens), probabilities(tokens);
    for (int query_token = 0; query_token < tokens; ++query_token) {
      // Large fixtures sample causal/window boundaries against the CPU oracle
      // instead of doing a quadratic full-CPU attention pass.
      if (tokens > 512 && query_token != 0 && query_token != 1 &&
          query_token != tokens - 1 && query_token != 1023 &&
          query_token != 1024 && query_token != 1025) continue;
      for (int head = 0; head < q_heads; ++head) {
        const int kv_head = head / group_size;
        float maximum = -INFINITY;
        for (int key_token = 0; key_token < tokens; ++key_token) {
          float dot = 0.0F;
          for (int column = 0; column < dim; ++column)
            dot = fmaf(
                __bfloat162float(host_q[
                    static_cast<std::size_t>(query_token) * q_width + head * dim + column]),
                __bfloat162float(host_k[
                    static_cast<std::size_t>(key_token) * kv_width + kv_head * dim + column]),
                dot);
          logits[key_token] = __bfloat162float(__float2bfloat16_rn(dot));
          const bool causal = key_token <= query_token &&
              (global || key_token > query_token - 1024);
          const bool block = block_sequence_ids[query_token] >= 0 &&
              block_sequence_ids[key_token] == block_sequence_ids[query_token];
          if (causal || block) maximum = std::max(maximum, logits[key_token]);
        }
        float denominator = 0.0F;
        for (int key_token = 0; key_token < tokens; ++key_token) {
          const bool causal = key_token <= query_token &&
              (global || key_token > query_token - 1024);
          const bool block = block_sequence_ids[query_token] >= 0 &&
              block_sequence_ids[key_token] == block_sequence_ids[query_token];
          probabilities[key_token] = causal || block
              ? std::exp(logits[key_token] - maximum) : 0.0F;
          denominator += probabilities[key_token];
        }
        for (float& probability : probabilities)
          probability = __bfloat162float(
              __float2bfloat16_rn(probability / denominator));
        for (int column = 0; column < dim; ++column) {
          float expected = 0.0F;
          for (int key_token = 0; key_token < tokens; ++key_token)
            expected = fmaf(
                probabilities[key_token],
                __bfloat162float(host_v[
                    static_cast<std::size_t>(key_token) * kv_width + kv_head * dim + column]),
                expected);
          const float observed = __bfloat162float(actual[
              static_cast<std::size_t>(query_token) * q_width + head * dim + column]);
          maximum_error = std::max(
              maximum_error,
              std::abs(observed - __bfloat162float(__float2bfloat16_rn(expected))));
        }
      }
    }
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(groups); cudaFree(packed_output); cudaFree(output); cudaFree(scores);
    cudaFree(v); cudaFree(k); cudaFree(q);
    return std::pair{maximum_error, microseconds};
  };
  const auto sliding = run_case(false);
  const auto global = run_case(true);
  return {sliding.first, sliding.second, global.first, global.second};
}

