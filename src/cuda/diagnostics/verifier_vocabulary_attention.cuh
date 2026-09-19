// Private implementation fragment; included only by src/gpu.cu.
TargetVerifierBatchFixedBenchmark benchmark_target_verifier_batch_fixed(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    const DeviceModel& fp8_model, int batch, int context_tokens,
    int repetitions, int drafts_per_cycle, const DeviceModel* nvfp4_vocab) {
  const int tokens = drafts_per_cycle + 1;
  const bool advance = std::getenv("GEVVA_ADVANCE_FIXED") != nullptr;
  const int reserved_tail = advance ? (repetitions + 1) * tokens : tokens;
  if (batch < 1 || batch > 8 || context_tokens < 1 ||
      context_tokens + reserved_tail > 262144 || repetitions < 1 ||
      repetitions > 100 || drafts_per_cycle < 0 || drafts_per_cycle > 4)
    throw std::runtime_error("invalid fixed verifier batch benchmark geometry");
  constexpr int kHidden = 2816;
  const char* short_context_env = std::getenv("GEVVA_FIXED_SHORT_CONTEXT");
  const int short_context = short_context_env ? std::atoi(short_context_env) : context_tokens;
  if (short_context < 1 || short_context > context_tokens)
    throw std::runtime_error("GEVVA_FIXED_SHORT_CONTEXT must be in [1, CONTEXT]");
  std::vector<int> contexts(batch, short_context);
  contexts[0] = context_tokens;
  std::vector<std::unique_ptr<KvCache>> cache_owners;
  std::vector<std::unique_ptr<CandidateKvCache>> candidate_owners;
  std::vector<KvCache*> caches;
  std::vector<CandidateKvCache*> candidates;
  for (int row = 0; row < batch; ++row) {
    cache_owners.push_back(
        std::make_unique<KvCache>(
            std::max(8192, context_tokens + reserved_tail)));
    candidate_owners.push_back(std::make_unique<CandidateKvCache>());
    auto* cache = cache_owners.back().get();
    cache->ensure_context(contexts[row] + reserved_tail);
    for (int layer = 0; layer < 30; ++layer) {
      const auto& view = cache->layer(layer);
      const int resident = layer % 6 == 5
          ? contexts[row] + reserved_tail
          : 3 * view.capacity + 4;
      const std::size_t bytes = static_cast<std::size_t>(resident) *
                                view.kv_heads * view.head_dim * 2;
      check(cudaMemset(view.keys, 0, bytes), "clear verifier benchmark keys");
      check(cudaMemset(view.values, 0, bytes),
            "clear verifier benchmark values");
    }
    caches.push_back(cache);
    candidates.push_back(candidate_owners.back().get());
  }
  GpuExecutionContext execution;
  DeviceScratchPool scratch;
  CompressedVocabRunner vocabulary(
      weights.model(), "model.language_model.embed_tokens.weight",
      compressed_vocab, execution.stream());
  std::unique_ptr<Nvfp4VocabRunner> nvfp4_vocabulary;
  if (nvfp4_vocab) {
    nvfp4_vocabulary = std::make_unique<Nvfp4VocabRunner>(
        weights.model(), "model.language_model.embed_tokens.weight",
        *nvfp4_vocab, batch * tokens);
  }
  const char* fixed_tile_env = std::getenv("GEVVA_FIXED_FP8_TILE_ROWS");
  int fixed_tile_rows = fixed_tile_env ? std::atoi(fixed_tile_env) : 16;
  if (!fixed_tile_env)
    while (fixed_tile_rows < batch * tokens) fixed_tile_rows <<= 1;
  if (fixed_tile_rows < batch * tokens || fixed_tile_rows > 64)
    throw std::runtime_error(
        "fixed verifier FP8 tile rows must cover active rows and fit the 64-row scratch");
  Fp8LinearRunner fp8(weights.model(), fp8_model, batch * tokens, false,
                      fixed_tile_rows);
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> expert_owners;
  std::vector<Nvfp4ExpertRunner*> experts;
  for (int layer = 0; layer < 30; ++layer) {
    expert_owners.push_back(std::make_unique<Nvfp4ExpertRunner>(
        weights.experts(), layer, batch * tokens));
    experts.push_back(expert_owners.back().get());
  }
  void* states{};
  int* drafts{};
  check(cudaMalloc(&states, static_cast<std::size_t>(batch) * kHidden * 2),
        "allocate verifier benchmark states");
  check(cudaMalloc(&drafts, static_cast<std::size_t>(batch) * 4 * sizeof(int)),
        "allocate verifier benchmark drafts");
  check(cudaMemset(states, 0, static_cast<std::size_t>(batch) * kHidden * 2),
        "clear verifier benchmark states");
  std::vector<int> host_drafts(batch * 4, 9259);
  check(cudaMemcpy(drafts, host_drafts.data(), host_drafts.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy verifier benchmark drafts");
  std::vector<int> previous(batch, 9259);
  auto launch = [&] {
    return launch_target_verifier_batch(
        weights, caches, candidates, contexts, previous, drafts, states,
        experts, scratch, vocabulary, nvfp4_vocabulary.get(), fp8, execution,
        drafts_per_cycle);
  };
  launch();
  const bool profile = std::getenv("GEVVA_PROFILE_FIXED") != nullptr;
  if (profile && cudaProfilerStart() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStart fixed verifier failed");
  const auto begin = std::chrono::steady_clock::now();
  TargetVerifierBatchResult observed;
  std::uint64_t sequence_hash = 1469598103934665603ULL;
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    observed = launch();
    for (int session = 0; session < batch; ++session) {
      const auto& result = observed.sessions[session];
      for (int token = 0; token < drafts_per_cycle + 1; ++token) {
        sequence_hash ^= static_cast<std::uint32_t>(result.selected_tokens[token]);
        sequence_hash *= 1099511628211ULL;
      }
      sequence_hash ^= static_cast<std::uint32_t>(result.output_count);
      sequence_hash *= 1099511628211ULL;
      if (advance) {
        contexts[session] += result.output_count;
        previous[session] = result.output_tokens[result.output_count - 1];
      }
    }
  }
  const float microseconds = static_cast<float>(
      std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - begin).count() / repetitions);
  if (profile && cudaProfilerStop() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStop fixed verifier failed");
  const auto first = observed.sessions.front();
  bool sessions_repeatable = true;
  for (int session = 1; session < batch; ++session) {
    const auto& other = observed.sessions[session];
    sessions_repeatable &= other.selected_tokens == first.selected_tokens &&
                           other.output_tokens == first.output_tokens &&
                           other.output_count == first.output_count &&
                           other.matched_drafts == first.matched_drafts;
  }
  std::vector<__nv_bfloat16> host_states(
      static_cast<std::size_t>(batch) * kHidden);
  check(cudaMemcpy(host_states.data(), states, host_states.size() * 2,
                   cudaMemcpyDeviceToHost),
        "copy verifier benchmark continuation states");
  for (int session = 1; session < batch; ++session)
    sessions_repeatable &=
        std::memcmp(host_states.data(),
                    host_states.data() + static_cast<std::size_t>(session) *
                                             kHidden,
                    kHidden * sizeof(__nv_bfloat16)) == 0;
  cudaFree(drafts);
  cudaFree(states);
  return {microseconds, first.selected_tokens, first.output_tokens,
          first.output_count, first.matched_drafts, sessions_repeatable,
          sequence_hash};
}

VocabHeadBenchmark benchmark_vocab_head(
    const DeviceModel& exact_model, const std::string& exact_tensor,
    const DeviceModel& int8_model) {
  constexpr int kVocab = 262144;
  constexpr int kBatch = 5;
  const auto exact = exact_model.tensor(exact_tensor);
  const auto packed = int8_model.tensor("weight");
  const auto row_scales = int8_model.tensor("scale");
  if (exact.info->dtype != "BF16" || exact.info->shape.size() != 2 ||
      exact.info->shape[0] != kVocab || packed.info->dtype != "I8" ||
      packed.info->shape != exact.info->shape || row_scales.info->dtype != "F32" ||
      row_scales.info->shape != std::vector<std::uint64_t>{kVocab})
    throw std::runtime_error("invalid INT8 vocabulary sidecar");
  const int columns = static_cast<int>(exact.info->shape[1]);
  if (columns != 1024 && columns != 2816)
    throw std::runtime_error("unexpected vocabulary hidden width");

  __nv_bfloat16* activation{};
  std::int8_t* quantized{};
  float* activation_scale{};
  std::int32_t* accumulated{};
  __nv_bfloat16 *exact_logits{}, *int8_logits{};
  float* block_maxima{};
  int* block_indices{};
  int* candidate_ids{};
  float* corrected{};
  int *exact_token{}, *approximate_token{}, *int8_token{};
  check(cudaMalloc(&activation, kBatch * columns * 2), "cudaMalloc(vocab activation)");
  check(cudaMalloc(&quantized, kBatch * columns), "cudaMalloc(vocab quantized)");
  check(cudaMalloc(&activation_scale, kBatch * sizeof(float)), "cudaMalloc(vocab input scale)");
  check(cudaMalloc(&accumulated, kBatch * kVocab * sizeof(std::int32_t)),
        "cudaMalloc(vocab accumulated)");
  check(cudaMalloc(&exact_logits, kBatch * kVocab * 2), "cudaMalloc(exact logits)");
  check(cudaMalloc(&int8_logits, kBatch * kVocab * 2), "cudaMalloc(int8 logits)");
  check(cudaMalloc(&block_maxima, 256 * sizeof(float)), "cudaMalloc(vocab maxima)");
  check(cudaMalloc(&block_indices, 256 * sizeof(int)), "cudaMalloc(vocab indices)");
  check(cudaMalloc(&candidate_ids, kBatch * 1024 * sizeof(int)),
        "cudaMalloc(vocab candidates)");
  check(cudaMalloc(&corrected, kBatch * 1024 * sizeof(float)),
        "cudaMalloc(vocab corrections)");
  check(cudaMalloc(&exact_token, kBatch * sizeof(int)), "cudaMalloc(exact token)");
  check(cudaMalloc(&approximate_token, kBatch * sizeof(int)), "cudaMalloc(approximate token)");
  check(cudaMalloc(&int8_token, kBatch * sizeof(int)), "cudaMalloc(int8 token)");
  std::vector<__nv_bfloat16> host_activation(kBatch * columns);
  for (int col = 0; col < columns; ++col)
    host_activation[col] = __float2bfloat16_rn(
        std::sin(col * 0.019F) * 0.6F + std::cos(col * 0.007F) * 0.2F);
  check(cudaMemcpy(activation, host_activation.data(), host_activation.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy vocab activation");

  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(vocab)");
  check(cublasCreate(&handle), "cublasCreate(vocab)");
  check(cublasSetStream(handle, stream), "cublasSetStream(vocab)");
  auto select = [&](const __nv_bfloat16* logits, int* token) {
    argmax_bf16_partial_kernel<<<256, 256, 0, stream>>>(
        logits, kVocab, block_maxima, block_indices);
    argmax_finish_kernel<<<1, 256, 0, stream>>>(
        block_maxima, block_indices, 256, token);
  };
  const float float_alpha = 1.0F;
  const float float_beta = 0.0F;
  auto launch_exact = [&] {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, 1, columns,
                       &float_alpha, exact.data, CUDA_R_16BF, columns,
                       activation, CUDA_R_16BF, columns, &float_beta,
                       exact_logits, CUDA_R_16BF, kVocab, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "exact vocabulary projection");
    select(exact_logits, exact_token);
  };
  const std::int32_t integer_alpha = 1;
  const std::int32_t integer_beta = 0;
  auto launch_int8 = [&] {
    quantize_int8_vector_kernel<<<1, 256, 0, stream>>>(
        activation, quantized, activation_scale, columns);
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, 1, columns,
                       &integer_alpha, packed.data, CUDA_R_8I, columns,
                       quantized, CUDA_R_8I, columns, &integer_beta,
                       accumulated, CUDA_R_32I, kVocab, CUBLAS_COMPUTE_32I,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "INT8 vocabulary projection");
    scale_int32_logits_kernel<<<(kVocab + 255) / 256, 256, 0, stream>>>(
        accumulated, reinterpret_cast<const float*>(row_scales.data),
        activation_scale, int8_logits, kVocab);
    select(int8_logits, approximate_token);
    local_top4_kernel<<<256, 256, 0, stream>>>(int8_logits, candidate_ids);
    correct_vocab_candidates_kernel<<<1024, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(exact.data), activation,
        columns, candidate_ids, corrected);
    argmax_corrected_kernel<<<1, 256, 0, stream>>>(
        corrected, candidate_ids, int8_token);
  };
  auto graph_time = [&](const std::function<void()>& body) {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin vocabulary graph");
    body();
    check(cudaStreamEndCapture(stream, &graph), "end vocabulary graph");
    check(cudaGraphInstantiate(&executable, graph), "instantiate vocabulary graph");
    const float result = event_benchmark(
        stream,
        [&] { check(cudaGraphLaunch(executable, stream), "launch vocabulary graph"); },
        10, 300);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    return result;
  };
  launch_exact();
  launch_int8();
  check(cudaStreamSynchronize(stream), "vocabulary benchmark synchronize");
  const float exact_us = graph_time(launch_exact);
  const float int8_us = graph_time(launch_int8);
  int host_exact{}, host_approximate{}, host_int8{};
  check(cudaMemcpy(&host_exact, exact_token, sizeof(int), cudaMemcpyDeviceToHost),
        "copy exact token");
  check(cudaMemcpy(&host_approximate, approximate_token, sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy approximate token");
  check(cudaMemcpy(&host_int8, int8_token, sizeof(int), cudaMemcpyDeviceToHost),
        "copy INT8 token");
  constexpr int kValidationTrials = 32;
  int approximate_matches = 0;
  int corrected_matches = 0;
  for (int trial = 0; trial < kValidationTrials; ++trial) {
    for (int col = 0; col < columns; ++col) {
      host_activation[col] = __float2bfloat16_rn(
          std::sin(col * (0.003F + trial * 0.00017F)) * 0.7F +
          std::cos(col * (0.011F + trial * 0.00009F)) * 0.23F);
    }
    check(cudaMemcpyAsync(activation, host_activation.data(), columns * 2,
                          cudaMemcpyHostToDevice, stream),
          "copy vocabulary validation activation");
    launch_exact();
    launch_int8();
    check(cudaMemcpyAsync(&host_exact, exact_token, sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
          "copy validation exact token");
    check(cudaMemcpyAsync(&host_approximate, approximate_token, sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
          "copy validation approximate token");
    check(cudaMemcpyAsync(&host_int8, int8_token, sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
          "copy validation corrected token");
    check(cudaStreamSynchronize(stream), "vocabulary validation synchronize");
    approximate_matches += host_exact == host_approximate;
    corrected_matches += host_exact == host_int8;
  }
  for (int token = 0; token < kBatch; ++token) {
    for (int col = 0; col < columns; ++col) {
      host_activation[token * columns + col] = __float2bfloat16_rn(
          std::sin(col * (0.005F + token * 0.0013F)) * 0.65F +
          std::cos(col * (0.009F + token * 0.0007F)) * 0.19F);
    }
  }
  check(cudaMemcpyAsync(activation, host_activation.data(), host_activation.size() * 2,
                        cudaMemcpyHostToDevice, stream),
        "copy vocabulary batch activation");
  auto launch_exact_batch = [&] {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, kBatch, columns,
                       &float_alpha, exact.data, CUDA_R_16BF, columns,
                       activation, CUDA_R_16BF, columns, &float_beta,
                       exact_logits, CUDA_R_16BF, kVocab, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "exact batched vocabulary projection");
    for (int token = 0; token < kBatch; ++token)
      select(exact_logits + token * kVocab, exact_token + token);
  };
  auto launch_int8_batch = [&] {
    for (int token = 0; token < kBatch; ++token) {
      quantize_int8_vector_kernel<<<1, 256, 0, stream>>>(
          activation + token * columns, quantized + token * columns,
          activation_scale + token, columns);
    }
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kVocab, kBatch, columns,
                       &integer_alpha, packed.data, CUDA_R_8I, columns,
                       quantized, CUDA_R_8I, columns, &integer_beta,
                       accumulated, CUDA_R_32I, kVocab, CUBLAS_COMPUTE_32I,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "INT8 batched vocabulary projection");
    for (int token = 0; token < kBatch; ++token) {
      scale_int32_logits_kernel<<<(kVocab + 255) / 256, 256, 0, stream>>>(
          accumulated + token * kVocab, reinterpret_cast<const float*>(row_scales.data),
          activation_scale + token, int8_logits + token * kVocab, kVocab);
      local_top4_kernel<<<256, 256, 0, stream>>>(
          int8_logits + token * kVocab, candidate_ids + token * 1024);
      correct_vocab_candidates_kernel<<<1024, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(exact.data),
          activation + token * columns, columns, candidate_ids + token * 1024,
          corrected + token * 1024);
      argmax_corrected_kernel<<<1, 256, 0, stream>>>(
          corrected + token * 1024, candidate_ids + token * 1024,
          int8_token + token);
    }
  };
  launch_exact_batch();
  launch_int8_batch();
  check(cudaStreamSynchronize(stream), "vocabulary batch synchronize");
  const float exact_batch_us = graph_time(launch_exact_batch);
  const float int8_batch_us = graph_time(launch_int8_batch);
  std::array<int, kBatch> host_exact_batch{}, host_int8_batch{};
  check(cudaMemcpy(host_exact_batch.data(), exact_token, kBatch * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy exact vocabulary batch");
  check(cudaMemcpy(host_int8_batch.data(), int8_token, kBatch * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy corrected vocabulary batch");
  int batch_matches = 0;
  for (int token = 0; token < kBatch; ++token)
    batch_matches += host_exact_batch[token] == host_int8_batch[token];

  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  cudaFree(int8_token); cudaFree(approximate_token); cudaFree(exact_token);
  cudaFree(corrected); cudaFree(candidate_ids); cudaFree(block_indices);
  cudaFree(block_maxima); cudaFree(int8_logits); cudaFree(exact_logits);
  cudaFree(accumulated); cudaFree(activation_scale); cudaFree(quantized);
  cudaFree(activation);
  return {exact_us, int8_us, host_exact, host_approximate, host_int8,
          kValidationTrials, approximate_matches, corrected_matches,
          exact_batch_us, int8_batch_us, batch_matches};
}

DecodeAttentionTestResult test_decode_attention(int context_tokens) {
  if (context_tokens < 1 || context_tokens > 262144)
    throw std::runtime_error("decode context must be in [1, 262144]");
  const auto sliding = test_decode_attention_case<256, 8, 1024>(context_tokens);
  const auto global = test_decode_attention_case<512, 2, 0>(context_tokens);
  if (sliding.first > 0.015625F || global.first > 0.015625F)
    throw std::runtime_error("decode attention differential exceeded BF16 tolerance");
  return {sliding.first, sliding.second, global.first, global.second};
}

TurboQuantKvBenchmark benchmark_turboquant_global_kv(int context_tokens) {
  if (context_tokens < 1 || context_tokens > 262144)
    throw std::runtime_error("TurboQuant context must be in [1, 262144]");
  constexpr int kQueryElements = 16 * 512;
  const std::size_t rows = static_cast<std::size_t>(context_tokens) * 2;
  const std::size_t cache_elements = rows * 512;
  const std::size_t score_elements =
      static_cast<std::size_t>(context_tokens) * 16;
  const int partitions = std::min(16, (context_tokens + 255) / 256);
  __nv_bfloat16 *query{}, *keys{}, *values{}, *bf16_output{}, *tq_output{};
  __nv_bfloat16 *key_scales{}, *value_scales{};
  unsigned char *packed_keys{}, *packed_values{};
  float *scores{}, *partials{}, *transformed_query{};
  auto allocate = [](auto** pointer, std::size_t bytes, const char* operation) {
    check(cudaMalloc(pointer, bytes), operation);
  };
  allocate(&query, kQueryElements * 2, "cudaMalloc(TurboQuant query)");
  allocate(&keys, cache_elements * 2, "cudaMalloc(TurboQuant BF16 K)");
  allocate(&values, cache_elements * 2, "cudaMalloc(TurboQuant BF16 V)");
  allocate(&bf16_output, kQueryElements * 2, "cudaMalloc(TurboQuant BF16 output)");
  allocate(&tq_output, kQueryElements * 2, "cudaMalloc(TurboQuant output)");
  allocate(&packed_keys, rows * 256, "cudaMalloc(TurboQuant packed K)");
  allocate(&packed_values, rows * 256, "cudaMalloc(TurboQuant packed V)");
  allocate(&key_scales, rows * 2 * 2, "cudaMalloc(TurboQuant K scales)");
  allocate(&value_scales, rows * 2 * 2, "cudaMalloc(TurboQuant V scales)");
  allocate(&scores, score_elements * sizeof(float), "cudaMalloc(TurboQuant scores)");
  allocate(&partials, static_cast<std::size_t>(partitions) * kQueryElements *
                          sizeof(float),
           "cudaMalloc(TurboQuant partials)");
  allocate(&transformed_query, kQueryElements * sizeof(float),
           "cudaMalloc(TurboQuant transformed query)");
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(TurboQuant)");
  const std::size_t initialization_elements =
      std::max(cache_elements, static_cast<std::size_t>(kQueryElements));
  initialize_tq_benchmark_kernel<<<
      (initialization_elements + 255) / 256, 256, 0, stream>>>(
      query, keys, values, context_tokens);
  auto quantize = [&] {
    tq4_quantize_512_kernel<<<rows, 256, 0, stream>>>(
        keys, packed_keys, key_scales, static_cast<int>(rows));
    tq4_quantize_512_kernel<<<rows, 256, 0, stream>>>(
        values, packed_values, value_scales, static_cast<int>(rows));
  };
  quantize();
  check(cudaStreamSynchronize(stream), "initialize TurboQuant benchmark");

  auto launch_bf16 = [&] {
    launch_decode_attention(true, query, keys, values, bf16_output,
                            context_tokens, scores, partials, stream);
  };
  auto launch_tq4 = [&] {
    tq_transform_queries_512_kernel<<<16, 256, 0, stream>>>(
        query, transformed_query);
    tq4_attention_scores_512_kernel<<<
        dim3((context_tokens + 7) / 8, 16), 256, 0, stream>>>(
        transformed_query, packed_keys, key_scales, scores, context_tokens);
    decode_attention_softmax_kernel<<<16, 256, 0, stream>>>(scores,
                                                            context_tokens);
    tq4_attention_values_512_kernel<<<dim3(16, partitions), 256, 0, stream>>>(
        scores, packed_values, value_scales, partials, context_tokens,
        partitions);
    tq_inverse_outputs_512_kernel<<<16, 256, 0, stream>>>(
        partials, tq_output, partitions);
  };
  launch_bf16();
  launch_tq4();
  check(cudaGetLastError(), "launch TurboQuant benchmark");
  check(cudaStreamSynchronize(stream), "warm up TurboQuant benchmark");
  const int repetitions = std::clamp(2000000 / context_tokens, 5, 100);
  const float bf16_us = event_benchmark(stream, launch_bf16, 2, repetitions);
  const float tq4_us = event_benchmark(stream, launch_tq4, 2, repetitions);
  const int quantize_repetitions = context_tokens <= 16384 ? 10 : 3;
  const float quantize_us =
      event_benchmark(stream, quantize, 1, quantize_repetitions);
  launch_bf16();
  launch_tq4();
  check(cudaStreamSynchronize(stream), "finalize TurboQuant benchmark");
  std::vector<__nv_bfloat16> host_bf16(kQueryElements), host_tq(kQueryElements);
  check(cudaMemcpy(host_bf16.data(), bf16_output, kQueryElements * 2,
                   cudaMemcpyDeviceToHost),
        "copy TurboQuant BF16 output");
  check(cudaMemcpy(host_tq.data(), tq_output, kQueryElements * 2,
                   cudaMemcpyDeviceToHost),
        "copy TurboQuant output");
  TurboQuantKvBenchmark result{};
  result.bf16_microseconds = bf16_us;
  result.tq4_microseconds = tq4_us;
  result.quantize_microseconds_per_token = quantize_us / context_tokens;
  double total_error = 0.0;
  double squared_error = 0.0;
  double reference_squared = 0.0;
  double candidate_squared = 0.0;
  double dot = 0.0;
  for (int index = 0; index < kQueryElements; ++index) {
    const float reference = __bfloat162float(host_bf16[index]);
    const float candidate = __bfloat162float(host_tq[index]);
    const float delta = reference - candidate;
    const float error = std::abs(delta);
    result.max_abs_error = std::max(result.max_abs_error, error);
    total_error += error;
    squared_error += static_cast<double>(delta) * delta;
    reference_squared += static_cast<double>(reference) * reference;
    candidate_squared += static_cast<double>(candidate) * candidate;
    dot += static_cast<double>(reference) * candidate;
  }
  result.mean_abs_error =
      static_cast<float>(total_error / static_cast<double>(kQueryElements));
  result.reference_rms = static_cast<float>(
      std::sqrt(reference_squared / static_cast<double>(kQueryElements)));
  result.normalized_rmse = static_cast<float>(
      std::sqrt(squared_error / std::max(reference_squared, 1e-30)));
  result.cosine_similarity = static_cast<float>(
      dot / std::sqrt(std::max(reference_squared * candidate_squared, 1e-30)));
  const double bf16_bytes = static_cast<double>(rows) * 512.0 * 2.0;
  const double tq_bytes = static_cast<double>(rows) * (256.0 + 4.0);
  result.compression_ratio = static_cast<float>(bf16_bytes / tq_bytes);
  cudaStreamDestroy(stream);
  cudaFree(transformed_query);
  cudaFree(partials); cudaFree(scores);
  cudaFree(value_scales); cudaFree(key_scales);
  cudaFree(packed_values); cudaFree(packed_keys);
  cudaFree(tq_output); cudaFree(bf16_output);
  cudaFree(values); cudaFree(keys); cudaFree(query);
  return result;
}

DecodeAttentionBatchTestResult test_decode_attention_batch(int context_tokens,
                                                           int tokens) {
  if (context_tokens < 1 || tokens < 1 || tokens > 5 ||
      context_tokens + tokens - 1 > 262144)
    throw std::runtime_error("batched decode context is invalid");
  const auto sliding =
      test_decode_attention_batch_case<256, 8, 1024>(context_tokens, tokens);
  const auto global =
      test_decode_attention_batch_case<512, 2, 0>(context_tokens, tokens);
  if (sliding.first > 0.015625F || global.first > 0.015625F)
    throw std::runtime_error(
        "batched decode attention differential exceeded BF16 tolerance");
  return {sliding.first, sliding.second, global.first, global.second};
}
