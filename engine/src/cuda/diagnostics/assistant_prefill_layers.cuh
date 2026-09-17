// Private implementation fragment; included only by src/gpu.cu.
AssistantBatchTestResult test_assistant_batch(
    const AssistantWeights& weights, const DeviceModel& compressed_vocab,
    const DeviceModel& target_model, int batch, int context_tokens,
    const DeviceModel* nvfp4_vocab,
    std::span<const int> hot_token_map) {
  if (batch < 1 || batch > 8 || context_tokens < 1 || context_tokens > 8192)
    throw std::runtime_error("invalid assistant batch test geometry");
  constexpr int kTargetHidden = 2816;
  std::vector<std::unique_ptr<KvCache>> owners;
  std::vector<KvCache*> caches;
  owners.reserve(batch);
  caches.reserve(batch);
  for (int row = 0; row < batch; ++row) {
    owners.push_back(std::make_unique<KvCache>(8192));
    caches.push_back(owners.back().get());
    for (const int layer : {28, 29}) {
      const auto& view = caches.back()->layer(layer);
      const std::size_t bytes = static_cast<std::size_t>(view.capacity) *
                                view.kv_heads * view.head_dim * 2;
      check(cudaMemset(view.keys, 0, bytes), "clear batch test keys");
      check(cudaMemset(view.values, 0, bytes), "clear batch test values");
    }
  }
  std::vector<float> seed(kTargetHidden);
  for (int column = 0; column < kTargetHidden; ++column)
    seed[column] = std::sin(column * 0.013F) * 0.25F;
  const int previous = 9259;
  AttentionWorkspace scalar_workspace(8192, 5);
  const auto scalar = benchmark_assistant_step(
      weights, *caches[0], scalar_workspace, context_tokens,
      &compressed_vocab, &target_model, 4, seed, previous);
  std::vector<__nv_bfloat16> host_states(
      static_cast<std::size_t>(batch) * kTargetHidden);
  for (int row = 0; row < batch; ++row)
    for (int column = 0; column < kTargetHidden; ++column)
      host_states[static_cast<std::size_t>(row) * kTargetHidden + column] =
          __float2bfloat16_rn(seed[column]);
  void* states{};
  int* drafts{};
  check(cudaMalloc(&states, host_states.size() * 2),
        "allocate assistant batch test states");
  check(cudaMalloc(&drafts, batch * 4 * sizeof(int)),
        "allocate assistant batch test drafts");
  check(cudaMemcpy(states, host_states.data(), host_states.size() * 2,
                   cudaMemcpyHostToDevice), "copy assistant batch test states");
  std::vector<int> contexts(batch, context_tokens);
  std::vector<int> previous_tokens(batch, previous);
  GpuExecutionContext execution;
  DeviceScratchPool scratch;
  CompressedVocabRunner vocabulary(weights.model(), "model.embed_tokens.weight",
                                   compressed_vocab, execution.stream());
  std::unique_ptr<Nvfp4VocabRunner> nvfp4_vocabulary;
  if (nvfp4_vocab)
    nvfp4_vocabulary = std::make_unique<Nvfp4VocabRunner>(
        weights.model(), "model.embed_tokens.weight", *nvfp4_vocab, 8,
        hot_token_map);
  launch_assistant_batch(weights, caches, contexts, compressed_vocab,
                         target_model, states, previous_tokens, drafts,
                         scratch, vocabulary, nvfp4_vocabulary.get(), execution,
                         nullptr);
  check(cudaStreamSynchronize(execution.stream()),
        "synchronize assistant batch test warmup");
  const bool profile = std::getenv("G4_PROFILE_ASSISTANT_BATCH") != nullptr;
  if (profile && cudaProfilerStart() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStart assistant batch failed");
  nvtxRangePushA(batch == 1 ? "assistant_batch/profile/b1"
                            : "assistant_batch/profile/bN");
  const float microseconds = event_benchmark(
      execution.stream(),
      [&] { launch_assistant_batch(weights, caches, contexts, compressed_vocab,
                                   target_model, states, previous_tokens,
                                   drafts, scratch, vocabulary,
                                   nvfp4_vocabulary.get(), execution,
                                   nullptr); },
      3, 20);
  nvtxRangePop();
  if (profile && cudaProfilerStop() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStop assistant batch failed");
  std::vector<int> host_drafts(batch * 4);
  check(cudaMemcpy(host_drafts.data(), drafts, host_drafts.size() * sizeof(int),
                   cudaMemcpyDeviceToHost), "copy assistant batch test drafts");
  std::array<int, 4> first_row{};
  bool repeatable = true;
  bool scalar_match = true;
  for (int draft = 0; draft < 4; ++draft) {
    first_row[draft] = host_drafts[draft];
    scalar_match &= first_row[draft] == scalar.drafted_tokens[draft];
  }
  for (int row = 1; row < batch; ++row)
    for (int draft = 0; draft < 4; ++draft)
      repeatable &= host_drafts[row * 4 + draft] == first_row[draft];
  cudaFree(drafts);
  cudaFree(states);
  return {batch, microseconds, repeatable, scalar_match, first_row,
          scalar.drafted_tokens};
}

SmallBatchBenchmark benchmark_target_small_batch(const DeviceModel& model) {
  constexpr int kHidden = 2816;
  constexpr int kMaximumTokens = 5;
  const auto q_view = model.tensor(
      "model.language_model.layers.0.self_attn.q_proj.weight");
  const auto mlp_view = model.tensor(
      "model.language_model.layers.0.mlp.gate_proj.weight");
  if (q_view.info->dtype != "BF16" ||
      q_view.info->shape != std::vector<std::uint64_t>{4096, kHidden} ||
      mlp_view.info->dtype != "BF16" ||
      mlp_view.info->shape != std::vector<std::uint64_t>{2112, kHidden})
    throw std::runtime_error("unexpected target small-batch projection geometry");

  __nv_bfloat16* input{};
  __nv_bfloat16* output{};
  check(cudaMalloc(&input, kMaximumTokens * kHidden * 2),
        "cudaMalloc(small batch input)");
  check(cudaMalloc(&output, kMaximumTokens * 4096 * 2),
        "cudaMalloc(small batch output)");
  std::vector<__nv_bfloat16> host_input(kMaximumTokens * kHidden);
  for (std::size_t index = 0; index < host_input.size(); ++index)
    host_input[index] = __float2bfloat16_rn(
        std::sin(index * 0.017F) * 0.5F + std::cos(index * 0.003F) * 0.2F);
  check(cudaMemcpy(input, host_input.data(), host_input.size() * 2,
                   cudaMemcpyHostToDevice),
        "copy small batch input");

  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(small batch)");
  check(cublasCreate(&handle), "cublasCreate(small batch)");
  check(cublasSetStream(handle, stream), "cublasSetStream(small batch)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto gemm = [&](const __nv_bfloat16* weight, int rows, int tokens,
                  const __nv_bfloat16* source, __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, kHidden,
                       &alpha, weight, CUDA_R_16BF, kHidden, source, CUDA_R_16BF,
                       kHidden, &beta, destination, CUDA_R_16BF, rows,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "target small-batch projection");
  };
  auto graph_time = [&](const std::function<void()>& body) {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin small-batch graph");
    body();
    check(cudaStreamEndCapture(stream, &graph), "end small-batch graph");
    check(cudaGraphInstantiate(&executable, graph), "instantiate small-batch graph");
    const float result = event_benchmark(
        stream,
        [&] { check(cudaGraphLaunch(executable, stream), "launch small-batch graph"); },
        10, 500);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    return result;
  };

  SmallBatchBenchmark result;
  auto measure = [&](const DeviceTensorView& view, int rows,
                     std::array<float, 5>& batched,
                     std::array<float, 5>& repeated) {
    const auto* weight = reinterpret_cast<const __nv_bfloat16*>(view.data);
    for (int tokens = 1; tokens <= kMaximumTokens; ++tokens) {
      batched[tokens - 1] = graph_time(
          [&] { gemm(weight, rows, tokens, input, output); });
      repeated[tokens - 1] = graph_time([&] {
        for (int token = 0; token < tokens; ++token)
          gemm(weight, rows, 1, input + token * kHidden,
               output + token * rows);
      });
    }
  };
  measure(q_view, 4096, result.q_batched_microseconds,
          result.q_repeated_microseconds);
  measure(mlp_view, 2112, result.mlp_batched_microseconds,
          result.mlp_repeated_microseconds);

  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  cudaFree(output);
  cudaFree(input);
  return result;
}

PrefillLayerBenchmark benchmark_prefill_layer(
    const RuntimeWeights& weights, int layer,
    std::span<const std::int32_t> block_sequence_ids) {
  const int tokens = static_cast<int>(block_sequence_ids.size());
  if (layer < 0 || layer >= 30 || tokens < 2 || tokens > 512)
    throw std::runtime_error("invalid prefill layer geometry");
  constexpr int kHidden = 2816;
  constexpr int kDense = 2112;
  constexpr int kExperts = 128;
  constexpr int kTop = 8;
  const bool global = layer % 6 == 5;
  const int q_width = global ? 8192 : 4096;
  const int kv_width = global ? 1024 : 2048;
  const auto& model = weights.model();
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  auto w = [&](const std::string& suffix) {
    return reinterpret_cast<const __nv_bfloat16*>(
        model.tensor(prefix + suffix).data);
  };

  std::vector<void*> allocations;
  auto allocate = [&](std::size_t elements, const char* name) {
    void* pointer{};
    check(cudaMalloc(&pointer, elements * 2), name);
    allocations.push_back(pointer);
    return reinterpret_cast<__nv_bfloat16*>(pointer);
  };
  auto* hidden = allocate(static_cast<std::size_t>(tokens) * kHidden,
                          "cudaMalloc(prefill hidden)");
  auto* output = allocate(static_cast<std::size_t>(tokens) * kHidden,
                          "cudaMalloc(prefill output)");
  auto* normalized = allocate(static_cast<std::size_t>(tokens) * kHidden,
                              "cudaMalloc(prefill norm)");
  auto* q = allocate(static_cast<std::size_t>(tokens) * q_width,
                     "cudaMalloc(prefill Q)");
  auto* k = allocate(static_cast<std::size_t>(tokens) * kv_width,
                     "cudaMalloc(prefill K)");
  auto* v = allocate(static_cast<std::size_t>(tokens) * kv_width,
                     "cudaMalloc(prefill V)");
  auto* attended = allocate(static_cast<std::size_t>(tokens) * q_width,
                            "cudaMalloc(prefill attended)");
  auto* packed_attended = allocate(
      (96 * sizeof(void*) + sizeof(__nv_bfloat16) - 1) /
          sizeof(__nv_bfloat16),
      "cudaMalloc(prefill attention workspace)");
  auto* attention = allocate(static_cast<std::size_t>(tokens) * kHidden,
                             "cudaMalloc(prefill attention)");
  auto* residual = allocate(static_cast<std::size_t>(tokens) * kHidden,
                            "cudaMalloc(prefill residual)");
  auto* dense_pre = allocate(static_cast<std::size_t>(tokens) * kHidden,
                             "cudaMalloc(prefill dense pre)");
  auto* expert_pre = allocate(static_cast<std::size_t>(tokens) * kHidden,
                              "cudaMalloc(prefill expert pre)");
  auto* router_input = allocate(static_cast<std::size_t>(tokens) * kHidden,
                                "cudaMalloc(prefill router input)");
  auto* router_logits = allocate(static_cast<std::size_t>(tokens) * kExperts,
                                 "cudaMalloc(prefill router logits)");
  auto* dense_gate = allocate(static_cast<std::size_t>(tokens) * kDense,
                              "cudaMalloc(prefill dense gate)");
  auto* dense_up = allocate(static_cast<std::size_t>(tokens) * kDense,
                            "cudaMalloc(prefill dense up)");
  auto* dense_product = allocate(static_cast<std::size_t>(tokens) * kDense,
                                 "cudaMalloc(prefill dense product)");
  auto* dense_output = allocate(static_cast<std::size_t>(tokens) * kHidden,
                                "cudaMalloc(prefill dense output)");
  auto* expert_output = allocate(static_cast<std::size_t>(tokens) * kHidden,
                                 "cudaMalloc(prefill expert output)");
  auto* scores = allocate(static_cast<std::size_t>(16) * tokens * tokens,
                          "cudaMalloc(prefill scores)");
  float* top_weights{};
  int *top_ids{}, *groups{};
  check(cudaMalloc(&top_weights, static_cast<std::size_t>(tokens) * kTop * 4),
        "cudaMalloc(prefill weights)");
  check(cudaMalloc(&top_ids, static_cast<std::size_t>(tokens) * kTop * 4),
        "cudaMalloc(prefill ids)");
  check(cudaMalloc(&groups, block_sequence_ids.size_bytes()),
        "cudaMalloc(prefill block ids)");
  allocations.push_back(top_weights);
  allocations.push_back(top_ids);
  allocations.push_back(groups);
  std::vector<__nv_bfloat16> host_hidden(
      static_cast<std::size_t>(tokens) * kHidden);
  for (std::size_t index = 0; index < host_hidden.size(); ++index)
    host_hidden[index] = __float2bfloat16_rn(
        std::sin(index * 0.0017F) * 0.7F + std::cos(index * 0.0003F) * 0.2F);
  check(cudaMemcpy(hidden, host_hidden.data(), host_hidden.size() * 2,
                   cudaMemcpyHostToDevice), "copy prefill hidden");
  check(cudaMemcpy(groups, block_sequence_ids.data(), block_sequence_ids.size_bytes(),
                   cudaMemcpyHostToDevice), "copy prefill block ids");

  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(prefill layer)");
  check(cublasCreate(&handle), "cublasCreate(prefill layer)");
  check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
        "cublasSetAtomicsMode(prefill layer)");
  check(cublasSetStream(handle, stream), "cublasSetStream(prefill layer)");
  const float alpha = 1.0F, beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int rows, int columns,
                     const __nv_bfloat16* source, __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, columns,
                       &alpha, matrix, CUDA_R_16BF, columns, source,
                       CUDA_R_16BF, columns, &beta, destination, CUDA_R_16BF,
                       rows, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "prefill layer projection");
  };
  Nvfp4ExpertRunner expert_runner(weights.experts(), layer, tokens);
  const auto* input_norm = w("input_layernorm.weight");
  const auto* q_weight = w("self_attn.q_proj.weight");
  const auto* k_weight = w("self_attn.k_proj.weight");
  const auto* v_weight = global ? nullptr : w("self_attn.v_proj.weight");
  const auto* o_weight = w("self_attn.o_proj.weight");
  const auto* q_norm = w("self_attn.q_norm.weight");
  const auto* k_norm = w("self_attn.k_norm.weight");
  const auto* post_attention = w("post_attention_layernorm.weight");
  const auto* dense_pre_w = w("pre_feedforward_layernorm.weight");
  const auto* expert_pre_w = w("pre_feedforward_layernorm_2.weight");
  const auto* router_scale = w("router.scale");
  const auto* router_weight = w("router.proj.weight");
  const auto* expert_scale = w("router.per_expert_scale");
  const auto* gate_weight = w("mlp.gate_proj.weight");
  const auto* up_weight = w("mlp.up_proj.weight");
  const auto* down_weight = w("mlp.down_proj.weight");
  const auto* dense_post = w("post_feedforward_layernorm_1.weight");
  const auto* expert_post = w("post_feedforward_layernorm_2.weight");
  const auto* combined_post = w("post_feedforward_layernorm.weight");
  const auto* layer_scalar = w("layer_scalar");
  auto launch = [&] {
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        hidden, input_norm, normalized, tokens, 1e-6F);
    project(q_weight, q_width, kHidden, normalized, q);
    project(k_weight, kv_width, kHidden, normalized, k);
    if (!global) project(v_weight, kv_width, kHidden, normalized, v);
    transform_qkv_batch(global, q, k, v, q_norm, k_norm, 0, tokens, stream);
    launch_prefill_attention_impl(handle, global, q, k, v, scores, groups,
                                  attended, packed_attended, tokens, tokens,
                                  0, 0, 0, tokens, stream);
    project(o_weight, kHidden, q_width, attended, attention);
    rmsnorm_add_2816_kernel<<<tokens, 256, 0, stream>>>(
        attention, post_attention, hidden, residual, tokens);
    dual_rmsnorm_router_2816_kernel<<<tokens, 256, 0, stream>>>(
        residual, dense_pre_w, expert_pre_w, router_scale, dense_pre,
        expert_pre, router_input, tokens);
    project(router_weight, kExperts, kHidden, router_input, router_logits);
    route_128_top8_kernel<<<tokens, kExperts, 0, stream>>>(
        router_logits, expert_scale, top_weights, top_ids, tokens);
    project(gate_weight, kDense, kHidden, dense_pre, dense_gate);
    project(up_weight, kDense, kHidden, dense_pre, dense_up);
    gelu_tanh_multiply_kernel<<<
        (tokens * kDense + 255) / 256, 256, 0, stream>>>(
        dense_gate, dense_up, dense_product, tokens * kDense);
    project(down_weight, kHidden, kDense, dense_product, dense_output);
    expert_runner.launch_batch(expert_pre, top_ids, top_weights,
                               expert_output, tokens, stream);
    finalize_feedforward_2816_kernel<<<tokens, 256, 0, stream>>>(
        dense_output, dense_post, expert_output, expert_post, combined_post,
        residual, layer_scalar, output, tokens);
  };
  launch();
  check(cudaGetLastError(), "launch prefill layer");
  check(cudaStreamSynchronize(stream), "synchronize prefill layer");
  // Exclude one-time library heuristic/cache initialization from the
  // repeatability comparison. Both samples below are steady-state launches.
  launch();
  check(cudaStreamSynchronize(stream), "warm prefill layer");
  std::vector<__nv_bfloat16> initial_output(host_hidden.size());
  check(cudaMemcpy(initial_output.data(), output, initial_output.size() * 2,
                   cudaMemcpyDeviceToHost),
        "copy initial prefill layer output");
  const bool profile_prefill =
      std::getenv("G4_PROFILE_PREFILL") != nullptr;
  if (profile_prefill && cudaProfilerStart() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStart prefill layer failed");
  const float microseconds = event_benchmark(stream, launch, 1, 5);
  if (profile_prefill && cudaProfilerStop() != cudaSuccess)
    throw std::runtime_error("cudaProfilerStop prefill layer failed");
  std::vector<__nv_bfloat16> host_output(host_hidden.size());
  check(cudaMemcpy(host_output.data(), output, host_output.size() * 2,
                   cudaMemcpyDeviceToHost), "copy prefill layer output");
  if (std::memcmp(initial_output.data(), host_output.data(),
                  host_output.size() * 2) != 0) {
    float maximum_difference = 0.0F;
    std::size_t different = 0;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float first = __bfloat162float(initial_output[index]);
      const float last = __bfloat162float(host_output[index]);
      maximum_difference =
          std::max(maximum_difference, std::abs(first - last));
      different += std::memcmp(&initial_output[index], &host_output[index],
                               sizeof(__nv_bfloat16)) != 0;
    }
    std::ostringstream message;
    message << "prefill layer is not bit-repeatable: max_diff="
            << maximum_difference << " different=" << different;
    throw std::runtime_error(message.str());
  }
  float checksum = 0.0F;
  for (const auto value : host_output) checksum += __bfloat162float(value);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  for (void* pointer : allocations) cudaFree(pointer);
  return {microseconds, checksum};
}

