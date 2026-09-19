// Private implementation fragment; included only by src/gpu.cu.
AssistantStepBenchmark benchmark_assistant_step(
    const AssistantWeights& weights, KvCache& target_cache,
    AttentionWorkspace& workspace, int context_tokens,
    const DeviceModel* compressed_vocab, const DeviceModel* target_model,
    int drafts, std::span<const float> initial_target_state,
    int previous_token, DeviceScratchPool* scratch,
    CompressedVocabRunner* persistent_vocab,
    GpuExecutionContext* context,
    const void* initial_target_state_device, int* drafted_tokens_device,
    Fp8LinearRunner* fp8_linears) {
  const bool seeded_cycle =
      !initial_target_state.empty() || initial_target_state_device;
  if (context_tokens < 1 || context_tokens > target_cache.maximum_context())
    throw std::runtime_error("invalid assistant step benchmark context");
  target_cache.ensure_context(context_tokens, context ? context->stream() : nullptr);
  if (drafts < 1 || drafts > 4 || (drafts > 1 && !target_model))
    throw std::runtime_error("assistant cycle requires 1..4 drafts and target weights");
  constexpr int kInput = 5632;
  constexpr int kHidden = 1024;
  constexpr int kIntermediate = 8192;
  constexpr int kTargetHidden = 2816;
  constexpr int kVocab = 262144;
  if (seeded_cycle &&
      ((!initial_target_state_device &&
        initial_target_state.size() != kTargetHidden) || !target_model ||
       previous_token < 0 || previous_token >= kVocab))
    throw std::runtime_error("invalid seeded assistant cycle state");
  const int position = context_tokens - (seeded_cycle ? 0 : 1);
  const auto& model = weights.model();
  auto bf16 = [](const void* pointer) {
    return reinterpret_cast<const __nv_bfloat16*>(pointer);
  };

  if (scratch) scratch->reset();
  std::vector<void*> allocations;
  auto allocate_bytes = [&](std::size_t bytes, const char* name) {
    if (scratch) return scratch->allocate(bytes);
    void* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return pointer;
  };
  auto allocate = [&](std::size_t elements, const char* name) {
    return reinterpret_cast<__nv_bfloat16*>(
        allocate_bytes(elements * sizeof(__nv_bfloat16), name));
  };
  auto* combined_input = allocate(kInput, "cudaMalloc(assistant input)");
  auto* hidden = allocate(kHidden, "cudaMalloc(assistant hidden)");
  auto* normalized = allocate(kHidden, "cudaMalloc(assistant normalized)");
  auto* attention = allocate(kHidden, "cudaMalloc(assistant attention)");
  auto* residual = allocate(kHidden, "cudaMalloc(assistant residual)");
  auto* gate = allocate(kIntermediate, "cudaMalloc(assistant gate)");
  auto* up = allocate(kIntermediate, "cudaMalloc(assistant up)");
  auto* product = allocate(kIntermediate, "cudaMalloc(assistant product)");
  auto* feedforward = allocate(kHidden, "cudaMalloc(assistant feedforward)");
  auto* q = allocate(8192, "cudaMalloc(assistant Q)");
  auto* attended = allocate(8192, "cudaMalloc(assistant attended)");
  auto* projected_state = allocate(kTargetHidden, "cudaMalloc(assistant state)");
  __nv_bfloat16* logits = nullptr;
  if (!compressed_vocab)
    logits = allocate(kVocab, "cudaMalloc(assistant logits)");
  int* selected_tokens{};
  float* argmax_maxima{};
  int* argmax_indices{};
  selected_tokens = static_cast<int*>(
      allocate_bytes(4 * sizeof(int), "cudaMalloc(assistant tokens)"));
  if (!compressed_vocab) {
    argmax_maxima = static_cast<float*>(allocate_bytes(
        256 * sizeof(float), "cudaMalloc(assistant argmax maxima)"));
    argmax_indices = static_cast<int*>(allocate_bytes(
        256 * sizeof(int), "cudaMalloc(assistant argmax indices)"));
  }

  std::vector<__nv_bfloat16> host_input(kInput);
  for (int col = 0; col < kInput; ++col)
    host_input[col] = __float2bfloat16_rn(
        std::sin(col * 0.013F) * 0.4F + std::cos(col * 0.007F) * 0.1F);
  check(cudaMemcpy(combined_input, host_input.data(), kInput * 2,
                   cudaMemcpyHostToDevice),
        "copy assistant input");

  // The target exports the final occurrence of each attention type: layer 28
  // for sliding attention and layer 29 for full attention.
  if (!seeded_cycle)
    for (const int target_layer : {28, 29}) {
      const auto& kv = target_cache.layer(target_layer);
      const std::size_t plane = static_cast<std::size_t>(kv.capacity) *
                                kv.kv_heads * kv.head_dim * 2;
      check(cudaMemset(kv.keys, 0, plane), "clear assistant shared K");
      check(cudaMemset(kv.values, 0, plane), "clear assistant shared V");
    }

  cudaStream_t stream{};
  cublasHandle_t handle{};
  const bool owns_context = context == nullptr;
  if (context) {
    stream = context->stream();
    handle = static_cast<cublasHandle_t>(context->blas_handle());
  } else {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate(assistant)");
    check(cublasCreate(&handle), "cublasCreate(assistant)");
  }
  check(cublasSetStream(handle, stream), "cublasSetStream(assistant)");
  std::unique_ptr<CompressedVocabRunner> compressed_runner;
  CompressedVocabRunner* vocabulary_runner = persistent_vocab;
  if (compressed_vocab && !vocabulary_runner) {
    compressed_runner = std::make_unique<CompressedVocabRunner>(
        model, "model.embed_tokens.weight", *compressed_vocab, stream);
    vocabulary_runner = compressed_runner.get();
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int rows, int cols,
                     const __nv_bfloat16* source, __nv_bfloat16* destination) {
    if (fp8_linears && fp8_linears->launch(
            matrix, rows, cols, source, destination, 1, stream, handle))
      return;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols,
                       &alpha, matrix, CUDA_R_16BF, cols, source, CUDA_R_16BF,
                       cols, &beta, destination, CUDA_R_16BF, rows,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "assistant projection");
  };

  const auto* pre_projection = bf16(weights.pre_projection());
  const auto* post_projection = bf16(weights.post_projection());
  const auto* final_norm = bf16(weights.final_norm());
  const auto* vocabulary = bf16(weights.embedding());
  const __nv_bfloat16* target_embedding = nullptr;
  if (target_model) {
    const auto view = target_model->tensor("model.language_model.embed_tokens.weight");
    if (view.info->dtype != "BF16" ||
        view.info->shape != std::vector<std::uint64_t>{kVocab, kTargetHidden})
      throw std::runtime_error("invalid target embedding for assistant cycle");
    target_embedding = reinterpret_cast<const __nv_bfloat16*>(view.data);
  }
  if (seeded_cycle) {
    if (initial_target_state_device) {
      check(cudaMemcpyAsync(projected_state, initial_target_state_device,
                            kTargetHidden * 2, cudaMemcpyDeviceToDevice, stream),
            "copy resident assistant seed state");
    } else {
      std::vector<__nv_bfloat16> seed(kTargetHidden);
      for (int col = 0; col < kTargetHidden; ++col)
        seed[col] = __float2bfloat16_rn(initial_target_state[col]);
      check(cudaMemcpyAsync(projected_state, seed.data(), kTargetHidden * 2,
                            cudaMemcpyHostToDevice, stream),
            "copy assistant seed state");
    }
    check(cudaMemcpyAsync(selected_tokens, &previous_token, sizeof(int),
                          cudaMemcpyHostToDevice, stream),
          "copy assistant seed token");
    prepare_next_assistant_input_kernel<<<(kTargetHidden + 255) / 256,
                                            256, 0, stream>>>(
        target_embedding, selected_tokens, projected_state, combined_input);
  }
  auto launch_logits = [&](int* selected_token) {
    if (vocabulary_runner) {
      vocabulary_runner->launch(normalized, selected_token, stream);
    } else {
      project(vocabulary, kVocab, kHidden, normalized, logits);
      argmax_bf16_partial_kernel<<<256, 256, 0, stream>>>(
          logits, kVocab, argmax_maxima, argmax_indices);
      argmax_finish_kernel<<<1, 256, 0, stream>>>(
          argmax_maxima, argmax_indices, 256, selected_token);
    }
  };
  auto launch = [&](int draft_position, int* selected_token) {
    project(pre_projection, kHidden, kInput, combined_input, hidden);
    for (int layer = 0; layer < 4; ++layer) {
      const bool global = layer == 3;
      const int q_width = global ? 8192 : 4096;
      const auto& layer_weights = weights.layer(layer);
      const auto* input_norm = bf16(layer_weights.input_norm);
      const auto* q_weight = bf16(layer_weights.q);
      const auto* q_norm = bf16(layer_weights.q_norm);
      const auto* o_weight = bf16(layer_weights.o);
      const auto* post_attention = bf16(layer_weights.post_attention);
      const auto* pre_feedforward = bf16(layer_weights.pre_feedforward);
      const auto* gate_weight = bf16(layer_weights.gate);
      const auto* up_weight = bf16(layer_weights.up);
      const auto* down_weight = bf16(layer_weights.down);
      const auto* post_feedforward = bf16(layer_weights.post_feedforward);
      const auto* layer_scalar = bf16(layer_weights.layer_scalar);
      const auto& kv = target_cache.layer(global ? 29 : 28);

      rmsnorm_1024_kernel<<<1, 256, 0, stream>>>(hidden, input_norm, normalized);
      if (fp8_linears) fp8_linears->invalidate_activation_cache();
      project(q_weight, q_width, kHidden, normalized, q);
      if (global)
        norm_rope_512_global_kernel<<<16, 512, 0, stream>>>(
            q, q_norm, 16, 1, 16, true, draft_position);
      else
        norm_rope_256_kernel<<<16, 256, 0, stream>>>(
            q, q_norm, 16, 1, 16, true, draft_position);
      launch_decode_attention(global, q, kv.keys, kv.values, attended,
                              context_tokens, workspace.scores(),
                              workspace.partials(), stream);
      project(o_weight, kHidden, q_width, attended, attention);
      rmsnorm_add_scale_1024_kernel<<<1, 256, 0, stream>>>(
          attention, post_attention, hidden, residual);

      rmsnorm_1024_kernel<<<1, 256, 0, stream>>>(
          residual, pre_feedforward, normalized);
      if (fp8_linears) fp8_linears->invalidate_activation_cache();
      project(gate_weight, kIntermediate, kHidden, normalized, gate);
      project(up_weight, kIntermediate, kHidden, normalized, up);
      gelu_tanh_multiply_kernel<<<32, 256, 0, stream>>>(
          gate, up, product, kIntermediate);
      project(down_weight, kHidden, kIntermediate, product, feedforward);
      rmsnorm_add_scale_1024_kernel<<<1, 256, 0, stream>>>(
          feedforward, post_feedforward, residual, hidden, layer_scalar);
    }
    rmsnorm_1024_kernel<<<1, 256, 0, stream>>>(hidden, final_norm, normalized);
    if (fp8_linears) fp8_linears->invalidate_activation_cache();
    project(post_projection, kTargetHidden, kHidden, normalized, projected_state);
    launch_logits(selected_token);
    if (target_embedding) {
      prepare_next_assistant_input_kernel<<<(kTargetHidden + 255) / 256,
                                              256, 0, stream>>>(
          target_embedding, selected_token, projected_state, combined_input);
    }
  };

  if (seeded_cycle && context) {
    constexpr std::uint64_t kAssistantGraphNamespace = 0x4153470000000000ULL;
    const auto graph_key = kAssistantGraphNamespace |
                           static_cast<std::uint64_t>(context_tokens);
    const bool cached = context->has_graph(graph_key);
    auto enqueue_cycle = [&] {
      for (int draft = 0; draft < drafts; ++draft)
        launch(position, selected_tokens + draft);
    };
    if (cached)
      context->launch_graph(graph_key);
    else
      enqueue_cycle();
    check(cudaGetLastError(), "assistant cycle launch");
    if (!drafted_tokens_device || !cached)
      check(cudaStreamSynchronize(stream), "seeded assistant cycle synchronize");
    if (!cached) context->capture_graph(graph_key, enqueue_cycle);
  } else {
    launch(position, selected_tokens);
    check(cudaGetLastError(), "assistant step launch");
  }
  if (seeded_cycle && !context) {
    // Gemma 4's MTP assistant shares the target model's unchanged K/V for the
    // whole drafting round. All autoregressive drafts therefore use the same
    // position (the last seen target token); only the concatenated token
    // embedding and projected assistant state advance between drafts.
    for (int draft = 1; draft < drafts; ++draft)
      launch(position, selected_tokens + draft);
    check(cudaStreamSynchronize(stream), "seeded assistant cycle synchronize");
  } else if (!seeded_cycle) {
    check(cudaStreamSynchronize(stream), "assistant step synchronize");
  }
  const float eager_us = seeded_cycle ? 0.0F : event_benchmark(
      stream, [&] { launch(position, selected_tokens); }, 10, 200);
  const float logits_us = seeded_cycle ? 0.0F : event_benchmark(
      stream, [&] { launch_logits(selected_tokens); }, 10, 500);
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  float graph_us = 0.0F;
  if (!seeded_cycle) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin assistant graph");
    launch(position, selected_tokens);
    check(cudaStreamEndCapture(stream, &graph), "end assistant graph");
    check(cudaGraphInstantiate(&graph_exec, graph), "instantiate assistant graph");
    graph_us = event_benchmark(
        stream,
        [&] { check(cudaGraphLaunch(graph_exec, stream), "launch assistant graph"); },
        10, 500);
  }
  float cycle_us = graph_us;
  if (drafts > 1 && !seeded_cycle) {
    cudaGraph_t cycle_graph{};
    cudaGraphExec_t cycle_exec{};
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin assistant cycle graph");
    for (int draft = 0; draft < drafts; ++draft)
      launch(position, selected_tokens + draft);
    check(cudaStreamEndCapture(stream, &cycle_graph), "end assistant cycle graph");
    check(cudaGraphInstantiate(&cycle_exec, cycle_graph),
          "instantiate assistant cycle graph");
    cycle_us = event_benchmark(
        stream,
        [&] { check(cudaGraphLaunch(cycle_exec, stream),
                    "launch assistant cycle graph"); },
        5, 200);
    check(cudaMemcpyAsync(combined_input, host_input.data(), kInput * 2,
                          cudaMemcpyHostToDevice, stream),
          "reset assistant cycle input");
    check(cudaGraphLaunch(cycle_exec, stream), "launch final assistant cycle");
    check(cudaStreamSynchronize(stream), "synchronize final assistant cycle");
    cudaGraphExecDestroy(cycle_exec);
    cudaGraphDestroy(cycle_graph);
  }

  std::array<int, 4> host_tokens{};
  std::vector<__nv_bfloat16> host_state(kTargetHidden);
  if (drafted_tokens_device) {
    check(cudaMemcpyAsync(drafted_tokens_device, selected_tokens,
                          drafts * sizeof(int), cudaMemcpyDeviceToDevice,
                          stream),
          "retain resident assistant tokens");
  } else {
    check(cudaMemcpy(host_tokens.data(), selected_tokens, drafts * sizeof(int),
                     cudaMemcpyDeviceToHost),
          "copy assistant tokens");
  }
  if (!initial_target_state_device)
    check(cudaMemcpy(host_state.data(), projected_state, kTargetHidden * 2,
                     cudaMemcpyDeviceToHost),
          "copy assistant state");
  float checksum = 0.0F;
  if (!initial_target_state_device)
    for (const auto value : host_state) checksum += __bfloat162float(value);
  float maximum_error = 0.0F;
  double mean_error = 0.0;
  bool oracle_available = false;
  bool token_match = false;
  const std::string oracle_suffix =
      drafts == 1 ? "" : "_drafts" + std::to_string(drafts);
  const auto oracle_path = std::filesystem::path("goldens/generated") /
      ("assistant_state_ctx" + std::to_string(context_tokens) + oracle_suffix + ".f32");
  const auto metadata_path = std::filesystem::path("goldens/generated") /
      ("assistant_step_ctx" + std::to_string(context_tokens) + oracle_suffix + ".json");
  if (!initial_target_state_device && std::filesystem::exists(oracle_path)) {
    oracle_available = true;
    std::ifstream input(oracle_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() !=
                      static_cast<std::streamoff>(kTargetHidden * sizeof(float)))
      throw std::runtime_error("invalid assistant oracle state");
    std::vector<float> expected(kTargetHidden);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(expected.data()),
               static_cast<std::streamsize>(expected.size() * sizeof(float)));
    for (int col = 0; col < kTargetHidden; ++col) {
      const float error = std::abs(expected[col] - __bfloat162float(host_state[col]));
      maximum_error = std::max(maximum_error, error);
      mean_error += error;
    }
    mean_error /= kTargetHidden;
  }
  if (std::filesystem::exists(metadata_path)) {
    std::ifstream input(metadata_path);
    nlohmann::json metadata;
    input >> metadata;
    if (drafts > 1 &&
        metadata.value("position_mode", std::string{}) != "single_position")
      throw std::runtime_error(
          "assistant cycle oracle does not declare single-position MTP");
    if (drafts == 1) {
      token_match = metadata.at("selected_token").get<int>() == host_tokens[0];
    } else {
      token_match = metadata.at("drafted_tokens").get<std::vector<int>>() ==
                    std::vector<int>(host_tokens.begin(), host_tokens.begin() + drafts);
    }
  }

  if (graph_exec) cudaGraphExecDestroy(graph_exec);
  if (graph) cudaGraphDestroy(graph);
  if (owns_context) {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }
  for (void* allocation : allocations) cudaFree(allocation);
  return {eager_us, graph_us, logits_us, cycle_us, host_tokens[drafts - 1],
          host_tokens, checksum,
          maximum_error, static_cast<float>(mean_error), oracle_available,
          token_match};
}

__global__ void prepare_next_assistant_input_batch_kernel(
    const __nv_bfloat16* target_embedding, const int* tokens,
    const __nv_bfloat16* projected_states, __nv_bfloat16* combined,
    int rows, int source_rows = 0) {
  constexpr int kHidden = 2816;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * kHidden) return;
  const int row = index / kHidden;
  const int col = index % kHidden;
  const auto scale = __float2bfloat16_rn(sqrtf(static_cast<float>(kHidden)));
  const auto embedding = target_embedding[
      static_cast<std::size_t>(tokens[row]) * kHidden + col];
  combined[static_cast<std::size_t>(row) * 2 * kHidden + col] =
      __float2bfloat16_rn(__bfloat162float(embedding) *
                          __bfloat162float(scale));
  combined[static_cast<std::size_t>(row) * 2 * kHidden + kHidden + col] =
      projected_states[static_cast<std::size_t>(
          source_rows ? min(row, source_rows - 1) : row) * kHidden + col];
}

__global__ void transpose_drafts_kernel(const int* draft_major,
                                        int* session_major, int sessions,
                                        int drafts) {
  const int index = threadIdx.x;
  if (index < sessions * drafts) {
    const int draft = index / sessions;
    const int session = index % sessions;
    session_major[session * drafts + draft] = draft_major[index];
  }
}

void launch_assistant_batch(
    const AssistantWeights& weights,
    std::span<KvCache* const> target_caches,
    std::span<const int> context_tokens,
    const DeviceModel& compressed_vocab,
    const DeviceModel& target_model,
    const void* initial_target_states_device,
    std::span<const int> previous_tokens,
    int* drafted_tokens_device,
    DeviceScratchPool& scratch,
    CompressedVocabRunner& vocabulary_runner,
    Nvfp4VocabRunner* nvfp4_vocabulary_runner,
    GpuExecutionContext& context, Fp8LinearRunner* fp8_linears, int drafts,
    int minimum_rows, int assistant_vocab_rows, DecodeBatchInputs* shared_inputs) {
  constexpr int kInput = 5632, kHidden = 1024, kIntermediate = 8192;
  constexpr int kTargetHidden = 2816, kVocab = 262144;
  const int logical_sessions = static_cast<int>(target_caches.size());
  const bool cull_transfers =
      std::getenv("GEVVA_DISABLE_DECODE_TRANSFER_CULL") == nullptr;
  if (logical_sessions < 1 || logical_sessions > 8 ||
      context_tokens.size() != target_caches.size() ||
      previous_tokens.size() != target_caches.size() ||
      !initial_target_states_device || !drafted_tokens_device ||
      drafts < 1 || drafts > 4 || minimum_rows < 1 || minimum_rows > 8)
    throw std::runtime_error("invalid assistant batch geometry");
  const int sessions = std::max(logical_sessions, minimum_rows);
  const int maximum_context = *std::max_element(
      context_tokens.begin(), context_tokens.end());
  const bool uniform_context = std::all_of(
      context_tokens.begin(), context_tokens.end(),
      [&](int length) { return length == maximum_context; });
  const bool assistant_gemm_attention =
      uniform_context &&
      std::getenv("GEVVA_DISABLE_ASSISTANT_GEMM_ATTENTION") == nullptr;
  for (int session = 0; session < logical_sessions; ++session) {
    if (!target_caches[session] || context_tokens[session] < 1 ||
        context_tokens[session] > target_caches[session]->maximum_context() ||
        previous_tokens[session] < 0 || previous_tokens[session] >= kVocab)
      throw std::runtime_error("invalid assistant batch session");
    target_caches[session]->ensure_context(context_tokens[session], context.stream());
  }
  scratch.reset();
  auto allocate_bytes = [&](std::size_t bytes) { return scratch.allocate(bytes); };
  auto allocate = [&](std::size_t elements) {
    return static_cast<__nv_bfloat16*>(allocate_bytes(elements * 2));
  };
  auto* combined = allocate(static_cast<std::size_t>(sessions) * kInput);
  auto* hidden = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* normalized = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* attention = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* residual = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* gate_up =
      allocate(static_cast<std::size_t>(sessions) * 2 * kIntermediate);
  auto* gate = gate_up;
  auto* up = gate_up + static_cast<std::size_t>(sessions) * kIntermediate;
  auto* product = allocate(static_cast<std::size_t>(sessions) * kIntermediate);
  auto* feedforward = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* q = allocate(static_cast<std::size_t>(sessions) * 8192);
  auto* attended = allocate(static_cast<std::size_t>(sessions) * 8192);
  auto* projected = allocate(static_cast<std::size_t>(sessions) * kTargetHidden);
  auto* selected = static_cast<int*>(allocate_bytes(sessions * 4 * sizeof(int)));
  auto* contexts_device = static_cast<int*>(allocate_bytes(2 * sessions * sizeof(int)));
  auto* previous_device = contexts_device + sessions;
  const DeviceKvView* sliding_views{};
  const DeviceKvView* global_views{};
  const int score_stride = maximum_context;
  auto* scores = static_cast<float*>(allocate_bytes(
      static_cast<std::size_t>(sessions) * 16 * score_stride * sizeof(float)));
  auto* partials = static_cast<float*>(allocate_bytes(
      static_cast<std::size_t>(sessions) * 16 * 16 * 512 * sizeof(float)));
  __nv_bfloat16* packed_queries{};
  __nv_bfloat16* probabilities{};
  __nv_bfloat16* packed_output{};
  const void** key_pointers{};
  const void** query_pointers{};
  void** score_pointers{};
  const void** value_pointers{};
  const void** probability_pointers{};
  void** output_pointers{};
  if (assistant_gemm_attention) {
    packed_queries = allocate(static_cast<std::size_t>(sessions) * 16 * 512);
    probabilities = allocate(
        static_cast<std::size_t>(sessions) * 16 * maximum_context);
    packed_output = allocate(static_cast<std::size_t>(sessions) * 16 * 512);
    const auto pointer_bytes =
        static_cast<std::size_t>(sessions) * 8 * sizeof(void*);
    key_pointers = static_cast<const void**>(allocate_bytes(pointer_bytes));
    query_pointers = static_cast<const void**>(allocate_bytes(pointer_bytes));
    score_pointers = static_cast<void**>(allocate_bytes(pointer_bytes));
    value_pointers = static_cast<const void**>(allocate_bytes(pointer_bytes));
    probability_pointers =
        static_cast<const void**>(allocate_bytes(pointer_bytes));
    output_pointers = static_cast<void**>(allocate_bytes(pointer_bytes));
  }
  std::array<KvCache*, 8> physical_caches{};
  std::array<int, 8> physical_contexts{}, physical_previous{};
  for (int session = 0; session < sessions; ++session) {
    const int source = std::min(session, logical_sessions - 1);
    physical_caches[session] = target_caches[source];
    physical_contexts[session] = context_tokens[source];
    physical_previous[session] = previous_tokens[source];
  }
  const auto stream = context.stream();
  scratch.prepare_assistant_views(
      std::span<KvCache* const>(physical_caches.data(), sessions), stream,
      sliding_views, global_views);
  auto handle = static_cast<cublasHandle_t>(context.blas_handle());
  if (shared_inputs) {
    auto* staging = static_cast<int*>(scratch.host_staging(2 * sessions * sizeof(int)));
    std::copy_n(physical_contexts.data(), sessions, staging);
    std::copy_n(physical_previous.data(), sessions, staging + sessions);
    check(cudaMemcpyAsync(contexts_device, staging, 2 * sessions * sizeof(int),
                          cudaMemcpyHostToDevice, stream), "copy shared decode inputs");
    *shared_inputs = {contexts_device, previous_device};
  } else {
    check(cudaMemcpyAsync(contexts_device, physical_contexts.data(),
                          sessions * sizeof(int), cudaMemcpyHostToDevice, stream),
          "copy assistant batch contexts");
    check(cudaMemcpyAsync(previous_device, physical_previous.data(),
                          sessions * sizeof(int), cudaMemcpyHostToDevice, stream),
          "copy assistant batch previous tokens");
  }
  if (!cull_transfers) {
    for (int session = 0; session < sessions; ++session) {
      const int source = std::min(session, logical_sessions - 1);
      check(cudaMemcpyAsync(
                projected + static_cast<std::size_t>(session) * kTargetHidden,
                static_cast<const __nv_bfloat16*>(initial_target_states_device) +
                    static_cast<std::size_t>(source) * kTargetHidden,
                kTargetHidden * 2, cudaMemcpyDeviceToDevice, stream),
            "copy assistant batch state");
    }
  }
  const auto target_embedding_view =
      target_model.tensor("model.language_model.embed_tokens.weight");
  const auto* target_embedding = reinterpret_cast<const __nv_bfloat16*>(
      target_embedding_view.data);
  prepare_next_assistant_input_batch_kernel<<<
      (sessions * kTargetHidden + 255) / 256, 256, 0, stream>>>(
      target_embedding, previous_device,
      cull_transfers ? static_cast<const __nv_bfloat16*>(initial_target_states_device)
                     : projected,
      combined, sessions, cull_transfers ? logical_sessions : 0);

  auto bf16 = [](const void* pointer) {
    return reinterpret_cast<const __nv_bfloat16*>(pointer);
  };
  check(cublasSetStream(handle, stream), "set assistant batch BLAS stream");
  const float alpha = 1.0F, beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int rows, int columns,
                     const __nv_bfloat16* source,
                     __nv_bfloat16* destination) {
    if (fp8_linears && fp8_linears->launch(
            matrix, rows, columns, source, destination, sessions, stream,
            handle))
      return;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, sessions,
                       columns, &alpha, matrix, CUDA_R_16BF, columns, source,
                       CUDA_R_16BF, columns, &beta, destination, CUDA_R_16BF,
                       rows, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "assistant batch projection");
  };
  const auto* pre_projection = bf16(weights.pre_projection());
  const auto* post_projection = bf16(weights.post_projection());
  const auto* final_norm = bf16(weights.final_norm());
  for (int draft = 0; draft < drafts; ++draft) {
    project(pre_projection, kHidden, kInput, combined, hidden);
    for (int layer = 0; layer < 4; ++layer) {
      const bool global = layer == 3;
      const int q_width = global ? 8192 : 4096;
      const auto& layer_weights = weights.layer(layer);
      rmsnorm_1024_kernel<<<sessions, 256, 0, stream>>>(
          hidden, bf16(layer_weights.input_norm), normalized);
      if (fp8_linears) fp8_linears->invalidate_activation_cache();
      project(bf16(layer_weights.q), q_width, kHidden,
              normalized, q);
      if (assistant_gemm_attention) {
        if (global)
          launch_assistant_direct_gemm_attention<512, 2, false>(
              handle, q, bf16(layer_weights.q_norm), global_views,
              contexts_device, attended,
              packed_queries, scores, probabilities, packed_output,
              key_pointers, query_pointers, score_pointers, value_pointers,
              probability_pointers, output_pointers, sessions,
              maximum_context, stream);
        else
          launch_assistant_direct_gemm_attention<256, 8, true>(
              handle, q, bf16(layer_weights.q_norm), sliding_views,
              contexts_device, attended,
              packed_queries, scores, probabilities, packed_output,
              key_pointers, query_pointers, score_pointers, value_pointers,
              probability_pointers, output_pointers, sessions,
              maximum_context, stream);
      } else {
        if (global)
          norm_rope_ragged_kernel<512><<<sessions * 16, 512, 0, stream>>>(
              q, bf16(layer_weights.q_norm), sessions, 16,
              contexts_device, 1, true);
        else
          norm_rope_ragged_kernel<256><<<sessions * 16, 256, 0, stream>>>(
              q, bf16(layer_weights.q_norm), sessions, 16,
              contexts_device, 1, false);
        launch_ragged_attention<false>(
            global, q, global ? global_views : sliding_views, nullptr,
            contexts_device, attended, scores, partials, sessions, sessions,
            1, maximum_context, stream);
      }
      project(bf16(layer_weights.o), kHidden, q_width,
              attended, attention);
      rmsnorm_add_scale_1024_kernel<<<sessions, 256, 0, stream>>>(
          attention, bf16(layer_weights.post_attention),
          hidden, residual);
      rmsnorm_1024_kernel<<<sessions, 256, 0, stream>>>(
          residual, bf16(layer_weights.pre_feedforward),
          normalized);
      if (fp8_linears) fp8_linears->invalidate_activation_cache();
      if (const auto* fused = bf16(weights.fused_gate_up(layer));
          fused && !fp8_linears) {
        project(fused, 2 * kIntermediate, kHidden, normalized, gate_up);
        assistant_gelu_tanh_multiply_packed_kernel<<<
            (sessions * kIntermediate + 255) / 256, 256, 0, stream>>>(
            gate_up, product, sessions);
      } else {
        project(bf16(layer_weights.gate), kIntermediate, kHidden,
                normalized, gate);
        project(bf16(layer_weights.up), kIntermediate, kHidden,
                normalized, up);
        gelu_tanh_multiply_kernel<<<
            (sessions * kIntermediate + 255) / 256, 256, 0, stream>>>(
            gate, up, product, sessions * kIntermediate);
      }
      project(bf16(layer_weights.down), kHidden, kIntermediate,
              product, feedforward);
      rmsnorm_add_scale_1024_kernel<<<sessions, 256, 0, stream>>>(
          feedforward, bf16(layer_weights.post_feedforward), residual, hidden,
          bf16(layer_weights.layer_scalar));
    }
    rmsnorm_1024_kernel<<<sessions, 256, 0, stream>>>(hidden, final_norm,
                                                       normalized);
    if (fp8_linears) fp8_linears->invalidate_activation_cache();
    // Only another draft consumes the projected continuation state. The
    // verifier supplies the next cycle's state, so the final projection is dead.
    if (!cull_transfers || draft + 1 < drafts)
      project(post_projection, kTargetHidden, kHidden, normalized, projected);
    if (nvfp4_vocabulary_runner)
      nvfp4_vocabulary_runner->launch_batch(
          normalized, selected + draft * sessions, sessions, stream,
          assistant_vocab_rows);
    else
      vocabulary_runner.launch_batch(normalized, selected + draft * sessions,
                                     sessions, stream);
    if (!cull_transfers || draft + 1 < drafts) {
      prepare_next_assistant_input_batch_kernel<<<
          (sessions * kTargetHidden + 255) / 256, 256, 0, stream>>>(
          target_embedding, selected + draft * sessions, projected, combined,
          sessions);
    }
  }
  transpose_drafts_kernel<<<1, 32, 0, stream>>>(selected,
                                                 drafted_tokens_device,
                                                 sessions, drafts);
  check(cudaGetLastError(), "assistant batch launch");
  (void)compressed_vocab;
}
