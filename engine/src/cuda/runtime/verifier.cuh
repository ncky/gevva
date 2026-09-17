// Private implementation fragment; included only by src/gpu.cu.
TargetVerifierBenchmark benchmark_target_verifier(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    KvCache& cache, AttentionWorkspace& workspace, int context_tokens,
    int tokens, std::span<const std::uint32_t> candidate_input_ids,
    std::span<const int> draft_tokens,
    std::span<Nvfp4ExpertRunner* const> persistent_experts,
    DeviceScratchPool* scratch,
    CompressedVocabRunner* persistent_vocab,
    CandidateKvCache* persistent_candidates,
    GpuExecutionContext* context,
    void* continuation_state_device,
    Fp8LinearRunner* fp8_linears, const int* draft_tokens_device,
    int previous_token) {
  const bool device_generation_cycle = draft_tokens_device != nullptr;
  const bool generation_cycle = !candidate_input_ids.empty() ||
                                device_generation_cycle;
  if (tokens < 1 || tokens > 5 || context_tokens < 1 ||
      context_tokens + tokens - 1 > cache.maximum_context() ||
      context_tokens + tokens - 1 > workspace.maximum_context() ||
      tokens > workspace.maximum_batch())
    throw std::runtime_error("invalid target verifier geometry");
  cache.ensure_context(context_tokens + tokens - 1, context ? context->stream() : nullptr);
  if (generation_cycle && !device_generation_cycle &&
      (candidate_input_ids.size() != static_cast<std::size_t>(tokens) ||
       draft_tokens.size() + 1 != static_cast<std::size_t>(tokens)))
    throw std::runtime_error("invalid target verifier generation inputs");
  if (device_generation_cycle && (previous_token < 0 ||
                                  previous_token >= 262144))
    throw std::runtime_error("invalid resident verifier generation inputs");
  constexpr int kHidden = 2816;
  constexpr int kDense = 2112;
  constexpr int kExperts = 128;
  constexpr int kTop = 8;
  const int position = context_tokens - 1;
  const auto& model = weights.model();
  std::unique_ptr<CandidateKvCache> owned_candidates;
  if (!persistent_candidates) {
    owned_candidates = std::make_unique<CandidateKvCache>();
    persistent_candidates = owned_candidates.get();
  }
  auto& candidates = *persistent_candidates;

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
  constexpr int kProjectionStorageTokens = 16;
  auto* hidden_a = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier hidden A)");
  auto* hidden_b = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier hidden B)");
  auto* normalized = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier normalized)");
  auto* q = allocate(kProjectionStorageTokens * 8192, "cudaMalloc(verifier Q)");
  auto* k = allocate(kProjectionStorageTokens * 2048, "cudaMalloc(verifier K)");
  auto* v = allocate(kProjectionStorageTokens * 2048, "cudaMalloc(verifier V)");
  auto* attended = allocate(kProjectionStorageTokens * 8192, "cudaMalloc(verifier attended)");
  auto* attention = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier attention)");
  auto* residual = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier residual)");
  auto* dense_pre = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier dense pre)");
  auto* expert_pre = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier expert pre)");
  auto* router_input = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier router input)");
  auto* router_logits = allocate(kProjectionStorageTokens * kExperts, "cudaMalloc(verifier router logits)");
  auto* dense_gate = allocate(kProjectionStorageTokens * kDense,
                              "cudaMalloc(verifier dense gate");
  auto* dense_up = allocate(kProjectionStorageTokens * kDense,
                            "cudaMalloc(verifier dense up");
  auto* dense_product = allocate(kProjectionStorageTokens * kDense, "cudaMalloc(verifier dense product)");
  auto* dense_output = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier dense output)");
  auto* expert_output = allocate(kProjectionStorageTokens * kHidden, "cudaMalloc(verifier expert output)");
  float* top_weights{};
  int *top_ids{}, *selected_tokens{};
  top_weights = static_cast<float*>(allocate_bytes(
      tokens * kTop * sizeof(float), "cudaMalloc(verifier route weights)"));
  top_ids = static_cast<int*>(allocate_bytes(
      tokens * kTop * sizeof(int), "cudaMalloc(verifier route ids)"));
  selected_tokens = static_cast<int*>(allocate_bytes(
      tokens * sizeof(int), "cudaMalloc(verifier tokens)"));
  auto* device_candidate_ids = static_cast<std::uint32_t*>(allocate_bytes(
      tokens * sizeof(std::uint32_t), "cudaMalloc(verifier candidate ids)"));
  auto* device_mtp_result = device_generation_cycle
      ? static_cast<DeviceMtpResult*>(allocate_bytes(
            sizeof(DeviceMtpResult), "cudaMalloc(verifier MTP result)"))
      : nullptr;

  std::vector<__nv_bfloat16> host_hidden;
  if (generation_cycle && !device_generation_cycle) {
    for (const auto id : candidate_input_ids)
      if (id >= 262144)
        throw std::runtime_error("target verifier token id is out of range");
  } else {
    host_hidden.resize(tokens * kHidden);
    for (std::size_t index = 0; index < host_hidden.size(); ++index)
      host_hidden[index] = __float2bfloat16_rn(
          std::sin(index * 0.017F) * 0.7F + std::cos(index * 0.003F) * 0.2F);
    check(cudaMemcpy(hidden_a, host_hidden.data(), host_hidden.size() * 2,
                     cudaMemcpyHostToDevice),
          "copy verifier hidden");
  }
  if (!generation_cycle)
    for (int layer = 0; layer < 30; ++layer) {
      const auto& kv = cache.layer(layer);
      const std::size_t plane = static_cast<std::size_t>(kv.capacity) *
                                kv.kv_heads * kv.head_dim * 2;
      check(cudaMemset(kv.keys, 0, plane), "clear verifier K cache");
      check(cudaMemset(kv.values, 0, plane), "clear verifier V cache");
    }

  cudaStream_t stream{};
  cublasHandle_t handle{};
  const bool owns_context = context == nullptr;
  if (context) {
    stream = context->stream();
    handle = static_cast<cublasHandle_t>(context->blas_handle());
  } else {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate(verifier)");
    check(cublasCreate(&handle), "cublasCreate(verifier)");
  }
  check(cublasSetStream(handle, stream), "cublasSetStream(verifier)");
  if (generation_cycle) {
    if (device_generation_cycle) {
      prepare_verifier_candidate_ids_kernel<<<1, 32, 0, stream>>>(
          previous_token, draft_tokens_device, device_candidate_ids, tokens);
    } else {
      check(cudaMemcpyAsync(device_candidate_ids, candidate_input_ids.data(),
                            candidate_input_ids.size_bytes(),
                            cudaMemcpyHostToDevice, stream),
            "copy verifier candidate ids");
    }
    const auto embedding = weights.embedding();
    gather_scaled_embeddings_kernel<<<
        (tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding.data),
        device_candidate_ids, hidden_a, tokens, kHidden);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int rows, int columns,
                     const __nv_bfloat16* source, __nv_bfloat16* destination,
                     bool rescale_output = true) {
    if (fp8_linears &&
        fp8_linears->launch(matrix, rows, columns, source, destination, tokens,
                            stream, handle, rescale_output))
      return;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, columns,
                       &alpha, matrix, CUDA_R_16BF, columns, source, CUDA_R_16BF,
                       columns, &beta, destination, CUDA_R_16BF, rows,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "target verifier projection");
  };
  if (!persistent_experts.empty() && persistent_experts.size() != 30)
    throw std::runtime_error("persistent verifier requires 30 expert runners");
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> owned_experts;
  std::array<Nvfp4ExpertRunner*, 30> expert_runners{};
  if (persistent_experts.empty()) {
    owned_experts.reserve(30);
    for (int layer = 0; layer < 30; ++layer) {
      owned_experts.push_back(
          std::make_unique<Nvfp4ExpertRunner>(weights.experts(), layer, tokens));
      expert_runners[layer] = owned_experts.back().get();
    }
  } else {
    std::copy(persistent_experts.begin(), persistent_experts.end(),
              expert_runners.begin());
  }
  std::unique_ptr<CompressedVocabRunner> owned_vocabulary;
  auto* vocabulary = persistent_vocab;
  if (!vocabulary) {
    owned_vocabulary = std::make_unique<CompressedVocabRunner>(
        model, "model.language_model.embed_tokens.weight", compressed_vocab,
        stream);
    vocabulary = owned_vocabulary.get();
  }

  // Thirty decoder layers perform an even number of ping-pong swaps, so the
  // final state always lands back in hidden_a. Keep this host-side value valid
  // when a cached graph launches without re-entering the enqueue lambda.
  __nv_bfloat16* final_hidden = hidden_a;
  auto launch = [&] {
    auto* current = hidden_a;
    auto* next = hidden_b;
    for (int layer = 0; layer < 30; ++layer) {
      const bool global = layer % 6 == 5;
      const int q_width = global ? 8192 : 4096;
      const int kv_width = global ? 1024 : 2048;
      const auto w = target_layer_weights(weights, layer);
      const auto* input_norm = w.input_norm;
      const auto* q_weight = w.q;
      const auto* k_weight = w.k;
      const auto* v_weight = w.v;
      const auto* o_weight = w.o;
      const auto* q_norm = w.q_norm;
      const auto* k_norm = w.k_norm;
      const auto* post_attention = w.post_attention;
      const auto* dense_pre_w = w.dense_pre;
      const auto* expert_pre_w = w.expert_pre;
      const auto* router_scale = w.router_scale;
      const auto* router_weight = w.router;
      const auto* expert_scale = w.expert_scale;
      const auto* gate_weight = w.gate;
      const auto* up_weight = w.up;
      const auto* down_weight = w.down;
      const auto* dense_post_w = w.dense_post;
      const auto* expert_post_w = w.expert_post;
      const auto* combined_post = w.combined_post;
      const auto* layer_scalar = w.layer_scalar;
      const auto& kv = cache.layer(layer);
      const auto& candidate = candidates.layer(layer);

      if (fp8_linears)
        fp8_linears->prepare_rmsnorm_2816(
            current, input_norm, normalized, tokens, stream);
      else
        rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
            current, input_norm, normalized, tokens, 1e-6F);
      Fp8OutputScale q_projection_scale{}, k_projection_scale{},
          v_projection_scale{};
      const bool concurrent_qkv = context && fp8_linears;
      cudaStream_t kv_stream = stream;
      cudaStream_t v_stream = stream;
      if (concurrent_qkv) {
        context->begin_auxiliary();
        kv_stream = context->auxiliary_stream();
        if (!global) {
          context->begin_tertiary();
          v_stream = context->tertiary_stream();
        }
      }
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              q_weight, q_width, kHidden, normalized, q, tokens, stream,
              handle, q_projection_scale))
        project(q_weight, q_width, kHidden, normalized, q);
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              k_weight, kv_width, kHidden, normalized, k, tokens, kv_stream,
              handle, k_projection_scale, concurrent_qkv))
        project(k_weight, kv_width, kHidden, normalized, k);
      if (!global &&
          (!fp8_linears || !fp8_linears->launch_unscaled(
              v_weight, kv_width, kHidden, normalized, v, tokens, v_stream,
              handle, v_projection_scale, 2)))
        project(v_weight, kv_width, kHidden, normalized, v);
      if (concurrent_qkv) {
        context->end_auxiliary();
        if (!global) context->end_tertiary();
      }
      if (q_projection_scale.activation_rows) {
        const int row_width = q_width + kv_width + (global ? 0 : kv_width);
        scale_fp8_qkv_output_kernel<<<
            (tokens * row_width + 255) / 256, 256, 0, stream>>>(
            q, q_width, q_projection_scale.weight, k, kv_width,
            k_projection_scale.weight, global ? nullptr : v,
            v_projection_scale.weight, q_projection_scale.activation_rows,
            tokens);
      }
      transform_qkv_batch(global, q, k, v, q_norm, k_norm,
                          position, tokens, stream);
      stage_candidate_kv(candidates, layer, k, v, tokens, stream);
      launch_decode_attention_batch(
          global, q, kv.keys, kv.values, candidate.keys, candidate.values,
          attended, context_tokens, tokens, workspace.scores(),
          workspace.partials(), stream);
      project(o_weight, kHidden, q_width, attended, attention);
      rmsnorm_add_2816_kernel<<<tokens, 256, 0, stream>>>(
          attention, post_attention, current, residual, tokens);
      if (fp8_linears)
        fp8_linears->prepare_dual_rmsnorm_router_2816(
            residual, dense_pre_w, expert_pre_w, router_scale, dense_pre,
            expert_pre, router_input, tokens, stream);
      else
        dual_rmsnorm_router_2816_kernel<<<tokens, 256, 0, stream>>>(
            residual, dense_pre_w, expert_pre_w, router_scale, dense_pre,
            expert_pre, router_input, tokens);
      Fp8OutputScale router_projection_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              router_weight, kExperts, kHidden, router_input, router_logits,
              tokens, stream, handle, router_projection_scale))
        project(router_weight, kExperts, kHidden, router_input, router_logits);
      route_128_top8_kernel<<<tokens, kExperts, 0, stream>>>(
          router_logits, expert_scale, top_weights, top_ids, tokens,
          router_projection_scale.activation_rows,
          router_projection_scale.weight);
      const bool concurrent_dense = context && fp8_linears;
      cudaStream_t dense_stream = stream;
      if (concurrent_dense) {
        context->begin_auxiliary();
        dense_stream = context->auxiliary_stream();
      }
      if (fp8_linears)
        fp8_linears->use_secondary_activation_2816(
            dense_pre, tokens, dense_stream);
      Fp8OutputScale gate_projection_scale{}, up_projection_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              gate_weight, kDense, kHidden, dense_pre, dense_gate, tokens,
              dense_stream, handle, gate_projection_scale,
              concurrent_dense))
        project(gate_weight, kDense, kHidden, dense_pre, dense_gate);
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              up_weight, kDense, kHidden, dense_pre, dense_up, tokens,
              dense_stream, handle, up_projection_scale,
              concurrent_dense))
        project(up_weight, kDense, kHidden, dense_pre, dense_up);
      if (fp8_linears)
        fp8_linears->prepare_gelu_2112(
            dense_gate, dense_up, gate_projection_scale, up_projection_scale,
            dense_product, tokens, dense_stream);
      else
        gelu_tanh_multiply_kernel<<<
            (tokens * kDense + 255) / 256, 256, 0, stream>>>(
                dense_gate, dense_up, dense_product, tokens * kDense);
      if (concurrent_dense) {
        if (!fp8_linears->launch(
                down_weight, kHidden, kDense, dense_product, dense_output,
                tokens, dense_stream, handle, true, true))
          throw std::runtime_error("FP8 dense branch projection unavailable");
      } else {
        project(down_weight, kHidden, kDense, dense_product, dense_output);
      }
      expert_runners[layer]->launch_batch(
          expert_pre, top_ids, top_weights, expert_output, tokens, stream);
      if (concurrent_dense) context->end_auxiliary();
      finalize_feedforward_2816_kernel<<<tokens, 256, 0, stream>>>(
          dense_output, dense_post_w, expert_output, expert_post_w,
          combined_post, residual, layer_scalar, next, tokens);
      std::swap(current, next);
    }
    final_hidden = current;
    const auto* final_norm =
        static_cast<const __nv_bfloat16*>(weights.final_norm());
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        current, final_norm, normalized, tokens, 1e-6F);
    vocabulary->launch_batch(normalized, selected_tokens, tokens, stream);
    if (device_generation_cycle) {
      commit_candidate_kv_kernel<<<30, 256, 0, stream>>>(
          candidates.device_layers(), cache.device_layers(), position, 0,
          draft_tokens_device, selected_tokens, final_hidden,
          static_cast<__nv_bfloat16*>(continuation_state_device),
          device_mtp_result, tokens - 1);
    }
  };

  constexpr std::uint64_t kVerifierGraphNamespace = 0x5456470000000000ULL;
  const std::uint64_t persistent_graph_key =
      (kVerifierGraphNamespace ^ experimental_expert_graph_key()) |
      (static_cast<std::uint64_t>(tokens) << 32) |
      static_cast<std::uint64_t>(context_tokens);
  const bool use_persistent_graph = generation_cycle && context;
  const bool graph_was_cached =
      use_persistent_graph && context->has_graph(persistent_graph_key);
  if (graph_was_cached)
    context->launch_graph(persistent_graph_key);
  else
    launch();
  check(cudaGetLastError(), "target verifier launch");
  check(cudaStreamSynchronize(stream), "target verifier synchronize");
  if (use_persistent_graph && !graph_was_cached)
    context->capture_graph(persistent_graph_key, launch);
  const float eager_us = generation_cycle ? 0.0F :
      event_benchmark(stream, launch, 2, 20);
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  float graph_us = 0.0F;
  if (!generation_cycle) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin target verifier graph");
    launch();
    check(cudaStreamEndCapture(stream, &graph), "end target verifier graph");
    check(cudaGraphInstantiate(&executable, graph),
          "instantiate target verifier graph");
    graph_us = event_benchmark(
        stream,
        [&] { check(cudaGraphLaunch(executable, stream),
                    "launch target verifier graph"); },
        2, 30);
  }
  std::array<int, 5> host_tokens{};
  std::array<int, 4> host_drafts{};
  DeviceMtpResult host_mtp_result{};
  if (device_generation_cycle) {
    check(cudaMemcpy(&host_mtp_result, device_mtp_result,
                     sizeof(host_mtp_result), cudaMemcpyDeviceToHost),
          "copy resident MTP result");
    std::copy(std::begin(host_mtp_result.target_tokens),
              std::end(host_mtp_result.target_tokens), host_tokens.begin());
    std::copy(std::begin(host_mtp_result.draft_tokens),
              std::end(host_mtp_result.draft_tokens), host_drafts.begin());
  } else {
    check(cudaMemcpy(host_tokens.data(), selected_tokens, tokens * sizeof(int),
                     cudaMemcpyDeviceToHost),
          "copy verifier tokens");
    std::copy(draft_tokens.begin(), draft_tokens.end(), host_drafts.begin());
  }
  std::vector<__nv_bfloat16> host_output;
  if (!continuation_state_device) {
    host_output.resize(tokens * kHidden);
    check(cudaMemcpy(host_output.data(), final_hidden, host_output.size() * 2,
                     cudaMemcpyDeviceToHost),
          "copy verifier hidden");
  }
  float checksum = 0.0F;
  for (const auto value : host_output) checksum += __bfloat162float(value);

  std::array<int, 5> output_tokens{};
  int output_count = 0;
  int matched_drafts = 0;
  std::vector<float> continuation_state;
  int commit_tokens = tokens;
  if (generation_cycle) {
    if (device_generation_cycle) {
      std::copy(std::begin(host_mtp_result.output_tokens),
                std::end(host_mtp_result.output_tokens), output_tokens.begin());
      output_count = host_mtp_result.output_count;
      matched_drafts = host_mtp_result.matched_drafts;
      commit_tokens = output_count;
    } else {
      const auto acceptance = resolve_greedy_mtp(
          host_drafts, std::span<const int>(
                           host_tokens.data(), static_cast<std::size_t>(tokens)));
      output_tokens = acceptance.output_tokens;
      output_count = acceptance.output_count;
      matched_drafts = acceptance.matched_drafts;
      commit_tokens = acceptance.output_count;
      const int state_row = acceptance.matched_drafts;
      if (continuation_state_device) {
      check(cudaMemcpyAsync(
                continuation_state_device, final_hidden + state_row * kHidden,
                kHidden * 2, cudaMemcpyDeviceToDevice, stream),
            "retain verifier continuation state");
      } else {
        continuation_state.resize(kHidden);
        for (int col = 0; col < kHidden; ++col)
          continuation_state[col] = __bfloat162float(
              host_output[state_row * kHidden + col]);
      }
    }
  }
  if (!device_generation_cycle)
    commit_candidate_kv(candidates, cache, position, commit_tokens, stream);

  if (executable) cudaGraphExecDestroy(executable);
  if (graph) cudaGraphDestroy(graph);
  if (owns_context) {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }
  for (void* pointer : allocations) cudaFree(pointer);
  return {eager_us, graph_us, host_tokens, host_drafts, checksum, output_tokens,
          output_count, matched_drafts, std::move(continuation_state)};
}

__global__ void prepare_verifier_candidate_ids_batch_kernel(
    const int* previous, const int* drafts, std::uint32_t* ids, int sessions,
    int draft_count) {
  const int tokens = draft_count + 1;
  const int row = threadIdx.x;
  if (row >= sessions * tokens) return;
  const int session = row / tokens;
  const int step = row % tokens;
  ids[row] = static_cast<std::uint32_t>(
      step == 0 ? previous[session]
                : drafts[session * draft_count + step - 1]);
}

__global__ void stage_candidate_kv_batch_kernel(
    const __nv_bfloat16* keys, const __nv_bfloat16* values,
    const DeviceKvView* candidates, int sessions, int plane, int tokens) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = sessions * tokens * plane;
  if (index >= elements) return;
  const int row = index / plane;
  const int column = index % plane;
  const int session = row / tokens;
  const int step = row % tokens;
  auto view = candidates[session];
  reinterpret_cast<__nv_bfloat16*>(view.keys)[step * plane + column] =
      keys[index];
  reinterpret_cast<__nv_bfloat16*>(view.values)[step * plane + column] =
      values[index];
}

__global__ void commit_candidate_kv_batch_kernel(
    const DeviceKvView* candidates, const DeviceKvView* destinations,
    const int* positions, const int* drafts, const int* targets,
    const __nv_bfloat16* hidden, __nv_bfloat16* continuation,
    DeviceMtpResult* results, int sessions, int draft_count) {
  const int tokens = draft_count + 1;
  const int session = blockIdx.x;
  const int layer = blockIdx.y;
  if (session >= sessions || layer >= 30) return;
  __shared__ int matched;
  if (threadIdx.x == 0) {
    int count = 0;
    while (count < draft_count &&
           drafts[session * draft_count + count] ==
               targets[session * tokens + count]) ++count;
    matched = count;
    if (layer == 0) {
      auto& result = results[session];
      for (int token = 0; token < 5; ++token) {
        result.target_tokens[token] = 0;
        result.output_tokens[token] = 0;
      }
      for (int token = 0; token < 4; ++token)
        result.draft_tokens[token] = 0;
      result.matched_drafts = count;
      result.output_count = count + 1;
      for (int token = 0; token < count; ++token)
        result.output_tokens[token] = drafts[session * draft_count + token];
      result.output_tokens[count] = targets[session * tokens + count];
      for (int token = 0; token < tokens; ++token)
        result.target_tokens[token] = targets[session * tokens + token];
      for (int token = 0; token < draft_count; ++token)
        result.draft_tokens[token] = drafts[session * draft_count + token];
    }
  }
  __syncthreads();
  if (layer == 0) {
    for (int column = threadIdx.x; column < 2816; column += blockDim.x)
      continuation[static_cast<std::size_t>(session) * 2816 + column] =
          hidden[(static_cast<std::size_t>(session) * tokens + matched) * 2816 +
                 column];
  }
  const auto source = candidates[layer * sessions + session];
  const auto destination = destinations[layer * sessions + session];
  const int plane = source.kv_heads * source.head_dim;
  for (int index = threadIdx.x; index < (matched + 1) * plane;
       index += blockDim.x) {
    const int token = index / plane;
    const int column = index % plane;
    const int slot = (positions[session] + token) % destination.capacity;
    const int destination_index = slot * plane + column;
    const int source_index = token * plane + column;
    const auto key =
        reinterpret_cast<const __nv_bfloat16*>(source.keys)[source_index];
    const auto value =
        reinterpret_cast<const __nv_bfloat16*>(source.values)[source_index];
    auto* destination_keys =
        reinterpret_cast<__nv_bfloat16*>(destination.keys);
    auto* destination_values =
        reinterpret_cast<__nv_bfloat16*>(destination.values);
    destination_keys[destination_index] = key;
    destination_values[destination_index] = value;
    if (destination.kv_heads == 8 && destination.capacity == 1024) {
      for (int copy = 1; copy < 3; ++copy) {
        const int mirrored = destination_index + copy * destination.capacity * plane;
        destination_keys[mirrored] = key;
        destination_values[mirrored] = value;
      }
      if (slot < 4) {
        const int mirrored = destination_index + 3 * destination.capacity * plane;
        destination_keys[mirrored] = key;
        destination_values[mirrored] = value;
      }
    }
  }
}

TargetVerifierBatchResult launch_target_verifier_batch(
    const RuntimeWeights& weights,
    std::span<KvCache* const> caches,
    std::span<CandidateKvCache* const> candidate_caches,
    std::span<const int> context_tokens,
    std::span<const int> previous_tokens,
    const int* drafted_tokens_device,
    void* continuation_states_device,
    std::span<Nvfp4ExpertRunner* const> experts,
    DeviceScratchPool& scratch,
    CompressedVocabRunner& vocabulary,
    Nvfp4VocabRunner* nvfp4_vocabulary,
    Fp8LinearRunner& fp8_linears,
    GpuExecutionContext& context, int drafts, const DecodeBatchInputs* shared_inputs) {
  constexpr int kHidden = 2816, kDense = 2112, kExperts = 128, kTop = 8;
  constexpr int kStorageRows = 64;
  const int sessions = static_cast<int>(caches.size());
  const int tokens = drafts + 1;
  const int rows = sessions * tokens;
  const int add_norm_threads =
      fused_norm_threads("G4_ATTENTION_NORM_THREADS");
  const int router_norm_threads =
      fused_norm_threads("G4_ROUTER_NORM_THREADS");
  const int final_norm_threads =
      fused_norm_threads("G4_FINAL_NORM_THREADS");
  const char* attention_router_setting = std::getenv("G4_FUSE_ATTENTION_ROUTER");
  const bool fuse_attention_router =
      (!attention_router_setting || std::string_view(attention_router_setting) != "0") &&
      add_norm_threads == router_norm_threads;
  if (sessions < 1 || sessions > 8 || candidate_caches.size() != caches.size() ||
      context_tokens.size() != caches.size() ||
      previous_tokens.size() != caches.size() || experts.size() != 30 ||
      !drafted_tokens_device || !continuation_states_device ||
      drafts < 0 || drafts > 4)
    throw std::runtime_error("invalid target verifier batch geometry");
  const int actual_maximum_context = *std::max_element(
      context_tokens.begin(), context_tokens.end()) + tokens;
  const char* gemm_attention_rows_env =
      std::getenv("G4_GEMM_ATTENTION_MAX_CONTEXT_ROWS");
  const long long gemm_attention_max_context_rows = gemm_attention_rows_env
      ? std::stoll(gemm_attention_rows_env)
      : 1048576LL;
  const bool gemm_attention =
      std::getenv("G4_SCALAR_ATTENTION") == nullptr &&
      static_cast<long long>(actual_maximum_context) * sessions <=
          gemm_attention_max_context_rows;
  const bool graph_decode = gemm_attention &&
      std::getenv("G4_DECODE_GRAPHS") != nullptr;
  // At four or more sessions, reading the token-major global KV cache with a
  // strided batched GEMM is faster than materializing a head-major copy.  The
  // packed path remains faster for the latency-oriented one/two-row cases.
  const bool tiled_attention = gemm_attention && (sessions == 1 || sessions == 8) &&
      std::getenv("G4_STAGED_ATTENTION") != nullptr;
  const bool exact_attention_opt = exact_attention_opt_enabled();
  const bool fast_gemm_softmax = exact_attention_opt || std::getenv("G4_FAST_GEMM_SOFTMAX") != nullptr;
  const bool fused_attention_unpack = gemm_attention &&
      (exact_attention_opt || std::getenv("G4_FUSED_ATTENTION_UNPACK") != nullptr);
  const bool tiled_global = tiled_attention && sessions == 1 &&
      actual_maximum_context >= 1024 && actual_maximum_context <= 8192;
  if (tiled_attention) prepare_serving_tiled_attention();
  bool direct_global_attention = gemm_attention && (sessions >= 4 || tiled_global) &&
      std::getenv("G4_DISABLE_DIRECT_GLOBAL_ATTENTION") == nullptr;
  const bool direct_sliding_attention = gemm_attention &&
      std::getenv("G4_DISABLE_DIRECT_SLIDING_ATTENTION") == nullptr;
  const bool fused_qkv_norm =
      std::getenv("G4_DISABLE_FUSED_QKV_NORM") == nullptr;
  const bool combined_qkv_transform =
      std::getenv("G4_DISABLE_COMBINED_QKV_TRANSFORM") == nullptr;
  const bool fused_projection_norms =
      std::getenv("G4_DISABLE_FUSED_PROJECTION_NORMS") == nullptr;
  const bool fused_expert_finalize =
      std::getenv("G4_DISABLE_VERIFIER_FUSED_EXPERT_FINALIZE") == nullptr;
  const bool fused_dense_gate_up = std::getenv("G4_FUSED_TARGET_GATE_UP") != nullptr;
  const bool parallel_dense_gate_up = !fused_dense_gate_up && sessions >= 4 &&
      std::getenv("G4_DISABLE_PARALLEL_DENSE_GATE_UP") == nullptr;
  const bool fused_direct_qkv_enabled =
      std::getenv("G4_DISABLE_FUSED_DIRECT_QKV") == nullptr;
  const bool scalar_global_attention =
      std::getenv("G4_SCALAR_GLOBAL_ATTENTION") != nullptr;
  const bool scalar_sliding_attention =
      std::getenv("G4_SCALAR_SLIDING_ATTENTION") != nullptr;
  // A stable bucket gives the captured decode graph fixed GEMM dimensions and
  // scratch addresses for many consecutive speculative cycles.
  const int maximum_context = graph_decode
      ? (actual_maximum_context + 255) & ~255
      : actual_maximum_context;
  const bool aligned_direct_attention =
      std::getenv("G4_ALIGNED_DIRECT_ATTENTION") != nullptr;
  const int attention_storage_context = aligned_direct_attention
      ? (maximum_context + 31) & ~31 : maximum_context;
  // Independent callers may reserve different capacities. In that case the
  // packed path is the safe fallback; do not grow beyond a row's reservation.
  direct_global_attention = direct_global_attention && std::all_of(
      caches.begin(), caches.end(), [&](const KvCache* cache) {
        return cache && cache->maximum_context() >= maximum_context;
      });
  for (int session = 0; session < sessions; ++session) {
    if (!caches[session] || !candidate_caches[session] ||
        context_tokens[session] < 1 ||
        context_tokens[session] + tokens > caches[session]->maximum_context())
      throw std::runtime_error("invalid target verifier batch session");
    // A direct batched GEMM uses the longest row's N for *every* cache.
    // Softmax masking happens after QK and cannot protect unmapped shorter
    // rows. Packed/scalar paths bound their reads per row instead.
    const int read_span = direct_global_attention && !scalar_global_attention
        ? maximum_context : context_tokens[session] + tokens;
    caches[session]->ensure_context(read_span, context.stream());
  }
  const auto stream = context.stream();
  auto handle = static_cast<cublasHandle_t>(context.blas_handle());

  scratch.reset();
  auto bytes = [&](std::size_t count) { return scratch.allocate(count); };
  auto bf16 = [&](std::size_t count) {
    return static_cast<__nv_bfloat16*>(bytes(count * 2));
  };
  auto* hidden_a = bf16(kStorageRows * kHidden);
  auto* hidden_b = bf16(kStorageRows * kHidden);
  auto* normalized = bf16(kStorageRows * kHidden);
  auto* q = bf16(kStorageRows * 8192);
  auto* k = bf16(kStorageRows * 2048);
  auto* v = bf16(kStorageRows * 2048);
  auto* attended = bf16(kStorageRows * 8192);
  auto* attention = bf16(kStorageRows * kHidden);
  auto* residual = bf16(kStorageRows * kHidden);
  auto* dense_pre = bf16(kStorageRows * kHidden);
  auto* expert_pre = bf16(kStorageRows * kHidden);
  auto* router_input = bf16(kStorageRows * kHidden);
  auto* router_logits = bf16(kStorageRows * kExperts);
  auto* dense_gate = bf16(kStorageRows * kDense);
  auto* dense_up = bf16(kStorageRows * kDense);
  auto* dense_gate_up = bf16(kStorageRows * kDense * 2);
  auto* dense_product = bf16(kStorageRows * kDense);
  auto* dense_output = bf16(kStorageRows * kHidden);
  auto* expert_output = bf16(kStorageRows * kHidden);
  auto* top_weights = static_cast<float*>(bytes(rows * kTop * sizeof(float)));
  auto* top_ids = static_cast<int*>(bytes(rows * kTop * sizeof(int)));
  auto* selected = static_cast<int*>(bytes(rows * sizeof(int)));
  auto* ids = static_cast<std::uint32_t*>(bytes(rows * sizeof(std::uint32_t)));
  auto* session_inputs_device =
      static_cast<int*>(bytes(2 * sessions * sizeof(int)));
  const int* contexts_device = shared_inputs ? shared_inputs->contexts : session_inputs_device;
  const int* previous_device = shared_inputs ? shared_inputs->previous : session_inputs_device + sessions;
  const DeviceKvView* live_views{};
  const DeviceKvView* candidate_views{};
  scratch.prepare_verifier_views(caches, candidate_caches, stream,
                                 live_views, candidate_views);
  auto* results_device = static_cast<DeviceMtpResult*>(
      bytes(sessions * sizeof(DeviceMtpResult)));
  auto* scores = static_cast<float*>(bytes(
      static_cast<std::size_t>(rows) * 16 * attention_storage_context * sizeof(float)));
  auto* partials = static_cast<float*>(bytes(
      static_cast<std::size_t>(rows) * 16 * 16 * 512 * sizeof(float)));
  float* tiled_partial = nullptr;
  if (tiled_attention) {
    const int tiled_keys = tiled_global ? maximum_context : std::min(maximum_context, 1024 + tokens - 1);
    // S=32 is the smallest selected tile; size conservatively for both heads.
    tiled_partial = static_cast<float*>(bytes(
        static_cast<std::size_t>(rows) * 16 * ((tiled_keys + 31) / 32) * 514 * sizeof(float)));
  }
  __nv_bfloat16 *packed_q{}, *packed_k{}, *packed_v{}, *probabilities{},
      *packed_attention{};
  const void **direct_keys{}, **direct_queries{}, **direct_values{},
      **direct_probabilities{};
  void **direct_scores{}, **direct_outputs{};
  if (gemm_attention) {
    const std::size_t query_elements =
        static_cast<std::size_t>(rows) * 16 * 512;
    const std::size_t global_kv_elements =
        static_cast<std::size_t>(sessions) * 2 * maximum_context * 512;
    const std::size_t sliding_kv_elements =
        static_cast<std::size_t>(sessions) * 8 *
        std::min(maximum_context, 1024 + tokens - 1) * 256;
    const std::size_t kv_elements =
        std::max(global_kv_elements, sliding_kv_elements);
    packed_q = bf16(query_elements);
    packed_k = bf16(kv_elements);
    packed_v = bf16(kv_elements);
    probabilities = bf16(
        static_cast<std::size_t>(rows) * 16 * attention_storage_context);
    packed_attention = bf16(query_elements);
    if (direct_global_attention || direct_sliding_attention) {
      const std::size_t pointer_bytes =
          static_cast<std::size_t>(sessions) * 8 * sizeof(void*);
      direct_keys = static_cast<const void**>(bytes(pointer_bytes));
      direct_queries = static_cast<const void**>(bytes(pointer_bytes));
      direct_scores = static_cast<void**>(bytes(pointer_bytes));
      direct_values = static_cast<const void**>(bytes(pointer_bytes));
      direct_probabilities = static_cast<const void**>(bytes(pointer_bytes));
      direct_outputs = static_cast<void**>(bytes(pointer_bytes));
    }
  }
  const auto embedding = weights.embedding();
  const auto* final_norm =
      static_cast<const __nv_bfloat16*>(weights.final_norm());
  constexpr std::size_t kResultOffset = 128;
  auto* host_staging = static_cast<std::byte*>(scratch.host_staging(
      kResultOffset + 8 * sizeof(DeviceMtpResult)));
  auto* session_inputs = reinterpret_cast<int*>(host_staging);
  if (!shared_inputs) {
    std::copy(context_tokens.begin(), context_tokens.end(), session_inputs);
    std::copy(previous_tokens.begin(), previous_tokens.end(),
              session_inputs + sessions);
    check(cudaMemcpyAsync(session_inputs_device, session_inputs,
                          2 * sessions * sizeof(int), cudaMemcpyHostToDevice,
                          stream),
          "copy verifier batch session inputs");
  }
  check(cublasSetStream(handle, stream), "set verifier batch BLAS stream");
  auto enqueue_compute = [&] {
  prepare_verifier_candidate_ids_batch_kernel<<<1, 64, 0, stream>>>(
      previous_device, drafted_tokens_device, ids, sessions, drafts);
  gather_scaled_embeddings_kernel<<<(rows * kHidden + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(embedding.data), ids, hidden_a,
      rows, kHidden);
  const float alpha = 1.0F, beta = 0.0F;
  auto fallback = [&](const __nv_bfloat16* matrix, int outputs, int inputs,
                      const __nv_bfloat16* source,
                      __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, outputs, rows, inputs,
                       &alpha, matrix, CUDA_R_16BF, inputs, source,
                       CUDA_R_16BF, inputs, &beta, destination, CUDA_R_16BF,
                       outputs, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "verifier batch fallback projection");
  };
  auto* current = hidden_a;
  auto* next = hidden_b;
  for (int layer = 0; layer < 30; ++layer) {
    const bool global = layer % 6 == 5;
    const int q_width = global ? 8192 : 4096;
    const int kv_width = global ? 1024 : 2048;
    const auto w = target_layer_weights(weights, layer);
    fp8_linears.prepare_rmsnorm_2816(current, w.input_norm,
                                      normalized, rows, stream);
    Fp8OutputScale qs{}, ks{}, vs{};
    context.begin_auxiliary();
    if (!global) context.begin_tertiary();
    if (!fp8_linears.launch_unscaled(w.q, q_width,
                                     kHidden, normalized, q, rows, stream,
                                     handle, qs, 0, true))
      fallback(w.q, q_width, kHidden, normalized, q);
    if (!fp8_linears.launch_unscaled(w.k, kv_width,
                                     kHidden, normalized, k, rows,
                                     context.auxiliary_stream(), handle, ks, 1,
                                     true))
      throw std::runtime_error("verifier batch FP8 K unavailable");
    if (!global && !fp8_linears.launch_unscaled(
                       w.v, kv_width, kHidden,
                       normalized, v, rows, context.tertiary_stream(), handle,
                       vs, 2, true))
      throw std::runtime_error("verifier batch FP8 V unavailable");
    context.end_auxiliary();
    if (!global) context.end_tertiary();
    if (!fused_qkv_norm)
      scale_fp8_qkv_output_kernel<<<
          (rows * (q_width + kv_width + (global ? 0 : kv_width)) + 255) / 256,
          256, 0, stream>>>(q, q_width, qs.weight, k, kv_width, ks.weight,
                            global ? nullptr : v, vs.weight,
                            qs.activation_rows, rows);
    const bool fused_direct_qkv = combined_qkv_transform &&
        (global ? direct_global_attention : direct_sliding_attention) &&
        fused_direct_qkv_enabled;
    if (combined_qkv_transform) {
      if (fused_direct_qkv && global)
        transform_qkv_direct_attention_kernel<512, 2, true, false>
            <<<rows * 18, 512, 0, stream>>>(
                q, k, v, w.q_norm, w.k_norm, rows, contexts_device, tokens,
                fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight,
                fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight,
                fused_qkv_norm ? vs.activation_rows : nullptr, vs.weight,
                live_views + layer * sessions,
                candidate_views + layer * sessions, packed_q, scores,
                probabilities, packed_attention, direct_keys, direct_queries,
                direct_scores, direct_values, direct_probabilities,
                direct_outputs, sessions, attention_storage_context);
      else if (fused_direct_qkv)
        transform_qkv_direct_attention_kernel<256, 8, false, true>
            <<<rows * 24, 256, 0, stream>>>(
                q, k, v, w.q_norm, w.k_norm, rows, contexts_device, tokens,
                fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight,
                fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight,
                fused_qkv_norm ? vs.activation_rows : nullptr, vs.weight,
                live_views + layer * sessions,
                candidate_views + layer * sessions, packed_q, scores,
                probabilities, packed_attention, direct_keys, direct_queries,
                direct_scores, direct_values, direct_probabilities,
                direct_outputs, sessions,
                aligned_direct_attention
                    ? (std::min(maximum_context, 1024 + tokens - 1) + 31) & ~31
                    : std::min(maximum_context, 1024 + tokens - 1));
      else if (global)
        transform_qkv_ragged_kernel<512, 2, true>
            <<<rows * 18, 512, 0, stream>>>(
                q, k, v, w.q_norm, w.k_norm, rows, contexts_device, tokens,
                fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight,
                fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight,
                fused_qkv_norm ? vs.activation_rows : nullptr, vs.weight, -1);
      else
        transform_qkv_ragged_kernel<256, 8, false>
            <<<rows * 24, 256, 0, stream>>>(
                q, k, v, w.q_norm, w.k_norm, rows, contexts_device, tokens,
                fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight,
                fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight,
                fused_qkv_norm ? vs.activation_rows : nullptr, vs.weight, -1);
    } else if (global) {
      check(cudaMemcpyAsync(v, k, rows * kv_width * 2,
                            cudaMemcpyDeviceToDevice, stream),
            "copy verifier batch global values");
      norm_rope_ragged_kernel<512><<<rows * 16, 512, 0, stream>>>(
          q, w.q_norm, rows, 16, contexts_device, tokens,
          true, fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight);
      norm_rope_ragged_kernel<512><<<rows * 2, 512, 0, stream>>>(
          k, w.k_norm, rows, 2, contexts_device, tokens,
          true, fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight);
      head_rmsnorm_kernel<512><<<rows * 2, 512, 0, stream>>>(
          v, rows * 2, fused_qkv_norm ? ks.activation_rows : nullptr,
          ks.weight, 2);
    } else {
      norm_rope_ragged_kernel<256><<<rows * 16, 256, 0, stream>>>(
          q, w.q_norm, rows, 16, contexts_device, tokens,
          false, fused_qkv_norm ? qs.activation_rows : nullptr, qs.weight);
      norm_rope_ragged_kernel<256><<<rows * 8, 256, 0, stream>>>(
          k, w.k_norm, rows, 8, contexts_device, tokens,
          false, fused_qkv_norm ? ks.activation_rows : nullptr, ks.weight);
      head_rmsnorm_kernel<256><<<rows * 8, 256, 0, stream>>>(
          v, rows * 8, fused_qkv_norm ? vs.activation_rows : nullptr,
          vs.weight, 8);
    }
    const int plane = global ? 2 * 512 : 8 * 256;
    const bool layer_gemm_attention = gemm_attention &&
        !(global && scalar_global_attention) &&
        !(!global && scalar_sliding_attention);
    if (layer_gemm_attention) {
      if (global && direct_global_attention)
        launch_direct_gemm_attention<512, 2, false>(
            handle, q, k, v, live_views + layer * sessions,
            candidate_views + layer * sessions, contexts_device, attended,
            packed_q, scores, probabilities, packed_attention, direct_keys,
            direct_queries, direct_scores, direct_values,
            direct_probabilities, direct_outputs, sessions, tokens,
            maximum_context, stream, fused_direct_qkv, tiled_global ? tiled_partial : nullptr, fused_attention_unpack, fast_gemm_softmax);
      else if (global)
        launch_ragged_gemm_attention<512, 2, false>(
            handle, q, live_views + layer * sessions,
            candidate_views + layer * sessions, contexts_device, attended,
            k, v,
            packed_q, packed_k, packed_v, scores, probabilities,
            packed_attention, sessions, tokens, maximum_context, stream, fused_attention_unpack, fast_gemm_softmax);
      else if (!global && direct_sliding_attention)
        launch_direct_gemm_attention<256, 8, true>(
            handle, q, k, v, live_views + layer * sessions,
            candidate_views + layer * sessions, contexts_device, attended,
            packed_q, scores, probabilities, packed_attention, direct_keys,
            direct_queries, direct_scores, direct_values,
            direct_probabilities, direct_outputs, sessions, tokens,
            std::min(maximum_context, 1024 + tokens - 1), stream,
            fused_direct_qkv, tiled_partial, fused_attention_unpack, fast_gemm_softmax);
      else
        launch_ragged_gemm_attention<256, 8, true>(
            handle, q, live_views + layer * sessions,
            candidate_views + layer * sessions, contexts_device, attended,
            k, v,
            packed_q, packed_k, packed_v, scores, probabilities,
            packed_attention, sessions, tokens, maximum_context, stream, fused_attention_unpack, fast_gemm_softmax);
      if (fused_attention_unpack)
        fp8_linears.prepare_packed_attention(packed_attention, attended,
            live_views + layer * sessions, contexts_device, sessions, tokens,
            global ? 512 : 256, !global && direct_sliding_attention, stream);
    } else {
      stage_candidate_kv_batch_kernel<<<
          (rows * plane + 255) / 256, 256, 0, stream>>>(
          k, v, candidate_views + layer * sessions, sessions, plane, tokens);
      launch_ragged_attention<true>(
          global, q, live_views + layer * sessions,
          candidate_views + layer * sessions, contexts_device, attended,
          scores, partials, rows, sessions, tokens, maximum_context, stream);
    }
    Fp8OutputScale attention_scale{};
    const bool attention_launched = fused_projection_norms
        ? fp8_linears.launch_unscaled(
              w.o, kHidden, q_width, attended,
              attention, rows, stream, handle, attention_scale, 0, true)
        : fp8_linears.launch(
              w.o, kHidden, q_width, attended,
              attention, rows, stream, handle, true, 0, true);
    if (!attention_launched)
      fallback(w.o, kHidden, q_width, attended,
               attention);
    if (!fuse_attention_router) {
      rmsnorm_add_2816_kernel<<<rows, add_norm_threads, 0, stream>>>(
          attention, w.post_attention, current, residual, rows,
          fused_projection_norms ? attention_scale.activation_rows : nullptr,
          attention_scale.weight);
    }
    fp8_linears.prepare_dual_rmsnorm_router_2816(
        fuse_attention_router ? current : residual,
        w.dense_pre, w.expert_pre, w.router_scale, dense_pre,
        expert_pre, router_input, rows, stream, router_norm_threads,
        fuse_attention_router ? attention : nullptr, w.post_attention,
        residual, fused_projection_norms ? attention_scale.activation_rows : nullptr,
        attention_scale.weight);
    Fp8OutputScale router_scale{};
    if (!fp8_linears.launch_unscaled(w.router, kExperts,
                                     kHidden, router_input, router_logits,
                                     rows, stream, handle, router_scale, 0,
                                     true))
      fallback(w.router, kExperts, kHidden, router_input,
               router_logits);
    route_128_top8_kernel<<<rows, kExperts, 0, stream>>>(
        router_logits, w.expert_scale, top_weights, top_ids,
        rows, router_scale.activation_rows, router_scale.weight);
    context.begin_auxiliary();
    if (parallel_dense_gate_up) context.begin_tertiary();
    fp8_linears.use_secondary_activation_2816(
        dense_pre, rows, context.auxiliary_stream());
    Fp8OutputScale gate_scale{}, up_scale{};
    if (fused_dense_gate_up) {
      if (!fp8_linears.launch_fused_gate_up(w.gate, w.up, dense_pre, dense_gate_up,
          rows, context.auxiliary_stream(), handle, gate_scale, up_scale))
        throw std::runtime_error("fused target gate/up was not prepared at load");
    } else {
    const bool gate_launched = fp8_linears.launch_unscaled(
        w.gate, kDense, kHidden, dense_pre, dense_gate, rows,
        context.auxiliary_stream(), handle, gate_scale, 1, true);
    const auto up_stream = parallel_dense_gate_up
        ? context.tertiary_stream()
        : context.auxiliary_stream();
    if (parallel_dense_gate_up)
      fp8_linears.assume_activation_ready_on(up_stream);
    const bool up_launched = fp8_linears.launch_unscaled(
        w.up, kDense, kHidden, dense_pre, dense_up, rows, up_stream, handle,
        up_scale, parallel_dense_gate_up ? 2 : 1, true);
    if (!gate_launched || !up_launched)
      throw std::runtime_error("verifier batch FP8 dense input unavailable");
    if (parallel_dense_gate_up) context.join_tertiary_to_auxiliary();
    }
    fp8_linears.prepare_gelu_2112(fused_dense_gate_up ? dense_gate_up : dense_gate,
                                  fused_dense_gate_up ? dense_gate_up + kDense : dense_up,
                                  gate_scale, up_scale,
                                  dense_product, rows,
                                  context.auxiliary_stream(), fused_dense_gate_up);
    Fp8OutputScale dense_output_scale{};
    const bool dense_output_launched = fused_projection_norms
        ? fp8_linears.launch_unscaled(
              w.down, kHidden, kDense, dense_product,
              dense_output, rows, context.auxiliary_stream(), handle,
              dense_output_scale, 1, true)
        : fp8_linears.launch(
              w.down, kHidden, kDense, dense_product,
              dense_output, rows, context.auxiliary_stream(), handle, true, 1,
              true);
    if (!dense_output_launched)
      throw std::runtime_error("verifier batch FP8 dense output unavailable");
    if (fused_expert_finalize)
      experts[layer]->launch_batch_unreduced(
          expert_pre, top_ids, rows, stream);
    else
      experts[layer]->launch_batch(expert_pre, top_ids, top_weights,
                                   expert_output, rows, stream);
    context.end_auxiliary();
    if (fused_expert_finalize)
      finalize_feedforward_routes_2816_kernel<<<
          rows, final_norm_threads, 0, stream>>>(
          dense_output, w.dense_post,
          static_cast<const __nv_bfloat16*>(experts[layer]->routed_output()),
          top_weights, experts[layer]->inverse_routes(), w.expert_post,
          w.combined_post, residual, w.layer_scalar, next, rows,
          fused_projection_norms ? dense_output_scale.activation_rows : nullptr,
          dense_output_scale.weight);
    else
      finalize_feedforward_2816_kernel<<<
          rows, final_norm_threads, 0, stream>>>(
          dense_output, w.dense_post, expert_output, w.expert_post,
          w.combined_post, residual, w.layer_scalar,
          next, rows,
          fused_projection_norms ? dense_output_scale.activation_rows : nullptr,
          dense_output_scale.weight);
    std::swap(current, next);
  }
  rmsnorm_2816_kernel<<<rows, 256, 0, stream>>>(
      current, final_norm,
      normalized, rows, 1e-6F);
  if (nvfp4_vocabulary)
    nvfp4_vocabulary->launch_batch(normalized, selected, rows, stream);
  else
    vocabulary.launch_batch(normalized, selected, rows, stream);
  commit_candidate_kv_batch_kernel<<<dim3(sessions, 30), 256, 0, stream>>>(
      candidate_views, live_views, contexts_device, drafted_tokens_device,
      selected, current, static_cast<__nv_bfloat16*>(continuation_states_device),
      results_device, sessions, drafts);
  check(cudaGetLastError(), "target verifier batch launch");
  };
  const std::uint64_t graph_key = 0x4241544300000000ULL ^
      experimental_expert_graph_key() ^
      (fused_attention_unpack ? 0x80000000000ULL : 0ULL) ^
      (fast_gemm_softmax ? 0x40000000000ULL : 0ULL) ^
      (tiled_attention ? 0x10000000000ULL : 0ULL) ^
      (tiled_global ? 0x20000000000ULL : 0ULL) ^
      (static_cast<std::uint64_t>(sessions) << 48) ^
      (static_cast<std::uint64_t>(drafts) << 44) ^
      (scratch.generation() * 0xd6e8feb86659fd93ULL) ^
      (static_cast<std::uint64_t>(maximum_context) * 0x9e3779b1ULL) ^
      (reinterpret_cast<std::uintptr_t>(continuation_states_device) >> 8);
  if (graph_decode && context.has_graph(graph_key)) {
    context.launch_graph(graph_key);
  } else {
    enqueue_compute();
    if (graph_decode) {
      check(cudaStreamSynchronize(stream),
            "synchronize verifier before graph capture");
      context.capture_graph(graph_key, enqueue_compute);
    }
  }
  auto* host_results = reinterpret_cast<DeviceMtpResult*>(
      host_staging + kResultOffset);
  check(cudaMemcpyAsync(host_results, results_device,
                        sessions * sizeof(DeviceMtpResult),
                        cudaMemcpyDeviceToHost, stream),
        "copy target verifier batch results");
  check(cudaStreamSynchronize(stream), "synchronize target verifier batch");
  TargetVerifierBatchResult result;
  result.sessions.resize(sessions);
  for (int session = 0; session < sessions; ++session) {
    const auto& source = host_results[session];
    auto& destination = result.sessions[session];
    std::copy(std::begin(source.target_tokens), std::end(source.target_tokens),
              destination.selected_tokens.begin());
    std::copy(std::begin(source.draft_tokens), std::end(source.draft_tokens),
              destination.draft_tokens.begin());
    std::copy(std::begin(source.output_tokens), std::end(source.output_tokens),
              destination.output_tokens.begin());
    destination.output_count = source.output_count;
    destination.matched_drafts = source.matched_drafts;
  }
  return result;
}
