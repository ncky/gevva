// Private implementation fragment; included only by src/gpu.cu.
TargetPrefillBenchmark benchmark_target_prefill(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    KvCache& cache, std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images, int layer_count, bool benchmark,
    std::span<Nvfp4ExpertRunner* const> persistent_experts,
    DeviceScratchPool* scratch, CompressedVocabRunner* persistent_vocab,
    GpuExecutionContext* context, void* device_final_state,
    int prefix_tokens, Fp8LinearRunner* fp8_linears,
    const void* device_projected_images) {
  constexpr int kHidden = 2816, kDense = 2112, kExperts = 128, kTop = 8;
  const int tokens = static_cast<int>(input_ids.size());
  const int total_tokens = prefix_tokens + tokens;
  if (tokens < 1 || tokens > 4608 || prefix_tokens < 0 || total_tokens < 2 ||
      layer_count < 1 || layer_count > 30 ||
      multimodal_types.size() != input_ids.size() ||
      total_tokens > cache.maximum_context())
    throw std::runtime_error("invalid target prefill geometry");
  const bool fused_expert_finalize =
      std::getenv("G4_DISABLE_PREFILL_FUSED_EXPERT_FINALIZE") == nullptr;
  cache.ensure_context(total_tokens, context ? context->stream() : nullptr);
  std::vector<int> image_rows(tokens, -1);
  std::vector<std::int32_t> block_ids(tokens, -1);
  int image_tokens = 0, block = -1;
  bool previous_image = false;
  for (int token = 0; token < tokens; ++token) {
    const bool image = multimodal_types[token] == 1;
    if (multimodal_types[token] > 1)
      throw std::runtime_error("unsupported multimodal type in prefill");
    if (image && !previous_image) ++block;
    if (image) {
      image_rows[token] = image_tokens++;
      block_ids[token] = block;
    }
    previous_image = image;
  }
  if ((!device_projected_images &&
       projected_images.size() != static_cast<std::size_t>(image_tokens) * kHidden) ||
      (device_projected_images && !projected_images.empty()))
    throw std::runtime_error("prefill image features do not match slots");
  const auto& model = weights.model();
  const auto embedding = weights.embedding();
  std::vector<__nv_bfloat16> host_images(projected_images.size());
  for (std::size_t i = 0; i < projected_images.size(); ++i)
    host_images[i] = __float2bfloat16_rn(projected_images[i]);

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
    return reinterpret_cast<__nv_bfloat16*>(allocate_bytes(elements * 2, name));
  };
  auto* hidden_a = allocate(static_cast<std::size_t>(tokens) * kHidden,
                            "cudaMalloc(prefill hidden A)");
  auto* hidden_b = allocate(static_cast<std::size_t>(tokens) * kHidden,
                            "cudaMalloc(prefill hidden B)");
  auto* normalized = allocate(static_cast<std::size_t>(tokens) * kHidden,
                              "cudaMalloc(prefill normalized)");
  auto* q = allocate(static_cast<std::size_t>(tokens) * 8192,
                     "cudaMalloc(prefill Q)");
  auto* k = allocate(static_cast<std::size_t>(tokens) * 2048,
                     "cudaMalloc(prefill K)");
  auto* v = allocate(static_cast<std::size_t>(tokens) * 2048,
                     "cudaMalloc(prefill V)");
  const int sliding_key_tokens = std::min(prefix_tokens, 1024) + tokens;
  auto* combined_k = prefix_tokens
      ? allocate(static_cast<std::size_t>(sliding_key_tokens) * 2048,
                 "cudaMalloc(prefill combined K)")
      : nullptr;
  auto* combined_v = prefix_tokens
      ? allocate(static_cast<std::size_t>(sliding_key_tokens) * 2048,
                 "cudaMalloc(prefill combined V)")
      : nullptr;
  const int attention_query_tile = total_tokens > 4608 ? 64 : tokens;
  auto* attended = allocate(static_cast<std::size_t>(tokens) * 8192,
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
  auto* expert_output = fused_expert_finalize
      ? nullptr
      : allocate(static_cast<std::size_t>(tokens) * kHidden,
                 "cudaMalloc(prefill expert output)");
  auto* scores = allocate(static_cast<std::size_t>(16) * attention_query_tile *
                              std::max(total_tokens, sliding_key_tokens),
                          "cudaMalloc(prefill scores)");
  auto* final_normalized = allocate(kHidden, "cudaMalloc(prefill final norm)");
  auto* d_ids = static_cast<std::uint32_t*>(allocate_bytes(
      input_ids.size_bytes(), "cudaMalloc(prefill input ids)"));
  auto* d_image_rows = static_cast<int*>(allocate_bytes(
      image_rows.size() * 4, "cudaMalloc(prefill image rows)"));
  auto* d_block_ids = static_cast<int*>(allocate_bytes(
      block_ids.size() * 4, "cudaMalloc(prefill block ids)"));
  // Keep a valid sentinel allocation for text-only prompts. The merge kernel
  // never dereferences it when every image row is -1.
  auto* d_images = device_projected_images
      ? const_cast<__nv_bfloat16*>(
            static_cast<const __nv_bfloat16*>(device_projected_images))
      : static_cast<__nv_bfloat16*>(allocate_bytes(
            std::max<std::size_t>(host_images.size() * 2, 2),
            "cudaMalloc(prefill images)"));
  auto* top_ids = static_cast<int*>(allocate_bytes(
      static_cast<std::size_t>(tokens) * kTop * 4,
      "cudaMalloc(prefill top ids)"));
  auto* top_weights = static_cast<float*>(allocate_bytes(
      static_cast<std::size_t>(tokens) * kTop * 4,
      "cudaMalloc(prefill top weights)"));
  auto* selected_token = static_cast<int*>(allocate_bytes(
      sizeof(int), "cudaMalloc(prefill selected token)"));
  check(cudaMemcpy(d_ids, input_ids.data(), input_ids.size_bytes(), cudaMemcpyHostToDevice),
        "copy prefill input ids");
  check(cudaMemcpy(d_image_rows, image_rows.data(), image_rows.size() * 4,
                   cudaMemcpyHostToDevice), "copy prefill image rows");
  check(cudaMemcpy(d_block_ids, block_ids.data(), block_ids.size() * 4,
                   cudaMemcpyHostToDevice), "copy prefill block ids");
  if (!device_projected_images && !host_images.empty())
    check(cudaMemcpy(d_images, host_images.data(), host_images.size() * 2,
                     cudaMemcpyHostToDevice), "copy prefill images");

  cudaStream_t stream{}; cublasHandle_t handle{};
  const bool owns_context = context == nullptr;
  if (context) {
    stream = context->stream();
    handle = static_cast<cublasHandle_t>(context->blas_handle());
  } else {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate(target prefill)");
    check(cublasCreate(&handle), "cublasCreate(target prefill)");
  }
  check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
        "cublasSetAtomicsMode(target prefill)");
  check(cublasSetStream(handle, stream), "cublasSetStream(target prefill)");
  const float alpha = 1.0F, beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int rows, int columns,
                     const __nv_bfloat16* source, __nv_bfloat16* destination,
                     cudaStream_t project_stream = nullptr,
                     int workspace_index = 0) {
    if (!project_stream) project_stream = stream;
    if (fp8_linears &&
        fp8_linears->launch(matrix, rows, columns, source, destination, tokens,
                            project_stream, handle, true, workspace_index))
      return;
    if (project_stream != stream)
      throw std::runtime_error(
          "concurrent prefill projection requires the FP8 sidecar");
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, columns,
                       &alpha, matrix, CUDA_R_16BF, columns, source,
                       CUDA_R_16BF, columns, &beta, destination, CUDA_R_16BF,
                       rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "target prefill projection");
  };
  if (!persistent_experts.empty() &&
      persistent_experts.size() != static_cast<std::size_t>(layer_count))
    throw std::runtime_error("persistent prefill requires one expert runner per layer");
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> owned_experts;
  std::vector<Nvfp4ExpertRunner*> expert_runners(layer_count);
  if (persistent_experts.empty()) {
    owned_experts.reserve(layer_count);
    for (int layer = 0; layer < layer_count; ++layer) {
      owned_experts.push_back(
          std::make_unique<Nvfp4ExpertRunner>(weights.experts(), layer,
                                             tokens));
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
  const auto* final_norm =
      static_cast<const __nv_bfloat16*>(weights.final_norm());
  __nv_bfloat16* final_hidden{};
  const bool prefill_concurrency = context && fp8_linears &&
      std::getenv("G4_DISABLE_PREFILL_CONCURRENCY") == nullptr;
  const bool prefill_combined_qkv =
      std::getenv("G4_DISABLE_PREFILL_COMBINED_QKV_TRANSFORM") == nullptr;
  const bool experimental_text_prefill = image_tokens == 0 &&
      std::getenv("G4_CUDNN_TEXT_PREFILL") != nullptr;
  const auto* rope_table = std::getenv("G4_PRECOMPUTED_ROPE")
      ? precomputed_target_rope() : nullptr;
  auto launch = [&] {
    compose_multimodal_embeddings_kernel<<<
        (tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding.data), d_ids,
        d_image_rows, d_images, hidden_a, tokens);
    auto* current = hidden_a;
    auto* next = hidden_b;
    for (int layer = 0; layer < layer_count; ++layer) {
      const bool global = layer % 6 == 5;
      const int q_width = global ? 8192 : 4096;
      const int kv_width = global ? 1024 : 2048;
      const auto w = target_layer_weights(weights, layer);
      if (fp8_linears)
        fp8_linears->prepare_rmsnorm_2816(
            current, w.input_norm, normalized, tokens, stream);
      else
        rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
            current, w.input_norm, normalized, tokens, 1e-6F);
      const bool concurrent_qkv = prefill_concurrency;
      if (concurrent_qkv) {
        context->begin_auxiliary();
        if (!global) context->begin_tertiary();
      }
      Fp8OutputScale q_scale{}, k_scale{}, v_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.q, q_width, kHidden, normalized, q,
              tokens, stream, handle, q_scale))
        project(w.q, q_width, kHidden, normalized, q);
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.k, kv_width, kHidden, normalized, k,
              tokens, concurrent_qkv ? context->auxiliary_stream() : stream,
              handle, k_scale, concurrent_qkv ? 1 : 0))
        project(w.k, kv_width, kHidden, normalized, k);
      if (!global &&
          (!fp8_linears || !fp8_linears->launch_unscaled(
              w.v, kv_width, kHidden, normalized, v,
              tokens, concurrent_qkv ? context->tertiary_stream() : stream,
              handle, v_scale, concurrent_qkv ? 2 : 0)))
        project(w.v, kv_width, kHidden, normalized, v);
      if (concurrent_qkv) {
        context->end_auxiliary();
        if (!global) context->end_tertiary();
      }
      const bool combined_qkv_transform = prefill_combined_qkv;
      if (combined_qkv_transform) {
        if (global)
          transform_qkv_ragged_kernel<512, 2, true>
              <<<tokens * 18, 512, 0, stream>>>(
                  q, k, v, w.q_norm,
                  w.k_norm, tokens, nullptr, tokens,
                  q_scale.activation_rows, q_scale.weight,
                  k_scale.activation_rows, k_scale.weight,
                  v_scale.activation_rows, v_scale.weight, prefix_tokens, rope_table);
        else
          transform_qkv_ragged_kernel<256, 8, false>
              <<<tokens * 24, 256, 0, stream>>>(
                  q, k, v, w.q_norm,
                  w.k_norm, tokens, nullptr, tokens,
                  q_scale.activation_rows, q_scale.weight,
                  k_scale.activation_rows, k_scale.weight,
                  v_scale.activation_rows, v_scale.weight, prefix_tokens, rope_table);
      } else {
        if (q_scale.activation_rows) {
          const int row_width = q_width + kv_width + (global ? 0 : kv_width);
          scale_fp8_qkv_output_kernel<<<
              (tokens * row_width + 255) / 256, 256, 0, stream>>>(
              q, q_width, q_scale.weight, k, kv_width, k_scale.weight,
              global ? nullptr : v, v_scale.weight, q_scale.activation_rows,
              tokens);
        }
        transform_qkv_batch(global, q, k, v, w.q_norm,
                            w.k_norm, prefix_tokens, tokens,
                            stream);
      }
      const auto& layer_cache = cache.layer(layer);
      const int plane = global ? 2 * 512 : 8 * 256;
      const auto* attention_k = k;
      const auto* attention_v = v;
      int attention_key_tokens = tokens;
      int key_position_base = prefix_tokens;
      if (global) {
        // Global KV is virtually contiguous and physically committed on
        // demand, so append first and attend directly without a prefix copy.
        append_qkv_batch(true, k, v, prefix_tokens, tokens,
                         layer_cache.keys, layer_cache.values, stream);
        attention_k = reinterpret_cast<const __nv_bfloat16*>(layer_cache.keys);
        attention_v = reinterpret_cast<const __nv_bfloat16*>(layer_cache.values);
        attention_key_tokens = total_tokens;
        key_position_base = 0;
      } else if (prefix_tokens) {
        // Materialize only the chronological sliding history. The physical
        // cache is a ring and may wrap at an arbitrary absolute position.
        const int past_tokens = std::min(prefix_tokens, 1024);
        const int past_begin = prefix_tokens - past_tokens;
        const int first_slot = past_begin & 1023;
        const int first_tokens = std::min(past_tokens, 1024 - first_slot);
        auto copy_past = [&](const void* source, __nv_bfloat16* destination,
                             const char* operation) {
          const auto* bytes = static_cast<const std::byte*>(source);
          if (first_tokens)
            check(cudaMemcpyAsync(
                      destination,
                      bytes + static_cast<std::size_t>(first_slot) * plane * 2,
                      static_cast<std::size_t>(first_tokens) * plane * 2,
                      cudaMemcpyDeviceToDevice, stream), operation);
          if (past_tokens > first_tokens)
            check(cudaMemcpyAsync(
                      destination + static_cast<std::size_t>(first_tokens) * plane,
                      bytes,
                      static_cast<std::size_t>(past_tokens - first_tokens) *
                          plane * 2,
                      cudaMemcpyDeviceToDevice, stream), operation);
        };
        copy_past(layer_cache.keys, combined_k,
                  "copy chronological sliding prefill keys");
        copy_past(layer_cache.values, combined_v,
                  "copy chronological sliding prefill values");
        check(cudaMemcpyAsync(
                  combined_k + static_cast<std::size_t>(past_tokens) * plane,
                  k, static_cast<std::size_t>(tokens) * plane * 2,
                  cudaMemcpyDeviceToDevice, stream),
              "append sliding suffix prefill keys");
        check(cudaMemcpyAsync(
                  combined_v + static_cast<std::size_t>(past_tokens) * plane,
                  v, static_cast<std::size_t>(tokens) * plane * 2,
                  cudaMemcpyDeviceToDevice, stream),
              "append sliding suffix prefill values");
        attention_k = combined_k;
        attention_v = combined_v;
        attention_key_tokens = past_tokens + tokens;
        key_position_base = past_begin;
      }
      if (!global)
        append_qkv_batch(false, k, v, prefix_tokens, tokens,
                         layer_cache.keys, layer_cache.values, stream);
      const int layer_query_tile = experimental_text_prefill && !global
          ? tokens : attention_query_tile;
      for (int first = 0; first < tokens; first += layer_query_tile) {
        const int count = std::min(tokens - first, layer_query_tile);
        launch_prefill_attention_impl(
            handle, global,
            q + static_cast<std::size_t>(first) * q_width,
            attention_k, attention_v, scores, d_block_ids,
            attended + static_cast<std::size_t>(first) * q_width,
            packed_attended, count, attention_key_tokens, prefix_tokens + first,
            key_position_base, prefix_tokens, tokens, stream,
            experimental_text_prefill);
      }
      Fp8OutputScale attention_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.o, kHidden, q_width, attended,
              attention, tokens, stream, handle, attention_scale))
        project(w.o, kHidden, q_width, attended,
                attention);
      rmsnorm_add_2816_kernel<<<tokens, 256, 0, stream>>>(
          attention, w.post_attention, current, residual,
          tokens, attention_scale.activation_rows, attention_scale.weight);
      if (fp8_linears)
        fp8_linears->prepare_dual_rmsnorm_router_2816(
            residual, w.dense_pre, w.expert_pre, w.router_scale,
            dense_pre, expert_pre, router_input, tokens, stream);
      else
        dual_rmsnorm_router_2816_kernel<<<tokens, 256, 0, stream>>>(
            residual, w.dense_pre, w.expert_pre, w.router_scale,
            dense_pre, expert_pre, router_input, tokens);
      Fp8OutputScale router_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.router, kExperts, kHidden, router_input,
              router_logits, tokens, stream, handle, router_scale))
        project(w.router, kExperts, kHidden, router_input,
                router_logits);
      route_128_top8_kernel<<<tokens, kExperts, 0, stream>>>(
          router_logits, w.expert_scale, top_weights, top_ids,
          tokens, router_scale.activation_rows, router_scale.weight);
      // The routed experts and dense branch are independent after routing.
      // Keep the much longer expert work on the primary stream while the
      // three dense projections execute on the auxiliary stream.
      const bool concurrent_feedforward = prefill_concurrency;
      if (concurrent_feedforward) context->begin_auxiliary();
      const cudaStream_t dense_stream = concurrent_feedforward
          ? context->auxiliary_stream() : stream;
      const int dense_workspace = concurrent_feedforward ? 1 : 0;
      if (fp8_linears)
        fp8_linears->use_secondary_activation_2816(
            dense_pre, tokens, dense_stream);
      Fp8OutputScale gate_scale{}, up_scale{}, down_scale{};
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.gate, kDense, kHidden, dense_pre,
              dense_gate, tokens, dense_stream, handle, gate_scale,
              dense_workspace))
        project(w.gate, kDense, kHidden, dense_pre,
                dense_gate, dense_stream, dense_workspace);
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.up, kDense, kHidden, dense_pre, dense_up,
              tokens, dense_stream, handle, up_scale, dense_workspace))
        project(w.up, kDense, kHidden, dense_pre, dense_up,
                dense_stream, dense_workspace);
      if (fp8_linears)
        fp8_linears->prepare_gelu_2112(
            dense_gate, dense_up, gate_scale, up_scale, dense_product, tokens,
            dense_stream);
      else
        gelu_tanh_multiply_kernel<<<
            (tokens * kDense + 255) / 256, 256, 0, dense_stream>>>(
            dense_gate, dense_up, dense_product, tokens * kDense);
      if (!fp8_linears || !fp8_linears->launch_unscaled(
              w.down, kHidden, kDense, dense_product,
              dense_output, tokens, dense_stream, handle, down_scale,
              dense_workspace))
        project(w.down, kHidden, kDense, dense_product,
                dense_output, dense_stream, dense_workspace);
      // Compact per-expert scale storage makes one route sort and one W13/W2
      // pair practical for the complete 4608-token prefill tile.
      if (fused_expert_finalize)
        expert_runners[layer]->launch_batch_unreduced(
            expert_pre, top_ids, tokens, stream);
      else
        expert_runners[layer]->launch_batch(
            expert_pre, top_ids, top_weights, expert_output, tokens, stream);
      if (concurrent_feedforward) context->end_auxiliary();
      if (fused_expert_finalize)
        finalize_feedforward_routes_2816_kernel<<<tokens, 256, 0, stream>>>(
            dense_output, w.dense_post,
            static_cast<const __nv_bfloat16*>(
                expert_runners[layer]->routed_output()),
            top_weights, expert_runners[layer]->inverse_routes(),
            w.expert_post, w.combined_post, residual, w.layer_scalar, next,
            tokens, down_scale.activation_rows, down_scale.weight);
      else
        finalize_feedforward_2816_kernel<<<tokens, 256, 0, stream>>>(
            dense_output, w.dense_post, expert_output, w.expert_post,
            w.combined_post, residual, w.layer_scalar,
            next, tokens, down_scale.activation_rows, down_scale.weight);
      std::swap(current, next);
    }
    final_hidden = current;
    rmsnorm_2816_kernel<<<1, 256, 0, stream>>>(
        current + static_cast<std::size_t>(tokens - 1) * kHidden,
        final_norm, final_normalized, 1, 1e-6F);
    vocabulary->launch(final_normalized, selected_token, stream);
  };
  std::vector<__nv_bfloat16> initial_last;
  float microseconds{};
  if (benchmark) {
    launch();
    check(cudaGetLastError(), "launch target prefill");
    check(cudaStreamSynchronize(stream), "synchronize target prefill");
    // Compare steady-state executions; the first cuBLAS call may populate its
    // heuristic cache and legitimately choose a different reduction kernel.
    launch();
    check(cudaStreamSynchronize(stream), "synchronize target prefill warmup");
    initial_last.resize(kHidden);
    check(cudaMemcpy(initial_last.data(),
                     final_hidden +
                         static_cast<std::size_t>(tokens - 1) * kHidden,
                     kHidden * 2, cudaMemcpyDeviceToHost),
          "copy initial prefill last hidden");
    microseconds = isolated_event_benchmark(
        stream, [&] { launch(); }, 1, 3);
  } else {
    microseconds = isolated_event_benchmark(
        stream, [&] { launch(); }, 0, 1);
  }
  if (device_final_state)
    check(cudaMemcpyAsync(
              device_final_state,
              final_hidden + static_cast<std::size_t>(tokens - 1) * kHidden,
              kHidden * 2, cudaMemcpyDeviceToDevice, stream),
          "retain prefill continuation state");
  int host_token{};
  if (scratch) {
    auto* pinned_token = static_cast<int*>(scratch->host_staging(sizeof(int)));
    check(cudaMemcpyAsync(pinned_token, selected_token, sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
          "copy prefill selected token");
    check(cudaStreamSynchronize(stream), "synchronize prefill selected token");
    host_token = *pinned_token;
  } else {
    check(cudaMemcpy(&host_token, selected_token, sizeof(int),
                     cudaMemcpyDeviceToHost),
          "copy prefill selected token");
  }
  const bool collect_host_state = benchmark || !device_final_state;
  std::vector<__nv_bfloat16> last;
  if (collect_host_state) {
    last.resize(kHidden);
    check(cudaMemcpy(last.data(),
                     final_hidden + static_cast<std::size_t>(tokens - 1) *
                                        kHidden,
                     kHidden * 2, cudaMemcpyDeviceToHost),
          "copy prefill last hidden");
  }
  if (benchmark &&
      std::memcmp(initial_last.data(), last.data(), kHidden * 2) != 0) {
    float first_sum = 0.0F, last_sum = 0.0F, maximum_difference = 0.0F;
    int different = 0;
    for (int col = 0; col < kHidden; ++col) {
      const float first_value = __bfloat162float(initial_last[col]);
      const float last_value = __bfloat162float(last[col]);
      first_sum += first_value;
      last_sum += last_value;
      maximum_difference =
          std::max(maximum_difference, std::abs(first_value - last_value));
      different += std::memcmp(&initial_last[col], &last[col],
                               sizeof(__nv_bfloat16)) != 0;
    }
    std::ostringstream message;
    message << "target prefill is not repeatable within one session: first="
            << first_sum << " last=" << last_sum << " max_diff="
            << maximum_difference << " different=" << different;
    throw std::runtime_error(message.str());
  }
  float checksum = 0.0F;
  std::vector<float> final_state;
  if (collect_host_state) {
    final_state.resize(kHidden);
    for (int col = 0; col < kHidden; ++col) {
      final_state[col] = __bfloat162float(last[col]);
      checksum += final_state[col];
    }
  }
  if (owns_context) {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }
  for (void* p : allocations) cudaFree(p);
  return {microseconds, checksum, host_token, total_tokens,
          std::move(final_state)};
}

__global__ void gather_prefill_batch_last_rows_kernel(
    const __nv_bfloat16* hidden, const int* last_rows, const int* output_rows,
    __nv_bfloat16* gathered, __nv_bfloat16* retained, int sessions) {
  const int session = blockIdx.x;
  if (session >= sessions) return;
  const auto* source = hidden +
      static_cast<std::size_t>(last_rows[session]) * 2816;
  for (int col = threadIdx.x; col < 2816; col += blockDim.x) {
    const auto value = source[col];
    gathered[static_cast<std::size_t>(session) * 2816 + col] = value;
    if (retained)
      retained[static_cast<std::size_t>(output_rows[session]) * 2816 + col] =
          value;
  }
}

std::vector<TargetPrefillBenchmark> benchmark_target_prefill_batch(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    std::span<const TargetPrefillBatchItem> items, int layer_count,
    std::span<Nvfp4ExpertRunner* const> persistent_experts,
    DeviceScratchPool* scratch, CompressedVocabRunner* persistent_vocab,
    GpuExecutionContext* context, void* device_final_states,
    Fp8LinearRunner* fp8_linears, Nvfp4VocabRunner* nvfp4_vocab) {
  constexpr int kHidden = 2816, kDense = 2112, kExperts = 128, kTop = 8;
  const int sessions = static_cast<int>(items.size());
  if (sessions < 1 || sessions > 8 || layer_count < 1 || layer_count > 30 ||
      !context || !scratch || !persistent_vocab || !fp8_linears ||
      persistent_experts.size() != static_cast<std::size_t>(layer_count))
    throw std::runtime_error("invalid target batch prefill resources");

  std::vector<int> row_offsets(sessions + 1), last_rows(sessions),
      output_rows(sessions), context_lengths(sessions);
  std::vector<std::uint32_t> flat_ids;
  std::vector<std::uint8_t> flat_types;
  std::vector<int> image_rows;
  std::vector<std::int32_t> block_ids;
  std::vector<__nv_bfloat16> host_images;
  struct DeviceImageCopy {
    const void* source{};
    std::size_t destination_value{};
    std::size_t values{};
  };
  std::vector<DeviceImageCopy> device_image_copies;
  std::size_t total_image_values = 0;
  bool has_host_image_values = false;
  int maximum_query_tile = 1, maximum_key_tokens = 1;
  int maximum_sliding_key_tokens = 1;
  for (int session = 0; session < sessions; ++session) {
    const auto& item = items[session];
    const int tokens = static_cast<int>(item.input_ids.size());
    const int total = item.prefix_tokens + tokens;
    if (!item.cache || tokens < 1 || item.multimodal_types.size() !=
            item.input_ids.size() || item.prefix_tokens < 0 || total < 2 ||
        total > item.cache->maximum_context())
      throw std::runtime_error("invalid target batch prefill item");
    item.cache->ensure_context(total, context->stream());
    row_offsets[session] = static_cast<int>(flat_ids.size());
    flat_ids.insert(flat_ids.end(), item.input_ids.begin(), item.input_ids.end());
    flat_types.insert(flat_types.end(), item.multimodal_types.begin(),
                      item.multimodal_types.end());
    int block = -1;
    bool previous_image = false;
    std::size_t local_image = 0;
    const int image_base = static_cast<int>(total_image_values / kHidden);
    for (int token = 0; token < tokens; ++token) {
      const bool image = item.multimodal_types[token] == 1;
      if (item.multimodal_types[token] > 1)
        throw std::runtime_error("unsupported multimodal type in batch prefill");
      if (image && !previous_image) ++block;
      image_rows.push_back(image ? image_base + static_cast<int>(local_image++)
                                 : -1);
      block_ids.push_back(image ? block : -1);
      previous_image = image;
    }
    if ((!item.device_projected_images &&
         item.projected_images.size() != local_image * kHidden) ||
        (item.device_projected_images && !item.projected_images.empty()))
      throw std::runtime_error("batch prefill image features do not match slots");
    const std::size_t local_image_values = local_image * kHidden;
    if (item.device_projected_images) {
      device_image_copies.push_back(
          {item.device_projected_images, total_image_values, local_image_values});
      host_images.resize(total_image_values + local_image_values);
    } else {
      has_host_image_values |= local_image_values != 0;
      host_images.reserve(total_image_values + local_image_values);
      for (const float value : item.projected_images)
        host_images.push_back(__float2bfloat16_rn(value));
    }
    total_image_values += local_image_values;
    last_rows[session] = row_offsets[session] + tokens - 1;
    if (item.final_state_row < 0 || item.final_state_row >= 8)
      throw std::runtime_error("invalid batch prefill final-state row");
    output_rows[session] = item.final_state_row;
    context_lengths[session] = item.prefix_tokens;
    maximum_query_tile = std::max(
        maximum_query_tile, total > 4608 ? std::min(tokens, 64) : tokens);
    maximum_key_tokens = std::max(
        maximum_key_tokens,
        std::max(total, std::min(item.prefix_tokens, 1024) + tokens));
    maximum_sliding_key_tokens = std::max(
        maximum_sliding_key_tokens,
        std::min(item.prefix_tokens, 1024) + tokens);
  }
  row_offsets[sessions] = static_cast<int>(flat_ids.size());
  const int rows = row_offsets.back();
  if (rows < 1 || rows > 4608)
    throw std::runtime_error("target batch prefill row budget exceeds 4608");
  const int uniform_tokens = row_offsets[1] - row_offsets[0];
  bool uniform_geometry = true;
  for (int session = 1; session < sessions; ++session)
    uniform_geometry &=
        row_offsets[session + 1] - row_offsets[session] == uniform_tokens;
  const int uniform_prefix = items[0].prefix_tokens;
  const bool experimental_text_prefill =
      std::getenv("G4_CUDNN_TEXT_PREFILL") != nullptr &&
      std::all_of(block_ids.begin(), block_ids.end(), [](int id) { return id < 0; });
  const auto* rope_table = std::getenv("G4_PRECOMPUTED_ROPE")
      ? precomputed_target_rope() : nullptr;
  const bool uniform_qkv_transform = uniform_geometry &&
      std::getenv("G4_DISABLE_UNIFORM_BATCH_QKV_TRANSFORM") == nullptr;
  bool uniform_attention = uniform_geometry &&
      uniform_prefix + uniform_tokens <= 4608 &&
      std::getenv("G4_DISABLE_UNIFORM_BATCH_PREFILL_ATTENTION") == nullptr;
  const bool fused_expert_finalize =
      std::getenv("G4_DISABLE_PREFILL_FUSED_EXPERT_FINALIZE") == nullptr;
  for (int session = 1; session < sessions; ++session)
    uniform_attention &= items[session].prefix_tokens == uniform_prefix;

  scratch->reset();
  auto allocate_bytes = [&](std::size_t bytes) { return scratch->allocate(bytes); };
  auto allocate = [&](std::size_t elements) {
    return static_cast<__nv_bfloat16*>(allocate_bytes(elements * 2));
  };
  auto* hidden_a = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* hidden_b = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* normalized = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* q = allocate(static_cast<std::size_t>(rows) * 8192);
  auto* k = allocate(static_cast<std::size_t>(rows) * 2048);
  auto* v = allocate(static_cast<std::size_t>(rows) * 2048);
  auto* combined_k = allocate(
      static_cast<std::size_t>(uniform_attention ? sessions : 1) *
      maximum_sliding_key_tokens * 2048);
  auto* combined_v = allocate(
      static_cast<std::size_t>(uniform_attention ? sessions : 1) *
      maximum_sliding_key_tokens * 2048);
  auto* attended = allocate(static_cast<std::size_t>(rows) * 8192);
  auto* attention = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* residual = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* dense_pre = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* expert_pre = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* router_input = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* router_logits = allocate(static_cast<std::size_t>(rows) * kExperts);
  auto* dense_gate = allocate(static_cast<std::size_t>(rows) * kDense);
  auto* dense_up = allocate(static_cast<std::size_t>(rows) * kDense);
  auto* dense_product = allocate(static_cast<std::size_t>(rows) * kDense);
  auto* dense_output = allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* expert_output = fused_expert_finalize
      ? nullptr : allocate(static_cast<std::size_t>(rows) * kHidden);
  auto* scores = allocate(static_cast<std::size_t>(
                              uniform_attention ? sessions : 1) *
                          16 * maximum_query_tile * maximum_key_tokens);
  auto* final_rows = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* final_normalized = allocate(static_cast<std::size_t>(sessions) * kHidden);
  auto* attention_pointers = allocate(
      (static_cast<std::size_t>(uniform_attention ? sessions : 1) * 96 *
           sizeof(void*) +
       sizeof(__nv_bfloat16) - 1) /
      sizeof(__nv_bfloat16));
  std::size_t metadata_bytes = 0;
  auto reserve_metadata = [&](std::size_t bytes, std::size_t alignment) {
    metadata_bytes = (metadata_bytes + alignment - 1) & ~(alignment - 1);
    const std::size_t offset = metadata_bytes;
    metadata_bytes += bytes;
    return offset;
  };
  const std::size_t ids_bytes = flat_ids.size() * sizeof(std::uint32_t);
  const std::size_t ids_offset = reserve_metadata(ids_bytes, 4);
  const std::size_t image_rows_offset =
      reserve_metadata(image_rows.size() * sizeof(int), 4);
  const std::size_t block_ids_offset =
      reserve_metadata(block_ids.size() * sizeof(std::int32_t), 4);
  const std::size_t last_rows_offset =
      reserve_metadata(last_rows.size() * sizeof(int), 4);
  const std::size_t output_rows_offset =
      reserve_metadata(output_rows.size() * sizeof(int), 4);
  const std::size_t context_lengths_offset =
      reserve_metadata(context_lengths.size() * sizeof(int), 4);
  const std::size_t cache_pointer_count =
      static_cast<std::size_t>(layer_count) * sessions;
  const std::size_t cache_keys_offset = reserve_metadata(
      cache_pointer_count * sizeof(__nv_bfloat16*), alignof(__nv_bfloat16*));
  const std::size_t cache_values_offset = reserve_metadata(
      cache_pointer_count * sizeof(__nv_bfloat16*), alignof(__nv_bfloat16*));
  auto* d_metadata = static_cast<std::byte*>(allocate_bytes(metadata_bytes));
  auto* d_ids = reinterpret_cast<std::uint32_t*>(d_metadata + ids_offset);
  auto* d_image_rows = reinterpret_cast<int*>(d_metadata + image_rows_offset);
  auto* d_block_ids =
      reinterpret_cast<std::int32_t*>(d_metadata + block_ids_offset);
  auto* d_last_rows = reinterpret_cast<int*>(d_metadata + last_rows_offset);
  auto* d_output_rows = reinterpret_cast<int*>(d_metadata + output_rows_offset);
  auto* d_context_lengths =
      reinterpret_cast<int*>(d_metadata + context_lengths_offset);
  auto* d_cache_keys = reinterpret_cast<__nv_bfloat16**>(
      d_metadata + cache_keys_offset);
  auto* d_cache_values = reinterpret_cast<__nv_bfloat16**>(
      d_metadata + cache_values_offset);
  auto* d_images = static_cast<__nv_bfloat16*>(allocate_bytes(
      std::max<std::size_t>(total_image_values * 2, 2)));
  auto* top_ids = static_cast<int*>(allocate_bytes(
      static_cast<std::size_t>(rows) * kTop * sizeof(int)));
  auto* top_weights = static_cast<float*>(allocate_bytes(
      static_cast<std::size_t>(rows) * kTop * sizeof(float)));
  auto* selected_tokens = static_cast<int*>(allocate_bytes(
      static_cast<std::size_t>(sessions) * sizeof(int)));

  const auto stream = context->stream();
  auto handle = static_cast<cublasHandle_t>(context->blas_handle());
  auto* host_metadata =
      static_cast<std::byte*>(scratch->host_staging(metadata_bytes));
  std::memcpy(host_metadata + ids_offset, flat_ids.data(), ids_bytes);
  std::memcpy(host_metadata + image_rows_offset, image_rows.data(),
              image_rows.size() * sizeof(int));
  std::memcpy(host_metadata + block_ids_offset, block_ids.data(),
              block_ids.size() * sizeof(std::int32_t));
  std::memcpy(host_metadata + last_rows_offset, last_rows.data(),
              last_rows.size() * sizeof(int));
  std::memcpy(host_metadata + output_rows_offset, output_rows.data(),
              output_rows.size() * sizeof(int));
  std::memcpy(host_metadata + context_lengths_offset, context_lengths.data(),
              context_lengths.size() * sizeof(int));
  auto* host_cache_keys = reinterpret_cast<__nv_bfloat16**>(
      host_metadata + cache_keys_offset);
  auto* host_cache_values = reinterpret_cast<__nv_bfloat16**>(
      host_metadata + cache_values_offset);
  for (int layer = 0; layer < layer_count; ++layer)
    for (int session = 0; session < sessions; ++session) {
      const auto index = static_cast<std::size_t>(layer) * sessions + session;
      const auto& cache = items[session].cache->layer(layer);
      host_cache_keys[index] = reinterpret_cast<__nv_bfloat16*>(cache.keys);
      host_cache_values[index] = reinterpret_cast<__nv_bfloat16*>(cache.values);
    }
  check(cudaMemcpyAsync(d_metadata, host_metadata, metadata_bytes,
                        cudaMemcpyHostToDevice, stream),
        "copy batch prefill metadata");
  if (has_host_image_values && device_image_copies.empty())
    check(cudaMemcpyAsync(d_images, host_images.data(), host_images.size() * 2,
                          cudaMemcpyHostToDevice, stream),
        "copy batch prefill images");
  else {
    // Serving items are device-resident. Mixed host/device batches are not a
    // current public path and rejecting them avoids copying uninitialized gaps.
    if (has_host_image_values)
      throw std::runtime_error("mixed host/device batch image features unsupported");
    for (const auto& copy : device_image_copies)
      if (copy.values && cudaMemcpyAsync(
              d_images + copy.destination_value, copy.source, copy.values * 2,
              cudaMemcpyDeviceToDevice, context->stream()) != cudaSuccess)
        throw std::runtime_error("copy device batch prefill images failed");
  }
  check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
        "set batch prefill atomics mode");
  check(cublasSetStream(handle, stream), "set batch prefill BLAS stream");

  const auto& model = weights.model();
  const auto embedding = weights.embedding();
  const float alpha = 1.0F, beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int outputs, int inputs,
                     const __nv_bfloat16* source, __nv_bfloat16* destination,
                     cudaStream_t project_stream = nullptr,
                     int workspace_index = 0) {
    if (!project_stream) project_stream = stream;
    if (fp8_linears->launch(matrix, outputs, inputs, source, destination, rows,
                            project_stream, handle, true, workspace_index))
      return;
    if (project_stream != stream)
      throw std::runtime_error("concurrent batch prefill requires FP8 weights");
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, outputs, rows, inputs,
                       &alpha, matrix, CUDA_R_16BF, inputs, source,
                       CUDA_R_16BF, inputs, &beta, destination, CUDA_R_16BF,
                       outputs, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "target batch prefill projection");
  };

  const auto* final_norm =
      static_cast<const __nv_bfloat16*>(weights.final_norm());
  __nv_bfloat16* final_hidden{};
  auto launch = [&] {
    compose_multimodal_embeddings_kernel<<<
        (rows * kHidden + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(embedding.data), d_ids,
        d_image_rows, d_images, hidden_a, rows);
    auto* current = hidden_a;
    auto* next = hidden_b;
    for (int layer = 0; layer < layer_count; ++layer) {
      const bool global = layer % 6 == 5;
      const int q_width = global ? 8192 : 4096;
      const int kv_width = global ? 1024 : 2048;
      const auto w = target_layer_weights(weights, layer);
      fp8_linears->prepare_rmsnorm_2816(
          current, w.input_norm, normalized, rows, stream);
      context->begin_auxiliary();
      if (!global) context->begin_tertiary();
      Fp8OutputScale q_scale{}, k_scale{}, v_scale{};
      if (!fp8_linears->launch_unscaled(
              w.q, q_width, kHidden, normalized, q, rows, stream, handle,
              q_scale))
        project(w.q, q_width, kHidden, normalized, q);
      if (!fp8_linears->launch_unscaled(
              w.k, kv_width, kHidden, normalized, k, rows,
              context->auxiliary_stream(), handle, k_scale, 1))
        throw std::runtime_error("batch prefill K FP8 projection missing");
      if (!global && !fp8_linears->launch_unscaled(
              w.v, kv_width, kHidden, normalized, v, rows,
              context->tertiary_stream(), handle, v_scale, 2))
        throw std::runtime_error("batch prefill V FP8 projection missing");
      context->end_auxiliary();
      if (!global) context->end_tertiary();

      if (uniform_qkv_transform) {
        if (global)
          transform_qkv_ragged_kernel<512, 2, true>
              <<<rows * 18, 512, 0, stream>>>(
                  q, k, v, w.q_norm, w.k_norm, rows, d_context_lengths,
                  uniform_tokens, q_scale.activation_rows, q_scale.weight,
                  k_scale.activation_rows, k_scale.weight,
                  v_scale.activation_rows, v_scale.weight, -1, rope_table);
        else
          transform_qkv_ragged_kernel<256, 8, false>
              <<<rows * 24, 256, 0, stream>>>(
                  q, k, v, w.q_norm, w.k_norm, rows, d_context_lengths,
                  uniform_tokens, q_scale.activation_rows, q_scale.weight,
                  k_scale.activation_rows, k_scale.weight,
                  v_scale.activation_rows, v_scale.weight, -1, rope_table);
      } else {
        for (int session = 0; session < sessions; ++session) {
          const int row = row_offsets[session];
          const int tokens = row_offsets[session + 1] - row;
          const auto scale_offset = [row](const float* pointer) {
            return pointer ? pointer + row : nullptr;
          };
          if (global)
            transform_qkv_ragged_kernel<512, 2, true>
                <<<tokens * 18, 512, 0, stream>>>(
                    q + static_cast<std::size_t>(row) * q_width,
                    k + static_cast<std::size_t>(row) * kv_width,
                    v + static_cast<std::size_t>(row) * kv_width,
                    w.q_norm, w.k_norm, tokens, nullptr, tokens,
                    scale_offset(q_scale.activation_rows), q_scale.weight,
                    scale_offset(k_scale.activation_rows), k_scale.weight,
                    scale_offset(v_scale.activation_rows), v_scale.weight,
                    items[session].prefix_tokens, rope_table);
          else
            transform_qkv_ragged_kernel<256, 8, false>
                <<<tokens * 24, 256, 0, stream>>>(
                    q + static_cast<std::size_t>(row) * q_width,
                    k + static_cast<std::size_t>(row) * kv_width,
                    v + static_cast<std::size_t>(row) * kv_width,
                    w.q_norm, w.k_norm, tokens, nullptr, tokens,
                    scale_offset(q_scale.activation_rows), q_scale.weight,
                    scale_offset(k_scale.activation_rows), k_scale.weight,
                    scale_offset(v_scale.activation_rows), v_scale.weight,
                    items[session].prefix_tokens, rope_table);
        }
      }

      for (int session = 0; session < sessions; ++session) {
        const auto& item = items[session];
        const int row = row_offsets[session];
        const int tokens = row_offsets[session + 1] - row;
        const int total = item.prefix_tokens + tokens;
        const auto& layer_cache = item.cache->layer(layer);
        auto* session_q = q + static_cast<std::size_t>(row) * q_width;
        auto* session_k = k + static_cast<std::size_t>(row) * kv_width;
        auto* session_v = v + static_cast<std::size_t>(row) * kv_width;
        const __nv_bfloat16* attention_k = session_k;
        const __nv_bfloat16* attention_v = session_v;
        int key_tokens = tokens;
        int key_position_base = item.prefix_tokens;
        if (global) {
          if (!uniform_attention)
            append_qkv_batch(true, session_k, session_v, item.prefix_tokens,
                             tokens, layer_cache.keys, layer_cache.values,
                             stream);
          attention_k = reinterpret_cast<const __nv_bfloat16*>(layer_cache.keys);
          attention_v = reinterpret_cast<const __nv_bfloat16*>(layer_cache.values);
          key_tokens = total;
          key_position_base = 0;
        } else if (item.prefix_tokens) {
          const int plane = 8 * 256;
          auto* session_combined_k = combined_k +
              static_cast<std::size_t>(uniform_attention ? session : 0) *
                  maximum_sliding_key_tokens * plane;
          auto* session_combined_v = combined_v +
              static_cast<std::size_t>(uniform_attention ? session : 0) *
                  maximum_sliding_key_tokens * plane;
          const int past_tokens = std::min(item.prefix_tokens, 1024);
          const int past_begin = item.prefix_tokens - past_tokens;
          const int first_slot = past_begin & 1023;
          const int first_tokens = std::min(past_tokens, 1024 - first_slot);
          auto copy_past = [&](const void* source, __nv_bfloat16* destination) {
            const auto* bytes = static_cast<const std::byte*>(source);
            if (first_tokens)
              check(cudaMemcpyAsync(
                        destination,
                        bytes + static_cast<std::size_t>(first_slot) * plane * 2,
                        static_cast<std::size_t>(first_tokens) * plane * 2,
                        cudaMemcpyDeviceToDevice, stream),
                    "copy batch chronological sliding prefix");
            if (past_tokens > first_tokens)
              check(cudaMemcpyAsync(
                        destination + static_cast<std::size_t>(first_tokens) * plane,
                        bytes,
                        static_cast<std::size_t>(past_tokens - first_tokens) *
                            plane * 2,
                        cudaMemcpyDeviceToDevice, stream),
                    "copy wrapped batch sliding prefix");
          };
          copy_past(layer_cache.keys, session_combined_k);
          copy_past(layer_cache.values, session_combined_v);
          check(cudaMemcpyAsync(
                    session_combined_k +
                        static_cast<std::size_t>(past_tokens) * plane,
                    session_k, static_cast<std::size_t>(tokens) * plane * 2,
                    cudaMemcpyDeviceToDevice, stream),
                "append batch sliding suffix K");
          check(cudaMemcpyAsync(
                    session_combined_v +
                        static_cast<std::size_t>(past_tokens) * plane,
                    session_v, static_cast<std::size_t>(tokens) * plane * 2,
                    cudaMemcpyDeviceToDevice, stream),
                "append batch sliding suffix V");
          attention_k = session_combined_k;
          attention_v = session_combined_v;
          key_tokens = past_tokens + tokens;
          key_position_base = past_begin;
        }
        if (!global && !uniform_attention)
          append_qkv_batch(false, session_k, session_v, item.prefix_tokens,
                           tokens, layer_cache.keys, layer_cache.values, stream);
        if (!uniform_attention) {
          const int query_tile = total > 4608 && !(experimental_text_prefill && !global)
              ? 64 : tokens;
          for (int first = 0; first < tokens; first += query_tile) {
            const int count = std::min(tokens - first, query_tile);
            launch_prefill_attention_impl(
                handle, global,
                session_q + static_cast<std::size_t>(first) * q_width,
                attention_k, attention_v, scores,
                d_block_ids + row,
                attended + static_cast<std::size_t>(row + first) * q_width,
                attention_pointers, count, key_tokens,
                item.prefix_tokens + first, key_position_base,
                item.prefix_tokens, tokens, stream, experimental_text_prefill);
          }
        }
      }

      if (uniform_attention) {
        auto* layer_cache_keys =
            d_cache_keys + static_cast<std::size_t>(layer) * sessions;
        auto* layer_cache_values =
            d_cache_values + static_cast<std::size_t>(layer) * sessions;
        if (global) {
          constexpr int kPlane = 2 * 512;
          const int elements = uniform_tokens * kPlane;
          append_uniform_global_kv_kernel<<<
              dim3(std::min(65535, (elements + 255) / 256), sessions), 256, 0,
              stream>>>(k, v, layer_cache_keys,
                         layer_cache_values, uniform_prefix, uniform_tokens,
                         kPlane);
        } else {
          constexpr int kPlane = 8 * 256;
          const int source_begin = std::max(0, uniform_tokens - 1024);
          const int elements = (uniform_tokens - source_begin) * kPlane;
          append_uniform_sliding_duplicated_kv_kernel<<<
              dim3(std::min(65535, (elements + 255) / 256), sessions), 256, 0,
              stream>>>(k, v, layer_cache_keys, layer_cache_values,
                         uniform_prefix, source_begin, uniform_tokens, kPlane);
        }
        const int key_tokens = global
            ? uniform_prefix + uniform_tokens
            : std::min(uniform_prefix, 1024) + uniform_tokens;
        if (!global && experimental_text_prefill)
          launch_cudnn_text_prefill(q,
                                    uniform_prefix ? combined_k : k,
                                    uniform_prefix ? combined_v : v,
                                    attended, sessions, uniform_tokens, stream,
                                    key_tokens);
        else if (global)
          launch_uniform_batch_prefill_attention<512, 2, true>(
              handle, q, k, v, layer_cache_keys, layer_cache_values,
              combined_k, combined_v, scores, d_block_ids, attended,
              attention_pointers, sessions, uniform_tokens, key_tokens,
              uniform_prefix, maximum_sliding_key_tokens, stream);
        else
          launch_uniform_batch_prefill_attention<256, 8, false>(
              handle, q, k, v, layer_cache_keys, layer_cache_values,
              combined_k, combined_v, scores, d_block_ids, attended,
              attention_pointers, sessions, uniform_tokens, key_tokens,
              uniform_prefix, maximum_sliding_key_tokens, stream);
      }

      Fp8OutputScale attention_scale{};
      if (!fp8_linears->launch_unscaled(
              w.o, kHidden, q_width, attended, attention, rows, stream, handle,
              attention_scale))
        project(w.o, kHidden, q_width, attended, attention);
      rmsnorm_add_2816_kernel<<<rows, 256, 0, stream>>>(
          attention, w.post_attention, current, residual, rows,
          attention_scale.activation_rows, attention_scale.weight);
      fp8_linears->prepare_dual_rmsnorm_router_2816(
          residual, w.dense_pre, w.expert_pre, w.router_scale,
          dense_pre, expert_pre, router_input, rows, stream);
      Fp8OutputScale router_scale{};
      if (!fp8_linears->launch_unscaled(
              w.router, kExperts, kHidden, router_input, router_logits, rows,
              stream, handle, router_scale))
        project(w.router, kExperts, kHidden, router_input, router_logits);
      route_128_top8_kernel<<<rows, kExperts, 0, stream>>>(
          router_logits, w.expert_scale, top_weights, top_ids, rows,
          router_scale.activation_rows, router_scale.weight);
      context->begin_auxiliary();
      const auto dense_stream = context->auxiliary_stream();
      fp8_linears->use_secondary_activation_2816(
          dense_pre, rows, dense_stream);
      Fp8OutputScale gate_scale{}, up_scale{}, down_scale{};
      if (!fp8_linears->launch_unscaled(
              w.gate, kDense, kHidden, dense_pre, dense_gate, rows,
              dense_stream, handle, gate_scale, 1))
        throw std::runtime_error("batch prefill gate FP8 projection missing");
      if (!fp8_linears->launch_unscaled(
              w.up, kDense, kHidden, dense_pre, dense_up, rows,
              dense_stream, handle, up_scale, 1))
        throw std::runtime_error("batch prefill up FP8 projection missing");
      fp8_linears->prepare_gelu_2112(
          dense_gate, dense_up, gate_scale, up_scale, dense_product, rows,
          dense_stream);
      if (!fp8_linears->launch_unscaled(
              w.down, kHidden, kDense, dense_product, dense_output, rows,
              dense_stream, handle, down_scale, 1))
        throw std::runtime_error("batch prefill down FP8 projection missing");
      if (fused_expert_finalize)
        persistent_experts[layer]->launch_batch_unreduced(
            expert_pre, top_ids, rows, stream);
      else
        persistent_experts[layer]->launch_batch(
            expert_pre, top_ids, top_weights, expert_output, rows, stream);
      context->end_auxiliary();
      if (fused_expert_finalize)
        finalize_feedforward_routes_2816_kernel<<<rows, 256, 0, stream>>>(
            dense_output, w.dense_post,
            static_cast<const __nv_bfloat16*>(
                persistent_experts[layer]->routed_output()),
            top_weights, persistent_experts[layer]->inverse_routes(),
            w.expert_post, w.combined_post, residual, w.layer_scalar, next,
            rows, down_scale.activation_rows, down_scale.weight);
      else
        finalize_feedforward_2816_kernel<<<rows, 256, 0, stream>>>(
            dense_output, w.dense_post, expert_output, w.expert_post,
            w.combined_post, residual, w.layer_scalar, next, rows,
            down_scale.activation_rows, down_scale.weight);
      std::swap(current, next);
    }
    final_hidden = current;
    gather_prefill_batch_last_rows_kernel<<<sessions, 256, 0, stream>>>(
        current, d_last_rows, d_output_rows, final_rows,
        static_cast<__nv_bfloat16*>(device_final_states), sessions);
    rmsnorm_2816_kernel<<<sessions, 256, 0, stream>>>(
        final_rows, final_norm, final_normalized, sessions, 1e-6F);
    if (nvfp4_vocab)
      nvfp4_vocab->launch_batch(
          final_normalized, selected_tokens, sessions, stream);
    else
      persistent_vocab->launch_batch(
          final_normalized, selected_tokens, sessions, stream);
  };

  const float microseconds = isolated_event_benchmark(
      stream, [&] { launch(); }, 0, 1);
  auto* host_tokens = static_cast<int*>(
      scratch->host_staging(sessions * sizeof(int)));
  check(cudaMemcpyAsync(host_tokens, selected_tokens,
                   sessions * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "copy batch prefill selected tokens");
  check(cudaStreamSynchronize(stream),
        "synchronize batch prefill selected tokens");
  std::vector<__nv_bfloat16> host_last;
  if (!device_final_states) {
    host_last.resize(static_cast<std::size_t>(sessions) * kHidden);
    check(cudaMemcpy(host_last.data(), final_rows, host_last.size() * 2,
                     cudaMemcpyDeviceToHost),
          "copy batch prefill final rows");
  }
  std::vector<TargetPrefillBenchmark> results(sessions);
  for (int session = 0; session < sessions; ++session) {
    auto& result = results[session];
    result.microseconds = microseconds;
    result.selected_token = host_tokens[session];
    result.tokens = items[session].prefix_tokens +
                    static_cast<int>(items[session].input_ids.size());
    if (!device_final_states) {
      result.final_state.resize(kHidden);
      for (int col = 0; col < kHidden; ++col) {
        const float value = __bfloat162float(
            host_last[static_cast<std::size_t>(session) * kHidden + col]);
        result.final_state[col] = value;
        result.hidden_checksum += value;
      }
    }
  }
  (void)final_hidden;
  return results;
}
