// Private implementation fragment; included only by src/gpu.cu.
AttentionSublayerBenchmark benchmark_attention_sublayer(
    const DeviceModel& model, KvCache& cache, AttentionWorkspace& workspace,
    int layer, int context_tokens) {
  if (layer < 0 || layer >= 30 || context_tokens < 1 ||
      context_tokens > cache.maximum_context())
    throw std::runtime_error("invalid attention sublayer benchmark arguments");
  constexpr int kHidden = 2816;
  const bool global = layer % 6 == 5;
  const int q_width = global ? 8192 : 4096;
  const int kv_width = global ? 1024 : 2048;
  const int position = context_tokens - 1;
  const auto prefix = "model.language_model.layers." + std::to_string(layer) + ".";
  auto weight = [&](const std::string& suffix) {
    return reinterpret_cast<const __nv_bfloat16*>(model.tensor(prefix + suffix).data);
  };
  const auto* input_norm = weight("input_layernorm.weight");
  const auto* q_weight = weight("self_attn.q_proj.weight");
  const auto* k_weight = weight("self_attn.k_proj.weight");
  const auto* v_weight = global ? nullptr : weight("self_attn.v_proj.weight");
  const auto* o_weight = weight("self_attn.o_proj.weight");
  const auto* q_norm = weight("self_attn.q_norm.weight");
  const auto* k_norm = weight("self_attn.k_norm.weight");
  __nv_bfloat16 *d_input{}, *d_normed{}, *d_q{}, *d_k{}, *d_v{}, *d_context{}, *d_output{};
  auto allocate = [](auto** pointer, std::size_t bytes, const char* name) {
    check(cudaMalloc(pointer, bytes), name);
  };
  allocate(&d_input, kHidden * 2, "cudaMalloc(sublayer input)");
  allocate(&d_normed, kHidden * 2, "cudaMalloc(sublayer normalized)");
  allocate(&d_q, q_width * 2, "cudaMalloc(sublayer Q)");
  allocate(&d_k, kv_width * 2, "cudaMalloc(sublayer K)");
  allocate(&d_v, kv_width * 2, "cudaMalloc(sublayer V)");
  allocate(&d_context, q_width * 2, "cudaMalloc(sublayer context)");
  allocate(&d_output, kHidden * 2, "cudaMalloc(sublayer output)");
  std::vector<__nv_bfloat16> input(kHidden);
  for (int col = 0; col < kHidden; ++col)
    input[col] = __float2bfloat16_rn(
        std::sin(col * 0.017F) * 0.7F + std::cos(col * 0.003F) * 0.2F);
  check(cudaMemcpy(d_input, input.data(), kHidden * 2, cudaMemcpyHostToDevice),
        "copy sublayer input");
  const auto& kv = cache.layer(layer);
  const std::size_t cache_plane_bytes =
      static_cast<std::size_t>(kv.capacity) * kv.kv_heads * kv.head_dim * 2;
  check(cudaMemset(kv.keys, 0, cache_plane_bytes), "clear K cache");
  check(cudaMemset(kv.values, 0, cache_plane_bytes), "clear V cache");
  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(attention sublayer)");
  check(cublasCreate(&handle), "cublasCreate(attention sublayer)");
  check(cublasSetStream(handle, stream), "cublasSetStream(attention sublayer)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int outputs,
                     const __nv_bfloat16* source, int inputs,
                     __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, outputs, 1, inputs,
                       &alpha, matrix, CUDA_R_16BF, inputs, source, CUDA_R_16BF,
                       inputs, &beta, destination, CUDA_R_16BF, outputs,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "attention sublayer projection");
  };
  auto launch = [&] {
    rmsnorm_2816_kernel<<<1, 256, 0, stream>>>(
        d_input, input_norm, d_normed, 1, 1e-6F);
    project(q_weight, q_width, d_normed, kHidden, d_q);
    project(k_weight, kv_width, d_normed, kHidden, d_k);
    if (!global) project(v_weight, kv_width, d_normed, kHidden, d_v);
    launch_qkv_transform(global, d_q, d_k, d_v, q_norm, k_norm, position,
                         kv.keys, kv.values, stream);
    launch_decode_attention(global, d_q, kv.keys, kv.values, d_context,
                            context_tokens, workspace.scores(), workspace.partials(), stream);
    project(o_weight, kHidden, d_context, q_width, d_output);
  };
  launch();
  check(cudaGetLastError(), "attention sublayer launch");
  check(cudaStreamSynchronize(stream), "attention sublayer synchronize");
  const float eager_us = event_benchmark(stream, launch, 10, 200);
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin attention sublayer graph");
  launch();
  check(cudaStreamEndCapture(stream, &graph), "end attention sublayer graph");
  check(cudaGraphInstantiate(&graph_exec, graph), "instantiate attention sublayer graph");
  const float graph_us = event_benchmark(
      stream, [&] { check(cudaGraphLaunch(graph_exec, stream), "launch attention sublayer graph"); },
      10, 500);
  std::vector<__nv_bfloat16> output(kHidden);
  check(cudaMemcpy(output.data(), d_output, kHidden * 2, cudaMemcpyDeviceToHost),
        "copy attention sublayer output");
  float checksum = 0.0F;
  for (const auto value : output) checksum += __bfloat162float(value);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  cudaFree(d_output); cudaFree(d_context); cudaFree(d_v); cudaFree(d_k);
  cudaFree(d_q); cudaFree(d_normed); cudaFree(d_input);
  return {eager_us, graph_us, checksum};
}

DecoderLayerBenchmark benchmark_decoder_layer(
    const RuntimeWeights& weights, KvCache& cache,
    AttentionWorkspace& workspace, int layer, int context_tokens, int tokens) {
  if (layer < 0 || layer >= 30 || context_tokens < 1 ||
      tokens < 1 || tokens > 5 ||
      context_tokens + tokens - 1 > cache.maximum_context() ||
      context_tokens + tokens - 1 > workspace.maximum_context() ||
      tokens > workspace.maximum_batch())
    throw std::runtime_error("invalid decoder layer benchmark arguments");
  constexpr int kHidden = 2816;
  constexpr int kDense = 2112;
  constexpr int kExperts = 128;
  constexpr int kTop = 8;
  const bool global = layer % 6 == 5;
  const int q_width = global ? 8192 : 4096;
  const int kv_width = global ? 1024 : 2048;
  const int position = context_tokens - 1;
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  const auto& model = weights.model();
  auto weight = [&](const std::string& suffix) {
    return reinterpret_cast<const __nv_bfloat16*>(model.tensor(prefix + suffix).data);
  };

  const auto* input_norm_w = weight("input_layernorm.weight");
  const auto* q_w = weight("self_attn.q_proj.weight");
  const auto* k_w = weight("self_attn.k_proj.weight");
  const auto* v_w = global ? nullptr : weight("self_attn.v_proj.weight");
  const auto* o_w = weight("self_attn.o_proj.weight");
  const auto* q_norm_w = weight("self_attn.q_norm.weight");
  const auto* k_norm_w = weight("self_attn.k_norm.weight");
  const auto* post_attention_w = weight("post_attention_layernorm.weight");
  const auto* dense_pre_w = weight("pre_feedforward_layernorm.weight");
  const auto* expert_pre_w = weight("pre_feedforward_layernorm_2.weight");
  const auto* router_scale = weight("router.scale");
  const auto* router_w = weight("router.proj.weight");
  const auto* per_expert_scale = weight("router.per_expert_scale");
  const auto* gate_w = weight("mlp.gate_proj.weight");
  const auto* up_w = weight("mlp.up_proj.weight");
  const auto* down_w = weight("mlp.down_proj.weight");
  const auto* dense_post_w = weight("post_feedforward_layernorm_1.weight");
  const auto* expert_post_w = weight("post_feedforward_layernorm_2.weight");
  const auto* combined_post_w = weight("post_feedforward_layernorm.weight");
  const auto* layer_scalar = weight("layer_scalar");

  std::vector<void*> allocations;
  auto allocate = [&](std::size_t bytes, const char* name) {
    void* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return reinterpret_cast<__nv_bfloat16*>(pointer);
  };
  auto rows = [&](int width) { return static_cast<std::size_t>(tokens) * width * 2; };
  auto* input = allocate(rows(kHidden), "cudaMalloc(layer input)");
  auto* normalized = allocate(rows(kHidden), "cudaMalloc(layer normalized)");
  auto* q = allocate(rows(q_width), "cudaMalloc(layer Q)");
  auto* k = allocate(rows(kv_width), "cudaMalloc(layer K)");
  auto* v = allocate(rows(kv_width), "cudaMalloc(layer V)");
  auto* context = allocate(rows(q_width), "cudaMalloc(layer context)");
  auto* attention = allocate(rows(kHidden), "cudaMalloc(layer attention)");
  auto* attention_norm = allocate(rows(kHidden), "cudaMalloc(layer attention norm)");
  auto* residual = allocate(rows(kHidden), "cudaMalloc(layer residual)");
  auto* dense_pre = allocate(rows(kHidden), "cudaMalloc(layer dense pre-norm)");
  auto* expert_pre = allocate(rows(kHidden), "cudaMalloc(layer expert pre-norm)");
  auto* router_input = allocate(rows(kHidden), "cudaMalloc(layer router input)");
  auto* router_logits = allocate(rows(kExperts), "cudaMalloc(layer router logits)");
  auto* dense_gate = allocate(rows(kDense), "cudaMalloc(layer dense gate)");
  auto* dense_up = allocate(rows(kDense), "cudaMalloc(layer dense up)");
  auto* dense_product = allocate(rows(kDense), "cudaMalloc(layer dense product)");
  auto* dense_output = allocate(rows(kHidden), "cudaMalloc(layer dense output)");
  auto* expert_output = allocate(rows(kHidden), "cudaMalloc(layer expert output)");
  auto* dense_post = allocate(rows(kHidden), "cudaMalloc(layer dense post-norm)");
  auto* expert_post = allocate(rows(kHidden), "cudaMalloc(layer expert post-norm)");
  auto* feedforward_sum = allocate(rows(kHidden), "cudaMalloc(layer ff sum)");
  auto* feedforward_norm = allocate(rows(kHidden), "cudaMalloc(layer ff norm)");
  auto* output = allocate(rows(kHidden), "cudaMalloc(layer output)");
  float* top_weights{};
  int* top_ids{};
  check(cudaMalloc(&top_weights, tokens * kTop * sizeof(float)), "cudaMalloc(layer route weights)");
  check(cudaMalloc(&top_ids, tokens * kTop * sizeof(int)), "cudaMalloc(layer route ids)");
  allocations.push_back(top_weights);
  allocations.push_back(top_ids);

  std::vector<__nv_bfloat16> host_input(tokens * kHidden);
  for (std::size_t col = 0; col < host_input.size(); ++col)
    host_input[col] = __float2bfloat16_rn(
        std::sin(col * 0.017F) * 0.7F + std::cos(col * 0.003F) * 0.2F);
  check(cudaMemcpy(input, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice),
        "copy decoder layer input");
  const auto& kv = cache.layer(layer);
  const std::size_t cache_plane_bytes =
      static_cast<std::size_t>(kv.capacity) * kv.kv_heads * kv.head_dim * 2;
  check(cudaMemset(kv.keys, 0, cache_plane_bytes), "clear decoder K cache");
  check(cudaMemset(kv.values, 0, cache_plane_bytes), "clear decoder V cache");

  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(decoder layer)");
  check(cublasCreate(&handle), "cublasCreate(decoder layer)");
  check(cublasSetStream(handle, stream), "cublasSetStream(decoder layer)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* matrix, int matrix_rows, int cols,
                     const __nv_bfloat16* source, __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_rows, tokens, cols,
                       &alpha, matrix, CUDA_R_16BF, cols, source, CUDA_R_16BF,
                       cols, &beta, destination, CUDA_R_16BF, matrix_rows,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "decoder layer projection");
  };
  Nvfp4ExpertRunner expert_runner(weights.experts(), layer, tokens);

  auto launch = [&] {
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        input, input_norm_w, normalized, tokens, 1e-6F);
    project(q_w, q_width, kHidden, normalized, q);
    project(k_w, kv_width, kHidden, normalized, k);
    if (!global) project(v_w, kv_width, kHidden, normalized, v);
    transform_qkv_batch(global, q, k, v, q_norm_w, k_norm_w,
                        position, tokens, stream);
    launch_decode_attention_batch(global, q, kv.keys, kv.values, k, v, context,
                                  context_tokens, tokens, workspace.scores(),
                                  workspace.partials(), stream);
    append_qkv_batch(global, k, v, position, tokens, kv.keys, kv.values, stream);
    project(o_w, kHidden, q_width, context, attention);

    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        attention, post_attention_w, attention_norm, tokens, 1e-6F);
    bf16_add_kernel<<<(tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        input, attention_norm, residual, tokens * kHidden);
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        residual, dense_pre_w, dense_pre, tokens, 1e-6F);
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        residual, expert_pre_w, expert_pre, tokens, 1e-6F);
    router_input_kernel<<<tokens, 256, 0, stream>>>(
        residual, router_scale, router_input, tokens);
    project(router_w, kExperts, kHidden, router_input, router_logits);
    route_128_top8_kernel<<<tokens, kExperts, 0, stream>>>(
        router_logits, per_expert_scale, top_weights, top_ids, tokens);

    project(gate_w, kDense, kHidden, dense_pre, dense_gate);
    project(up_w, kDense, kHidden, dense_pre, dense_up);
    gelu_tanh_multiply_kernel<<<(tokens * kDense + 255) / 256, 256, 0, stream>>>(
        dense_gate, dense_up, dense_product, tokens * kDense);
    project(down_w, kHidden, kDense, dense_product, dense_output);
    expert_runner.launch_batch(expert_pre, top_ids, top_weights,
                               expert_output, tokens, stream);

    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        dense_output, dense_post_w, dense_post, tokens, 1e-6F);
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        expert_output, expert_post_w, expert_post, tokens, 1e-6F);
    bf16_add_kernel<<<(tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        dense_post, expert_post, feedforward_sum, tokens * kHidden);
    rmsnorm_2816_kernel<<<tokens, 256, 0, stream>>>(
        feedforward_sum, combined_post_w, feedforward_norm, tokens, 1e-6F);
    bf16_add_kernel<<<(tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        residual, feedforward_norm, output, tokens * kHidden);
    bf16_scale_pointer_kernel<<<(tokens * kHidden + 255) / 256, 256, 0, stream>>>(
        output, tokens * kHidden, layer_scalar);
  };

  launch();
  check(cudaGetLastError(), "decoder layer launch");
  check(cudaStreamSynchronize(stream), "decoder layer synchronize");
  const float eager_us = event_benchmark(stream, launch, 10, 200);
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin decoder layer graph");
  launch();
  check(cudaStreamEndCapture(stream, &graph), "end decoder layer graph");
  check(cudaGraphInstantiate(&graph_exec, graph), "instantiate decoder layer graph");
  const float graph_us = event_benchmark(
      stream, [&] { check(cudaGraphLaunch(graph_exec, stream), "launch decoder layer graph"); },
      10, 500);
  std::vector<__nv_bfloat16> host_output(tokens * kHidden);
  check(cudaMemcpy(host_output.data(), output, host_output.size() * 2, cudaMemcpyDeviceToHost),
        "copy decoder layer output");
  float checksum = 0.0F;
  for (const auto value : host_output) checksum += __bfloat162float(value);

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  for (void* allocation : allocations) cudaFree(allocation);
  return {eager_us, graph_us, checksum};
}

