// Private implementation fragment; included only by src/gpu.cu.
VisionPatchTestResult test_vision_patch_embedding(
    const DeviceModel& model, std::span<const float> pixel_values,
    std::span<const std::int32_t> position_ids,
    std::span<const float> expected) {
  constexpr int kPatches = 2520;
  constexpr int kPatchValues = 768;
  constexpr int kHidden = 1152;
  if (pixel_values.size() !=
          static_cast<std::size_t>(kPatches) * kPatchValues ||
      position_ids.size() != static_cast<std::size_t>(kPatches) * 2 ||
      (!expected.empty() &&
       expected.size() != static_cast<std::size_t>(kPatches) * kHidden))
    throw std::runtime_error("vision patch differential shape mismatch");
  const auto weight = model.tensor(
      "model.vision_tower.patch_embedder.input_proj.weight");
  const auto positions = model.tensor(
      "model.vision_tower.patch_embedder.position_embedding_table");
  if (weight.info->dtype != "BF16" ||
      weight.info->shape != std::vector<std::uint64_t>{kHidden, kPatchValues} ||
      positions.info->dtype != "BF16" ||
      positions.info->shape !=
          std::vector<std::uint64_t>{2, 10240, kHidden})
    throw std::runtime_error("unexpected vision patch weight geometry");
  float* d_pixels{};
  std::int32_t* d_positions{};
  __nv_bfloat16 *d_patches{}, *d_hidden{};
  check(cudaMalloc(&d_pixels, pixel_values.size_bytes()),
        "cudaMalloc(vision pixels)");
  check(cudaMalloc(&d_positions, position_ids.size_bytes()),
        "cudaMalloc(vision positions)");
  check(cudaMalloc(&d_patches, pixel_values.size() * 2),
        "cudaMalloc(vision patches)");
  check(cudaMalloc(&d_hidden,
                   static_cast<std::size_t>(kPatches) * kHidden * 2),
        "cudaMalloc(vision hidden)");
  check(cudaMemcpy(d_pixels, pixel_values.data(), pixel_values.size_bytes(),
                   cudaMemcpyHostToDevice),
        "copy vision pixels");
  check(cudaMemcpy(d_positions, position_ids.data(), position_ids.size_bytes(),
                   cudaMemcpyHostToDevice),
        "copy vision positions");
  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(vision patch)");
  check(cublasCreate(&handle), "cublasCreate(vision patch)");
  check(cublasSetStream(handle, stream), "cublasSetStream(vision patch)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto launch = [&] {
    prepare_vision_patches_kernel<<<
        (pixel_values.size() + 255) / 256, 256, 0, stream>>>(
        d_pixels, d_patches, static_cast<int>(pixel_values.size()));
    check(cublasGemmEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, kHidden, kPatches,
              kPatchValues, &alpha,
              reinterpret_cast<const __nv_bfloat16*>(weight.data), CUDA_R_16BF,
              kPatchValues, d_patches, CUDA_R_16BF, kPatchValues, &beta,
              d_hidden, CUDA_R_16BF, kHidden, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "vision patch projection");
    add_vision_positions_kernel<<<
        (kPatches * kHidden + 255) / 256, 256, 0, stream>>>(
        d_hidden,
        reinterpret_cast<const __nv_bfloat16*>(positions.data), d_positions,
        kPatches);
  };
  launch();
  check(cudaStreamSynchronize(stream), "vision patch synchronize");
  const float microseconds = event_benchmark(stream, launch, 2, 20);
  std::vector<__nv_bfloat16> hidden(
      static_cast<std::size_t>(kPatches) * kHidden);
  check(cudaMemcpy(hidden.data(), d_hidden, hidden.size() * 2,
                   cudaMemcpyDeviceToHost),
        "copy vision patch hidden");
  VisionPatchTestResult result{};
  double total_error = 0.0;
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const float actual = __bfloat162float(hidden[index]);
    result.checksum += actual;
    if (!expected.empty()) {
      const float error = std::abs(actual - expected[index]);
      result.max_abs_error = std::max(result.max_abs_error, error);
      total_error += error;
    }
  }
  if (!expected.empty())
    result.mean_abs_error = static_cast<float>(total_error / hidden.size());
  result.microseconds = microseconds;
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  cudaFree(d_hidden);
  cudaFree(d_patches);
  cudaFree(d_positions);
  cudaFree(d_pixels);
  return result;
}

VisionLayerTestResult test_vision_layer0(
    const DeviceModel& model, std::span<const float> input,
    std::span<const std::int32_t> position_ids,
    std::span<const float> expected) {
  constexpr int kHidden = 1152;
  constexpr int kIntermediate = 4304;
  constexpr int kHeads = 16;
  constexpr int kHeadDim = 72;
  if (input.empty() || input.size() % kHidden != 0 ||
      position_ids.size() != (input.size() / kHidden) * 2 ||
      (!expected.empty() && expected.size() != input.size()))
    throw std::runtime_error("vision layer differential shape mismatch");
  const int tokens = static_cast<int>(input.size() / kHidden);
  int valid_tokens = tokens;
  while (valid_tokens > 0 && position_ids[(valid_tokens - 1) * 2] == -1 &&
         position_ids[(valid_tokens - 1) * 2 + 1] == -1)
    --valid_tokens;
  for (int token = 0; token < valid_tokens; ++token)
    if (position_ids[token * 2] < 0 || position_ids[token * 2 + 1] < 0)
      throw std::runtime_error("vision padding must be a contiguous tail");
  const std::string prefix = "model.vision_tower.encoder.layers.0.";
  auto tensor = [&](const std::string& suffix) {
    return model.tensor(prefix + suffix);
  };
  auto bf16 = [&](const std::string& suffix,
                  std::initializer_list<std::uint64_t> shape) {
    const auto result = tensor(suffix);
    if (result.info->dtype != "BF16" ||
        result.info->shape != std::vector<std::uint64_t>(shape))
      throw std::runtime_error("unexpected vision tensor: " + suffix);
    return reinterpret_cast<const __nv_bfloat16*>(result.data);
  };
  const auto* input_norm = bf16("input_layernorm.weight", {kHidden});
  const auto* q_weight = bf16("self_attn.q_proj.linear.weight", {kHidden, kHidden});
  const auto* k_weight = bf16("self_attn.k_proj.linear.weight", {kHidden, kHidden});
  const auto* v_weight = bf16("self_attn.v_proj.linear.weight", {kHidden, kHidden});
  const auto* o_weight = bf16("self_attn.o_proj.linear.weight", {kHidden, kHidden});
  const auto* q_norm = bf16("self_attn.q_norm.weight", {kHeadDim});
  const auto* k_norm = bf16("self_attn.k_norm.weight", {kHeadDim});
  const auto* post_attention = bf16("post_attention_layernorm.weight", {kHidden});
  const auto* pre_feedforward = bf16("pre_feedforward_layernorm.weight", {kHidden});
  const auto* gate_weight = bf16("mlp.gate_proj.linear.weight", {kIntermediate, kHidden});
  const auto* up_weight = bf16("mlp.up_proj.linear.weight", {kIntermediate, kHidden});
  const auto* down_weight = bf16("mlp.down_proj.linear.weight", {kHidden, kIntermediate});
  const auto* post_feedforward = bf16("post_feedforward_layernorm.weight", {kHidden});

  std::vector<void*> allocations;
  auto allocate = [&](std::size_t bytes, const char* name) {
    void* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return reinterpret_cast<__nv_bfloat16*>(pointer);
  };
  const std::size_t hidden_bytes = input.size() * 2;
  auto* hidden = allocate(hidden_bytes, "cudaMalloc(vision layer input)");
  auto* normalized = allocate(hidden_bytes, "cudaMalloc(vision layer norm)");
  auto* q = allocate(hidden_bytes, "cudaMalloc(vision Q)");
  auto* k = allocate(hidden_bytes, "cudaMalloc(vision K)");
  auto* v = allocate(hidden_bytes, "cudaMalloc(vision V)");
  auto* attention = allocate(hidden_bytes, "cudaMalloc(vision attention)");
  auto* residual = allocate(hidden_bytes, "cudaMalloc(vision residual)");
  auto* gate = allocate(static_cast<std::size_t>(tokens) * kIntermediate * 2,
                        "cudaMalloc(vision gate)");
  auto* up = allocate(static_cast<std::size_t>(tokens) * kIntermediate * 2,
                      "cudaMalloc(vision up)");
  auto* product = allocate(static_cast<std::size_t>(tokens) * kIntermediate * 2,
                           "cudaMalloc(vision product)");
  auto* mlp = allocate(hidden_bytes, "cudaMalloc(vision MLP)");
  auto* output = allocate(hidden_bytes, "cudaMalloc(vision output)");
  auto* scores = allocate(static_cast<std::size_t>(kHeads) * tokens * tokens * 2,
                          "cudaMalloc(vision scores)");
  std::int32_t* positions{};
  check(cudaMalloc(&positions, position_ids.size_bytes()),
        "cudaMalloc(vision layer positions)");
  allocations.push_back(positions);
  std::vector<__nv_bfloat16> host_input(input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    host_input[i] = __float2bfloat16_rn(input[i]);
  check(cudaMemcpy(hidden, host_input.data(), hidden_bytes, cudaMemcpyHostToDevice),
        "copy vision layer input");
  check(cudaMemcpy(positions, position_ids.data(), position_ids.size_bytes(),
                   cudaMemcpyHostToDevice),
        "copy vision layer positions");
  cudaStream_t stream{};
  cublasHandle_t handle{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate(vision layer)");
  check(cublasCreate(&handle), "cublasCreate(vision layer)");
  check(cublasSetStream(handle, stream), "cublasSetStream(vision layer)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* weight, int rows, int columns,
                     const __nv_bfloat16* source, __nv_bfloat16* destination) {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, columns,
                       &alpha, weight, CUDA_R_16BF, columns, source,
                       CUDA_R_16BF, columns, &beta, destination, CUDA_R_16BF,
                       rows, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "vision projection");
  };
  auto launch = [&] {
    vision_rmsnorm_1152_kernel<<<tokens, 256, 0, stream>>>(
        hidden, input_norm, normalized, tokens);
    project(q_weight, kHidden, kHidden, normalized, q);
    project(k_weight, kHidden, kHidden, normalized, k);
    project(v_weight, kHidden, kHidden, normalized, v);
    vision_qkv_norm_rope_kernel<<<tokens * kHeads, 128, 0, stream>>>(
        q, k, v, q_norm, k_norm, positions, tokens);
    check(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, tokens, tokens, kHeadDim,
              &alpha, k, CUDA_R_16BF, kHidden, kHeadDim, q, CUDA_R_16BF,
              kHidden, kHeadDim, &beta, scores, CUDA_R_16BF, tokens,
              static_cast<long long>(tokens) * tokens, kHeads,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "vision QK projection");
    vision_softmax_kernel<<<tokens * kHeads, 256, 0, stream>>>(
        scores, tokens, valid_tokens);
    check(cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, tokens, tokens,
              &alpha, v, CUDA_R_16BF, kHidden, kHeadDim, scores, CUDA_R_16BF,
              tokens, static_cast<long long>(tokens) * tokens, &beta, q,
              CUDA_R_16BF, kHidden, kHeadDim, kHeads, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "vision PV projection");
    project(o_weight, kHidden, kHidden, q, attention);
    vision_rmsnorm_add_1152_kernel<<<tokens, 256, 0, stream>>>(
        attention, post_attention, hidden, residual, tokens);
    vision_rmsnorm_1152_kernel<<<tokens, 256, 0, stream>>>(
        residual, pre_feedforward, normalized, tokens);
    project(gate_weight, kIntermediate, kHidden, normalized, gate);
    project(up_weight, kIntermediate, kHidden, normalized, up);
    gelu_tanh_multiply_kernel<<<
        (tokens * kIntermediate + 255) / 256, 256, 0, stream>>>(
        gate, up, product, tokens * kIntermediate);
    project(down_weight, kHidden, kIntermediate, product, mlp);
    vision_rmsnorm_add_1152_kernel<<<tokens, 256, 0, stream>>>(
        mlp, post_feedforward, residual, output, tokens);
  };
  launch();
  check(cudaStreamSynchronize(stream), "vision layer synchronize");
  const float microseconds = event_benchmark(stream, launch, 2, 10);
  std::vector<__nv_bfloat16> host_output(input.size());
  check(cudaMemcpy(host_output.data(), output, hidden_bytes,
                   cudaMemcpyDeviceToHost),
        "copy vision layer output");
  VisionLayerTestResult result{};
  double total_error = 0.0;
  const std::size_t compared_elements =
      static_cast<std::size_t>(valid_tokens) * kHidden;
  for (std::size_t index = 0; index < host_output.size(); ++index) {
    const float actual = __bfloat162float(host_output[index]);
    result.checksum += actual;
    if (!expected.empty() && index < compared_elements) {
      const float error = std::abs(actual - expected[index]);
      result.max_abs_error = std::max(result.max_abs_error, error);
      total_error += error;
    }
  }
  if (!expected.empty())
    result.mean_abs_error = static_cast<float>(total_error / compared_elements);
  result.microseconds = microseconds;
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  for (void* pointer : allocations) cudaFree(pointer);
  return result;
}

