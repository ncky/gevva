// Private implementation fragment; included only by src/gpu.cu.
VisionEncoderBenchmark benchmark_vision_encoder(
    const DeviceModel& model, std::span<const float> pixel_values,
    std::span<const std::int32_t> position_ids, int valid_patches,
    std::span<const float> expected_projected, bool benchmark,
    DeviceScratchPool* scratch, GpuExecutionContext* context,
    int image_batch, void* device_projected_output,
    bool copy_projected_to_host, Fp8LinearRunner* fp8_linears,
    std::span<const float> second_pixel_values,
    std::span<const std::int32_t> second_position_ids,
    const VisionFusedWeights* fused_weights, const float* device_pixel_values,
    const std::int32_t* device_position_ids) {
  constexpr int kPatchValues = 768;
  constexpr int kHidden = 1152;
  constexpr int kTextHidden = 2816;
  constexpr int kIntermediate = 4304;
  constexpr int kHeads = 16;
  constexpr int kHeadDim = 72;
  if (image_batch < 1 || image_batch > 8)
    throw std::runtime_error("vision image batch must be in [1, 8]");
  const bool device_input = device_pixel_values != nullptr;
  if (device_input && (!device_position_ids || !pixel_values.empty() || image_batch != 1 ||
                       !second_pixel_values.empty() || !context))
    throw std::runtime_error("device vision input requires one compact image and a context");
  const bool split_pair_input = !second_pixel_values.empty();
  if (split_pair_input &&
      (image_batch != 2 || second_position_ids.empty() ||
       second_pixel_values.size() != pixel_values.size() ||
       second_position_ids.size() != position_ids.size()))
    throw std::runtime_error("split vision input requires two equal images");
  const std::size_t total_pixel_values =
      device_input ? static_cast<std::size_t>(valid_patches) * kPatchValues
                   : pixel_values.size() + second_pixel_values.size();
  const std::size_t total_position_values =
      position_ids.size() + second_position_ids.size();
  const int maximum_patches = static_cast<int>(
      total_pixel_values / (static_cast<std::size_t>(image_batch) *
                            kPatchValues));
  // Preprocessing reserves a fixed upload envelope, but valid patches are a
  // dense prefix. Padding participates in neither attention nor pooling, so
  // projecting it and forming its quadratic query rows is pure waste.
  const int patches_per_image = valid_patches;
  const int kComputePatches = valid_patches * image_batch;
  if (total_pixel_values !=
          static_cast<std::size_t>(image_batch) * maximum_patches *
              kPatchValues ||
      (maximum_patches != 2520 && maximum_patches != 5040 &&
       maximum_patches != valid_patches) ||
      total_position_values !=
          static_cast<std::size_t>(image_batch) * maximum_patches * 2 ||
      valid_patches < 9 || valid_patches > maximum_patches ||
      valid_patches % 9 != 0)
    throw std::runtime_error("vision encoder input geometry is invalid");
  int patch_width = 0;
  int patch_height = 0;
  for (int image = 0; image < image_batch; ++image) {
    const auto image_positions = split_pair_input && image == 1
        ? second_position_ids : position_ids;
    const std::size_t position_base = split_pair_input
        ? 0 : static_cast<std::size_t>(image) * maximum_patches * 2;
    int image_width = 0, image_height = 0;
    for (int patch = 0; patch < valid_patches; ++patch) {
      image_width = std::max(
          image_width, image_positions[position_base + patch * 2] + 1);
      image_height = std::max(
          image_height, image_positions[position_base + patch * 2 + 1] + 1);
    }
    if (image == 0) {
      patch_width = image_width;
      patch_height = image_height;
    } else if (image_width != patch_width || image_height != patch_height) {
      throw std::runtime_error("uniform vision batch geometry mismatch");
    }
  }
  if (patch_width * patch_height != valid_patches || patch_width % 3 != 0 ||
      patch_height % 3 != 0)
    throw std::runtime_error("vision patch positions are not a dense 3x3-poolable grid");
  const int soft_tokens_per_image = valid_patches / 9;
  const int soft_tokens = soft_tokens_per_image * image_batch;
  const bool preferred_native_layout = std::getenv("GEVVA_CUDNN_PACKED_LAYOUT") == nullptr;
  const bool preferred_fused_projections = fused_weights && !fp8_linears && preferred_native_layout &&
      std::getenv("GEVVA_DISABLE_FUSED_VISION_PROJECTIONS") == nullptr;
  const bool fused_vision_attention =
      std::getenv("GEVVA_LEGACY_VISION_ATTENTION") == nullptr &&
      cudnn_vision_plan_available(image_batch, kHeads, patches_per_image, kHeadDim,
          preferred_native_layout, preferred_fused_projections ? 3 * kHidden : kHidden);
  if (!expected_projected.empty() &&
      expected_projected.size() !=
          static_cast<std::size_t>(soft_tokens) * kTextHidden)
    throw std::runtime_error("vision projected oracle shape mismatch");

  auto require = [&](const std::string& name,
                     std::initializer_list<std::uint64_t> shape) {
    const auto result = model.tensor(name);
    if (result.info->dtype != "BF16" ||
        result.info->shape != std::vector<std::uint64_t>(shape))
      throw std::runtime_error("unexpected vision tensor: " + name);
    return reinterpret_cast<const __nv_bfloat16*>(result.data);
  };
  const auto* patch_weight = require(
      "model.vision_tower.patch_embedder.input_proj.weight",
      {kHidden, kPatchValues});
  const auto* position_table = require(
      "model.vision_tower.patch_embedder.position_embedding_table",
      {2, 10240, kHidden});
  const auto* standard_bias =
      require("model.vision_tower.std_bias", {kHidden});
  const auto* standard_scale =
      require("model.vision_tower.std_scale", {kHidden});
  const auto* projector = require(
      "model.embed_vision.embedding_projection.weight", {kTextHidden, kHidden});

  if (scratch) scratch->reset();
  std::vector<void*> allocations;
  auto allocate_bytes = [&](std::size_t bytes, const char* name) {
    if (scratch) return scratch->allocate(bytes);
    void* pointer{};
    check(cudaMalloc(&pointer, bytes), name);
    allocations.push_back(pointer);
    return pointer;
  };
  auto allocate = [&](std::size_t bytes, const char* name) {
    return reinterpret_cast<__nv_bfloat16*>(allocate_bytes(bytes, name));
  };
  std::vector<float> packed_pixels;
  std::vector<std::int32_t> packed_positions;
  std::span<const float> upload_pixels = pixel_values;
  std::span<const std::int32_t> upload_positions = position_ids;
  if (image_batch > 1 && maximum_patches != valid_patches) {
    packed_pixels.resize(static_cast<std::size_t>(kComputePatches) * kPatchValues);
    packed_positions.resize(static_cast<std::size_t>(kComputePatches) * 2);
    for (int image = 0; image < image_batch; ++image) {
      std::copy_n(pixel_values.begin() +
                      static_cast<std::size_t>(image) * maximum_patches * kPatchValues,
                  static_cast<std::size_t>(valid_patches) * kPatchValues,
                  packed_pixels.begin() +
                      static_cast<std::size_t>(image) * valid_patches * kPatchValues);
      std::copy_n(position_ids.begin() +
                      static_cast<std::size_t>(image) * maximum_patches * 2,
                  static_cast<std::size_t>(valid_patches) * 2,
                  packed_positions.begin() +
                      static_cast<std::size_t>(image) * valid_patches * 2);
    }
    upload_pixels = packed_pixels;
    upload_positions = packed_positions;
  }
  const std::size_t upload_pixel_count = (split_pair_input || device_input)
      ? total_pixel_values : upload_pixels.size();
  const std::size_t upload_position_count = split_pair_input
      ? total_position_values : upload_positions.size();
  auto* d_pixels = device_input ? const_cast<float*>(device_pixel_values) : static_cast<float*>(allocate_bytes(
      upload_pixel_count * sizeof(float), "cudaMalloc(vision encoder pixels)"));
  auto* d_positions = device_input ? const_cast<std::int32_t*>(device_position_ids) : static_cast<std::int32_t*>(allocate_bytes(
      upload_position_count * sizeof(std::int32_t),
      "cudaMalloc(vision encoder positions)"));
  auto* patches = allocate(upload_pixel_count * 2,
                           "cudaMalloc(vision encoder patches)");
  const std::size_t hidden_bytes =
      static_cast<std::size_t>(kComputePatches) * kHidden * 2;
  const bool native_cudnn_layout = fused_vision_attention &&
      std::getenv("GEVVA_CUDNN_PACKED_LAYOUT") == nullptr;
  const bool fused_projections = fused_weights && !fp8_linears &&
      native_cudnn_layout &&
      std::getenv("GEVVA_DISABLE_FUSED_VISION_PROJECTIONS") == nullptr;
  auto* hidden_a = allocate(hidden_bytes, "cudaMalloc(vision encoder hidden A)");
  auto* hidden_b = allocate(hidden_bytes, "cudaMalloc(vision encoder hidden B)");
  auto* normalized = allocate(hidden_bytes, "cudaMalloc(vision encoder norm)");
  auto* qkv = fused_projections
      ? allocate(3 * hidden_bytes, "cudaMalloc(vision encoder packed QKV)")
      : nullptr;
  auto* q = fused_projections
      ? qkv : allocate(hidden_bytes, "cudaMalloc(vision encoder Q)");
  auto* k = fused_projections
      ? qkv + kHidden : allocate(hidden_bytes, "cudaMalloc(vision encoder K)");
  auto* v = fused_projections
      ? qkv + 2 * kHidden : allocate(hidden_bytes, "cudaMalloc(vision encoder V)");
  auto* packed_q = fused_vision_attention && !native_cudnn_layout
      ? allocate(hidden_bytes, "cudaMalloc(packed vision Q)") : nullptr;
  auto* packed_k = fused_vision_attention && !native_cudnn_layout
      ? allocate(hidden_bytes, "cudaMalloc(packed vision K)") : nullptr;
  auto* packed_v = fused_vision_attention && !native_cudnn_layout
      ? allocate(hidden_bytes, "cudaMalloc(packed vision V)") : nullptr;
  auto* packed_attention = fused_vision_attention
      ? allocate(hidden_bytes, "cudaMalloc(packed vision attention)") : nullptr;
  auto* attention = allocate(hidden_bytes, "cudaMalloc(vision encoder attention)");
  auto* residual = allocate(hidden_bytes, "cudaMalloc(vision encoder residual)");
  const std::size_t intermediate_bytes =
      static_cast<std::size_t>(kComputePatches) * kIntermediate * 2;
  auto* gate_up = fused_projections
      ? allocate(2 * intermediate_bytes,
                 "cudaMalloc(vision encoder packed gate/up)")
      : nullptr;
  auto* gate = fused_projections
      ? gate_up : allocate(intermediate_bytes, "cudaMalloc(vision encoder gate)");
  auto* up = fused_projections
      ? gate_up + kIntermediate
      : allocate(intermediate_bytes, "cudaMalloc(vision encoder up)");
  auto* product = allocate(
      static_cast<std::size_t>(kComputePatches) * kIntermediate * 2,
      "cudaMalloc(vision encoder product)");
  auto* mlp = allocate(hidden_bytes, "cudaMalloc(vision encoder MLP)");
  auto* scores = allocate(
      fused_vision_attention
          ? sizeof(__nv_bfloat16)
          : static_cast<std::size_t>(image_batch) * kHeads *
                patches_per_image * patches_per_image * 2,
      "cudaMalloc(vision encoder scores)");
  auto* attention_pointers = allocate_bytes(
      static_cast<std::size_t>(image_batch) * kHeads * 6 * sizeof(void*),
      "cudaMalloc(vision encoder attention pointers)");
  auto* soft = allocate(static_cast<std::size_t>(soft_tokens) * kHidden * 2,
                        "cudaMalloc(vision soft tokens)");
  auto* soft_norm = allocate(static_cast<std::size_t>(soft_tokens) * kHidden * 2,
                             "cudaMalloc(vision soft norm)");
  auto* projected = device_projected_output
      ? static_cast<__nv_bfloat16*>(device_projected_output)
      : allocate(static_cast<std::size_t>(soft_tokens) * kTextHidden * 2,
                 "cudaMalloc(vision projected)");
  if (!device_input) {
  if (scratch && context && std::getenv("GEVVA_PAGEABLE_VISION_UPLOAD") == nullptr) {
    const auto pixel_bytes = upload_pixels.size_bytes();
    const auto position_bytes = upload_positions.size_bytes();
    auto* staging = static_cast<std::byte*>(scratch->host_staging(pixel_bytes + position_bytes));
    std::memcpy(staging, upload_pixels.data(), pixel_bytes);
    std::memcpy(staging + pixel_bytes, upload_positions.data(), position_bytes);
    check(cudaMemcpyAsync(d_pixels, staging, pixel_bytes, cudaMemcpyHostToDevice, context->stream()),
          "upload pinned vision pixels");
    check(cudaMemcpyAsync(d_positions, staging + pixel_bytes, position_bytes, cudaMemcpyHostToDevice, context->stream()),
          "upload pinned vision positions");
  } else {
  check(cudaMemcpy(d_pixels, upload_pixels.data(), upload_pixels.size_bytes(),
                   cudaMemcpyHostToDevice), "copy vision encoder pixels");
  check(cudaMemcpy(d_positions, upload_positions.data(),
                   upload_positions.size_bytes(), cudaMemcpyHostToDevice),
        "copy vision encoder positions");
  }
  }
  if (split_pair_input) {
    check(cudaMemcpy(d_pixels + pixel_values.size(), second_pixel_values.data(),
                     second_pixel_values.size_bytes(), cudaMemcpyHostToDevice),
          "copy second vision encoder pixels");
    check(cudaMemcpy(d_positions + position_ids.size(),
                     second_position_ids.data(), second_position_ids.size_bytes(),
                     cudaMemcpyHostToDevice),
          "copy second vision encoder positions");
  }
  cudaStream_t stream{};
  cublasHandle_t handle{};
  const bool owns_context = context == nullptr;
  if (context) {
    stream = context->stream();
    handle = static_cast<cublasHandle_t>(context->blas_handle());
  } else {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate(vision encoder)");
    check(cublasCreate(&handle), "cublasCreate(vision encoder)");
  }
  check(cublasSetStream(handle, stream), "cublasSetStream(vision encoder)");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto project = [&](const __nv_bfloat16* weight, int rows, int columns,
                     int tokens, const __nv_bfloat16* source,
                     __nv_bfloat16* destination, bool allow_fp8 = false) {
    if (allow_fp8 && fp8_linears && fp8_linears->launch(
            weight, rows, columns, source, destination, tokens, stream,
            handle, true, 0, false))
      return;
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, columns,
                       &alpha, weight, CUDA_R_16BF, columns, source,
                       CUDA_R_16BF, columns, &beta, destination, CUDA_R_16BF,
                       rows, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "vision encoder projection");
  };
  auto launch = [&] {
    const char* first_layer_env = std::getenv("GEVVA_VISION_FP8_FIRST_LAYER");
    const int fp8_first_layer = first_layer_env
        ? std::clamp(std::atoi(first_layer_env), 0, 27) : 0;
    const char* fp8_mask_env = std::getenv("GEVVA_VISION_FP8_MASK");
    const char* fp8_mask = fp8_mask_env ? fp8_mask_env : "qkvogud";
    auto fp8_type = [&](char type) { return std::strchr(fp8_mask, type); };
    prepare_vision_patches_kernel<<<
        (static_cast<std::size_t>(kComputePatches) * kPatchValues + 255) / 256,
        256, 0, stream>>>(d_pixels, patches,
                          kComputePatches * kPatchValues);
    project(patch_weight, kHidden, kPatchValues, kComputePatches, patches,
            hidden_a);
    add_vision_positions_kernel<<<
        (kComputePatches * kHidden + 255) / 256, 256, 0, stream>>>(
        hidden_a, position_table, d_positions, kComputePatches);
    auto* current = hidden_a;
    auto* next = hidden_b;
    for (int layer = 0; layer < 27; ++layer) {
      if (fp8_linears) fp8_linears->invalidate_activation_cache();
      const std::string prefix =
          "model.vision_tower.encoder.layers." + std::to_string(layer) + ".";
      auto weight = [&](const std::string& suffix) {
        return reinterpret_cast<const __nv_bfloat16*>(
            model.tensor(prefix + suffix).data);
      };
      const auto* input_norm = weight("input_layernorm.weight");
      const auto* q_weight = weight("self_attn.q_proj.linear.weight");
      const auto* k_weight = weight("self_attn.k_proj.linear.weight");
      const auto* v_weight = weight("self_attn.v_proj.linear.weight");
      const auto* o_weight = weight("self_attn.o_proj.linear.weight");
      const auto* q_norm = weight("self_attn.q_norm.weight");
      const auto* k_norm = weight("self_attn.k_norm.weight");
      const auto* post_attention = weight("post_attention_layernorm.weight");
      const auto* pre_feedforward = weight("pre_feedforward_layernorm.weight");
      const auto* gate_weight = weight("mlp.gate_proj.linear.weight");
      const auto* up_weight = weight("mlp.up_proj.linear.weight");
      const auto* down_weight = weight("mlp.down_proj.linear.weight");
      const auto* post_feedforward = weight("post_feedforward_layernorm.weight");
      vision_rmsnorm_1152_kernel<<<kComputePatches, 256, 0, stream>>>(
          current, input_norm, normalized, kComputePatches);
      const bool layer_fp8 = layer >= fp8_first_layer;
      if (fused_projections) {
        project(static_cast<const __nv_bfloat16*>(fused_weights->qkv(layer)),
                3 * kHidden, kHidden, kComputePatches, normalized, qkv);
      } else {
        project(q_weight, kHidden, kHidden, kComputePatches, normalized, q,
                layer_fp8 && fp8_type('q'));
        project(k_weight, kHidden, kHidden, kComputePatches, normalized, k,
                layer_fp8 && fp8_type('k'));
        project(v_weight, kHidden, kHidden, kComputePatches, normalized, v,
                layer_fp8 && fp8_type('v'));
      }
      vision_qkv_norm_rope_kernel<<<kComputePatches * kHeads, 128, 0, stream>>>(
          q, k, v, q_norm, k_norm, d_positions, kComputePatches,
          fused_projections ? 3 * kHidden : kHidden);
      if (fused_vision_attention) {
        const char* scale_env = std::getenv("GEVVA_CUDNN_VISION_SCALE");
        const float attention_scale =
            scale_env ? std::strtof(scale_env, nullptr) : 1.0F;
        if (native_cudnn_layout) {
          launch_cudnn_sdpa_bshd(q, k, v, packed_attention, image_batch,
                                 kHeads, patches_per_image, kHeadDim,
                                 attention_scale, stream, true,
                                 fused_projections ? 3 * kHidden : kHidden);
        } else {
          const int values = kComputePatches * kHidden;
          vision_qkv_bshd_to_bhsd_kernel<<<(values + 255) / 256, 256, 0,
                                             stream>>>(
              q, k, v, packed_q, packed_k, packed_v, kComputePatches,
              patches_per_image);
          launch_cudnn_sdpa_bshd(packed_q, packed_k, packed_v,
                                 packed_attention, image_batch, kHeads,
                                 patches_per_image, kHeadDim, attention_scale,
                                 stream, false);
          vision_bhsd_to_bshd_kernel<<<(values + 255) / 256, 256, 0, stream>>>(
              packed_attention, q, kComputePatches, patches_per_image);
        }
      } else if (image_batch == 1) {
        check(cublasGemmStridedBatchedEx(
                  handle, CUBLAS_OP_T, CUBLAS_OP_N, patches_per_image,
                  patches_per_image, kHeadDim, &alpha, k, CUDA_R_16BF,
                  kHidden, kHeadDim, q, CUDA_R_16BF, kHidden, kHeadDim, &beta,
                  scores, CUDA_R_16BF, patches_per_image,
                  static_cast<long long>(patches_per_image) * patches_per_image,
                  kHeads, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "vision encoder QK");
      } else {
        const int matrices = image_batch * kHeads;
        auto* pointer_bytes = static_cast<std::byte*>(attention_pointers);
        auto** key_pointers = reinterpret_cast<const void**>(pointer_bytes);
        auto** query_pointers = reinterpret_cast<const void**>(
            pointer_bytes + matrices * sizeof(void*));
        auto** score_pointers = reinterpret_cast<void**>(
            pointer_bytes + 2 * matrices * sizeof(void*));
        auto** value_pointers = reinterpret_cast<const void**>(
            pointer_bytes + 3 * matrices * sizeof(void*));
        auto** probability_pointers = reinterpret_cast<const void**>(
            pointer_bytes + 4 * matrices * sizeof(void*));
        auto** output_pointers = reinterpret_cast<void**>(
            pointer_bytes + 5 * matrices * sizeof(void*));
        prepare_vision_batch_attention_pointers_kernel<<<
            (matrices + 127) / 128, 128, 0, stream>>>(
            q, k, v, scores, key_pointers, query_pointers, score_pointers,
            value_pointers, probability_pointers, output_pointers,
            patches_per_image, image_batch);
        check(cublasGemmBatchedEx(
                  handle, CUBLAS_OP_T, CUBLAS_OP_N, patches_per_image,
                  patches_per_image, kHeadDim, &alpha, key_pointers,
                  CUDA_R_16BF, kHidden, query_pointers, CUDA_R_16BF, kHidden,
                  &beta, score_pointers, CUDA_R_16BF, patches_per_image,
                  matrices, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "batched vision encoder QK");
      }
      if (!fused_vision_attention)
        vision_softmax_kernel<<<image_batch * patches_per_image * kHeads, 256,
                                0, stream>>>(scores, patches_per_image,
                                             patches_per_image);
      if (fused_vision_attention) {
        // cuDNN writes a token-major attention result to packed_attention.
      } else if (image_batch == 1) {
        check(cublasGemmStridedBatchedEx(
                  handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim,
                  patches_per_image, patches_per_image, &alpha, v,
                  CUDA_R_16BF, kHidden, kHeadDim, scores, CUDA_R_16BF,
                  patches_per_image,
                  static_cast<long long>(patches_per_image) * patches_per_image,
                  &beta, q, CUDA_R_16BF, kHidden, kHeadDim, kHeads,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "vision encoder PV");
      } else {
        const int matrices = image_batch * kHeads;
        auto* pointer_bytes = static_cast<std::byte*>(attention_pointers);
        auto** value_pointers = reinterpret_cast<const void**>(
            pointer_bytes + 3 * matrices * sizeof(void*));
        auto** probability_pointers = reinterpret_cast<const void**>(
            pointer_bytes + 4 * matrices * sizeof(void*));
        auto** output_pointers = reinterpret_cast<void**>(
            pointer_bytes + 5 * matrices * sizeof(void*));
        check(cublasGemmBatchedEx(
                  handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim,
                  patches_per_image, patches_per_image, &alpha,
                  value_pointers, CUDA_R_16BF, kHidden, probability_pointers,
                  CUDA_R_16BF, patches_per_image, &beta, output_pointers,
                  CUDA_R_16BF, kHidden, matrices, CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "batched vision encoder PV");
      }
      project(o_weight, kHidden, kHidden, kComputePatches,
              native_cudnn_layout ? packed_attention : q, attention,
              layer_fp8 && fp8_type('o'));
      vision_rmsnorm_add_1152_kernel<<<kComputePatches, 256, 0, stream>>>(
          attention, post_attention, current, residual, kComputePatches);
      vision_rmsnorm_1152_kernel<<<kComputePatches, 256, 0, stream>>>(
          residual, pre_feedforward, normalized, kComputePatches);
      if (fused_projections) {
        project(static_cast<const __nv_bfloat16*>(
                    fused_weights->gate_up(layer)),
                2 * kIntermediate, kHidden, kComputePatches, normalized,
                gate_up);
        vision_gelu_tanh_multiply_packed_kernel<<<
            (kComputePatches * kIntermediate + 255) / 256, 256, 0, stream>>>(
            gate_up, product, kComputePatches);
      } else {
        project(gate_weight, kIntermediate, kHidden, kComputePatches,
                normalized, gate, layer_fp8 && fp8_type('g'));
        project(up_weight, kIntermediate, kHidden, kComputePatches, normalized,
                up, layer_fp8 && fp8_type('u'));
        gelu_tanh_multiply_kernel<<<
            (kComputePatches * kIntermediate + 255) / 256, 256, 0, stream>>>(
            gate, up, product, kComputePatches * kIntermediate);
      }
      project(down_weight, kHidden, kIntermediate, kComputePatches, product, mlp,
              layer_fp8 && fp8_type('d'));
      vision_rmsnorm_add_1152_kernel<<<kComputePatches, 256, 0, stream>>>(
          mlp, post_feedforward, residual, next, kComputePatches);
      std::swap(current, next);
    }
    if (image_batch == 1)
      vision_pool_standardize_kernel<<<soft_tokens, 256, 0, stream>>>(
          current, standard_bias, standard_scale, soft, soft_tokens,
          patch_width, patch_width / 3);
    else
      vision_pool_standardize_batch_kernel<<<soft_tokens, 256, 0, stream>>>(
          current, standard_bias, standard_scale, soft,
          soft_tokens_per_image, patches_per_image, patch_width,
          patch_width / 3, image_batch);
    vision_rmsnorm_noscale_1152_kernel<<<soft_tokens, 256, 0, stream>>>(
        soft, soft_norm, soft_tokens);
    project(projector, kTextHidden, kHidden, soft_tokens, soft_norm, projected);
  };
  float microseconds{};
  if (benchmark) {
    launch();
    check(cudaStreamSynchronize(stream), "vision encoder synchronize");
    microseconds = event_benchmark(stream, launch, 1, 5);
  } else if (context && scratch && (std::getenv("GEVVA_VISION_GRAPHS") ||
                                    std::getenv("GEVVA_DECISION_GRAPHS"))) {
    auto signature = [&] {
      return std::vector<std::uint64_t>{0x564953494f4eULL,
          static_cast<std::uint64_t>(image_batch), static_cast<std::uint64_t>(valid_patches),
          static_cast<std::uint64_t>(patch_width), static_cast<std::uint64_t>(patch_height),
          static_cast<std::uint64_t>(fused_vision_attention), static_cast<std::uint64_t>(native_cudnn_layout),
          static_cast<std::uint64_t>(fused_projections), scratch->generation(),
          cudnn_plan_cache_stats().evictions, fp8_linears ? fp8_linears->instance_id() : 0,
          reinterpret_cast<std::uintptr_t>(scratch), reinterpret_cast<std::uintptr_t>(&model),
          reinterpret_cast<std::uintptr_t>(fused_weights),
          reinterpret_cast<std::uintptr_t>(d_pixels), reinterpret_cast<std::uintptr_t>(projected)};
    };
    auto key = signature();
    if (context->has_graph(key))
      microseconds = event_benchmark(stream, [&] { context->launch_graph(key); }, 0, 1);
    else {
      microseconds = event_benchmark(stream, launch, 0, 1);
      key = signature();
      if (context->should_capture_graph(key)) context->capture_graph(key, launch);
    }
  } else {
    microseconds = event_benchmark(stream, launch, 0, 1);
  }
  VisionEncoderBenchmark result{};
  std::vector<__nv_bfloat16> host_projected;
  if (copy_projected_to_host || !expected_projected.empty()) {
    host_projected.resize(static_cast<std::size_t>(soft_tokens) * kTextHidden);
    check(cudaMemcpy(host_projected.data(), projected, host_projected.size() * 2,
                     cudaMemcpyDeviceToHost),
          "copy projected vision tokens");
  }
  if (copy_projected_to_host) result.projected.resize(host_projected.size());
  double total_error = 0.0;
  for (std::size_t index = 0; index < host_projected.size(); ++index) {
    const float actual = __bfloat162float(host_projected[index]);
    if (copy_projected_to_host) result.projected[index] = actual;
    result.projected_checksum += actual;
    if (!expected_projected.empty()) {
      const float error = std::abs(actual - expected_projected[index]);
      result.projected_max_abs_error =
          std::max(result.projected_max_abs_error, error);
      total_error += error;
    }
  }
  if (!expected_projected.empty())
    result.projected_mean_abs_error =
        static_cast<float>(total_error / host_projected.size());
  result.microseconds = microseconds;
  if (owns_context) {
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
  }
  for (void* pointer : allocations) cudaFree(pointer);
  return result;
}

