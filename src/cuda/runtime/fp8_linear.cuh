// Private implementation fragment; included only by src/gpu.cu.
struct Fp8LinearRunner::Impl {
  static std::uint64_t next_instance_id() {
    static std::atomic<std::uint64_t> next{0};
    return ++next;
  }
  const std::uint64_t instance_id = next_instance_id();
  struct Workspaces {
    std::array<void*, 3> pointers{};
    int count{};
    cublasLtHandle_t handle{};
    static constexpr std::size_t bytes = 64ULL << 20;
    explicit Workspaces(int requested) : count(requested) {
      try {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("create shared FP8 cuBLASLt handle failed");
        for (int i = 0; i < count; ++i)
          check(cudaMalloc(&pointers[i], bytes), "allocate shared FP8 workspace");
      } catch (...) {
        for (void* p : pointers) if (p) cudaFree(p);
        if (handle) cublasLtDestroy(handle);
        throw;
      }
    }
    ~Workspaces() {
      if (handle) cublasLtDestroy(handle);
      for (void* p : pointers) if (p) cudaFree(p);
    }
  };
  struct PackedPair {
    void* data{};
    ~PackedPair() { if (data) cudaFree(data); }
  };
  static std::shared_ptr<PackedPair> pack_pair(const void* gate, const void* up) {
    static std::mutex mutex;
    static std::unordered_map<const void*, std::weak_ptr<PackedPair>> cache;
    std::lock_guard lock(mutex);
    if (auto existing = cache[gate].lock()) return existing;
    auto result = std::make_shared<PackedPair>();
    constexpr std::size_t bytes = 2112 * 2816;
    check(cudaMalloc(&result->data, 2 * bytes), "allocate fused target gate/up");
    check(cudaMemcpy(result->data, gate, bytes, cudaMemcpyDeviceToDevice), "pack target gate");
    check(cudaMemcpy(static_cast<std::byte*>(result->data) + bytes, up, bytes,
                     cudaMemcpyDeviceToDevice), "pack target up");
    cache[gate] = result;
    return result;
  }
  struct Weight {
    const void* fp8{};
    const float* scale{};
    int outputs{};
    int inputs{};
    bool per_output_scale{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight_layout{};
    cublasLtMatrixLayout_t activation_layout{};
    cublasLtMatrixLayout_t output_layout{};
    cublasLtMatmulAlgo_t algorithm{};
    bool initialized{};
  };

  Impl(const DeviceModel& exact_model, const DeviceModel& fp8_model,
       int token_limit, bool assistant, int projection_tile_tokens,
       int workspace_count, bool vision, std::shared_ptr<Workspaces> shared_workspace)
      : maximum_tokens(token_limit),
        padded_tokens(projection_tile_tokens > 0
                          ? (projection_tile_tokens + 15) & ~15
                          : (token_limit + 15) & ~15),
        storage_tokens(((token_limit + padded_tokens - 1) / padded_tokens) *
                       padded_tokens),
        concurrent_workspaces(workspace_count) {
    if (maximum_tokens < 1 || maximum_tokens > 5040)
      throw std::runtime_error("FP8 linear runner supports 1..5040 tokens");
    if (padded_tokens < 16 || padded_tokens > storage_tokens)
      throw std::runtime_error("invalid FP8 projection tile size");
    if (concurrent_workspaces < 1 || concurrent_workspaces > 3)
      throw std::runtime_error("FP8 runner requires 1..3 workspaces");
    auto add_weight = [&](const std::string& exact_name,
                          const std::string& fp8_name,
                          const std::string& scale_name, bool optional) {
      try {
          const auto exact = exact_model.tensor(exact_name);
          const auto fp8 = fp8_model.tensor(fp8_name);
          const auto scale = fp8_model.tensor(scale_name);
          if (exact.info->shape.size() != 2 || exact.info->dtype != "BF16" ||
              fp8.info->shape != exact.info->shape ||
              (fp8.info->dtype != "F8_E4M3" && fp8.info->dtype != "U8") ||
              scale.info->dtype != "F32" ||
              (scale.info->shape != std::vector<std::uint64_t>{1} &&
               scale.info->shape !=
                   std::vector<std::uint64_t>{exact.info->shape[0]}))
            throw std::runtime_error("invalid FP8 dense tensor: " + fp8_name);
          weights.emplace(exact.data,
                          Weight{fp8.data,
                                 reinterpret_cast<const float*>(scale.data),
                                 static_cast<int>(exact.info->shape[0]),
                                 static_cast<int>(exact.info->shape[1]),
                                 scale.info->shape[0] != 1});
          maximum_width = std::max(
              maximum_width, static_cast<int>(exact.info->shape[1]));
          maximum_output = std::max(
              maximum_output, static_cast<int>(exact.info->shape[0]));
      } catch (const std::runtime_error&) {
        if (!optional) throw;
      }
    };
    if (vision) {
      auto add_vision = [&](const std::string& name) {
        add_weight(name, name + ".fp8", name + ".scale", false);
      };
      add_vision("model.vision_tower.patch_embedder.input_proj.weight");
      add_vision("model.embed_vision.embedding_projection.weight");
      constexpr std::array<const char*, 7> suffixes{
          "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight",
          "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight",
          "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight",
          "mlp.down_proj.linear.weight"};
      for (int layer = 0; layer < 27; ++layer)
        for (const auto* suffix : suffixes)
          add_vision("model.vision_tower.encoder.layers." +
                     std::to_string(layer) + "." + suffix);
    } else if (assistant) {
      for (const auto* name : {"pre_projection.weight", "post_projection.weight"})
        add_weight(name, std::string(name) + ".fp8",
                   std::string(name) + ".scale", false);
      constexpr std::array<const char*, 5> suffixes{
          "self_attn.q_proj.weight", "self_attn.o_proj.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight",
          "mlp.down_proj.weight"};
      for (int layer = 0; layer < 4; ++layer)
        for (const auto* suffix : suffixes) {
          const std::string name = "model.layers." + std::to_string(layer) +
                                   "." + suffix;
          add_weight(name, name + ".fp8", name + ".scale", false);
        }
    } else {
      constexpr std::array<const char*, 8> suffixes{
          "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
          "self_attn.o_proj", "router.proj", "mlp.gate_proj",
          "mlp.up_proj", "mlp.down_proj"};
      for (int layer = 0; layer < 30; ++layer)
        for (const auto* suffix : suffixes) {
          const std::string exact_name = "model.language_model.layers." +
              std::to_string(layer) + "." + suffix + ".weight";
          const std::string fp8_base = "layers." + std::to_string(layer) +
              "." + suffix;
          add_weight(exact_name, fp8_base + ".weight", fp8_base + ".scale",
                     std::string_view(suffix) == "self_attn.v_proj");
        }
    }
    const std::size_t expected_weights = vision ? 191 : (assistant ? 22 : 235);
    if (weights.size() != expected_weights)
      throw std::runtime_error("FP8 dense sidecar projection count mismatch");
    if (!vision && !assistant &&
        (std::getenv("GEVVA_FUSED_TARGET_GATE_UP") || std::getenv("GEVVA_PREPARE_FUSED_TARGET_GATE_UP"))) {
      for (int layer = 0; layer < 30; ++layer) {
        const auto prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        const auto* gate = exact_model.tensor(prefix + "gate_proj.weight").data;
        const auto* up = exact_model.tensor(prefix + "up_proj.weight").data;
        const auto gate_scale = weights.at(gate).scale;
        auto packed = pack_pair(weights.at(gate).fp8, weights.at(up).fp8);
        paired_keys[gate] = packed->data;
        weights.emplace(packed->data, Weight{packed->data, gate_scale, 4224, 2816, false});
        packed_pairs.push_back(std::move(packed));
      }
    }
    check(cudaMalloc(&activation,
                     static_cast<std::size_t>(storage_tokens) * maximum_width),
          "cudaMalloc(FP8 activation)");
    check(cudaMalloc(&activation_scales, storage_tokens * sizeof(float)),
          "cudaMalloc(FP8 activation scales)");
    std::vector<float> initial_scales(storage_tokens, 1.0F);
    check(cudaMemcpy(activation_scales, initial_scales.data(),
                     initial_scales.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "initialize FP8 activation scales");
    check(cudaMalloc(&padded_output,
                     static_cast<std::size_t>(concurrent_workspaces) *
                         padded_tokens *
                         maximum_output * sizeof(__nv_bfloat16)),
          "cudaMalloc(FP8 padded output)");
    check(cudaMalloc(&secondary_activation,
                     static_cast<std::size_t>(storage_tokens) * maximum_width),
          "cudaMalloc(secondary FP8 activation)");
    check(cudaMalloc(&secondary_activation_scales,
                     storage_tokens * sizeof(float)),
          "cudaMalloc(secondary FP8 activation scales)");
    workspace_owner = shared_workspace ? std::move(shared_workspace)
                                       : std::make_shared<Workspaces>(concurrent_workspaces);
    if (workspace_owner->count < concurrent_workspaces)
      throw std::runtime_error("shared FP8 workspace count is too small");
    workspace = workspace_owner->pointers[0];
    auxiliary_workspace = workspace_owner->pointers[1];
    tertiary_workspace = workspace_owner->pointers[2];
    check(cudaMalloc(&unit_scale, sizeof(float)),
          "cudaMalloc(FP8 unit scale)");
    const float one = 1.0F;
    check(cudaMemcpy(unit_scale, &one, sizeof(one), cudaMemcpyHostToDevice),
          "copy FP8 unit scale");
    lt_handle = workspace_owner->handle;
    check(cudaEventCreateWithFlags(&activation_ready, cudaEventDisableTiming),
          "create FP8 activation ready event");
    check(cudaEventCreateWithFlags(&secondary_activation_ready,
                                   cudaEventDisableTiming),
          "create secondary FP8 activation ready event");
  }

  ~Impl() {
    for (auto& [_, weight] : weights) {
      if (weight.output_layout) cublasLtMatrixLayoutDestroy(weight.output_layout);
      if (weight.activation_layout)
        cublasLtMatrixLayoutDestroy(weight.activation_layout);
      if (weight.weight_layout) cublasLtMatrixLayoutDestroy(weight.weight_layout);
      if (weight.operation) cublasLtMatmulDescDestroy(weight.operation);
    }
    if (secondary_activation_ready) cudaEventDestroy(secondary_activation_ready);
    if (activation_ready) cudaEventDestroy(activation_ready);
    if (unit_scale) cudaFree(unit_scale);
    if (padded_output) cudaFree(padded_output);
    if (secondary_activation_scales) cudaFree(secondary_activation_scales);
    if (secondary_activation) cudaFree(secondary_activation);
    if (activation_scales) cudaFree(activation_scales);
    if (activation) cudaFree(activation);
  }

  bool launch(const void* exact_weight, int outputs, int inputs,
              const void* source, void* destination, int tokens,
              cudaStream_t stream, cublasHandle_t,
              bool rescale_output, int workspace_index,
              bool output_has_padded_capacity) const {
    const auto found = weights.find(exact_weight);
    if (found == weights.end()) return false;
    auto& weight = found->second;
    if (weight.outputs != outputs || weight.inputs != inputs || tokens < 1 ||
        tokens > maximum_tokens || workspace_index < 0 ||
        workspace_index >= concurrent_workspaces)
      throw std::runtime_error("FP8 dense launch geometry mismatch");
    const bool reuse_activation = source == last_source &&
                                  tokens == last_tokens &&
                                  inputs == last_inputs;
    if (!reuse_activation) {
      quantize_fp8_rows_kernel<<<tokens, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(source), activation,
          activation_scales, tokens, inputs);
      check(cudaEventRecord(activation_ready, stream),
            "record FP8 activation readiness");
      activation_stream = stream;
    } else if (stream != activation_stream) {
      check(cudaStreamWaitEvent(stream, activation_ready),
            "wait for shared FP8 activation");
    }
    last_source = source;
    last_tokens = tokens;
    last_inputs = inputs;
    const float alpha = 1.0F, beta = 0.0F;
    void* selected_workspace = workspace_index == 2
        ? tertiary_workspace
        : (workspace_index == 1 ? auxiliary_workspace : workspace);
    auto* output_base = static_cast<__nv_bfloat16*>(destination);
    bool first_tile_launched = false;
    if (!weight.initialized) {
      if (cublasLtMatmulDescCreate(&weight.operation, CUBLAS_COMPUTE_32F,
                                   CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("create FP8 matmul descriptor failed");
      const cublasOperation_t transpose = CUBLAS_OP_T;
      if (cublasLtMatmulDescSetAttribute(
              weight.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
              sizeof(transpose)) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatmulDescSetAttribute(
              weight.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
              &unit_scale, sizeof(unit_scale)) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatmulDescSetAttribute(
              weight.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
              &unit_scale, sizeof(unit_scale)) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatrixLayoutCreate(&weight.weight_layout, CUDA_R_8F_E4M3,
                                     inputs, outputs, inputs) !=
              CUBLAS_STATUS_SUCCESS ||
          cublasLtMatrixLayoutCreate(&weight.activation_layout,
                                     CUDA_R_8F_E4M3, inputs, padded_tokens,
                                     inputs) !=
              CUBLAS_STATUS_SUCCESS ||
          cublasLtMatrixLayoutCreate(&weight.output_layout, CUDA_R_16BF,
                                     outputs, padded_tokens, outputs) !=
              CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("create FP8 matrix layout failed");
      const std::uint64_t algorithm_key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(outputs))
           << 32) |
          static_cast<std::uint32_t>(inputs);
      auto* candidate_output = output_has_padded_capacity ||
                               tokens >= padded_tokens
          ? output_base
          : padded_output + static_cast<std::size_t>(workspace_index) *
                                padded_tokens * maximum_output;
      int returned = 0;
      cublasStatus_t last_candidate_status = CUBLAS_STATUS_SUCCESS;
      if (const auto cached = algorithms.find(algorithm_key);
          cached != algorithms.end()) {
        last_candidate_status = cublasLtMatmul(
            lt_handle, weight.operation, &alpha, weight.fp8,
            weight.weight_layout, activation, weight.activation_layout, &beta,
            nullptr, weight.output_layout, candidate_output,
            weight.output_layout, &cached->second, selected_workspace,
            workspace_bytes, stream);
        if (last_candidate_status == CUBLAS_STATUS_SUCCESS) {
          weight.algorithm = cached->second;
          first_tile_launched = true;
        }
      }
      if (!first_tile_launched) {
      cublasLtMatmulPreference_t preference{};
      if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatmulPreferenceSetAttribute(
              preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &workspace_bytes, sizeof(workspace_bytes)) !=
              CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("create FP8 matmul preference failed");
      std::array<cublasLtMatmulHeuristicResult_t, 32> heuristics{};
      const char* algorithm_start_env = std::getenv("GEVVA_FP8_ALGO_START");
      int algorithm_start =
          algorithm_start_env ? std::max(0, std::atoi(algorithm_start_env))
                              : (padded_tokens == 64 && outputs == 2816 &&
                                         inputs == 2112
                                     ? 1
                                     : 0);
      if (!algorithm_start_env && padded_tokens == 64 &&
          std::getenv("GEVVA_TUNED_B8_PROJECTIONS")) {
        if (outputs == 4096 && inputs == 2816) algorithm_start = 2;
        if (outputs == 2816 && inputs == 4096) algorithm_start = 3;
      }
      const auto status = cublasLtMatmulAlgoGetHeuristic(
          lt_handle, weight.operation, weight.weight_layout,
          weight.activation_layout, weight.output_layout,
          weight.output_layout, preference, heuristics.size(),
          heuristics.data(), &returned);
      cublasLtMatmulPreferenceDestroy(preference);
      if (status != CUBLAS_STATUS_SUCCESS || returned < 1)
        throw std::runtime_error("no FP8 cuBLASLt algorithm for projection");
      const bool tune = tokens <= padded_tokens &&
          ((padded_tokens <= 64 && std::getenv("GEVVA_TUNE_FP8_PROJECTIONS")) ||
           (padded_tokens > 64 && std::getenv("GEVVA_TUNE_FP8_PREFILL")));
      float best_ms = INFINITY;
      int best_index = -1;
      std::vector<__nv_bfloat16> reference_output;
      // Tuning is an opt-in plan-creation operation, never part of steady decode.
      // Only alternatives that reproduce the current projection bytes qualify.
      for (int index = algorithm_start; index < returned; ++index) {
        // The heuristic's first choice for these M=16 FP8 projections often
        // uses split-K.  At decode sizes its tiny reduction kernel costs more
        // than the extra parallelism and adds one launch per projection.
        int split_k = 0;
        std::size_t split_k_size = 0;
        if (cublasLtMatmulAlgoConfigGetAttribute(
                &heuristics[index].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                &split_k, sizeof(split_k), &split_k_size) !=
                CUBLAS_STATUS_SUCCESS)
          continue;
        if (split_k_size == sizeof(split_k) && split_k > 1) {
          split_k = 1;
          if (cublasLtMatmulAlgoConfigSetAttribute(
                  &heuristics[index].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                  &split_k, sizeof(split_k)) != CUBLAS_STATUS_SUCCESS)
            continue;
        }
        const auto candidate_status = cublasLtMatmul(
            lt_handle, weight.operation, &alpha, weight.fp8,
            weight.weight_layout, activation, weight.activation_layout, &beta,
            nullptr, weight.output_layout, candidate_output,
            weight.output_layout,
            &heuristics[index].algo, selected_workspace, workspace_bytes,
            stream);
        last_candidate_status = candidate_status;
        if (candidate_status == CUBLAS_STATUS_SUCCESS) {
          if (tune) {
            std::vector<__nv_bfloat16> observed(static_cast<std::size_t>(tokens) * outputs);
            check(cudaMemcpyAsync(observed.data(), candidate_output, observed.size() * 2,
                                  cudaMemcpyDeviceToHost, stream), "copy FP8 tuning result");
            check(cudaStreamSynchronize(stream), "synchronize FP8 tuning result");
            if (reference_output.empty()) reference_output = observed;
            else if (std::memcmp(reference_output.data(), observed.data(), observed.size() * 2))
              continue;
            cudaEvent_t begin{}, end{};
            check(cudaEventCreate(&begin), "create FP8 tuning start");
            check(cudaEventCreate(&end), "create FP8 tuning end");
            check(cudaEventRecord(begin, stream), "record FP8 tuning start");
            for (int repeat = 0; repeat < 8; ++repeat)
              check(cublasLtMatmul(lt_handle, weight.operation, &alpha, weight.fp8,
                    weight.weight_layout, activation, weight.activation_layout, &beta,
                    nullptr, weight.output_layout, candidate_output, weight.output_layout,
                    &heuristics[index].algo, selected_workspace, workspace_bytes, stream),
                    "time FP8 projection candidate");
            check(cudaEventRecord(end, stream), "record FP8 tuning end");
            check(cudaEventSynchronize(end), "wait FP8 tuning end");
            float elapsed{};
            check(cudaEventElapsedTime(&elapsed, begin, end), "FP8 tuning elapsed");
            cudaEventDestroy(end);
            cudaEventDestroy(begin);
            if (elapsed < best_ms) {
              best_ms = elapsed;
              best_index = index;
              weight.algorithm = heuristics[index].algo;
            }
            first_tile_launched = true;
            continue;
          }
          weight.algorithm = heuristics[index].algo;
          algorithms[algorithm_key] = weight.algorithm;
          first_tile_launched = true;
          break;
        }
      }
      if (tune && first_tile_launched) {
        algorithms[algorithm_key] = weight.algorithm;
        // The last candidate might have failed the byte comparison; restore
        // the selected result before its downstream consumer runs.
        check(cublasLtMatmul(lt_handle, weight.operation, &alpha, weight.fp8,
              weight.weight_layout, activation, weight.activation_layout, &beta,
              nullptr, weight.output_layout, candidate_output, weight.output_layout,
              &weight.algorithm, selected_workspace, workspace_bytes, stream),
              "restore tuned FP8 projection output");
        std::fprintf(stderr, "fp8_tune m=%d n=%d k=%d index=%d us=%.3f\n",
                     padded_tokens, outputs, inputs, best_index, best_ms * 125.0F);
      }
      if (!first_tile_launched)
        throw std::runtime_error(
            "no runnable FP8 cuBLASLt algorithm for projection " +
            std::to_string(outputs) + "x" + std::to_string(inputs) +
            " at padded M=" + std::to_string(padded_tokens) +
            ", candidates=" + std::to_string(returned) +
            ", last status=" +
            std::to_string(static_cast<int>(last_candidate_status)) +
            ", dst alignment=" + std::to_string(
                reinterpret_cast<std::uintptr_t>(destination) & 255));
      }
      weight.initialized = true;
    }
    // Reuse one tensor-core-friendly M shape for arbitrarily long prefill.
    // A partial final tile lands in private padding so callers only need to
    // allocate their active rows.
    for (int row = first_tile_launched ? padded_tokens : 0; row < tokens;
         row += padded_tokens) {
      const int active_rows = std::min(padded_tokens, tokens - row);
      auto* destination_row = output_base +
                              static_cast<std::size_t>(row) * outputs;
      auto* tile_output = output_has_padded_capacity ||
                          active_rows == padded_tokens
          ? destination_row
          : padded_output + static_cast<std::size_t>(workspace_index) *
                                padded_tokens * maximum_output;
      const auto matmul_status = cublasLtMatmul(
          lt_handle, weight.operation, &alpha, weight.fp8,
          weight.weight_layout, activation + static_cast<std::size_t>(row) * inputs,
          weight.activation_layout, &beta, nullptr, weight.output_layout,
          tile_output,
          weight.output_layout, &weight.algorithm, selected_workspace,
          workspace_bytes, stream);
      if (matmul_status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(
            "FP8 cuBLASLt target projection failed: status " +
          std::to_string(static_cast<int>(matmul_status)));
      if (active_rows != padded_tokens && !output_has_padded_capacity)
        check(cudaMemcpyAsync(destination_row, tile_output,
                              static_cast<std::size_t>(active_rows) * outputs *
                                  sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice, stream),
              "copy partial FP8 projection tile");
    }
    if (first_tile_launched && tokens < padded_tokens &&
        !output_has_padded_capacity)
      check(cudaMemcpyAsync(
                output_base,
                padded_output + static_cast<std::size_t>(workspace_index) *
                                    padded_tokens * maximum_output,
                static_cast<std::size_t>(tokens) * outputs *
                    sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream),
            "copy initial partial FP8 projection tile");
    if (rescale_output)
      scale_fp8_gemm_output_kernel<<<
          (tokens * outputs + 255) / 256, 256, 0, stream>>>(
          static_cast<__nv_bfloat16*>(destination), activation_scales,
          weight.scale, tokens, outputs, weight.per_output_scale);
    return true;
  }

  bool launch_unscaled(const void* exact_weight, int outputs, int inputs,
                       const void* source, void* destination, int tokens,
                       cudaStream_t stream, cublasHandle_t handle,
                       Fp8OutputScale& scale,
                       int workspace_index,
                       bool output_has_padded_capacity) const {
    if (!launch(exact_weight, outputs, inputs, source, destination, tokens,
                stream, handle, false, workspace_index,
                output_has_padded_capacity)) {
      scale = {};
      return false;
    }
    const auto found = weights.find(exact_weight);
    scale = {activation_scales, found->second.scale};
    return true;
  }

  void prepare_rmsnorm_2816(const void* source, const void* norm_weight,
                            const void* cache_key, int tokens,
                            cudaStream_t stream) const {
    if (tokens < 1 || tokens > maximum_tokens || maximum_width < 2816)
      throw std::runtime_error("FP8 RMSNorm preparation geometry mismatch");
    rmsnorm_quantize_fp8_2816_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source),
        static_cast<const __nv_bfloat16*>(norm_weight), activation,
        activation_scales, tokens);
    check(cudaEventRecord(activation_ready, stream),
          "record FP8 RMSNorm activation readiness");
    activation_stream = stream;
    last_source = cache_key;
    last_tokens = tokens;
    last_inputs = 2816;
  }

  void prepare_dual_rmsnorm_router_2816(
      const void* source, const void* dense_weight, const void* expert_weight,
      const void* router_weight, void* dense_output, void* expert_output,
      const void* cache_key, int tokens, cudaStream_t stream,
      int threads, const void* attention, const void* attention_weight,
      void* residual_output, const float* attention_scales,
      const float* attention_projection_scale) const {
    if (tokens < 1 || tokens > maximum_tokens || maximum_width < 2816)
      throw std::runtime_error("FP8 router preparation geometry mismatch");
    if (threads != 128 && threads != 256 && threads != 512)
      throw std::runtime_error("invalid FP8 router preparation threads");
    if (attention)
      dual_rmsnorm_router_quantize_fp8_2816_kernel<true><<<
          tokens, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(source),
          static_cast<const __nv_bfloat16*>(dense_weight),
          static_cast<const __nv_bfloat16*>(expert_weight),
          static_cast<const __nv_bfloat16*>(router_weight),
          static_cast<__nv_bfloat16*>(dense_output),
          static_cast<__nv_bfloat16*>(expert_output), activation,
          activation_scales, secondary_activation, secondary_activation_scales,
          tokens, static_cast<const __nv_bfloat16*>(attention),
          static_cast<const __nv_bfloat16*>(attention_weight),
          static_cast<__nv_bfloat16*>(residual_output), attention_scales,
          attention_projection_scale);
    else
    dual_rmsnorm_router_quantize_fp8_2816_kernel<false><<<
        tokens, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source),
        static_cast<const __nv_bfloat16*>(dense_weight),
        static_cast<const __nv_bfloat16*>(expert_weight),
        static_cast<const __nv_bfloat16*>(router_weight),
        static_cast<__nv_bfloat16*>(dense_output),
        static_cast<__nv_bfloat16*>(expert_output), activation,
        activation_scales, secondary_activation,
        secondary_activation_scales, tokens);
    check(cudaEventRecord(activation_ready, stream),
          "record FP8 router activation readiness");
    activation_stream = stream;
    secondary_activation_stream = stream;
    last_source = cache_key;
    last_tokens = tokens;
    last_inputs = 2816;
  }

  void use_secondary_activation_2816(const void* cache_key, int tokens,
                                     cudaStream_t dependency_ordered_stream) const {
    if (tokens < 1 || tokens > maximum_tokens)
      throw std::runtime_error("secondary FP8 activation geometry mismatch");
    std::swap(activation, secondary_activation);
    std::swap(activation_scales, secondary_activation_scales);
    std::swap(activation_ready, secondary_activation_ready);
    // Callers fork dependency_ordered_stream from the producer stream after
    // prepare_dual_rmsnorm_router_2816.  Re-recording and waiting on the
    // per-runner event here duplicated that stronger stream dependency.
    activation_stream = dependency_ordered_stream;
    last_source = cache_key;
    last_tokens = tokens;
    last_inputs = 2816;
  }

  void prepare_gelu_2112(const void* gate, const void* up,
                         Fp8OutputScale gate_scale,
                         Fp8OutputScale up_scale,
                         const void* cache_key, int tokens,
                         cudaStream_t stream, bool packed) const {
    if (tokens < 1 || tokens > maximum_tokens || maximum_width < 2112)
      throw std::runtime_error("FP8 GeGLU preparation geometry mismatch");
    gelu_quantize_fp8_2112_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up), secondary_activation,
        secondary_activation_scales, tokens, gate_scale.activation_rows,
        gate_scale.weight, up_scale.activation_rows, up_scale.weight, packed ? 4224 : 2112);
    std::swap(activation, secondary_activation);
    std::swap(activation_scales, secondary_activation_scales);
    std::swap(activation_ready, secondary_activation_ready);
    // The down projection is enqueued immediately on this same stream.
    activation_stream = stream;
    last_source = cache_key;
    last_tokens = tokens;
    last_inputs = 2112;
  }

  int maximum_tokens{};
  int padded_tokens{};
  int storage_tokens{};
  int concurrent_workspaces{};
  int maximum_width{};
  int maximum_output{};
  mutable __nv_fp8_e4m3* activation{};
  mutable float* activation_scales{};
  mutable __nv_fp8_e4m3* secondary_activation{};
  mutable float* secondary_activation_scales{};
  __nv_bfloat16* padded_output{};
  std::shared_ptr<Workspaces> workspace_owner;
  void* workspace{};
  void* auxiliary_workspace{};
  void* tertiary_workspace{};
  float* unit_scale{};
  mutable cudaEvent_t activation_ready{};
  mutable cudaEvent_t secondary_activation_ready{};
  mutable cudaStream_t activation_stream{};
  mutable cudaStream_t secondary_activation_stream{};
  std::size_t workspace_bytes{64ULL << 20};
  cublasLtHandle_t lt_handle{};
  mutable const void* last_source{};
  mutable int last_tokens{};
  mutable int last_inputs{};
  mutable std::unordered_map<const void*, Weight> weights;
  std::unordered_map<const void*, const void*> paired_keys;
  std::vector<std::shared_ptr<PackedPair>> packed_pairs;
  mutable std::unordered_map<std::uint64_t, cublasLtMatmulAlgo_t> algorithms;
};

Fp8LinearRunner::Fp8LinearRunner(const DeviceModel& exact_model,
                                 const DeviceModel& fp8_model,
                                 int maximum_tokens, bool assistant,
                                 int projection_tile_tokens,
                                 int concurrent_workspaces, bool vision,
                                 Fp8LinearRunner* shared_workspace)
    : impl_(std::make_unique<Impl>(exact_model, fp8_model, maximum_tokens,
                                  assistant, projection_tile_tokens,
                                  concurrent_workspaces, vision,
                                  shared_workspace ? shared_workspace->impl_->workspace_owner : nullptr)) {}
int Fp8LinearRunner::padded_output_rows(int tokens) const {
  return ((tokens + impl_->padded_tokens - 1) / impl_->padded_tokens) * impl_->padded_tokens;
}
std::uint64_t Fp8LinearRunner::instance_id() const { return impl_->instance_id; }
Fp8LinearRunner::~Fp8LinearRunner() = default;

void Fp8LinearRunner::invalidate_activation_cache() const {
  impl_->last_source = nullptr;
  impl_->last_tokens = 0;
  impl_->last_inputs = 0;
}

bool Fp8LinearRunner::launch(const void* exact_weight, int outputs, int inputs,
                             const void* source, void* output, int tokens,
                             cudaStream_t stream, void* blas_handle,
                             bool rescale_output,
                             int workspace_index,
                             bool output_has_padded_capacity) const {
  return impl_->launch(exact_weight, outputs, inputs, source, output, tokens,
                       stream, static_cast<cublasHandle_t>(blas_handle),
                       rescale_output, workspace_index,
                       output_has_padded_capacity);
}

bool Fp8LinearRunner::launch_unscaled(
    const void* exact_weight, int outputs, int inputs, const void* source,
    void* output, int tokens, cudaStream_t stream, void* blas_handle,
    Fp8OutputScale& scale, int workspace_index,
    bool output_has_padded_capacity) const {
  return impl_->launch_unscaled(
      exact_weight, outputs, inputs, source, output, tokens, stream,
      static_cast<cublasHandle_t>(blas_handle), scale, workspace_index,
      output_has_padded_capacity);
}

void Fp8LinearRunner::prepare_rmsnorm_2816(
    const void* input, const void* norm_weight,
    const void* activation_cache_key, int tokens, cudaStream_t stream) const {
  impl_->prepare_rmsnorm_2816(
      input, norm_weight, activation_cache_key, tokens, stream);
}

