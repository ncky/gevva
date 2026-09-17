// Private implementation fragment; included only by src/gpu.cu.
bool exact_attention_opt_enabled() {
  const char* setting = std::getenv("G4_EXACT_ATTENTION_OPT");
  return !setting || std::strcmp(setting, "0") != 0;
}

std::uint64_t experimental_expert_graph_key() {
  if (!std::getenv("G4_PREPARE_EXPERT_TILES")) return 0;
  if (std::getenv("G4_EXPERT_COOPERATIVE")) return 0x0030000000000000ULL;
  if (std::getenv("G4_EXPERT_WIDE_TILES")) return 0x0010000000000000ULL;
  if (std::getenv("G4_EXPERT_NARROW_TILES")) return 0x0020000000000000ULL;
  return 0;
}

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
  }
}

struct Bf16TargetLayerWeights {
  const __nv_bfloat16 *input_norm{}, *q{}, *k{}, *v{}, *o{}, *q_norm{},
      *k_norm{}, *post_attention{}, *dense_pre{}, *expert_pre{},
      *router_scale{}, *router{}, *expert_scale{}, *gate{}, *up{}, *down{},
      *dense_post{}, *expert_post{}, *combined_post{}, *layer_scalar{};
};

Bf16TargetLayerWeights target_layer_weights(
    const g4::RuntimeWeights& weights, int layer) {
  const auto& source = weights.layer(layer);
  const auto ptr = [](const void* value) {
    return static_cast<const __nv_bfloat16*>(value);
  };
  return {ptr(source.input_norm), ptr(source.q), ptr(source.k), ptr(source.v),
          ptr(source.o), ptr(source.q_norm), ptr(source.k_norm),
          ptr(source.post_attention), ptr(source.dense_pre),
          ptr(source.expert_pre), ptr(source.router_scale), ptr(source.router),
          ptr(source.expert_scale), ptr(source.gate), ptr(source.up),
          ptr(source.down), ptr(source.dense_post), ptr(source.expert_post),
          ptr(source.combined_post), ptr(source.layer_scalar)};
}

