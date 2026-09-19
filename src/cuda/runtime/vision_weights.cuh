// Private implementation fragment; included only by src/gpu.cu.
struct VisionFusedWeights::Impl {
  static constexpr int kLayers = 27;
  static constexpr int kHidden = 1152;
  static constexpr int kIntermediate = 4304;
  static constexpr std::size_t kQkvElements =
      3ULL * kHidden * kHidden;
  static constexpr std::size_t kGateUpElements =
      2ULL * kIntermediate * kHidden;

  explicit Impl(const DeviceModel& model) {
    const std::size_t layer_elements = kQkvElements + kGateUpElements;
    check(cudaMalloc(&storage,
                     kLayers * layer_elements * sizeof(__nv_bfloat16)),
          "cudaMalloc(fused vision weights)");
    auto* cursor = static_cast<__nv_bfloat16*>(storage);
    auto copy_weight = [&](const std::string& name, __nv_bfloat16* destination,
                           std::uint64_t rows, std::uint64_t columns) {
      const auto tensor = model.tensor(name);
      if (tensor.info->dtype != "BF16" ||
          tensor.info->shape != std::vector<std::uint64_t>{rows, columns})
        throw std::runtime_error("unexpected fused vision tensor: " + name);
      check(cudaMemcpy(destination, tensor.data,
                       static_cast<std::size_t>(rows * columns) *
                           sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToDevice),
            "pack fused vision weight");
    };
    try {
      for (int layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "model.vision_tower.encoder.layers." +
            std::to_string(layer) + ".";
        qkv_weights[layer] = cursor;
        copy_weight(prefix + "self_attn.q_proj.linear.weight", cursor,
                    kHidden, kHidden);
        cursor += static_cast<std::size_t>(kHidden) * kHidden;
        copy_weight(prefix + "self_attn.k_proj.linear.weight", cursor,
                    kHidden, kHidden);
        cursor += static_cast<std::size_t>(kHidden) * kHidden;
        copy_weight(prefix + "self_attn.v_proj.linear.weight", cursor,
                    kHidden, kHidden);
        cursor += static_cast<std::size_t>(kHidden) * kHidden;
        gate_up_weights[layer] = cursor;
        copy_weight(prefix + "mlp.gate_proj.linear.weight", cursor,
                    kIntermediate, kHidden);
        cursor += static_cast<std::size_t>(kIntermediate) * kHidden;
        copy_weight(prefix + "mlp.up_proj.linear.weight", cursor,
                    kIntermediate, kHidden);
        cursor += static_cast<std::size_t>(kIntermediate) * kHidden;
      }
    } catch (...) {
      cudaFree(storage);
      storage = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (storage) cudaFree(storage);
  }

  void* storage{};
  std::array<const void*, kLayers> qkv_weights{};
  std::array<const void*, kLayers> gate_up_weights{};
};

VisionFusedWeights::VisionFusedWeights(const DeviceModel& model)
    : impl_(std::make_unique<Impl>(model)) {}
VisionFusedWeights::~VisionFusedWeights() = default;

const void* VisionFusedWeights::qkv(int layer) const {
  if (layer < 0 || layer >= Impl::kLayers)
    throw std::runtime_error("invalid fused vision QKV layer");
  return impl_->qkv_weights[layer];
}

const void* VisionFusedWeights::gate_up(int layer) const {
  if (layer < 0 || layer >= Impl::kLayers)
    throw std::runtime_error("invalid fused vision gate/up layer");
  return impl_->gate_up_weights[layer];
}

