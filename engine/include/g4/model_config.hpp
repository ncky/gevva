#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace g4 {

struct TextConfig {
  std::uint32_t vocab_size{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t moe_intermediate_size{};
  std::uint32_t num_layers{};
  std::uint32_t num_attention_heads{};
  std::uint32_t num_kv_heads{};
  std::uint32_t num_global_kv_heads{};
  std::uint32_t head_dim{};
  std::uint32_t num_experts{};
  std::uint32_t top_k_experts{};
  std::uint32_t sliding_window{};
  std::uint32_t max_positions{};
  bool attention_k_eq_v{};
  bool moe{};
  float rms_norm_eps{};
  float final_logit_softcap{};
  std::vector<std::string> layer_types;
  std::vector<std::uint32_t> layer_head_dims;
  std::vector<std::uint32_t> layer_kv_heads;
};

struct ModelConfig {
  std::string architecture;
  std::string model_type;
  std::string dtype;
  bool quantized{};
  std::string quant_method;
  TextConfig text;
};

ModelConfig load_model_config(const std::filesystem::path& model_dir);
void validate_target_model(const ModelConfig& config);

}  // namespace g4
