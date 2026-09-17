#include "g4/model_config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace g4 {

ModelConfig load_model_config(const std::filesystem::path& model_dir) {
  const auto path = model_dir / "config.json";
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  nlohmann::json root;
  input >> root;

  ModelConfig out;
  out.model_type = root.at("model_type").get<std::string>();
  out.dtype = root.value("dtype", "unknown");
  if (root.contains("architectures") && !root["architectures"].empty()) {
    out.architecture = root["architectures"][0].get<std::string>();
  }
  out.quantized = root.contains("quantization_config");
  if (out.quantized) out.quant_method = root["quantization_config"].value("quant_method", "modelopt-nvfp4");

  const auto& t = root.at("text_config");
  auto& c = out.text;
  c.vocab_size = t.at("vocab_size");
  c.hidden_size = t.at("hidden_size");
  c.intermediate_size = t.at("intermediate_size");
  c.moe_intermediate_size = t.at("moe_intermediate_size");
  c.num_layers = t.at("num_hidden_layers");
  c.num_attention_heads = t.at("num_attention_heads");
  c.num_kv_heads = t.at("num_key_value_heads");
  c.num_global_kv_heads = t.at("num_global_key_value_heads");
  c.head_dim = t.at("head_dim");
  c.num_experts = t.at("num_experts");
  c.top_k_experts = t.at("top_k_experts");
  c.sliding_window = t.at("sliding_window");
  c.max_positions = t.at("max_position_embeddings");
  c.attention_k_eq_v = t.at("attention_k_eq_v");
  c.moe = t.at("enable_moe_block");
  c.rms_norm_eps = t.at("rms_norm_eps");
  c.final_logit_softcap = t.at("final_logit_softcapping");
  c.layer_types = t.at("layer_types").get<std::vector<std::string>>();
  c.layer_head_dims.assign(c.num_layers, c.head_dim);
  c.layer_kv_heads.assign(c.num_layers, c.num_kv_heads);
  const auto global_head_dim = t.value("global_head_dim", c.head_dim);
  const auto global_kv_heads = t.value("num_global_key_value_heads", c.num_kv_heads);
  for (std::size_t i = 0; i < c.layer_types.size(); ++i) {
    if (c.layer_types[i] == "full_attention") {
      c.layer_head_dims[i] = global_head_dim;
      if (c.attention_k_eq_v) c.layer_kv_heads[i] = global_kv_heads;
    }
  }
  // Newer exported configs may materialize the same overrides by numeric layer.
  if (t.contains("per_layer_config")) {
    for (auto it = t["per_layer_config"].begin(); it != t["per_layer_config"].end(); ++it) {
      const auto layer = static_cast<std::size_t>(std::stoul(it.key()));
      if (layer >= c.num_layers) throw std::runtime_error("invalid per-layer config index");
      c.layer_head_dims[layer] = it.value().value("head_dim", c.layer_head_dims[layer]);
      c.layer_kv_heads[layer] = it.value().value("num_key_value_heads", c.layer_kv_heads[layer]);
    }
  }
  return out;
}

void validate_target_model(const ModelConfig& m) {
  const auto& c = m.text;
  const bool exact = m.model_type == "gemma4" && c.vocab_size == 262144 &&
      c.hidden_size == 2816 && c.num_layers == 30 && c.num_attention_heads == 16 &&
      c.num_kv_heads == 8 && c.num_global_kv_heads == 2 && c.head_dim == 256 &&
      c.num_experts == 128 && c.top_k_experts == 8 && c.sliding_window == 1024 &&
      c.moe && c.attention_k_eq_v && c.layer_types.size() == c.num_layers;
  const bool exact_layers = c.layer_head_dims.size() == c.num_layers &&
      c.layer_kv_heads.size() == c.num_layers && [&] {
        for (std::size_t i = 0; i < c.num_layers; ++i) {
          const bool full = c.layer_types[i] == "full_attention";
          if (c.layer_head_dims[i] != (full ? 512U : 256U)) return false;
          if (c.layer_kv_heads[i] != (full ? 2U : 8U)) return false;
        }
        return true;
      }();
  if (!exact || !exact_layers) {
    throw std::runtime_error("checkpoint is not the supported Gemma 4 26B-A4B architecture");
  }
}

}  // namespace g4
