#include "gevva/decisions.hpp"
#include "gevva/cudnn_attention.hpp"
#include "gevva/gpu.hpp"
#include "gevva/runtime.hpp"
#include "gevva/multimodal.hpp"
#include "gevva/image.hpp"
#include "gevva/image_gpu.hpp"
#include "gevva/tokenizer.hpp"

#include <nlohmann/json.hpp>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <malloc.h>
#include <numeric>
#include <set>
#include <stdexcept>

namespace gevva {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr int kHidden = 2816, kVocab = 262144, kChunk = 4608;
struct ProfileRange {
  explicit ProfileRange(const char* name) { nvtxRangePushA(name); }
  ~ProfileRange() { nvtxRangePop(); }
};
struct RequestError : std::runtime_error { using std::runtime_error::runtime_error; };
struct DecisionModelPaths {
  std::filesystem::path target, experts, vocab, dense;
};
DecisionModelPaths decision_model_paths() {
  const char* configured = std::getenv("GEVVA_MODEL_ROOT");
  const std::filesystem::path root = configured ? configured : "models";
  auto component = [&](const char* variable, const char* directory) {
    const char* override = std::getenv(variable);
    const auto path = override ? std::filesystem::path(override) : root / directory;
    if (!std::filesystem::is_directory(path))
      throw std::runtime_error(std::string("Missing model directory for ") + variable + ": " +
          path.string() + ". Launch with a Gevva TOML config or set GEVVA_MODEL_ROOT.");
    return path;
  };
  return {component("GEVVA_MODEL_TARGET", "gemma4-26b-a4b-nvfp4"),
          component("GEVVA_MODEL_EXPERTS", "gemma4-26b-a4b-trtllm"),
          component("GEVVA_MODEL_VOCAB", "gemma4-26b-a4b-target-vocab-int8"),
          component("GEVVA_MODEL_DENSE", "gemma4-26b-a4b-dense-fp8")};
}

double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

std::vector<std::uint8_t> image_bytes(const std::string& source) {
  constexpr std::size_t limit = 32ULL * 1024 * 1024;
  if (source.starts_with("data:")) {
    const auto marker = source.find(";base64,");
    if (!source.starts_with("data:image/") || marker == std::string::npos || source.size() > limit * 4 / 3 + 256)
      throw RequestError("image data URL must contain base64 image data, at most 32 MiB");
    std::vector<std::uint8_t> result;
    result.reserve((source.size() - marker - 8) * 3 / 4);
    unsigned accumulator = 0;
    int bits = 0;
    bool padding = false;
    for (const unsigned char ch : std::string_view(source).substr(marker + 8)) {
      if (ch == '=') { padding = true; continue; }
      if (padding) throw RequestError("invalid base64 padding");
      const int value = ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26 :
                        ch >= '0' && ch <= '9' ? ch - '0' + 52 : ch == '+' ? 62 : ch == '/' ? 63 : -1;
      if (value < 0) throw RequestError("invalid base64 image");
      accumulator = (accumulator << 6) | static_cast<unsigned>(value);
      bits += 6;
      if (bits >= 8) { bits -= 8; result.push_back(static_cast<std::uint8_t>(accumulator >> bits)); accumulator &= (1U << bits) - 1U; }
    }
    if (result.empty() || result.size() > limit) throw RequestError("empty or oversized image");
    return result;
  }
  std::ifstream input(source, std::ios::binary | std::ios::ate);
  if (!input) throw RequestError("cannot open image: " + source);
  const auto size = input.tellg();
  if (size <= 0 || static_cast<std::uint64_t>(size) > limit) throw RequestError("image must contain 1 byte..32 MiB");
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(result.data()), result.size())) throw RequestError("cannot read image");
  return result;
}

struct Question {
  std::string id, type, instruction;
  std::vector<std::string> options;
  std::vector<double> levels;
  std::vector<std::uint32_t> prompt, aliases;
  std::vector<std::uint8_t> types;
};

class DecisionWorker {
 public:
  explicit DecisionWorker(const DecisionModelPaths& paths)
      : tokenizer_(paths.target),
        weights_(paths.target, paths.experts,
                 std::getenv("GEVVA_DECISION_FULL_WEIGHTS") == nullptr),
        compressed_(paths.vocab),
        dense_(paths.dense),
        vocab_(weights_.model(), "model.language_model.embed_tokens.weight", compressed_, context_.stream()) {
    std::ifstream config(paths.target / "config.json");
    const auto text_config = Json::parse(config).at("text_config");
    if (text_config.at("final_logit_softcapping") != 30.0 ||
        text_config.at("hidden_size") != kHidden || text_config.at("vocab_size") != kVocab)
      throw std::runtime_error("unsupported decision model configuration");
    // The rendered prompt ends at this special token, an exact BPE boundary.
    // Verify aliases once here rather than retokenizing the entire state for
    // every candidate of every question.
    const auto boundary = tokenizer_.encode("<channel|>");
    if (boundary.size() != 1) throw std::runtime_error("missing answer boundary token");
    answer_boundary_ = boundary.front();
    auto add_alias = [&](const std::string& label) {
      const auto encoded = tokenizer_.encode("<channel|>" + label);
      if (encoded.size() != 2 || encoded.front() != answer_boundary_ || encoded.back() >= kVocab) return;
      if (std::find(alias_ids_.begin(), alias_ids_.end(), encoded.back()) != alias_ids_.end()) return;
      alias_ids_.push_back(encoded.back()); alias_labels_.push_back(label);
    };
    for (char label = 'A'; label <= 'Z'; ++label) add_alias(std::string(1, label));
    for (char a = 'A'; a <= 'Z' && alias_ids_.size() < 255; ++a)
      for (char b = 'A'; b <= 'Z' && alias_ids_.size() < 255; ++b) add_alias(std::string{a, b});
    if (alias_ids_.size() != 255) throw std::runtime_error("tokenizer lacks 255 distinct single-token option labels");
    decision_head_ = prepare_decision_head(weights_, alias_ids_, context_, head_storage_);
    for (int layer = 0; layer < 30; ++layer) {
      expert_owners_.push_back(std::make_unique<Nvfp4ExpertRunner>(weights_.experts(), layer, kChunk,
          layer && !std::getenv("GEVVA_DECISION_PRIVATE_WORKSPACES") ? experts_[0] : nullptr));
      experts_[layer] = expert_owners_.back().get();
    }
  }

  Json run(const Json& request) {
    ProfileRange profile("decision/request");
    const auto started = Clock::now();
    const auto mode = request.value("prefix_mode", std::string("readonly"));
    if (mode != "copy" && mode != "none" && mode != "shared" && mode != "readonly") throw RequestError("prefix_mode must be readonly, copy, shared, or none");
    const int batch_size = request.value("batch_size", mode == "readonly" ? 64 : 8);
    if (batch_size < 1 || batch_size > (mode == "readonly" ? 256 : 8)) throw RequestError("batch_size must be 1..256 for readonly, 1..8 otherwise");
    const auto readout_mode = request.value("readout_mode", std::string("full"));
    if (readout_mode != "full" && readout_mode != "options") throw RequestError("readout_mode must be full or options");
    const double temperature = request.value("temperature", 1.0);
    if (!std::isfinite(temperature) || temperature <= 0)
      throw RequestError("temperature must be finite and positive");
    const auto& raw_questions = request.at("questions");
    if (!raw_questions.is_array() || raw_questions.empty() || raw_questions.size() > 256)
      throw RequestError("questions must contain 1..256 entries");
    const auto& state_value = request.at("state");
    const std::string state = state_value.is_string() ? state_value.get<std::string>() : state_value.dump();
    const std::string directive =
        "Read the following state as evidence. Answer the question by selecting exactly one "
        "of its listed options. Respond with only the option label, without explanation.\n\n";
    const std::string shared = directive + "STATE:\n" + state + "\n\n";
    const auto layout = request.value("prompt_layout", std::string("system_state"));
    const bool compact = request.value("compact_prompt", true);
    if (layout != "system_state" && layout != "user_state")
      throw RequestError("prompt_layout must be system_state or user_state");
    const auto image_metrics = prepare_images(request);
    const bool has_images = image_metrics.at("image_count").get<int>() != 0;
    if (has_images && layout != "system_state") throw RequestError("images require system_state layout");
    std::string framed_state = "<bos><|turn>system\n" + shared + "<turn|>";
    if (has_images) {
      framed_state = "<bos><|turn>system\n" + directive + "<turn|>\n<|turn>user\nSTATE:\n" + state;
      for (std::size_t i = 0; i < image_counts_.size(); ++i)
        framed_state += "\nImage " + std::to_string(i + 1) + ": <|image|>";
    }
    auto state_ids = layout == "system_state" ? tokenizer_.encode(framed_state) : std::vector<std::uint32_t>{};
    std::vector<std::uint8_t> state_types(state_ids.size(), 0);
    if (has_images) {
      auto multimodal = prepare_image_prompt(tokenizer_, framed_state, image_counts_);
      state_ids = std::move(multimodal.input_ids);
      state_types = std::move(multimodal.multimodal_types);
    }
    std::vector<Question> questions;
    std::set<std::string> ids;
    for (const auto& raw : raw_questions) {
      Question q;
      q.id = raw.at("id").get<std::string>();
      if (q.id.empty() || !ids.insert(q.id).second) throw RequestError("question IDs must be nonempty and unique");
      q.type = raw.value("type", std::string("choice"));
      q.instruction = raw.at("question").get<std::string>();
      if (q.instruction.empty()) throw RequestError("question must not be empty");
      if (q.type == "boolean") {
        if (raw.contains("options")) throw RequestError("boolean options are fixed to false/true");
        q.options = {"False", "True"};
      } else if (q.type == "choice" || q.type == "score") {
        q.options = raw.at("options").get<std::vector<std::string>>();
      } else throw RequestError("type must be choice, boolean, or score");
      if (q.options.empty() || q.options.size() > 255)
        throw RequestError("each question requires 1..255 options");
      for (const auto& option : q.options)
        if (option.empty())
          throw RequestError("options must be nonempty");
      if (q.type == "score") {
        q.levels = raw.value("levels", std::vector<double>{});
        if (q.levels.empty()) {
          q.levels.resize(q.options.size());
          std::iota(q.levels.begin(), q.levels.end(), 0.0);
        }
        if (q.levels.size() != q.options.size()) throw RequestError("levels must match options");
        for (std::size_t i = 0; i < q.levels.size(); ++i)
          if (!std::isfinite(q.levels[i]) || (i && q.levels[i] <= q.levels[i-1]))
            throw RequestError("score levels must be finite and strictly increasing");
      }
      std::string content = (layout == "user_state" ? shared : "") +
          (compact ? "" : "QUESTION:\n") + q.instruction + (compact ? "\n" : "\nOPTIONS:\n");
      for (std::size_t i = 0; i < q.options.size(); ++i)
        content += alias_labels_[i] + (compact ? " " : ": ") + q.options[i] + "\n";
      if (!compact) content += "Return only the letter of the best option.";
      if (layout == "system_state") {
        const std::string suffix = (has_images ? "\n" : "\n<|turn>user\n") + content + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        q.prompt = state_ids;
        const auto suffix_ids = tokenizer_.encode(suffix);
        q.prompt.insert(q.prompt.end(), suffix_ids.begin(), suffix_ids.end());
        q.types = state_types;
        q.types.resize(q.prompt.size(), 0);
        if (request.value("verify_prompt", false) && q.prompt != (has_images
                ? prepare_image_prompt(tokenizer_, framed_state + suffix, image_counts_).input_ids
                : tokenizer_.encode(framed_state + suffix)))
          throw RequestError("split prompt tokenization mismatch");
      } else { q.prompt = tokenizer_.encode(format_gemma4_user_turn(content)); q.types.resize(q.prompt.size(), 0); }
      if (q.prompt.size() < 2 || q.prompt.size() > kMaximumContext)
        throw RequestError("question prompt exceeds the 32768-token experiment limit");
      if (q.prompt.back() != answer_boundary_)
        throw RequestError("rendered prompt lacks the verified answer boundary");
      q.aliases.assign(alias_ids_.begin(), alias_ids_.begin() + q.options.size());
      questions.push_back(std::move(q));
    }
    const double preparation_ms = milliseconds(started);
    std::size_t common = mode != "none" && questions.size() > 1 ? questions.front().prompt.size() - 1 : 0;
    if (mode != "none" && questions.size() == 1 && layout == "system_state")
      common = state_ids.size();
    for (const auto& q : questions) {
      common = std::min(common, q.prompt.size() - 1);
      std::size_t i = 0;
      while (i < common && q.prompt[i] == questions.front().prompt[i]) ++i;
      common = i;
    }
    if (common < 2) common = 0;
    double prefix_ms = 0, clone_ms = 0, suffix_ms = 0, readout_ms = 0;
    if (request.value("reset_prefix", false)) {
      branches_.clear(); cached_ids_.clear(); // Invalidate content, retain overwriteable storage.
    }
    double readout_error = 0;
    if (branch_mode_ != mode) {
      branches_.clear();
      branch_mode_ = mode;
    }
    if (!common && !branches_.empty() && branches_.front()->shared_prefix_tokens())
      branches_.clear();
    bool cache_hit = false, prefix_storage_reused = false;
    if (common) {
      std::vector<std::uint32_t> prefix(questions.front().prompt.begin(), questions.front().prompt.begin() + common);
      const auto image_revision = has_images ? image_revision_ : 0;
      cache_hit = prefix == cached_ids_ && cached_prefix_image_revision_ == image_revision;
      if (!cache_hit) {
        branches_.clear();
        if (!prefix_ || prefix_sealed_) {
          prefix_ = std::make_unique<KvCache>(kMaximumContext);
          prefix_sealed_ = false;
        } else {
          // Prefill starts at position zero and overwrites every attended row.
          // Retain allocations, never prior-state K/V content or probabilities.
          prefix_storage_reused = true;
        }
        ProfileRange profile("decision/prefix");
        const auto begin = Clock::now();
        cached_ids_.clear(); // Failure cannot leave a stale cache entry valid.
        prefill(*prefix_, prefix, 0, std::span(questions.front().types).first(common),
            std::getenv("GEVVA_DECISION_PREFIX_READOUT") == nullptr);
        cached_prefix_image_revision_ = image_revision;
        cached_ids_ = std::move(prefix);
        prefix_ms = milliseconds(begin);
      }
    }
    if (mode == "readonly" && !prefix_) prefix_ = std::make_unique<KvCache>(kMaximumContext);
    Json answers = Json::array();
    int waves = 0;
    std::size_t suffix_tokens = 0;
    for (std::size_t first = 0; first < questions.size();) {
      std::size_t end = first, rows = 0, max_suffix = 0;
      while (end < questions.size() && end - first < static_cast<std::size_t>(batch_size)) {
        const auto count = questions[end].prompt.size() - common;
        const auto candidate_max = std::max(max_suffix, count);
        const auto attention_elements = (end - first + 1) * 16 * candidate_max * (common + candidate_max);
        if (end > first && (rows + count > kChunk || (mode == "readonly" && attention_elements > 256ULL * 1024 * 1024))) break;
        max_suffix = candidate_max;
        rows += count;
        ++end;
      }
      const int count = static_cast<int>(end - first);
      const auto allocation_begin = Clock::now();
      const bool readonly = mode == "readonly" && rows <= kChunk;
      while (!readonly && branches_.size() < static_cast<std::size_t>(count)) {
        auto branch = std::make_unique<KvCache>(kMaximumContext);
        if (common && mode == "shared") {
          prefix_sealed_ = true; // Includes a partially failed physical fork.
          branch->share_global_prefix_from(*prefix_, common);
        }
        branches_.push_back(std::move(branch));
      }
      clone_ms += milliseconds(allocation_begin);
      if (common && !readonly) {
        const auto begin = Clock::now();
        for (int slot = 0; slot < count; ++slot) clone(*prefix_, *branches_[slot], common, mode != "shared");
        check(cudaStreamSynchronize(context_.stream()));
        clone_ms += milliseconds(begin);
      }
      std::vector<TargetPrefillBenchmark> results;
      const bool device_readout = rows <= kChunk && !request.value("verify_readout", false);
      final_states_.reset();
      void* device_states = device_readout ? final_states_.allocate(count * kHidden * 2) : nullptr;
      const auto suffix_begin = Clock::now();
      {
      ProfileRange profile("decision/suffix");
      if (rows > kChunk) {
        const auto suffix = std::span(questions[first].prompt).subspan(common);
        results.push_back(prefill(*branches_[0], suffix, common, std::span(questions[first].types).subspan(common)));
      } else {
        std::vector<std::vector<std::uint8_t>> types(count);
        std::vector<TargetPrefillBatchItem> items;
        for (int slot = 0; slot < count; ++slot) {
          const auto suffix = std::span(questions[first + slot].prompt).subspan(common);
          const auto mask = std::span(questions[first + slot].types).subspan(common);
          types[slot].assign(mask.begin(), mask.end());
          const void* images = std::count(mask.begin(), mask.end(), 1) ? projected_images_ : nullptr;
          items.push_back({readonly ? prefix_.get() : branches_[slot].get(), suffix, types[slot], {}, images, static_cast<int>(common), slot});
        }
        results = benchmark_target_prefill_batch(weights_, compressed_, items, 30, experts_,
            &prefill_scratch_, &vocab_, &context_, device_states, &linears(rows), nullptr, true, readonly, has_images);
      }
      }
      suffix_ms += milliseconds(suffix_begin);
      suffix_tokens += rows;
      ++waves;
      std::vector<float> states;
      for (const auto& result : results) states.insert(states.end(), result.final_state.begin(), result.final_state.end());
      const auto readout_begin = Clock::now();
      std::size_t active_options = 0;
      for (std::size_t index = first; index < end; ++index) active_options = std::max(active_options, questions[index].options.size());
      const auto readouts = decision_scores(weights_, states, std::span(alias_ids_).first(active_options), context_, readout_scratch_, device_states, count, readout_mode == "options", &decision_head_);
      for (int slot = 0; slot < count; ++slot) {
        if (request.value("verify_readout", false)) {
          for (std::size_t option = 0; option < questions[first + slot].aliases.size(); ++option) {
            const auto alias = questions[first + slot].aliases[option];
            const double reference = cpu_logit(std::span(states).subspan(slot * kHidden, kHidden), alias);
            readout_error = std::max(readout_error, std::abs(reference - readouts[slot].option_logits[option]));
          }
        }
        answers.push_back(answer(questions[first + slot], readouts[slot], temperature));
      }
      readout_ms += milliseconds(readout_begin);
      first = end;
    }
    const double wall = milliseconds(started);
    std::size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    const auto plans = cudnn_plan_cache_stats();
    const auto heap = mallinfo2();
    std::size_t shared_bytes = 0;
    for (const auto& branch : branches_) shared_bytes += branch->shared_global_bytes();
    return {{"answers", answers}, {"probabilities_calibrated", false}, {"temperature", temperature}, {"readout_mode", readout_mode},
      {"readout_reference_max_error", request.value("verify_readout", false) ? Json(readout_error) : Json(nullptr)},
      {"image_metrics", image_metrics},
      {"metrics", {{"wall_ms", wall}, {"preparation_ms", preparation_ms}, {"prefix_ms", prefix_ms},
        {"prefix_clone_ms", clone_ms}, {"suffix_ms", suffix_ms}, {"readout_ms", readout_ms},
        {"prefix_tokens", common}, {"suffix_tokens", suffix_tokens}, {"prefix_cache_hit", cache_hit}, {"prefix_storage_reused", prefix_storage_reused},
        {"prefix_mode", mode}, {"prompt_layout", layout}, {"compact_prompt", compact}, {"shared_global_bytes", shared_bytes}, {"batch_size", batch_size}, {"waves", waves},
        {"decisions_per_second", questions.size() * 1000.0 / wall}, {"device_used_bytes", total - free},
        {"runtime_weight_bytes", weights_.device_bytes()},
        {"host_heap_used_bytes", heap.uordblks}, {"host_heap_free_bytes", heap.fordblks},
        {"host_heap_mapped_bytes", heap.hblkhd},
        {"workspace_sharing", std::getenv("GEVVA_DECISION_PRIVATE_WORKSPACES") == nullptr},
        {"fp8_runner_entries", linears_.size()},
        {"execution_graph_entries", context_.prefill_graph_count()},
        {"execution_graph_hits", context_.prefill_graph_hits()},
        {"execution_graph_builds", context_.prefill_graph_builds()},
        {"scratch_bytes", {{"prefill", prefill_scratch_.bytes()}, {"vision", vision_scratch_.bytes()},
          {"readout", readout_scratch_.bytes()}, {"final_states", final_states_.bytes()}, {"vision_output", vision_output_.bytes()}}},
        {"attention_plan_cache", {{"text_entries", plans.text_entries}, {"vision_entries", plans.vision_entries},
          {"text_limit", plans.text_limit}, {"vision_limit", plans.vision_limit},
          {"builds", plans.builds}, {"evictions", plans.evictions}, {"fallbacks", plans.fallbacks}}}}}};
  }

 private:
  static constexpr int kMaximumContext = 32768;
  Json prepare_images(const Json& request) {
    const auto sources = request.value("images", Json::array());
    if (!sources.is_array() || sources.size() > 8) throw RequestError("images must be an array of at most 8 local paths or data URLs");
    Json metrics = {{"image_count", sources.size()}, {"images_encoded", 0}, {"image_cache_hit", false},
                    {"image_load_ms", 0.0}, {"image_preprocess_ms", 0.0}, {"vision_ms", 0.0}, {"image_tokens", 0}};
    if (sources.empty()) return metrics;
    const int soft_limit = request.value("image_soft_tokens", 280);
    if (soft_limit != 70 && soft_limit != 140 && soft_limit != 280 && soft_limit != 560 && soft_limit != 1120)
      throw RequestError("image_soft_tokens must be 70, 140, 280, 560 or 1120");
    const auto load_start = Clock::now();
    std::vector<std::vector<std::uint8_t>> encoded;
    for (const auto& source : sources) encoded.push_back(image_bytes(source.get<std::string>()));
    metrics["image_load_ms"] = milliseconds(load_start);
    if (!request.value("reset_images", false) && encoded == cached_images_ && soft_limit == image_soft_limit_) {
      metrics["image_cache_hit"] = true;
    } else {
      cached_images_.clear(); // A failed encode cannot leave stale cache identity.
      ++image_revision_;
      const bool gpu_preprocess = std::getenv("GEVVA_CPU_IMAGE_PREPROCESS") == nullptr;
      metrics["gpu_preprocess"] = gpu_preprocess;
      metrics["hardware_jpeg_images"] = 0;
      if (gpu_preprocess) {
        if (!image_preprocessor_) image_preprocessor_ = std::make_unique<GpuImagePreprocessor>();
        image_counts_.clear(); vision_output_.reset();
        projected_images_ = vision_output_.allocate(encoded.size() * soft_limit * kHidden * 2);
        std::size_t offset = 0;
        for (const auto& bytes : encoded) {
          const auto preprocess_start = Clock::now();
          DeviceVisionInput input;
          try { input = image_preprocessor_->prepare(bytes, soft_limit, context_); }
          catch (const ImageInputError& error) { throw RequestError(error.what()); }
          metrics["image_preprocess_ms"] = metrics["image_preprocess_ms"].get<double>() + milliseconds(preprocess_start);
          metrics["hardware_jpeg_images"] = metrics["hardware_jpeg_images"].get<int>() + int(input.hardware_jpeg);
          image_counts_.push_back(input.geometry.soft_token_count);
          const auto vision_start = Clock::now();
          if (!vision_weights_) vision_weights_ = std::make_unique<VisionFusedWeights>(weights_.model());
          benchmark_vision_encoder(weights_.model(), {}, input.geometry.position_ids, input.geometry.patch_count,
              {}, false, &vision_scratch_, &context_, 1, static_cast<std::byte*>(projected_images_) + offset * kHidden * 2,
              false, nullptr, {}, {}, vision_weights_.get(), input.pixels, input.positions);
          metrics["vision_ms"] = metrics["vision_ms"].get<double>() + milliseconds(vision_start);
          offset += input.geometry.soft_token_count;
        }
        metrics["images_encoded"] = encoded.size();
      } else {
      const auto preprocess_start = Clock::now();
      std::vector<VisionInput> inputs;
      image_counts_.clear();
      for (const auto& bytes : encoded) {
        try { inputs.push_back(preprocess_gemma4_image(load_rgb_image(bytes), soft_limit, true)); }
        catch (const std::runtime_error& error) { throw RequestError(error.what()); }
        image_counts_.push_back(inputs.back().soft_token_count);
      }
      metrics["image_preprocess_ms"] = milliseconds(preprocess_start);
      const auto vision_start = Clock::now();
      if (!vision_weights_) vision_weights_ = std::make_unique<VisionFusedWeights>(weights_.model());
      vision_output_.reset();
      projected_images_ = vision_output_.allocate(std::accumulate(image_counts_.begin(), image_counts_.end(), 0ULL) * kHidden * 2);
      std::size_t offset = 0;
      for (const auto& input : inputs) {
        benchmark_vision_encoder(weights_.model(), input.pixel_values, input.position_ids, input.patch_count,
            {}, false, &vision_scratch_, &context_, 1, static_cast<std::byte*>(projected_images_) + offset * kHidden * 2,
            false, nullptr, {}, {}, vision_weights_.get());
        offset += input.soft_token_count;
      }
      metrics["vision_ms"] = milliseconds(vision_start);
      metrics["images_encoded"] = inputs.size();
      }
      cached_images_ = std::move(encoded);
      image_soft_limit_ = soft_limit;
    }
    metrics["image_tokens"] = std::accumulate(image_counts_.begin(), image_counts_.end(), 0);
    return metrics;
  }
  double cpu_logit(std::span<const float> state, std::uint32_t token) {
    const auto norm = weights_.model().host_tensor("model.language_model.norm.weight");
    const auto embedding = weights_.model().host_tensor("model.language_model.embed_tokens.weight");
    auto bf16 = [](const std::byte* pointer) {
      std::uint16_t bits;
      std::memcpy(&bits, pointer, sizeof(bits));
      return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    };
    double squares = 0;
    for (float value : state) squares += static_cast<double>(value) * value;
    const float inverse = 1.0 / std::sqrt(squares / kHidden + 1e-6);
    double dot = 0;
    for (int col = 0; col < kHidden; ++col) {
      const float normalized = state[col] * inverse * bf16(norm.bytes.data() + col * 2);
      auto bits = std::bit_cast<std::uint32_t>(normalized);
      bits += 0x7fff + ((bits >> 16) & 1); // BF16 round-to-nearest-even.
      const float rounded = std::bit_cast<float>(bits & 0xffff0000);
      dot += static_cast<double>(rounded) * bf16(embedding.bytes.data() + (static_cast<std::size_t>(token) * kHidden + col) * 2);
    }
    return 30.0 * std::tanh(dot / 30.0);
  }
  Fp8LinearRunner& linears(int rows) {
    const int bucket = (rows + 15) & ~15;
    if (!linears_.contains(bucket)) {
      if (linears_.size() >= 32) {
        const auto oldest = std::min_element(linear_usage_.begin(), linear_usage_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        linears_.erase(oldest->first);
        linear_usage_.erase(oldest);
      }
      linears_[bucket] = std::make_unique<Fp8LinearRunner>(weights_.model(), dense_, bucket, false, 0, 3, false,
          !linears_.empty() && !std::getenv("GEVVA_DECISION_PRIVATE_WORKSPACES") ? linears_.begin()->second.get() : nullptr);
    }
    linear_usage_[bucket] = ++linear_clock_;
    return *linears_.at(bucket);
  }
  TargetPrefillBenchmark prefill(KvCache& cache, std::span<const std::uint32_t> ids, int prefix,
                                  std::span<const std::uint8_t> types, bool kv_only = false) {
    TargetPrefillBenchmark result;
    std::size_t image_rows = 0;
    for (std::size_t first = 0; first < ids.size();) {
      auto end = std::min<std::size_t>(first + kChunk, ids.size());
      // Preserve each bidirectional image island within one prefill chunk.
      if (end < ids.size() && types[end] == 1)
        while (end > first && types[end - 1] == 1) --end;
      if (end == first) throw std::runtime_error("image island exceeds prefill chunk");
      const auto chunk = ids.subspan(first, end - first);
      const auto mask = types.subspan(first, end - first);
      const auto images = std::count(mask.begin(), mask.end(), 1);
      const void* projected = images ? static_cast<const std::byte*>(projected_images_) + image_rows * kHidden * 2 : nullptr;
      result = benchmark_target_prefill(weights_, compressed_, cache, chunk, mask, {}, 30, false,
          experts_, &prefill_scratch_, &vocab_, &context_, nullptr, prefix + first, &linears(chunk.size()), projected, kv_only);
      image_rows += images;
      first = end;
    }
    return result;
  }
  void clone(KvCache& source, KvCache& destination, int tokens, bool copy_global) {
    destination.ensure_context(tokens, context_.stream());
    if (tokens <= 1024) copy_sliding_prefix_arena(source.sliding_arena(), destination.sliding_arena(), tokens, context_.stream());
    else check(cudaMemcpyAsync(destination.sliding_arena(), source.sliding_arena(), source.sliding_bytes(), cudaMemcpyDeviceToDevice, context_.stream()));
    if (!copy_global) return;
    for (int layer = 5; layer < 30; layer += 6) {
      const std::size_t bytes = static_cast<std::size_t>(tokens) * 2 * 512 * 2;
      check(cudaMemcpyAsync(destination.layer(layer).keys, source.layer(layer).keys, bytes, cudaMemcpyDeviceToDevice, context_.stream()));
      check(cudaMemcpyAsync(destination.layer(layer).values, source.layer(layer).values, bytes, cudaMemcpyDeviceToDevice, context_.stream()));
    }
  }
  Json answer(const Question& q, const DecisionReadout& readout, double temperature) {
    std::vector<double> selected_logits, full_logprobs, probabilities;
    for (std::size_t i = 0; i < q.aliases.size(); ++i) {
      selected_logits.push_back(readout.option_logits[i]);
      if (readout.full_vocabulary)
        full_logprobs.push_back(static_cast<double>(readout.option_logits[i]) - readout.log_normalizer);
    }
    const double best = *std::max_element(selected_logits.begin(), selected_logits.end());
    double denominator = 0, allowed_mass = 0;
    for (std::size_t i = 0; i < selected_logits.size(); ++i) {
      probabilities.push_back(std::exp((selected_logits[i] - best) / temperature));
      denominator += probabilities.back();
      if (readout.full_vocabulary) allowed_mass += std::exp(full_logprobs[i]);
    }
    double entropy = 0;
    for (double& p : probabilities) {
      p /= denominator;
      if (p > 0) entropy -= p * std::log(p);
    }
    const auto chosen = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    auto sorted = probabilities;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    const auto unconstrained = readout.argmax_token;
    Json result = {{"id", q.id}, {"type", q.type}, {"options", q.options}, {"choice_index", chosen},
      {"choice", q.options[chosen]}, {"probabilities", probabilities}, {"option_logits", selected_logits},
      {"full_vocab_logprobs", readout.full_vocabulary ? Json(full_logprobs) : Json(nullptr)},
      {"allowed_mass", readout.full_vocabulary ? Json(std::min(1.0, allowed_mass)) : Json(nullptr)}, {"entropy_nats", entropy},
      {"max_probability", sorted[0]}, {"probability_margin", sorted[0] - (sorted.size() > 1 ? sorted[1] : 0.0)},
      {"alias_token_ids", q.aliases}, {"unconstrained_token_id", readout.full_vocabulary ? Json(unconstrained) : Json(nullptr)},
      {"unconstrained_token", readout.full_vocabulary ? Json(tokenizer_.decode_token(unconstrained)) : Json(nullptr)}, {"prompt_tokens", q.prompt.size()}};
    if (q.type == "boolean") { result["value"] = chosen == 1; result["p_true"] = probabilities[1]; }
    if (q.type == "score") {
      result["levels"] = q.levels;
      result["score"] = std::inner_product(probabilities.begin(), probabilities.end(), q.levels.begin(), 0.0);
    }
    return result;
  }
  Tokenizer tokenizer_;
  std::uint32_t answer_boundary_{};
  std::vector<std::uint32_t> alias_ids_;
  std::vector<std::string> alias_labels_;
  RuntimeWeights weights_;
  DeviceModel compressed_, dense_;
  GpuExecutionContext context_;
  CompressedVocabRunner vocab_;
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> expert_owners_;
  std::array<Nvfp4ExpertRunner*, 30> experts_{};
  DeviceScratchPool prefill_scratch_, readout_scratch_, final_states_, vision_scratch_, vision_output_;
  DeviceScratchPool head_storage_;
  DecisionHeadView decision_head_;
  std::unique_ptr<VisionFusedWeights> vision_weights_;
  std::unique_ptr<GpuImagePreprocessor> image_preprocessor_;
  std::vector<std::vector<std::uint8_t>> cached_images_;
  std::vector<int> image_counts_;
  int image_soft_limit_{};
  std::uint64_t image_revision_{}, cached_prefix_image_revision_{};
  void* projected_images_{};
  std::map<int, std::uint64_t> linear_usage_;
  std::uint64_t linear_clock_{};
  std::map<int, std::unique_ptr<Fp8LinearRunner>> linears_;
  std::unique_ptr<KvCache> prefix_;
  bool prefix_sealed_{};
  std::vector<std::unique_ptr<KvCache>> branches_;
  std::vector<std::uint32_t> cached_ids_;
  std::string branch_mode_;
};
} // namespace

int run_decision_worker() {
  // JSONL requests can contain large base64 images. Buffered C++ input avoids
  // per-character stdio synchronization; responses still flush explicitly.
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  const auto started = Clock::now();
  const auto paths = decision_model_paths();
  const auto gpu = select_target_gpu();
  if (!std::getenv("GEVVA_CUDNN_TEXT_PREFILL") && !std::getenv("GEVVA_DECISION_REFERENCE_ATTENTION"))
    setenv("GEVVA_CUDNN_TEXT_PREFILL", "1", 0);
  DecisionWorker worker(paths);
  std::cerr << Json({{"status", "ready"}, {"gpu", gpu.name}, {"startup_ms", milliseconds(started)}}).dump() << '\n';
  const char* profile_start = std::getenv("GEVVA_PROFILE_REQUEST_START");
  const std::size_t profile_request = profile_start ? std::stoull(profile_start) : 0;
  std::size_t request_number = 0;
  for (std::string line; std::getline(std::cin, line);) {
    if (line.empty()) continue;
    if (++request_number == profile_request) check(cudaProfilerStart());
    try { std::cout << worker.run(Json::parse(line)).dump() << std::endl; }
    catch (const RequestError& error) {
      std::cout << Json({{"error", error.what()}, {"error_kind", "validation"}}).dump() << std::endl;
    } catch (const Json::exception& error) {
      std::cout << Json({{"error", error.what()}, {"error_kind", "validation"}}).dump() << std::endl;
    } catch (const std::exception& error) {
      std::cout << Json({{"error", error.what()}, {"error_kind", "execution"}}).dump() << std::endl;
    }
  }
  if (profile_request && request_number >= profile_request) check(cudaProfilerStop());
  return 0;
}

int test_decision_kv_fork() {
  select_target_gpu();
  constexpr int prefix_tokens = 12345; // Beyond initial allocation, partial page.
  constexpr std::size_t prefix_bytes = prefix_tokens * 2048ULL;
  auto source = std::make_unique<KvCache>(32768);
  source->ensure_context(prefix_tokens);
  for (int layer = 5; layer < 30; layer += 6) {
    check(cudaMemset(source->layer(layer).keys, 0x3c, prefix_bytes));
    check(cudaMemset(source->layer(layer).values, 0x5a, prefix_bytes));
  }
  KvCache first(32768), second(32768);
  first.share_global_prefix_from(*source, prefix_tokens);
  second.share_global_prefix_from(*source, prefix_tokens);
  if (!first.shared_global_bytes() || first.shared_global_bytes() != second.shared_global_bytes())
    throw std::runtime_error("KV fork did not share physical pages");
  bool sealed = false;
  try { source->ensure_context(prefix_tokens + 1); }
  catch (const std::runtime_error&) { sealed = true; }
  if (!sealed) throw std::runtime_error("KV fork source was not sealed");
  first.ensure_context(20000); // Private growth must preserve all shared pages.
  second.ensure_context(17000);
  for (int layer = 5; layer < 30; layer += 6) {
    check(cudaMemset(first.layer(layer).keys + prefix_bytes, 0x11, 2048));
    check(cudaMemset(second.layer(layer).keys + prefix_bytes, 0x22, 2048));
  }
  source.reset(); // Physical handles must outlive the source virtual mapping.
  for (int layer = 5; layer < 30; layer += 6) {
    for (const auto* branch : {&first, &second}) {
      for (const auto offset : {std::size_t{0}, std::size_t{8192 * 2048}, prefix_bytes - 1}) {
        std::uint8_t key{}, value{};
        check(cudaMemcpy(&key, branch->layer(layer).keys + offset, 1, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&value, branch->layer(layer).values + offset, 1, cudaMemcpyDeviceToHost));
        if (key != 0x3c || value != 0x5a) throw std::runtime_error("shared KV data corrupted");
      }
    }
    std::uint8_t a{}, b{};
    check(cudaMemcpy(&a, first.layer(layer).keys + prefix_bytes, 1, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&b, second.layer(layer).keys + prefix_bytes, 1, cudaMemcpyDeviceToHost));
    if (a != 0x11 || b != 0x22) throw std::runtime_error("branch suffixes are not isolated");
  }
  std::cout << Json({{"passed", true}, {"shared_bytes_per_branch", first.shared_global_bytes()},
                    {"prefix_tokens", prefix_tokens}}).dump() << '\n';
  return 0;
}
} // namespace gevva
