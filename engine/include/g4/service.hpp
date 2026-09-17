#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace g4 {

struct MultimodalRuntimePaths {
  std::filesystem::path target;
  std::filesystem::path experts;
  std::filesystem::path assistant;
  std::filesystem::path target_vocab;
  std::filesystem::path target_vocab_nvfp4;
  std::filesystem::path assistant_vocab;
  std::filesystem::path assistant_vocab_nvfp4;
  std::filesystem::path target_dense_fp8;
  std::filesystem::path assistant_dense_fp8;
};

struct GenerationCycle {
  std::array<int, 4> drafts{};
  std::array<int, 5> target_tokens{};
  int matched_drafts{};
  int draft_count{4};
};

struct GenerationResult {
  std::vector<std::uint32_t> token_ids;
  std::string text;
  std::vector<GenerationCycle> cycles;
  int prompt_tokens{};
  int prefill_token{};
  float prefill_checksum{};
  double image_decode_milliseconds{};
  double image_preprocess_milliseconds{};
  float vision_microseconds{};
  float prefill_microseconds{};
  double first_token_milliseconds{};
  int accepted_drafts{};
  double assistant_milliseconds{};
  double verifier_milliseconds{};
  double wall_milliseconds{};
  bool length_limited{};

  double draft_acceptance() const;
  double decode_postprefill_tokens_per_second() const;
};

struct BatchGenerationRequest {
  std::span<const std::uint8_t> encoded_image;
  std::string_view system;
  std::string_view prompt;
  int maximum_tokens{};
  bool stop_on_json_object{};
  std::span<const std::string> expected_json_ids;
  std::function<void(GenerationResult)> on_complete;
  std::function<void(std::string)> on_tokens;
};

using BatchRefillCallback =
    std::function<std::optional<BatchGenerationRequest>()>;

// One long-lived, single-GPU serving runtime. It owns exactly one target
// checkpoint representation plus its expert sidecar and the official MTP
// assistant. CUDA context, weights, expert plans, KV arenas, and scratch memory
// remain resident across requests.
class MultimodalRuntime {
 public:
  explicit MultimodalRuntime(MultimodalRuntimePaths paths,
                             int maximum_context = 262144);
  ~MultimodalRuntime();
  MultimodalRuntime(const MultimodalRuntime&) = delete;
  MultimodalRuntime& operator=(const MultimodalRuntime&) = delete;

  GenerationResult generate(const std::filesystem::path& image,
                            std::string_view prompt, int maximum_tokens);
  GenerationResult generate(std::span<const std::uint8_t> encoded_image,
                            std::string_view prompt, int maximum_tokens);
  GenerationResult generate(std::span<const std::uint8_t> encoded_image,
                            std::string_view system, std::string_view prompt,
                            int maximum_tokens,
                            bool stop_on_json_object = false,
                            std::span<const std::string> expected_json_ids = {});
  GenerationResult generate(std::string_view prompt, int maximum_tokens);
  std::vector<GenerationResult> generate_batch(
      std::span<const BatchGenerationRequest> requests);
  // Keeps the admitted GPU batch resident, completes finished slots
  // immediately, and asks refill for replacement work at MTP boundaries.
  void generate_continuous(
      std::span<const BatchGenerationRequest> initial_requests,
      BatchRefillCallback refill);
  double startup_milliseconds() const;
  int maximum_context() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g4
