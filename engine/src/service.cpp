#include "g4/service.hpp"

#include "g4/gpu.hpp"
#include "g4/image.hpp"
#include "g4/multimodal.hpp"
#include "g4/runtime.hpp"
#include "g4/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <nvtx3/nvToolsExt.h>

namespace g4 {

namespace {

constexpr int kServingDrafts = 4;

int serving_drafts(int sessions) {
  if (sessions < 1 || sessions > 8)
    throw std::runtime_error("invalid serving batch for MTP policy");
  static const std::array<int, 8> configured = [] {
    // B1 benefits from stopping after the third assistant token on both text
    // and multimodal prompts. Throughput-oriented B8 retains the official
    // four-token chain. Every batch can be independently overridden.
    std::array<int, 8> result{3, 4, 4, 4, 4, 4, 4, 4};
    const char* global = std::getenv("G4_MTP_DRAFTS");
    for (int batch = 1; batch <= 8; ++batch) {
      const std::string name = "G4_MTP_DRAFTS_B" + std::to_string(batch);
      const char* value = std::getenv(name.c_str());
      if (!value) value = global;
      if (!value) continue;
      const int drafts = std::atoi(value);
      if (drafts < 0 || drafts > kServingDrafts)
        throw std::runtime_error(name + " must be in [0, 4]");
      result[batch - 1] = drafts;
    }
    return result;
  }();
  return configured[sessions - 1];
}

int assistant_minimum_rows() {
  static const int rows = [] {
    const char* value = std::getenv("G4_ASSISTANT_MIN_ROWS");
    const int parsed = value ? std::atoi(value) : 1;
    if (parsed < 1 || parsed > 8)
      throw std::runtime_error("G4_ASSISTANT_MIN_ROWS must be in [1, 8]");
    return parsed;
  }();
  return rows;
}

constexpr int kNativeContextTokens = 262144;

bool is_stop(std::uint32_t token) {
  return token == 1 || token == 50 || token == 106;
}

struct JsonObjectBoundary {
  bool feed(std::string_view piece,
            std::span<const std::string> expected_ids = {}) {
    for (const char character : piece) {
      if (!started) {
        if (character == '{') {
          started = true;
          depth = 1;
          candidate = "{";
        }
        continue;
      }
      candidate.push_back(character);
      if (in_string) {
        if (escaped) escaped = false;
        else if (character == '\\') escaped = true;
        else if (character == '"') in_string = false;
        continue;
      }
      if (character == '"') in_string = true;
      else if (character == '{') ++depth;
      else if (character == '}' && --depth == 0) {
        collect_ids();
        const auto parsed = nlohmann::json::parse(candidate, nullptr, false);
        const bool valid_object = !parsed.is_discarded() && parsed.is_object();
        const bool complete = expected_ids.empty()
            ? valid_object
            : std::all_of(expected_ids.begin(), expected_ids.end(),
                          [&](const auto& id) { return seen_ids.contains(id); });
        started = false;
        in_string = false;
        escaped = false;
        candidate.clear();
        if (complete) return true;
      }
    }
    return false;
  }
  void collect_ids() {
    std::size_t position = 0;
    while ((position = candidate.find("\"id\"", position)) !=
           std::string::npos) {
      const auto colon = candidate.find(':', position + 4);
      const auto quote = colon == std::string::npos
          ? std::string::npos : candidate.find('"', colon + 1);
      const auto end = quote == std::string::npos
          ? std::string::npos : candidate.find('"', quote + 1);
      if (end == std::string::npos) break;
      seen_ids.emplace(candidate.substr(quote + 1, end - quote - 1));
      position = end + 1;
    }
  }
  bool started{};
  bool in_string{};
  bool escaped{};
  int depth{};
  std::string candidate;
  std::unordered_set<std::string> seen_ids;
};

}  // namespace

struct MultimodalRuntime::Impl {
  struct BatchResources {
    BatchResources(Impl& owner) : owner(owner) {
      int least_priority = 0, greatest_priority = 0;
      if (cudaDeviceGetStreamPriorityRange(&least_priority,
                                           &greatest_priority) != cudaSuccess ||
          cudaStreamCreateWithPriority(&prefix_copy_stream,
                                       cudaStreamNonBlocking,
                                       least_priority) != cudaSuccess ||
          cudaEventCreateWithFlags(&prefix_source_ready,
                                   cudaEventDisableTiming) != cudaSuccess ||
          cudaEventCreateWithFlags(&prefix_copy_ready,
                                   cudaEventDisableTiming) != cudaSuccess)
        throw std::runtime_error("create shared-prefix copy stream failed");
      for (int slot = 0; slot < 8; ++slot) {
        caches.push_back(std::make_unique<KvCache>(owner.context_limit));
        candidates.push_back(std::make_unique<CandidateKvCache>());
      }
      prefix_cache = std::make_unique<KvCache>(owner.context_limit);
      for (int layer = 0; layer < 30; ++layer) {
        expert_owners.push_back(std::make_unique<Nvfp4ExpertRunner>(
            owner.weights.experts(), layer, 40));
        experts[layer] = expert_owners.back().get();
      }
      // Verifier projections use three physical M buckets for every legal
      // B1..B8/tail-cycle shape.  Own exactly one plan per bucket up front;
      // separate plans for every logical row count duplicated descriptors,
      // workspaces, tensor lookups, and first-use setup without changing the
      // GEMM geometry.
      for (const int tile_rows : {16, 32, 64})
        fp8_by_rows.emplace(
            tile_rows, std::make_unique<Fp8LinearRunner>(
                           owner.weights.model(), owner.target_dense_fp8_model,
                           tile_rows, false, tile_rows));
      if (owner.target_vocab_nvfp4_model)
        for (const int rows : {4, 10, 15, 20, 25, 30, 35, 40})
          nvfp4_vocab_by_rows.emplace(
              rows, std::make_unique<Nvfp4VocabRunner>(
                        owner.weights.model(),
                        "model.language_model.embed_tokens.weight",
                        *owner.target_vocab_nvfp4_model, rows));
      const std::size_t state_bytes = 8ULL * 2816 * sizeof(std::uint16_t);
      if (cudaMalloc(&states_a, state_bytes) != cudaSuccess ||
          cudaMalloc(&states_b, state_bytes) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&drafts),
                     8 * 4 * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("allocate batch generation state failed");
      for (void*& output : vision_projected)
        if (cudaMalloc(&output, 560ULL * 2816 * sizeof(std::uint16_t)) !=
            cudaSuccess)
          throw std::runtime_error("allocate device vision output failed");
      if (cudaMalloc(&vision_projected_packed,
                     8ULL * 560 * 2816 * sizeof(std::uint16_t)) != cudaSuccess)
        throw std::runtime_error("allocate packed device vision output failed");
    }
    Fp8LinearRunner& fp8_for_sessions(int sessions) {
      auto& runner = fp8_by_sessions.at(sessions - 1);
      if (!runner)
        runner = std::make_unique<Fp8LinearRunner>(
            owner.weights.model(), owner.target_dense_fp8_model,
            sessions * (kServingDrafts + 1));
      return *runner;
    }
    Fp8LinearRunner& fp8_for_rows(int sessions, int tokens) {
      if (owner.padded_verifier_fp8_rows)
        return fp8_for_sessions(sessions);
      const int rows = sessions * tokens;
      int tile_rows = 16;
      while (tile_rows < rows) tile_rows <<= 1;
      auto found = fp8_by_rows.find(tile_rows);
      if (found != fp8_by_rows.end()) return *found->second;
      throw std::runtime_error("missing prebuilt verifier FP8 bucket");
    }
    Nvfp4VocabRunner* nvfp4_vocab_for_rows(int sessions, int tokens) {
      const int rows = sessions * tokens;
      const auto found = nvfp4_vocab_by_rows.find(rows);
      return found == nvfp4_vocab_by_rows.end() ? nullptr
                                                : found->second.get();
    }
    Impl& owner;
    ~BatchResources() {
      if (prefix_copy_stream) cudaStreamSynchronize(prefix_copy_stream);
      if (prefix_copy_ready) cudaEventDestroy(prefix_copy_ready);
      if (prefix_source_ready) cudaEventDestroy(prefix_source_ready);
      if (prefix_copy_stream) cudaStreamDestroy(prefix_copy_stream);
      if (vision_projected_packed) cudaFree(vision_projected_packed);
      for (void* output : vision_projected)
        if (output) cudaFree(output);
      if (drafts) cudaFree(drafts);
      if (states_b) cudaFree(states_b);
      if (states_a) cudaFree(states_a);
    }
    std::vector<std::unique_ptr<KvCache>> caches;
    std::unique_ptr<KvCache> prefix_cache;
    std::vector<std::uint32_t> cached_prefix_ids;
    cudaStream_t prefix_copy_stream{};
    cudaEvent_t prefix_source_ready{};
    cudaEvent_t prefix_copy_ready{};
    bool prefix_copy_pending{};
    int prefix_copy_source_slot{-1};
    std::vector<std::unique_ptr<CandidateKvCache>> candidates;
    std::vector<std::unique_ptr<Nvfp4ExpertRunner>> expert_owners;
    std::array<Nvfp4ExpertRunner*, 30> experts{};
    std::array<std::unique_ptr<Fp8LinearRunner>, 8> fp8_by_sessions;
    std::unordered_map<int, std::unique_ptr<Fp8LinearRunner>> fp8_by_rows;
    std::unordered_map<int, std::unique_ptr<Nvfp4VocabRunner>>
        nvfp4_vocab_by_rows;
    DeviceScratchPool assistant_scratch;
    DeviceScratchPool verifier_scratch;
    void* states_a{};
    void* states_b{};
    int* drafts{};
    std::array<void*, 8> vision_projected{};
    void* vision_projected_packed{};
  };

  explicit Impl(MultimodalRuntimePaths runtime_paths, int context_limit)
      : load_begin(std::chrono::steady_clock::now()),
        paths(std::move(runtime_paths)),
        context_limit(context_limit),
        tokenizer(paths.target),
        weights(paths.target, paths.experts),
        assistant(paths.assistant),
        target_vocab_model(paths.target_vocab),
        assistant_vocab_model(paths.assistant_vocab),
        target_dense_fp8_model(paths.target_dense_fp8),
        execution(true),
        target_vocab(weights.model(),
                     "model.language_model.embed_tokens.weight",
                     target_vocab_model, execution.stream()),
        assistant_vocab(assistant.model(), "model.embed_tokens.weight",
                        assistant_vocab_model, execution.stream()),
        target_fp8_linears(weights.model(), target_dense_fp8_model, 5),
        cache(context_limit),
        candidates(),
        workspace(context_limit, 5) {
    if (context_limit < 16 || context_limit > 262144)
      throw std::runtime_error("serving context is too small for prompt prefill");
    if (std::getenv("G4_ASSISTANT_FP8") != nullptr) {
      assistant_dense_fp8_model =
          std::make_unique<DeviceModel>(paths.assistant_dense_fp8);
      assistant_fp8_linears = std::make_unique<Fp8LinearRunner>(
          assistant.model(), *assistant_dense_fp8_model, 8, true);
    }
    if (std::getenv("G4_DISABLE_NVFP4_VOCAB") == nullptr &&
        std::filesystem::exists(paths.target_vocab_nvfp4)) {
      target_vocab_nvfp4_model =
          std::make_unique<DeviceModel>(paths.target_vocab_nvfp4);
      for (int rows = 1; rows <= 8; ++rows)
        prefill_vocab_nvfp4[rows - 1] = std::make_unique<Nvfp4VocabRunner>(
            weights.model(), "model.language_model.embed_tokens.weight",
            *target_vocab_nvfp4_model, rows);
    }
    if (std::getenv("G4_DISABLE_ASSISTANT_NVFP4_VOCAB") == nullptr &&
        std::filesystem::exists(paths.assistant_vocab_nvfp4)) {
      assistant_vocab_nvfp4_model =
          std::make_unique<DeviceModel>(paths.assistant_vocab_nvfp4);
      auto load_hot_vocab = [](const char* map_path) {
        std::vector<int> hot_vocab;
        if (!map_path) return hot_vocab;
        std::ifstream input(map_path, std::ios::binary | std::ios::ate);
        if (!input)
          throw std::runtime_error("cannot open assistant hot-vocabulary map");
        const auto bytes = input.tellg();
        if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int)))
          throw std::runtime_error("invalid assistant hot-vocabulary map size");
        hot_vocab.resize(static_cast<std::size_t>(bytes) / sizeof(int));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(hot_vocab.data()), bytes);
        if (!input)
          throw std::runtime_error("cannot read assistant hot-vocabulary map");
        return hot_vocab;
      };
      const auto hot_vocab =
          load_hot_vocab(std::getenv("G4_ASSISTANT_HOT_VOCAB"));
      const auto text_hot_vocab =
          load_hot_vocab(std::getenv("G4_ASSISTANT_HOT_VOCAB_TEXT"));
      const auto multimodal_hot_vocab = load_hot_vocab(
          std::getenv("G4_ASSISTANT_HOT_VOCAB_MULTIMODAL"));
      if (!hot_vocab.empty() || !text_hot_vocab.empty() ||
          !multimodal_hot_vocab.empty()) {
        assistant_hot_vocab_min_batch = 1;
        if (const char* value =
                std::getenv("G4_ASSISTANT_HOT_VOCAB_MIN_BATCH"))
          assistant_hot_vocab_min_batch = std::stoi(value);
        if (assistant_hot_vocab_min_batch < 1 ||
            assistant_hot_vocab_min_batch > 8)
          throw std::runtime_error(
              "G4_ASSISTANT_HOT_VOCAB_MIN_BATCH must be in [1, 8]");
        auto make_hot = [&](std::span<const int> map) {
          return map.empty() ? std::unique_ptr<Nvfp4VocabRunner>{}
                             : std::make_unique<Nvfp4VocabRunner>(
                                   assistant.model(),
                                   "model.embed_tokens.weight",
                                   *assistant_vocab_nvfp4_model, 8, map);
        };
        if (!text_hot_vocab.empty() && !multimodal_hot_vocab.empty()) {
          std::unordered_set<int> combined_ids(text_hot_vocab.begin(),
                                               text_hot_vocab.end());
          std::vector<int> combined = text_hot_vocab;
          combined.reserve(multimodal_hot_vocab.size());
          for (const int token : multimodal_hot_vocab)
            if (combined_ids.insert(token).second) combined.push_back(token);
          if (combined.size() != multimodal_hot_vocab.size())
            throw std::runtime_error(
                "text hot vocabulary must be a subset of multimodal map");
          assistant_combined_hot_vocab_nvfp4 =
              std::make_unique<Nvfp4VocabRunner>(
                  assistant.model(), "model.embed_tokens.weight",
                  *assistant_vocab_nvfp4_model, 8, combined,
                  static_cast<int>(text_hot_vocab.size()));
          assistant_combined_text_vocab_rows =
              static_cast<int>(text_hot_vocab.size());
          assistant_combined_multimodal_vocab_rows =
              static_cast<int>(combined.size());
        } else {
          assistant_hot_vocab_nvfp4 = make_hot(hot_vocab);
          assistant_text_hot_vocab_nvfp4 = make_hot(text_hot_vocab);
          assistant_multimodal_hot_vocab_nvfp4 =
              make_hot(multimodal_hot_vocab);
        }
      }
      const bool generic_covers_missing_specializations =
          assistant_hot_vocab_nvfp4 != nullptr;
      const bool mapped_covers_every_request =
          assistant_hot_vocab_min_batch == 1 &&
          (assistant_combined_hot_vocab_nvfp4 != nullptr ||
           generic_covers_missing_specializations ||
           (assistant_text_hot_vocab_nvfp4 != nullptr &&
            assistant_multimodal_hot_vocab_nvfp4 != nullptr));
      if (!mapped_covers_every_request) {
        assistant_vocab_nvfp4 = std::make_unique<Nvfp4VocabRunner>(
            assistant.model(), "model.embed_tokens.weight",
            *assistant_vocab_nvfp4_model, 8);
      }
    }
    verifier_expert_owners.reserve(30);
    prefill_expert_owners.reserve(30);
    for (int layer = 0; layer < 30; ++layer) {
      verifier_expert_owners.push_back(
          std::make_unique<Nvfp4ExpertRunner>(weights.experts(), layer, 5));
      verifier_experts[layer] = verifier_expert_owners.back().get();
      prefill_expert_owners.push_back(
          std::make_unique<Nvfp4ExpertRunner>(weights.experts(), layer,
                                             4608));
      prefill_experts[layer] = prefill_expert_owners.back().get();
    }
    if (std::getenv("G4_VISION_FP8") != nullptr)
      vision_fp8_model = std::make_unique<DeviceModel>(
          "/mnt/SSD/g4-models/gemma4-26b-a4b-vision-fp8");
    else if (std::getenv("G4_DISABLE_FUSED_VISION_PROJECTIONS") == nullptr)
      vision_fused_weights =
          std::make_unique<VisionFusedWeights>(weights.model());
    if (cudaMalloc(&continuation_state_device, 2816 * sizeof(std::uint16_t)) !=
        cudaSuccess)
      throw std::runtime_error("cudaMalloc(continuation state) failed");
    if (cudaMalloc(&vision_projected_device,
                   560ULL * 2816 * sizeof(std::uint16_t)) != cudaSuccess)
      throw std::runtime_error("cudaMalloc(vision projected state) failed");
    if (cudaMalloc(reinterpret_cast<void**>(&drafted_tokens_device),
                   4 * sizeof(int)) != cudaSuccess)
      throw std::runtime_error("cudaMalloc(drafted tokens) failed");
    // Pay the cuBLAS INT8 dispatch/JIT cost before the server announces
    // readiness. Packed prefill selects one final row per admitted session,
    // so warming 1..8 covers every serving batch without synthetic model work.
    target_vocab.warmup(8, execution.stream());
    // These resources are invariant for this hardware-specific server. Build
    // them before readiness instead of charging cache arenas, verifier plans,
    // and packed expert runners to the first client request.
    batch = std::make_unique<BatchResources>(*this);
    if (std::getenv("G4_DISABLE_STARTUP_PREFILL_WARMUP") == nullptr) {
      // Prime the exact B1/B8 text serving geometry, including pointer-batched
      // prefill attention and one live MTP cycle. These isolated caches are
      // overwritten by the first real request; no generated result escapes.
      constexpr std::string_view warm_prompt =
          "Explain why local inference benchmarks should report prompt "
          "prefill throughput, decode throughput, and time to first token.";
      for (const int sessions : {1, 8}) {
        std::vector<BatchGenerationRequest> warm_requests(sessions);
        for (auto& request : warm_requests) {
          request.prompt = warm_prompt;
          request.maximum_tokens = 6;
        }
        (void)generate_batch(warm_requests);
      }
    }
    load_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - load_begin)
                  .count();
  }

  ~Impl() {
    if (vision_projected_device) cudaFree(vision_projected_device);
    if (drafted_tokens_device) cudaFree(drafted_tokens_device);
    if (continuation_state_device) cudaFree(continuation_state_device);
  }

  GenerationResult generate_prepared(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint8_t> multimodal_types,
      std::span<const float> projected_images, int maximum_tokens,
      std::chrono::steady_clock::time_point wall_begin,
      double image_decode_milliseconds, double image_preprocess_milliseconds,
      float vision_microseconds, bool stop_on_json_object = false,
      std::span<const std::string> expected_json_ids = {},
      const void* device_projected_images = nullptr);
  std::vector<GenerationResult> generate_batch(
      std::span<const BatchGenerationRequest> requests,
      BatchRefillCallback refill = {});
  TargetPrefillBenchmark prefill_chunks(
      KvCache& destination, std::span<const std::uint32_t> input_ids,
      std::span<const std::uint8_t> multimodal_types,
      std::span<const float> projected_images, void* device_final_state,
      int initial_prefix = 0,
      const void* device_projected_images = nullptr);
  Fp8LinearRunner& prefill_fp8_for_tokens(int tokens) {
    if (tokens < 1 || tokens > 4608)
      throw std::runtime_error("invalid FP8 prefill token count");
    const int bucket = (tokens + 15) & ~15;
    auto found = prefill_fp8_by_tokens.find(bucket);
    if (found != prefill_fp8_by_tokens.end()) return *found->second;
    // Bound unusual long-running workloads while retaining every shape in
    // the fixed benchmark corpus. Prefill synchronizes before this point, so
    // evicting an inactive runner is safe.
    if (prefill_fp8_by_tokens.size() >= 64)
      prefill_fp8_by_tokens.erase(prefill_fp8_by_tokens.begin());
    auto runner = std::make_unique<Fp8LinearRunner>(
        weights.model(), target_dense_fp8_model, bucket, false, 0, 3);
    auto* result = runner.get();
    prefill_fp8_by_tokens.emplace(bucket, std::move(runner));
    return *result;
  }
  Fp8LinearRunner* vision_fp8_for_patches(int patches) {
    if (!vision_fp8_model) return nullptr;
    auto found = vision_fp8_by_patches.find(patches);
    if (found != vision_fp8_by_patches.end()) return found->second.get();
    if (vision_fp8_by_patches.size() >= 2)
      vision_fp8_by_patches.erase(vision_fp8_by_patches.begin());
    auto runner = std::make_unique<Fp8LinearRunner>(
        weights.model(), *vision_fp8_model, patches, false,
        (patches + 15) & ~15, 1, true);
    auto* result = runner.get();
    vision_fp8_by_patches.emplace(patches, std::move(runner));
    return result;
  }

  std::chrono::steady_clock::time_point load_begin;
  MultimodalRuntimePaths paths;
  int context_limit;
  bool padded_verifier_fp8_rows =
      std::getenv("G4_PADDED_VERIFIER_FP8_ROWS") != nullptr;
  Tokenizer tokenizer;
  RuntimeWeights weights;
  AssistantWeights assistant;
  DeviceModel target_vocab_model;
  DeviceModel assistant_vocab_model;
  DeviceModel target_dense_fp8_model;
  std::unique_ptr<DeviceModel> target_vocab_nvfp4_model;
  std::unique_ptr<DeviceModel> assistant_vocab_nvfp4_model;
  std::unique_ptr<DeviceModel> assistant_dense_fp8_model;
  std::unique_ptr<DeviceModel> vision_fp8_model;
  std::unique_ptr<VisionFusedWeights> vision_fused_weights;
  GpuExecutionContext execution;
  GpuExecutionContext admission_execution;
  CompressedVocabRunner target_vocab;
  CompressedVocabRunner assistant_vocab;
  std::unique_ptr<Nvfp4VocabRunner> assistant_vocab_nvfp4;
  std::unique_ptr<Nvfp4VocabRunner> assistant_hot_vocab_nvfp4;
  std::unique_ptr<Nvfp4VocabRunner> assistant_text_hot_vocab_nvfp4;
  std::unique_ptr<Nvfp4VocabRunner> assistant_multimodal_hot_vocab_nvfp4;
  std::unique_ptr<Nvfp4VocabRunner> assistant_combined_hot_vocab_nvfp4;
  int assistant_combined_text_vocab_rows{};
  int assistant_combined_multimodal_vocab_rows{};
  int assistant_hot_vocab_min_batch{9};
  std::array<std::unique_ptr<Nvfp4VocabRunner>, 8> prefill_vocab_nvfp4;
  std::unique_ptr<Fp8LinearRunner> assistant_fp8_linears;
  Fp8LinearRunner target_fp8_linears;
  std::unordered_map<int, std::unique_ptr<Fp8LinearRunner>>
      prefill_fp8_by_tokens;
  std::unordered_map<int, std::unique_ptr<Fp8LinearRunner>>
      vision_fp8_by_patches;
  KvCache cache;
  CandidateKvCache candidates;
  AttentionWorkspace workspace;
  DeviceScratchPool assistant_scratch;
  DeviceScratchPool verifier_scratch;
  DeviceScratchPool vision_scratch;
  DeviceScratchPool admission_vision_scratch;
  DeviceScratchPool prefill_scratch;
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> verifier_expert_owners;
  std::vector<std::unique_ptr<Nvfp4ExpertRunner>> prefill_expert_owners;
  std::array<Nvfp4ExpertRunner*, 30> verifier_experts{};
  std::array<Nvfp4ExpertRunner*, 30> prefill_experts{};
  void* continuation_state_device{};
  void* vision_projected_device{};
  int* drafted_tokens_device{};
  std::unique_ptr<BatchResources> batch;
  double load_ms{};
};

double GenerationResult::draft_acceptance() const {
  const int proposed = std::accumulate(
      cycles.begin(), cycles.end(), 0,
      [](int total, const GenerationCycle& cycle) {
        return total + cycle.draft_count;
      });
  return proposed == 0 ? 0.0
                       : static_cast<double>(accepted_drafts) / proposed;
}

double GenerationResult::decode_postprefill_tokens_per_second() const {
  const double decode_ms = assistant_milliseconds + verifier_milliseconds;
  return decode_ms > 0.0 && token_ids.size() > 1
             ? 1000.0 * static_cast<double>(token_ids.size() - 1) / decode_ms
             : 0.0;
}

MultimodalRuntime::MultimodalRuntime(MultimodalRuntimePaths paths,
                                     int maximum_context)
    : impl_(std::make_unique<Impl>(std::move(paths), maximum_context)) {}

MultimodalRuntime::~MultimodalRuntime() = default;

double MultimodalRuntime::startup_milliseconds() const { return impl_->load_ms; }

int MultimodalRuntime::maximum_context() const { return impl_->context_limit; }

GenerationResult MultimodalRuntime::generate(const std::filesystem::path& image_path,
                                             std::string_view prompt_text,
                                             int maximum_tokens) {
  if (maximum_tokens < 1 || maximum_tokens > kNativeContextTokens)
    throw std::runtime_error("maximum_tokens must be in [1, 262144]");
  const auto wall_begin = std::chrono::steady_clock::now();
  const auto image = load_rgb_image(image_path);
  const auto image_loaded = std::chrono::steady_clock::now();
  const auto vision_input = preprocess_gemma4_image(
      image, 280, std::getenv("G4_PADDED_VISION_INPUT") == nullptr);
  const auto image_preprocessed = std::chrono::steady_clock::now();
  const std::array counts{vision_input.soft_token_count};
  const auto chat_prompt = format_gemma4_user_turn(prompt_text);
  const auto prompt = prepare_image_prompt(impl_->tokenizer, chat_prompt, counts);
  const auto vision = benchmark_vision_encoder(
      impl_->weights.model(), vision_input.pixel_values,
      vision_input.position_ids, vision_input.patch_count, {}, false,
      &impl_->vision_scratch, &impl_->execution, 1,
      impl_->vision_projected_device, false,
      impl_->vision_fp8_for_patches(vision_input.patch_count), {}, {},
      impl_->vision_fused_weights.get());

  return impl_->generate_prepared(
      prompt.input_ids, prompt.multimodal_types, {},
      maximum_tokens, wall_begin,
      std::chrono::duration<double, std::milli>(image_loaded - wall_begin)
          .count(),
      std::chrono::duration<double, std::milli>(image_preprocessed - image_loaded)
          .count(),
      vision.microseconds, false, {}, impl_->vision_projected_device);
}

GenerationResult MultimodalRuntime::generate(
    std::span<const std::uint8_t> encoded_image, std::string_view prompt_text,
    int maximum_tokens) {
  return generate(encoded_image, {}, prompt_text, maximum_tokens);
}

GenerationResult MultimodalRuntime::generate(
    std::span<const std::uint8_t> encoded_image, std::string_view system,
    std::string_view prompt_text, int maximum_tokens,
    bool stop_on_json_object,
    std::span<const std::string> expected_json_ids) {
  if (maximum_tokens < 1 || maximum_tokens > kNativeContextTokens)
    throw std::runtime_error("maximum_tokens must be in [1, 262144]");
  const auto wall_begin = std::chrono::steady_clock::now();
  const auto image = load_rgb_image(encoded_image);
  const auto image_loaded = std::chrono::steady_clock::now();
  const auto vision_input = preprocess_gemma4_image(
      image, 560, std::getenv("G4_PADDED_VISION_INPUT") == nullptr);
  const auto image_preprocessed = std::chrono::steady_clock::now();
  const std::array counts{vision_input.soft_token_count};
  std::string content(prompt_text);
  content += "<|image|>";
  const auto chat_prompt = system.empty()
      ? format_gemma4_user_turn(content)
      : format_gemma4_system_user_turn(system, content);
  const auto prompt = prepare_image_prompt(impl_->tokenizer, chat_prompt, counts);
  const auto vision = benchmark_vision_encoder(
      impl_->weights.model(), vision_input.pixel_values,
      vision_input.position_ids, vision_input.patch_count, {}, false,
      &impl_->vision_scratch, &impl_->execution, 1,
      impl_->vision_projected_device, false,
      impl_->vision_fp8_for_patches(vision_input.patch_count), {}, {},
      impl_->vision_fused_weights.get());
  return impl_->generate_prepared(
      prompt.input_ids, prompt.multimodal_types, {},
      maximum_tokens, wall_begin,
      std::chrono::duration<double, std::milli>(image_loaded - wall_begin).count(),
      std::chrono::duration<double, std::milli>(image_preprocessed - image_loaded).count(),
      vision.microseconds, stop_on_json_object, expected_json_ids,
      impl_->vision_projected_device);
}

GenerationResult MultimodalRuntime::generate(std::string_view prompt_text,
                                             int maximum_tokens) {
  if (maximum_tokens < 1 || maximum_tokens > kNativeContextTokens)
    throw std::runtime_error("maximum_tokens must be in [1, 262144]");
  const auto wall_begin = std::chrono::steady_clock::now();
  const auto input_ids =
      impl_->tokenizer.encode(format_gemma4_user_turn(prompt_text));
  const std::vector<std::uint8_t> multimodal_types(input_ids.size(), 0);
  return impl_->generate_prepared(input_ids, multimodal_types, {},
                                  maximum_tokens, wall_begin, 0.0, 0.0, 0.0F);
}

std::vector<GenerationResult> MultimodalRuntime::generate_batch(
    std::span<const BatchGenerationRequest> requests) {
  return impl_->generate_batch(requests);
}

void MultimodalRuntime::generate_continuous(
    std::span<const BatchGenerationRequest> initial_requests,
    BatchRefillCallback refill) {
  impl_->generate_batch(initial_requests, std::move(refill));
}

TargetPrefillBenchmark MultimodalRuntime::Impl::prefill_chunks(
    KvCache& destination, std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images, void* device_final_state,
    int initial_prefix, const void* device_projected_images) {
  constexpr int kChunkTokens = 4608;
  constexpr int kHidden = 2816;
  if (input_ids.empty() || input_ids.size() != multimodal_types.size())
    throw std::runtime_error("invalid chunked prefill inputs");
  const std::size_t image_tokens = static_cast<std::size_t>(std::count(
      multimodal_types.begin(), multimodal_types.end(), std::uint8_t{1}));
  if ((!device_projected_images && projected_images.size() != image_tokens * kHidden) ||
      (device_projected_images && !projected_images.empty()))
    throw std::runtime_error("chunked prefill image features do not match slots");
  const bool force_chunked = std::getenv("G4_FORCE_CHUNKED_PREFILL") != nullptr;
  if (!force_chunked &&
      initial_prefix + static_cast<int>(input_ids.size()) <= 4608)
    return benchmark_target_prefill(
        weights, target_vocab_model, destination, input_ids, multimodal_types,
        projected_images, 30, false, prefill_experts, &prefill_scratch,
        &target_vocab, &execution, device_final_state, initial_prefix,
        &prefill_fp8_for_tokens(static_cast<int>(input_ids.size())),
        device_projected_images);

  TargetPrefillBenchmark result;
  float total_microseconds = 0.0F;
  std::size_t first = 0;
  std::size_t image_first = 0;
  while (first < input_ids.size()) {
    std::size_t end = std::min(input_ids.size(), first + kChunkTokens);
    // Gemma's image tokens form a bidirectional attention island. If the
    // boundary lands inside one, move it to the island's beginning so the
    // chunk never exceeds the 4608-row execution plan. Only an image island
    // larger than the complete plan is intrinsically unsupported.
    if (end < input_ids.size() && end > first &&
        multimodal_types[end - 1] == 1 && multimodal_types[end] == 1) {
      std::size_t island_begin = end;
      while (island_begin > first && multimodal_types[island_begin - 1] == 1)
        --island_begin;
      if (island_begin > first) {
        end = island_begin;
      } else {
        while (end < input_ids.size() && multimodal_types[end] == 1) ++end;
        if (end - first > kChunkTokens)
          throw std::runtime_error("multimodal island exceeds prefill plan");
      }
    }
    const auto chunk_types = multimodal_types.subspan(first, end - first);
    const std::size_t chunk_images = static_cast<std::size_t>(std::count(
        chunk_types.begin(), chunk_types.end(), std::uint8_t{1}));
    result = benchmark_target_prefill(
        weights, target_vocab_model, destination,
        input_ids.subspan(first, end - first), chunk_types,
        device_projected_images
            ? std::span<const float>{}
            : projected_images.subspan(image_first * kHidden,
                                       chunk_images * kHidden),
        30, false, prefill_experts, &prefill_scratch, &target_vocab,
        &execution, device_final_state,
        initial_prefix + static_cast<int>(first),
        &prefill_fp8_for_tokens(static_cast<int>(end - first)),
        device_projected_images
            ? static_cast<const std::uint16_t*>(device_projected_images) +
                  image_first * kHidden
            : nullptr);
    total_microseconds += result.microseconds;
    image_first += chunk_images;
    first = end;
  }
  result.microseconds = total_microseconds;
  result.tokens = initial_prefix + static_cast<int>(input_ids.size());
  return result;
}

std::vector<GenerationResult> MultimodalRuntime::Impl::generate_batch(
    std::span<const BatchGenerationRequest> requests,
    BatchRefillCallback refill) {
  if (requests.empty() || requests.size() > 8)
    throw std::runtime_error("generation batch must contain 1..8 requests");
  if (!batch) batch = std::make_unique<BatchResources>(*this);
  const int count = static_cast<int>(requests.size());
  const int capacity = refill ? 8 : count;
  std::vector<BatchGenerationRequest> session_requests(requests.begin(),
                                                       requests.end());
  session_requests.resize(capacity);
  struct PreparedRequest {
    MultimodalPrompt prompt;
    std::vector<float> projected;
    const void* device_projected{};
    VisionInput vision_input;
    std::chrono::steady_clock::time_point loaded;
    std::chrono::steady_clock::time_point preprocessed;
    float vision_microseconds{};
    double cpu_prepare_milliseconds{};
  };
  std::vector<PreparedRequest> prepared(capacity);
  std::vector<GenerationResult> results(capacity);
  std::vector<int> cache_tokens(capacity);
  std::vector<int> maximum_tokens(capacity);
  std::vector<JsonObjectBoundary> json_boundaries(capacity);
  std::vector<bool> json_complete(capacity);
  std::vector<std::chrono::steady_clock::time_point> wall_begin(capacity);
  const bool async_admission =
      std::getenv("G4_SERIAL_ADMISSION") == nullptr;
  double prepare_wall_ms = 0.0;
  std::mutex prepare_wall_mutex;
  double prefix_clone_wall_ms = 0.0;
  double prefill_wall_ms = 0.0;
  std::uint64_t packed_prefill_waves = 0;
  std::uint64_t packed_prefill_rows = 0;
  std::uint64_t packed_prefill_sessions = 0;

  auto prepare_cpu = [&](int slot) {
    const auto prepare_begin = std::chrono::steady_clock::now();
    const auto& request = session_requests[slot];
    if (request.maximum_tokens < 1 ||
        request.maximum_tokens > kNativeContextTokens)
      throw std::runtime_error("maximum_tokens must be in [1, 262144]");
    wall_begin[slot] = std::chrono::steady_clock::now();
    RgbImage image;
    if (!request.encoded_image.empty()) {
      nvtxRangePushA("refill/image_decode");
      image = load_rgb_image(request.encoded_image);
      nvtxRangePop();
    }
    const auto loaded = std::chrono::steady_clock::now();
    VisionInput vision_input;
    if (!request.encoded_image.empty()) {
      nvtxRangePushA("refill/image_preprocess");
      vision_input = preprocess_gemma4_image(
          image, 560, std::getenv("G4_PADDED_VISION_INPUT") == nullptr,
          std::getenv("G4_IMAGE_WORKERS")
              ? std::clamp(std::atoi(std::getenv("G4_IMAGE_WORKERS")), 1, 8)
              : 8);
      nvtxRangePop();
    }
    const auto preprocessed = std::chrono::steady_clock::now();
    std::string content(request.prompt);
    if (!request.encoded_image.empty()) content += "<|image|>";
    const auto rendered = request.system.empty()
        ? format_gemma4_user_turn(content)
        : format_gemma4_system_user_turn(request.system, content);
    auto& item = prepared[slot];
    if (request.encoded_image.empty()) {
      item.prompt.input_ids = tokenizer.encode(rendered);
      item.prompt.multimodal_types.assign(item.prompt.input_ids.size(), 0);
    } else {
      const std::array image_counts{vision_input.soft_token_count};
      item.prompt = prepare_image_prompt(tokenizer, rendered, image_counts);
    }
    item.vision_input = std::move(vision_input);
    item.loaded = loaded;
    item.preprocessed = preprocessed;
    if (item.prompt.input_ids.size() + 1U >
        static_cast<std::size_t>(context_limit))
      throw std::runtime_error("prompt exceeds the configured model context");
    maximum_tokens[slot] = std::min(
        request.maximum_tokens,
        context_limit - static_cast<int>(item.prompt.input_ids.size()));
    if (maximum_tokens[slot] < 1)
      throw std::runtime_error("prompt leaves no serving context for generation");
    item.cpu_prepare_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prepare_begin).count();
  };
  auto prepare_gpu = [&](int slot, int lane = 0) {
    const auto gpu_begin = std::chrono::steady_clock::now();
    auto& item = prepared[slot];
    if (item.vision_input.pixel_values.empty()) {
      item.projected.clear();
      item.device_projected = nullptr;
      item.vision_microseconds = 0.0F;
      std::lock_guard lock(prepare_wall_mutex);
      prepare_wall_ms += item.cpu_prepare_milliseconds;
      return;
    }
    auto& lane_scratch = lane == 0 ? vision_scratch : admission_vision_scratch;
    auto& lane_execution = lane == 0 ? execution : admission_execution;
    nvtxRangePushA("refill/vision");
    auto vision = benchmark_vision_encoder(
        weights.model(), item.vision_input.pixel_values,
        item.vision_input.position_ids, item.vision_input.patch_count, {},
        false, &lane_scratch, &lane_execution, 1,
        batch->vision_projected[slot], false,
        vision_fp8_for_patches(item.vision_input.patch_count), {}, {},
        vision_fused_weights.get());
    nvtxRangePop();
    item.projected.clear();
    item.device_projected = batch->vision_projected[slot];
    item.vision_microseconds = vision.microseconds;
    item.vision_input = {};
    const double elapsed = item.cpu_prepare_milliseconds +
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
    std::lock_guard lock(prepare_wall_mutex);
    prepare_wall_ms += elapsed;
  };
  auto prepare_request = [&](int slot) {
    prepare_cpu(slot);
    prepare_gpu(slot);
  };
  if (async_admission && count > 1) {
    std::vector<std::future<void>> initial_cpu;
    initial_cpu.reserve(count);
    for (int slot = 0; slot < count; ++slot)
      initial_cpu.push_back(std::async(std::launch::async,
                                      [&, slot] { prepare_cpu(slot); }));
    const bool pair_vision_requested =
        std::getenv("G4_PAIR_VISION") != nullptr;
    const bool batch_vision_requested =
        std::getenv("G4_BATCH_VISION") != nullptr || pair_vision_requested;
    if (pair_vision_requested) {
      // Each pair below consumes its CPU futures just before GPU submission,
      // overlapping later image preprocessing with the first vision pair.
    } else if (batch_vision_requested)
      for (auto& task : initial_cpu) task.get();
    else if (std::getenv("G4_CONCURRENT_VISION") != nullptr) {
      for (int slot = 0; slot < count; slot += 2) {
        initial_cpu[slot].get();
        auto first = std::async(std::launch::async,
                                [&, slot] { prepare_gpu(slot, 0); });
        if (slot + 1 < count) {
          initial_cpu[slot + 1].get();
          prepare_gpu(slot + 1, 1);
        }
        first.get();
      }
    } else {
      for (int slot = 0; slot < count; ++slot) {
        initial_cpu[slot].get();
        prepare_gpu(slot);
      }
    }
    bool uniform_vision = !pair_vision_requested &&
        !prepared[0].vision_input.pixel_values.empty();
    if (!pair_vision_requested)
      for (int slot = 1; slot < count; ++slot) {
        uniform_vision &=
            prepared[slot].vision_input.patch_count ==
                prepared[0].vision_input.patch_count &&
            prepared[slot].vision_input.pixel_values.size() ==
                prepared[0].vision_input.pixel_values.size() &&
            prepared[slot].vision_input.position_ids.size() ==
                prepared[0].vision_input.position_ids.size();
      }
    // The true batched path is retained for profiler-driven development, but
    // larger BF16 GEMM geometry is not token-equivalent to the validated
    // per-image schedule yet.
    if (pair_vision_requested) {
      for (int slot = 0; slot < count; slot += 2) {
        initial_cpu[slot].get();
        if (slot + 1 < count) initial_cpu[slot + 1].get();
        const bool paired = slot + 1 < count &&
            prepared[slot].vision_input.patch_count ==
                prepared[slot + 1].vision_input.patch_count &&
            prepared[slot].vision_input.pixel_values.size() ==
                prepared[slot + 1].vision_input.pixel_values.size() &&
            prepared[slot].vision_input.position_ids.size() ==
                prepared[slot + 1].vision_input.position_ids.size();
        if (!paired) {
          prepare_gpu(slot);
          continue;
        }
        const auto gpu_begin = std::chrono::steady_clock::now();
        const auto& first_input = prepared[slot].vision_input;
        const auto& second_input = prepared[slot + 1].vision_input;
        auto* pair_output =
            static_cast<std::uint16_t*>(batch->vision_projected_packed) +
            static_cast<std::size_t>(slot) * 560 * 2816;
        nvtxRangePushA("batch/vision_pair");
        auto vision = benchmark_vision_encoder(
            weights.model(), first_input.pixel_values, first_input.position_ids,
            first_input.patch_count, {}, false, &vision_scratch, &execution, 2,
            pair_output, false, nullptr, second_input.pixel_values,
            second_input.position_ids, vision_fused_weights.get());
        nvtxRangePop();
        const std::size_t projected_per_image =
            static_cast<std::size_t>(
                prepared[slot].vision_input.patch_count / 9) * 2816;
        for (int local = 0; local < 2; ++local) {
          auto& item = prepared[slot + local];
          item.projected.clear();
          item.device_projected = pair_output +
              static_cast<std::size_t>(local) * projected_per_image;
          item.vision_microseconds = vision.microseconds;
          item.vision_input = {};
          prepare_wall_ms += item.cpu_prepare_milliseconds;
        }
        prepare_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpu_begin).count();
      }
    } else if (uniform_vision && batch_vision_requested) {
      const auto gpu_begin = std::chrono::steady_clock::now();
      std::vector<float> pixels;
      std::vector<std::int32_t> positions;
      pixels.reserve(prepared[0].vision_input.pixel_values.size() * count);
      positions.reserve(prepared[0].vision_input.position_ids.size() * count);
      for (int slot = 0; slot < count; ++slot) {
        const auto& input = prepared[slot].vision_input;
        pixels.insert(pixels.end(), input.pixel_values.begin(),
                      input.pixel_values.end());
        positions.insert(positions.end(), input.position_ids.begin(),
                         input.position_ids.end());
      }
      nvtxRangePushA("batch/vision");
      auto vision = benchmark_vision_encoder(
          weights.model(), pixels, positions,
          prepared[0].vision_input.patch_count, {}, false, &vision_scratch,
          &execution, count, batch->vision_projected_packed, false, nullptr,
          {}, {}, vision_fused_weights.get());
      nvtxRangePop();
      const std::size_t projected_per_image =
          static_cast<std::size_t>(prepared[0].vision_input.patch_count / 9) *
          2816;
      for (int slot = 0; slot < count; ++slot) {
        auto& item = prepared[slot];
        item.projected.clear();
        item.device_projected =
            static_cast<const std::uint16_t*>(batch->vision_projected_packed) +
            static_cast<std::size_t>(slot) * projected_per_image;
        item.vision_microseconds = vision.microseconds;
        item.vision_input = {};
        prepare_wall_ms += item.cpu_prepare_milliseconds;
      }
      prepare_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - gpu_begin).count();
    } else if (batch_vision_requested) {
      for (int slot = 0; slot < count; ++slot) prepare_gpu(slot);
    }
  } else {
    for (int slot = 0; slot < count; ++slot) prepare_request(slot);
  }

  // Cache the longest all-text prefix shared by the complete admitted cohort.
  // Global layers retain all of it; sliding layers clone only their physical
  // 1024-token ring, whose slots already use absolute-position modulo layout.
  int common_prefix =
      static_cast<int>(prepared.front().prompt.input_ids.size()) - 1;
  for (int slot = 1; slot < count; ++slot)
    common_prefix = std::min<int>(
        common_prefix,
        static_cast<int>(prepared[slot].prompt.input_ids.size()) - 1);
  for (int token = 0; token < common_prefix; ++token) {
    const auto id = prepared.front().prompt.input_ids[token];
    bool common = prepared.front().prompt.multimodal_types[token] == 0;
    for (int slot = 1; slot < count && common; ++slot)
      common = prepared[slot].prompt.input_ids[token] == id &&
               prepared[slot].prompt.multimodal_types[token] == 0;
    if (!common) {
      common_prefix = token;
      break;
    }
  }
  auto copy_prefix = [&](KvCache& source, KvCache& destination, int tokens,
                         cudaStream_t stream) {
    source.ensure_context(tokens);
    destination.ensure_context(tokens);
    if (source.sliding_bytes() != destination.sliding_bytes())
      throw std::runtime_error("shared prefix sliding arena mismatch");
    if (tokens <= 1024 &&
        std::getenv("G4_FULL_SLIDING_PREFIX_COPY") == nullptr) {
      copy_sliding_prefix_arena(source.sliding_arena(),
                                destination.sliding_arena(), tokens, stream);
    } else if (cudaMemcpyAsync(destination.sliding_arena(),
                               source.sliding_arena(), source.sliding_bytes(),
                               cudaMemcpyDeviceToDevice, stream) !=
               cudaSuccess) {
      throw std::runtime_error("clone shared sliding prefix arena failed");
    }
    for (int layer = 0; layer < 30; ++layer) {
      const auto& source_layer = source.layer(layer);
      const auto& destination_layer = destination.layer(layer);
      const bool sliding =
          source_layer.kv_heads == 8 && source_layer.capacity == 1024;
      if (sliding) continue;
      const int resident_tokens =
          std::min(tokens, std::min(source_layer.capacity,
                                    destination_layer.capacity));
      const std::size_t bytes = static_cast<std::size_t>(resident_tokens) *
                                source_layer.kv_heads * source_layer.head_dim *
                                2;
      if (cudaMemcpyAsync(destination_layer.keys, source_layer.keys, bytes,
                          cudaMemcpyDeviceToDevice, stream) !=
              cudaSuccess ||
          cudaMemcpyAsync(destination_layer.values, source_layer.values,
                          bytes, cudaMemcpyDeviceToDevice, stream) !=
              cudaSuccess)
        throw std::runtime_error("clone shared prefix KV failed");
    }
  };
  auto clone_prefix = [&](int slot, int tokens) {
    const auto clone_begin = std::chrono::steady_clock::now();
    if (batch->prefix_copy_pending &&
        cudaStreamWaitEvent(execution.stream(), batch->prefix_copy_ready, 0) !=
            cudaSuccess)
      throw std::runtime_error("wait for shared prefix copy failed");
    copy_prefix(*batch->prefix_cache, *batch->caches[slot], tokens,
                execution.stream());
    prefix_clone_wall_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - clone_begin).count();
  };
  auto cache_prefix_after_response = [&](int slot, int tokens) {
    if (batch->prefix_copy_pending &&
        cudaStreamWaitEvent(batch->prefix_copy_stream,
                            batch->prefix_copy_ready, 0) != cudaSuccess)
      throw std::runtime_error("serialize shared prefix copies failed");
    if (cudaEventRecord(batch->prefix_source_ready, execution.stream()) !=
            cudaSuccess ||
        cudaStreamWaitEvent(batch->prefix_copy_stream,
                            batch->prefix_source_ready, 0) != cudaSuccess)
      throw std::runtime_error("order shared prefix source copy failed");
    copy_prefix(*batch->caches[slot], *batch->prefix_cache, tokens,
                batch->prefix_copy_stream);
    if (cudaEventRecord(batch->prefix_copy_ready,
                        batch->prefix_copy_stream) != cudaSuccess)
      throw std::runtime_error("record shared prefix copy completion failed");
    batch->prefix_copy_pending = true;
    batch->prefix_copy_source_slot = slot;
  };
  int deferred_prefix_tokens = 0;
  bool prioritize_first_shared_request = false;
  const bool shared_prefix_cache =
      std::getenv("G4_DISABLE_SHARED_PREFIX_CACHE") == nullptr;
  const char* shared_prefix_minimum_text =
      std::getenv("G4_SHARED_PREFIX_MIN_TOKENS");
  const int shared_prefix_minimum = shared_prefix_minimum_text
      ? std::clamp(std::atoi(shared_prefix_minimum_text), 1, 1024)
      : 128;
  if (shared_prefix_cache && common_prefix >= shared_prefix_minimum) {
    const auto& first_ids = prepared.front().prompt.input_ids;
    const bool cache_hit =
        batch->cached_prefix_ids.size() == static_cast<std::size_t>(common_prefix) &&
        std::equal(batch->cached_prefix_ids.begin(),
                   batch->cached_prefix_ids.end(), first_ids.begin());
    // On this GPU, warmed B8 measurements put the cold construct-and-clone
    // crossover around 153 tokens. Require a clear margin: shorter cohorts
    // use the ordinary packed prefill, while an existing warm hit can still
    // clone its already-computed prefix.
    if (!cache_hit && count > 1 && common_prefix < 256) {
      common_prefix = 0;
    } else if (!cache_hit && count == 1) {
      // A first request is never delayed to construct a cache for later work.
      // Serve it through the ordinary prefill path, then copy its immutable
      // prefix after the completion callback has made the response visible.
      deferred_prefix_tokens = common_prefix;
      common_prefix = 0;
    } else if (!cache_hit) {
      if (batch->prefix_copy_pending &&
          cudaStreamWaitEvent(execution.stream(), batch->prefix_copy_ready,
                              0) != cudaSuccess)
        throw std::runtime_error("wait before replacing shared prefix failed");
      const std::vector<std::uint8_t> prefix_types(common_prefix, 0);
      prefill_chunks(*batch->prefix_cache,
                     std::span(first_ids).first(common_prefix), prefix_types,
                     {}, batch->states_a);
      batch->cached_prefix_ids.assign(first_ids.begin(),
                                      first_ids.begin() + common_prefix);
      prioritize_first_shared_request = common_prefix >= 4096;
    }
    if (common_prefix) {
      const int immediate_clones = prioritize_first_shared_request ? 1 : count;
      for (int slot = 0; slot < immediate_clones; ++slot)
        clone_prefix(slot, common_prefix);
    }
  } else {
    common_prefix = 0;
  }

  std::array<float, 8> prefill_accumulated_us{};
  auto apply_prefill = [&](int slot, const TargetPrefillBenchmark& prefill) {
    const auto& request = session_requests[slot];
    const auto& item = prepared[slot];
    auto& result = results[slot];
    result = {};
    json_boundaries[slot] = {};
    json_complete[slot] = false;
    result.prompt_tokens = static_cast<int>(item.prompt.input_ids.size());
    result.token_ids.reserve(maximum_tokens[slot] + 4);
    result.token_ids.push_back(prefill.selected_token);
    if (request.on_tokens)
      request.on_tokens(tokenizer.decode_token(result.token_ids.back(), true));
    if (request.stop_on_json_object)
      json_complete[slot] = json_boundaries[slot].feed(
          tokenizer.decode_token(result.token_ids.back(), true),
          request.expected_json_ids);
    result.prefill_token = prefill.selected_token;
    result.prefill_checksum = prefill.hidden_checksum;
    result.first_token_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_begin[slot]).count();
    result.image_decode_milliseconds =
        std::chrono::duration<double, std::milli>(item.loaded - wall_begin[slot]).count();
    result.image_preprocess_milliseconds =
        std::chrono::duration<double, std::milli>(item.preprocessed - item.loaded).count();
    result.vision_microseconds = item.vision_microseconds;
    result.prefill_microseconds = prefill_accumulated_us[slot];
    cache_tokens[slot] = result.prompt_tokens;
  };
  auto prefill_request = [&](int slot, void* state_base, int state_row,
                             int prefix_tokens) {
    const auto prefill_begin = std::chrono::steady_clock::now();
    if (batch->prefix_copy_pending &&
        batch->prefix_copy_source_slot == slot &&
        cudaStreamWaitEvent(execution.stream(), batch->prefix_copy_ready, 0) !=
            cudaSuccess)
      throw std::runtime_error("wait before reusing prefix source cache failed");
    const auto& item = prepared[slot];
    const auto suffix_ids =
        std::span(item.prompt.input_ids).subspan(prefix_tokens);
    const auto suffix_types =
        std::span(item.prompt.multimodal_types).subspan(prefix_tokens);
    nvtxRangePushA("refill/prefill");
    auto prefill = prefill_chunks(
        *batch->caches[slot], suffix_ids, suffix_types, item.projected,
        static_cast<std::uint16_t*>(state_base) +
            static_cast<std::size_t>(state_row) * 2816,
        prefix_tokens,
        item.device_projected
            ? static_cast<const std::uint16_t*>(item.device_projected) +
                  static_cast<std::size_t>(std::count(
                      item.prompt.multimodal_types.begin(),
                      item.prompt.multimodal_types.begin() + prefix_tokens,
                      std::uint8_t{1})) * 2816
            : nullptr);
    nvtxRangePop();
    prefill_accumulated_us[slot] = prefill.microseconds;
    apply_prefill(slot, prefill);
    prefill_wall_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prefill_begin).count();
  };
  const bool batch_prefill = count > 1 &&
      std::getenv("G4_DISABLE_BATCH_PREFILL") == nullptr;
  auto prefill_slot_batch = [&](std::span<const int> slots,
                                std::span<const int> initial_prefixes,
                                void* state_base, int first_state_row) {
    if (slots.empty() || slots.size() != initial_prefixes.size() ||
        first_state_row < 0 ||
        first_state_row + static_cast<int>(slots.size()) > 8)
      throw std::runtime_error("invalid packed prefill slot group");
    if (batch->prefix_copy_pending &&
        std::find(slots.begin(), slots.end(),
                  batch->prefix_copy_source_slot) != slots.end() &&
        cudaStreamWaitEvent(execution.stream(), batch->prefix_copy_ready, 0) !=
            cudaSuccess)
      throw std::runtime_error("wait before reusing prefix source batch failed");
    std::vector<int> cursor(initial_prefixes.begin(), initial_prefixes.end());
    for (const int slot : slots) prefill_accumulated_us[slot] = 0.0F;
    int unfinished = static_cast<int>(slots.size());
    while (unfinished) {
      const auto wave_begin = std::chrono::steady_clock::now();
      std::vector<TargetPrefillBatchItem> batch_items;
      std::vector<int> batch_slots;
      std::vector<int> batch_local_rows;
      int budget = 4608;
      int candidates = unfinished;
      for (int local = 0; local < static_cast<int>(slots.size()) && budget > 0;
           ++local) {
        const int slot = slots[local];
        const auto& item = prepared[slot];
        const int total = static_cast<int>(item.prompt.input_ids.size());
        if (cursor[local] >= total) continue;
        const int share = std::max(1, budget / std::max(1, candidates));
        const int begin = cursor[local];
        int end = std::min(total, begin + share);
        const auto& types = item.prompt.multimodal_types;
        if (end < total && end > begin &&
            types[end - 1] == 1 && types[end] == 1) {
          int island_begin = end;
          while (island_begin > begin && types[island_begin - 1] == 1)
            --island_begin;
          if (island_begin > begin) {
            end = island_begin;
          } else {
            while (end < total && types[end] == 1) ++end;
            if (end - begin > budget) {
              if (!batch_items.empty()) break;
              throw std::runtime_error(
                  "multimodal island exceeds batch prefill budget");
            }
          }
        }
        if (end <= begin) continue;
        const std::size_t image_begin = static_cast<std::size_t>(std::count(
            types.begin(), types.begin() + begin, std::uint8_t{1}));
        const std::size_t image_count = static_cast<std::size_t>(std::count(
            types.begin() + begin, types.begin() + end, std::uint8_t{1}));
        batch_items.push_back({
            batch->caches[slot].get(),
            std::span(item.prompt.input_ids).subspan(begin, end - begin),
            std::span(item.prompt.multimodal_types).subspan(begin, end - begin),
            item.device_projected
                ? std::span<const float>{}
                : std::span(item.projected)
                      .subspan(image_begin * 2816, image_count * 2816),
            item.device_projected && image_count
                ? static_cast<const std::uint16_t*>(item.device_projected) +
                      image_begin * 2816
                : nullptr,
            begin, first_state_row + local});
        batch_slots.push_back(slot);
        batch_local_rows.push_back(local);
        budget -= end - begin;
        --candidates;
      }
      if (batch_items.empty())
        throw std::runtime_error("batch prefill scheduler made no progress");
      const int rows = std::accumulate(
          batch_items.begin(), batch_items.end(), 0,
          [](int total, const auto& item) {
            return total + static_cast<int>(item.input_ids.size());
          });
      ++packed_prefill_waves;
      packed_prefill_rows += rows;
      packed_prefill_sessions += batch_items.size();
      nvtxRangePushA("batch/prefill");
      auto batch_results = benchmark_target_prefill_batch(
          weights, target_vocab_model, batch_items, 30, prefill_experts,
          &prefill_scratch, &target_vocab, &execution, state_base,
          &prefill_fp8_for_tokens(rows),
          prefill_vocab_nvfp4[batch_items.size() - 1].get());
      nvtxRangePop();
      for (std::size_t index = 0; index < batch_items.size(); ++index) {
        const int slot = batch_slots[index];
        const int local = batch_local_rows[index];
        prefill_accumulated_us[slot] += batch_results[index].microseconds;
        cursor[local] += static_cast<int>(batch_items[index].input_ids.size());
        if (cursor[local] ==
            static_cast<int>(prepared[slot].prompt.input_ids.size())) {
          apply_prefill(slot, batch_results[index]);
          --unfinished;
        }
      }
      prefill_wall_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wave_begin).count();
    }
  };
  if (prioritize_first_shared_request) {
    // On a cold, long cohort cache miss, make the earliest request visible
    // before its independent peers pay for their KV clones. This changes only
    // admission order; all sessions join the same decode batch afterward.
    prefill_request(0, batch->states_a, 0, common_prefix);
    for (int slot = 1; slot < count; ++slot) clone_prefix(slot, common_prefix);
    std::vector<int> slots(count - 1), prefixes(count - 1, common_prefix);
    std::iota(slots.begin(), slots.end(), 1);
    if (batch_prefill)
      prefill_slot_batch(slots, prefixes, batch->states_a, 1);
    else
      for (int slot = 1; slot < count; ++slot)
        prefill_request(slot, batch->states_a, slot, common_prefix);
  } else if (batch_prefill) {
    std::vector<int> slots(count), prefixes(count, common_prefix);
    std::iota(slots.begin(), slots.end(), 0);
    prefill_slot_batch(slots, prefixes, batch->states_a, 0);
  } else {
    for (int slot = 0; slot < count; ++slot)
      prefill_request(slot, batch->states_a, slot, common_prefix);
  }

  std::vector<int> active;
  void* current_states = batch->states_a;
  void* compact_states = batch->states_b;
  struct PendingAdmission {
    int slot{};
    std::future<void> prepared;
  };
  std::vector<PendingAdmission> pending_admissions;
  std::vector<int> vacant_slots;
  for (int slot = count; slot < capacity; ++slot)
    vacant_slots.push_back(slot);
  std::mutex admission_mutex;
  std::array<std::uint64_t, 9> cycles_by_batch{};
  std::array<double, 9> assistant_ms_by_batch{};
  std::array<double, 9> verifier_ms_by_batch{};
  std::array<std::uint64_t, 9> output_tokens_by_batch{};
  std::uint64_t refill_attempts = 0, refill_hits = 0;
  auto finish_request = [&](int slot) {
    auto& result = results[slot];
    const bool populate_deferred_prefix =
        slot == 0 && deferred_prefix_tokens >= 32 &&
        deferred_prefix_tokens + static_cast<int>(result.token_ids.size()) <=
            1024;
    result.length_limited =
        static_cast<int>(result.token_ids.size()) >= maximum_tokens[slot] &&
        !json_complete[slot] && !is_stop(result.token_ids.back());
    result.text = tokenizer.decode(result.token_ids);
    result.wall_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_begin[slot]).count();
    if (session_requests[slot].on_complete)
      session_requests[slot].on_complete(std::move(result));
    if (populate_deferred_prefix) {
      cache_prefix_after_response(slot, deferred_prefix_tokens);
      batch->cached_prefix_ids.assign(
          prepared[slot].prompt.input_ids.begin(),
          prepared[slot].prompt.input_ids.begin() + deferred_prefix_tokens);
      deferred_prefix_tokens = 0;
    }
  };
  auto reusable_prefix = [&](int slot) {
    const auto& ids = prepared[slot].prompt.input_ids;
    const int tokens = std::min<int>(
        batch->cached_prefix_ids.size(), static_cast<int>(ids.size()) - 1);
    if (tokens < 32 ||
        !std::equal(batch->cached_prefix_ids.begin(),
                    batch->cached_prefix_ids.begin() + tokens, ids.begin()))
      return 0;
    return tokens;
  };
  auto request_done = [&](int slot) {
    return json_complete[slot] || is_stop(results[slot].token_ids.back()) ||
           static_cast<int>(results[slot].token_ids.size()) >=
               maximum_tokens[slot];
  };
  for (int slot = 0; slot < count; ++slot) {
    bool live = true;
    while (request_done(slot)) {
      finish_request(slot);
      ++refill_attempts;
      if (!refill) {
        live = false;
        break;
      }
      auto replacement = refill();
      if (!replacement) {
        vacant_slots.push_back(slot);
        live = false;
        break;
      }
      ++refill_hits;
      session_requests[slot] = std::move(*replacement);
      prepare_request(slot);
      const int prefix_tokens = reusable_prefix(slot);
      if (prefix_tokens) clone_prefix(slot, prefix_tokens);
      prefill_request(slot, batch->states_a, slot, prefix_tokens);
    }
    if (live) active.push_back(slot);
  }
  if (!active.empty() && active.size() != static_cast<std::size_t>(count)) {
    for (int row = 0; row < static_cast<int>(active.size()); ++row)
      if (cudaMemcpyAsync(
              static_cast<std::uint16_t*>(compact_states) +
                  static_cast<std::size_t>(row) * 2816,
              static_cast<const std::uint16_t*>(current_states) +
                  static_cast<std::size_t>(active[row]) * 2816,
              2816 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
              execution.stream()) != cudaSuccess)
        throw std::runtime_error("compact initial continuation state failed");
    std::swap(current_states, compact_states);
  }
  auto admit_prepared = [&](bool wait_for_one) {
    if (wait_for_one && !pending_admissions.empty())
      pending_admissions.front().prepared.wait();
    std::vector<int> ready_slots;
    std::vector<int> ready_prefixes;
    for (auto item = pending_admissions.begin();
         item != pending_admissions.end();) {
      if (item->prepared.wait_for(std::chrono::seconds(0)) !=
          std::future_status::ready) {
        ++item;
        continue;
      }
      item->prepared.get();
      const int slot = item->slot;
      prepare_gpu(slot);
      const int prefix_tokens = reusable_prefix(slot);
      if (prefix_tokens) clone_prefix(slot, prefix_tokens);
      ready_slots.push_back(slot);
      ready_prefixes.push_back(prefix_tokens);
      item = pending_admissions.erase(item);
    }
    if (ready_slots.empty()) return;
    const int first_row = static_cast<int>(active.size());
    if (batch_prefill && ready_slots.size() > 1) {
      prefill_slot_batch(ready_slots, ready_prefixes, current_states,
                         first_row);
    } else {
      for (std::size_t index = 0; index < ready_slots.size(); ++index)
        prefill_request(ready_slots[index], current_states,
                        first_row + static_cast<int>(index),
                        ready_prefixes[index]);
    }
    for (std::size_t index = 0; index < ready_slots.size(); ++index) {
      const int slot = ready_slots[index];
      if (request_done(slot)) {
        finish_request(slot);
        ++refill_attempts;
        vacant_slots.push_back(slot);
        continue;
      }
      const int source_row = first_row + static_cast<int>(index);
      const int destination_row = static_cast<int>(active.size());
      if (destination_row != source_row && cudaMemcpyAsync(
              static_cast<std::uint16_t*>(current_states) + destination_row * 2816,
              static_cast<const std::uint16_t*>(current_states) + source_row * 2816,
              2816 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
              execution.stream()) != cudaSuccess)
        throw std::runtime_error("compact admitted continuation state failed");
      active.push_back(slot);
    }
  };
  auto poll_refills = [&] {
    if (!refill || vacant_slots.empty()) return;
    std::vector<int> completed_refill_slots;
    for (auto item = vacant_slots.begin(); item != vacant_slots.end();) {
      auto replacement = refill();
      if (!replacement) break;
      const int slot = *item;
      ++refill_hits;
      session_requests[slot] = std::move(*replacement);
      item = vacant_slots.erase(item);
      if (async_admission) {
        pending_admissions.push_back(PendingAdmission{
            slot, std::async(std::launch::async, [&, slot] {
              std::lock_guard lock(admission_mutex);
              prepare_cpu(slot);
            })});
      } else {
        prepare_request(slot);
        const int prefix_tokens = reusable_prefix(slot);
        if (prefix_tokens) clone_prefix(slot, prefix_tokens);
        prefill_request(slot, current_states, static_cast<int>(active.size()),
                        prefix_tokens);
        if (request_done(slot)) {
          finish_request(slot);
          ++refill_attempts;
          completed_refill_slots.push_back(slot);
        } else {
          active.push_back(slot);
        }
      }
    }
    vacant_slots.insert(vacant_slots.end(), completed_refill_slots.begin(),
                        completed_refill_slots.end());
  };
  while (!active.empty() || !pending_admissions.empty()) {
    poll_refills();
    admit_prepared(active.empty());
    if (active.empty()) continue;
    ++cycles_by_batch[active.size()];
    std::vector<KvCache*> active_caches;
    std::vector<CandidateKvCache*> active_candidates;
    std::vector<int> contexts;
    std::vector<int> previous;
    for (const int slot : active) {
      active_caches.push_back(batch->caches[slot].get());
      active_candidates.push_back(batch->candidates[slot].get());
      contexts.push_back(cache_tokens[slot]);
      previous.push_back(static_cast<int>(results[slot].token_ids.back()));
    }
    int cycle_drafts = serving_drafts(static_cast<int>(active.size()));
    for (const int slot : active) {
      const int output_room = maximum_tokens[slot] -
                              static_cast<int>(results[slot].token_ids.size());
      const int context_room = context_limit - cache_tokens[slot];
      cycle_drafts = std::min(
          cycle_drafts, std::max(0, std::min(output_room, context_room) - 1));
    }
    const auto assistant_begin = std::chrono::steady_clock::now();
    static constexpr std::array assistant_ranges{
        "", "decode/assistant/b1", "decode/assistant/b2",
        "decode/assistant/b3", "decode/assistant/b4", "decode/assistant/b5",
        "decode/assistant/b6", "decode/assistant/b7", "decode/assistant/b8"};
    nvtxRangePushA(assistant_ranges[active.size()]);
    Nvfp4VocabRunner* cycle_assistant_vocab = assistant_vocab_nvfp4.get();
    int cycle_assistant_vocab_rows = 0;
    if (static_cast<int>(active.size()) >= assistant_hot_vocab_min_batch) {
      const bool has_image = std::any_of(
          active.begin(), active.end(), [&](int slot) {
            return !session_requests[slot].encoded_image.empty();
          });
      if (assistant_combined_hot_vocab_nvfp4) {
        cycle_assistant_vocab = assistant_combined_hot_vocab_nvfp4.get();
        cycle_assistant_vocab_rows = has_image
            ? assistant_combined_multimodal_vocab_rows
            : assistant_combined_text_vocab_rows;
      } else {
        auto* specialized = has_image
            ? assistant_multimodal_hot_vocab_nvfp4.get()
            : assistant_text_hot_vocab_nvfp4.get();
        if (specialized)
          cycle_assistant_vocab = specialized;
        else if (assistant_hot_vocab_nvfp4)
          cycle_assistant_vocab = assistant_hot_vocab_nvfp4.get();
      }
    }
    DecodeBatchInputs shared_inputs;
    const bool cull_transfers =
        std::getenv("G4_DISABLE_DECODE_TRANSFER_CULL") == nullptr;
    const bool share_inputs = cull_transfers && cycle_drafts > 0 &&
        std::getenv("G4_DECODE_GRAPHS") == nullptr;
    if (cycle_drafts > 0)
      launch_assistant_batch(
          assistant, active_caches, contexts, assistant_vocab_model,
          weights.model(), current_states, previous, batch->drafts,
          batch->assistant_scratch, assistant_vocab,
          cycle_assistant_vocab, execution,
          assistant_fp8_linears.get(),
          cycle_drafts, assistant_minimum_rows(),
          cycle_assistant_vocab_rows, share_inputs ? &shared_inputs : nullptr);
    nvtxRangePop();
    const auto verifier_begin = std::chrono::steady_clock::now();
    static constexpr std::array verifier_ranges{
        "", "decode/verifier/b1", "decode/verifier/b2",
        "decode/verifier/b3", "decode/verifier/b4", "decode/verifier/b5",
        "decode/verifier/b6", "decode/verifier/b7", "decode/verifier/b8"};
    nvtxRangePushA(verifier_ranges[active.size()]);
    auto verified = launch_target_verifier_batch(
        weights, active_caches, active_candidates, contexts, previous,
        batch->drafts, current_states, batch->experts,
        batch->verifier_scratch, target_vocab,
        batch->nvfp4_vocab_for_rows(static_cast<int>(active.size()),
                                    cycle_drafts + 1),
        batch->fp8_for_rows(static_cast<int>(active.size()), cycle_drafts + 1),
        execution,
        cycle_drafts, share_inputs ? &shared_inputs : nullptr);
    nvtxRangePop();
    const auto cycle_end = std::chrono::steady_clock::now();
    const double assistant_ms = std::chrono::duration<double, std::milli>(
        verifier_begin - assistant_begin).count();
    const double verifier_ms = std::chrono::duration<double, std::milli>(
        cycle_end - verifier_begin).count();
    assistant_ms_by_batch[active.size()] += assistant_ms;
    verifier_ms_by_batch[active.size()] += verifier_ms;
    std::vector<int> remaining;
    std::vector<int> remaining_rows;
    std::vector<int> finished_slots;
    for (int row = 0; row < static_cast<int>(active.size()); ++row) {
      const int slot = active[row];
      auto& result = results[slot];
      const auto& value = verified.sessions[row];
      result.assistant_milliseconds += assistant_ms;
      result.verifier_milliseconds += verifier_ms;
      result.accepted_drafts += value.matched_drafts;
      result.cycles.push_back(
          {value.draft_tokens, value.selected_tokens, value.matched_drafts,
           cycle_drafts});
      cache_tokens[slot] += value.output_count;
      output_tokens_by_batch[active.size()] += value.output_count;
      std::string committed_text;
      for (int token = 0; token < value.output_count; ++token) {
        result.token_ids.push_back(value.output_tokens[token]);
        if (session_requests[slot].on_tokens)
          committed_text += tokenizer.decode_token(result.token_ids.back(), true);
        if (session_requests[slot].stop_on_json_object &&
            json_boundaries[slot].feed(
                tokenizer.decode_token(result.token_ids.back(), true),
                session_requests[slot].expected_json_ids)) {
          json_complete[slot] = true;
          break;
        }
        if (is_stop(result.token_ids.back()) ||
            static_cast<int>(result.token_ids.size()) >= maximum_tokens[slot])
          break;
      }
      if (!committed_text.empty())
        session_requests[slot].on_tokens(std::move(committed_text));
      if (!json_complete[slot] && !is_stop(result.token_ids.back()) &&
          static_cast<int>(result.token_ids.size()) < maximum_tokens[slot]) {
        remaining.push_back(slot);
        remaining_rows.push_back(row);
      } else {
        finished_slots.push_back(slot);
      }
    }
    if (refill) {
      // Keep continuation rows dense and write replacement prefills directly
      // behind the survivors. Completion is visible to the HTTP client before
      // we briefly wait for its next concurrency-limited request.
      const bool compact = !cull_transfers || remaining.size() != active.size();
      if (compact) {
        for (int next_row = 0; next_row < static_cast<int>(remaining.size());
             ++next_row) {
          if (cudaMemcpyAsync(
                  static_cast<std::uint16_t*>(compact_states) +
                      static_cast<std::size_t>(next_row) * 2816,
                  static_cast<const std::uint16_t*>(current_states) +
                      static_cast<std::size_t>(remaining_rows[next_row]) * 2816,
                  2816 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                  execution.stream()) != cudaSuccess)
            throw std::runtime_error("compact batch continuation state failed");
        }
      }
      for (const int slot : finished_slots) {
        finish_request(slot);
        ++refill_attempts;
        vacant_slots.push_back(slot);
      }
      if (compact && !remaining.empty()) std::swap(current_states, compact_states);
    } else if (!remaining.empty() && remaining.size() != active.size()) {
      for (int next_row = 0; next_row < static_cast<int>(remaining.size());
           ++next_row) {
        if (cudaMemcpyAsync(
                static_cast<std::uint16_t*>(compact_states) +
                    static_cast<std::size_t>(next_row) * 2816,
                static_cast<const std::uint16_t*>(current_states) +
                    static_cast<std::size_t>(remaining_rows[next_row]) * 2816,
                2816 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                execution.stream()) != cudaSuccess)
          throw std::runtime_error("compact batch continuation state failed");
      }
      std::swap(current_states, compact_states);
    }
    active = std::move(remaining);
  }
  for (int slot = 0; slot < count; ++slot) {
    if (!refill) finish_request(slot);
  }
  if (refill) {
    nlohmann::json stats{{"event", "continuous_batch_summary"},
                         {"refill_attempts", refill_attempts},
                         {"refill_hits", refill_hits},
                         {"prepare_wall_ms", prepare_wall_ms},
                         {"prefix_clone_wall_ms", prefix_clone_wall_ms},
                         {"prefill_wall_ms", prefill_wall_ms},
                         {"packed_prefill_waves", packed_prefill_waves},
                         {"packed_prefill_rows", packed_prefill_rows},
                         {"packed_prefill_sessions", packed_prefill_sessions}};
    for (int size = 1; size <= 8; ++size) {
      stats["cycles_b" + std::to_string(size)] = cycles_by_batch[size];
      stats["assistant_ms_b" + std::to_string(size)] =
          assistant_ms_by_batch[size];
      stats["verifier_ms_b" + std::to_string(size)] =
          verifier_ms_by_batch[size];
      stats["output_tokens_b" + std::to_string(size)] =
          output_tokens_by_batch[size];
    }
    std::cerr << stats.dump() << '\n';
  }
  results.resize(count);
  return results;
}

GenerationResult MultimodalRuntime::Impl::generate_prepared(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images, int maximum_tokens,
    std::chrono::steady_clock::time_point wall_begin,
    double image_decode_milliseconds, double image_preprocess_milliseconds,
    float vision_microseconds, bool stop_on_json_object,
    std::span<const std::string> expected_json_ids,
    const void* device_projected_images) {
  if (input_ids.size() + 1U > static_cast<std::size_t>(context_limit))
    throw std::runtime_error("prompt exceeds the configured model context");
  maximum_tokens = std::min(
      maximum_tokens,
      context_limit - static_cast<int>(input_ids.size()));
  if (maximum_tokens < 1)
    throw std::runtime_error("prompt leaves no serving context for generation");

  auto prefill = prefill_chunks(cache, input_ids, multimodal_types,
                                projected_images, continuation_state_device, 0,
                                device_projected_images);

  GenerationResult result;
  result.prompt_tokens = static_cast<int>(input_ids.size());
  result.token_ids.reserve(maximum_tokens + 4);
  result.token_ids.push_back(static_cast<std::uint32_t>(prefill.selected_token));
  result.prefill_token = prefill.selected_token;
  result.prefill_checksum = prefill.hidden_checksum;
  result.image_decode_milliseconds = image_decode_milliseconds;
  result.image_preprocess_milliseconds = image_preprocess_milliseconds;
  result.vision_microseconds = vision_microseconds;
  result.prefill_microseconds = prefill.microseconds;
  result.first_token_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_begin).count();
  int cache_tokens = static_cast<int>(input_ids.size());
  JsonObjectBoundary json_boundary;
  bool json_complete = stop_on_json_object && json_boundary.feed(
      tokenizer.decode_token(result.token_ids.back(), true), expected_json_ids);

  while (static_cast<int>(result.token_ids.size()) < maximum_tokens &&
         !json_complete && !is_stop(result.token_ids.back())) {
    const int output_room = maximum_tokens -
                            static_cast<int>(result.token_ids.size());
    const int context_room = context_limit - cache_tokens;
    const int cycle_drafts = std::max(
        0, std::min(serving_drafts(1),
                    std::min(output_room, context_room) - 1));
    const int previous_token = static_cast<int>(result.token_ids.back());
    const auto assistant_begin = std::chrono::steady_clock::now();
    if (cycle_drafts > 0)
      benchmark_assistant_step(
          assistant, cache, workspace, cache_tokens,
          &assistant_vocab_model, &weights.model(), cycle_drafts,
          {}, previous_token, &assistant_scratch,
          &assistant_vocab, &execution, continuation_state_device,
          drafted_tokens_device, assistant_fp8_linears.get());
    const auto assistant_end = std::chrono::steady_clock::now();
    const auto verified = benchmark_target_verifier(
        weights, target_vocab_model, cache,
        workspace, cache_tokens + 1, cycle_drafts + 1, {}, {},
        verifier_experts, &verifier_scratch,
        &target_vocab, &candidates, &execution, continuation_state_device,
        &target_fp8_linears, drafted_tokens_device, previous_token);
    const auto verifier_end = std::chrono::steady_clock::now();

    result.assistant_milliseconds +=
        std::chrono::duration<double, std::milli>(assistant_end - assistant_begin)
            .count();
    result.verifier_milliseconds +=
        std::chrono::duration<double, std::milli>(verifier_end - assistant_end)
            .count();
    result.accepted_drafts += verified.matched_drafts;
    result.cycles.push_back(
        {verified.draft_tokens, verified.selected_tokens,
         verified.matched_drafts, cycle_drafts});
    cache_tokens += verified.output_count;
    for (int index = 0; index < verified.output_count; ++index) {
      result.token_ids.push_back(
          static_cast<std::uint32_t>(verified.output_tokens[index]));
      if (stop_on_json_object && json_boundary.feed(
              tokenizer.decode_token(result.token_ids.back(), true),
              expected_json_ids)) {
        json_complete = true;
        break;
      }
      if (is_stop(result.token_ids.back()) ||
          static_cast<int>(result.token_ids.size()) >= maximum_tokens)
        break;
    }
  }
  result.text = tokenizer.decode(result.token_ids);
  result.length_limited =
      static_cast<int>(result.token_ids.size()) >= maximum_tokens &&
      !json_complete && !is_stop(result.token_ids.back());
  result.wall_milliseconds =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - wall_begin)
          .count();
  return result;
}

}  // namespace g4
