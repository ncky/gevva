#include "g4/gpu.hpp"
#include "g4/tiled_attention.hpp"
#include "g4/image.hpp"
#include "g4/model_config.hpp"
#include "g4/multimodal.hpp"
#include "g4/oracle.hpp"
#include "g4/openai_server.hpp"
#include "g4/runtime.hpp"
#include "g4/safetensors.hpp"
#include "g4/service.hpp"
#include "g4/tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <cuda_profiler_api.h>

namespace {

constexpr const char* kDefaultModel =
    "/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4";
constexpr const char* kDefaultExperts =
    "/mnt/SSD/g4-models/gemma4-26b-a4b-trtllm";
constexpr const char* kDefaultAssistant =
    "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant";

void usage() {
  std::cerr << "usage:\n"
            << "  g4 gpu-info\n"
            << "  g4 mtp-accept-test\n"
            << "  g4 image-preprocess IMAGE [SOFT_TOKENS] [ORACLE_DIR]\n"
            << "  g4 multimodal-layout TEXT IMAGE_SOFT_TOKENS...\n"
            << "  g4 multimodal-frontend IMAGE TEXT\n"
            << "  g4 multimodal-prefill IMAGE TEXT [LAYERS]\n"
            << "  g4 multimodal-generate IMAGE TEXT [MAX_TOKENS]\n"
            << "  g4 batch-serve-test IMAGE [BATCH]\n"
            << "  g4 multimodal-serve\n"
            << "  g4 oai-serve [PORT]\n"
            << "  g4 prefill-attention-test\n"
            << "  g4 text-prefill-attention-test [TOKENS]\n"
            << "  g4 rope-table-test\n"
            << "  g4 prefill-layer-bench [LAYER] [TOKENS]\n"
            << "  g4 vision-patch-bench IMAGE [ORACLE_DIR]\n"
            << "  g4 vision-layer-bench ORACLE_DIR\n"
            << "  g4 vision-encoder-bench IMAGE [ORACLE_DIR]\n"
            << "  g4 inspect [--model PATH]\n"
            << "  g4 tensor-info NAME [--model PATH]\n"
            << "  g4 upload-bench [--model PATH]\n"
            << "  g4 runtime-load-bench\n"
            << "  g4 assistant-load-bench\n"
            << "  g4 assistant-step-bench [CONTEXT]\n"
            << "  g4 assistant-cycle-bench [CONTEXT]\n"
            << "  g4 assistant-batch-bench [BATCH] [CONTEXT]\n"
            << "  g4 small-batch-bench\n"
            << "  g4 target-verifier-bench [CONTEXT]\n"
            << "  g4 target-batch-bench [BATCH] [CONTEXT] [DRAFTS]\n"
            << "  g4 target-throughput-sweep [CONTEXT] [REPETITIONS]\n"
            << "  g4 attention-router-test\n"
            << "  g4 attention-softmax-test\n"
            << "  g4 fast-softmax-test\n"
            << "  g4 attention-unpack-test\n"
            << "  g4 target-mtp-sweep [CONTEXT] [REPETITIONS]\n"
            << "  g4 vocab-head-bench [assistant|target]\n"
            << "  g4 nvfp4-vocab-bench [ROWS] [assistant] [HOT_MAP]\n"
            << "  g4 kv-cache-info [CONTEXT]\n"
            << "  g4 primitive-bench\n"
            << "  g4 decode-attention-bench [CONTEXT]\n"
            << "  g4 turboquant-kv-bench [CONTEXT]\n"
            << "  g4 decode-attention-batch-bench [CONTEXT]\n"
            << "  g4 attention-sublayers-bench [CONTEXT]\n"
            << "  g4 decoder-layer-bench [LAYER] [CONTEXT]\n"
            << "  g4 decoder-layer-batch-bench [LAYER] [CONTEXT]\n"
            << "  g4 matvec-bench [TENSOR] [--model PATH]\n"
            << "  g4 nvfp4-bench [TENSOR_BASE] [--model PATH]\n"
            << "  g4 nvfp4-grouped-bench [--model PATH]\n"
            << "  g4 nvfp4-expert-block-bench [--model PATH]\n"
            << "  g4 cutlass-expert-bench [LAYER]\n"
            << "  g4 expert-repeat-test [LAYER] [TOKENS] [REPETITIONS]\n"
            << "  g4 differential [--model PATH] [--oracle PATH]\n"
            << "  g4 tokenize TEXT [--model PATH]\n";
}

std::filesystem::path tensor_model_arg(int argc, char** argv) {
  if (argc == 3) return kDefaultModel;
  if (argc == 5 && std::string(argv[3]) == "--model") return argv[4];
  throw std::runtime_error("expected NAME and optional --model PATH");
}

struct MatvecArgs {
  std::string tensor{"model.language_model.layers.0.router.proj.weight"};
  std::filesystem::path model{kDefaultModel};
};

MatvecArgs matvec_args(int argc, char** argv) {
  MatvecArgs result;
  if (argc == 2) return result;
  if (argc == 3) {
    result.tensor = argv[2];
    return result;
  }
  if (argc == 4 && std::string(argv[2]) == "--model") {
    result.model = argv[3];
    return result;
  }
  if (argc == 5 && std::string(argv[3]) == "--model") {
    result.tensor = argv[2];
    result.model = argv[4];
    return result;
  }
  throw std::runtime_error("expected optional TENSOR and --model PATH");
}

std::filesystem::path model_arg(int argc, char** argv) {
  if (argc == 2) return kDefaultModel;
  if (argc == 4 && std::string(argv[2]) == "--model") return argv[3];
  throw std::runtime_error("expected optional --model PATH");
}

struct DifferentialArgs {
  std::filesystem::path model{
      "/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model"};
  std::filesystem::path oracle{"goldens/generated"};
};

DifferentialArgs differential_args(int argc, char** argv) {
  DifferentialArgs result;
  for (int i = 2; i < argc; i += 2) {
    if (i + 1 >= argc) throw std::runtime_error("missing differential option value");
    const std::string option = argv[i];
    if (option == "--model") result.model = argv[i + 1];
    else if (option == "--oracle") result.oracle = argv[i + 1];
    else throw std::runtime_error("unknown differential option: " + option);
  }
  return result;
}

void print_gpu(const g4::GpuInfo& gpu) {
  std::cout << "gpu.name=" << gpu.name << '\n'
            << "gpu.ordinal=" << gpu.ordinal << '\n'
            << "gpu.pci=" << gpu.pci_bus_id << '\n'
            << "gpu.uuid=GPU-" << gpu.uuid << '\n'
            << "gpu.compute=" << gpu.compute_major << '.' << gpu.compute_minor << '\n'
            << "gpu.vram_gib=" << std::fixed << std::setprecision(2)
            << static_cast<double>(gpu.total_memory) / (1ULL << 30) << '\n';
}

nlohmann::json generation_json(const g4::GenerationResult& result) {
  nlohmann::json cycles = nlohmann::json::array();
  for (const auto& cycle : result.cycles) {
    cycles.push_back({{"drafts", cycle.drafts},
                      {"target_tokens", cycle.target_tokens},
                      {"matched_drafts", cycle.matched_drafts}});
  }
  return {{"token_ids", result.token_ids},
          {"text", result.text},
          {"prompt_tokens", result.prompt_tokens},
          {"prefill_token", result.prefill_token},
          {"prefill_checksum", result.prefill_checksum},
          {"image_decode_ms", result.image_decode_milliseconds},
          {"image_preprocess_ms", result.image_preprocess_milliseconds},
          {"vision_us", result.vision_microseconds},
          {"prefill_us", result.prefill_microseconds},
          {"first_token_ms", result.first_token_milliseconds},
          {"cycles", std::move(cycles)},
          {"accepted_drafts", result.accepted_drafts},
          {"draft_acceptance", result.draft_acceptance()},
          {"assistant_ms", result.assistant_milliseconds},
          {"verifier_ms", result.verifier_milliseconds},
          {"decode_postprefill_tps",
           result.decode_postprefill_tokens_per_second()},
          {"wall_ms", result.wall_milliseconds}};
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string command = argv[1];
  if (command == "mtp-accept-test") {
    const std::array<int, 4> drafts{10, 20, 30, 40};
    const std::array<int, 5> all_match{10, 20, 30, 40, 50};
    const std::array<int, 5> reject_second{10, 99, 31, 41, 51};
    const auto accepted = g4::resolve_greedy_mtp(drafts, all_match);
    const auto rejected = g4::resolve_greedy_mtp(drafts, reject_second);
    const auto chat =
        g4::format_gemma4_user_turn("<|image|>Describe this image.");
    const std::string expected_chat =
        "<bos><|turn>user\n<|image|>Describe this image.<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>";
    if (accepted.output_count != 5 || accepted.matched_drafts != 4 ||
        accepted.output_tokens[4] != 50 || rejected.output_count != 2 ||
        rejected.matched_drafts != 1 || rejected.output_tokens[1] != 99 ||
        chat != expected_chat)
      throw std::runtime_error("MTP greedy acceptance self-test failed");
    std::cout << "mtp_accept.all_match_outputs=5\n"
              << "mtp_accept.early_reject_outputs=2\n"
              << "mtp_accept.chat_template=pass\n"
              << "mtp_accept.status=pass\n";
    return 0;
  }
  if (command == "gpu-info") {
    print_gpu(g4::select_target_gpu());
    return 0;
  }
  if (command == "rope-table-test") {
    print_gpu(g4::select_target_gpu());
    std::cout << "rope_table.exact_cases=" << g4::test_precomputed_rope() << '\n';
    return 0;
  }
  if (command == "prefill-attention-test" || command == "text-prefill-attention-test") {
    if (argc > (command == "text-prefill-attention-test" ? 3 : 2))
      throw std::runtime_error("text-prefill-attention-test accepts optional TOKENS");
    const int text_tokens = argc == 3 ? std::stoi(argv[2]) : 257;
    if (text_tokens < 2 || text_tokens > 4608)
      throw std::runtime_error("text prefill test TOKENS must be in [2, 4608]");
    print_gpu(g4::select_target_gpu());
    const std::vector<std::int32_t> blocks = command == "text-prefill-attention-test"
        ? std::vector<std::int32_t>(text_tokens, -1)
        : std::vector<std::int32_t>{
        -1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1};
    const auto result = g4::test_prefill_attention(blocks);
    std::cout << std::setprecision(8)
              << "prefill.sliding_us=" << result.sliding_microseconds << '\n'
              << "prefill.sliding_max_abs_error="
              << result.sliding_max_abs_error << '\n'
              << "prefill.global_us=" << result.global_microseconds << '\n'
              << "prefill.global_max_abs_error="
              << result.global_max_abs_error << '\n';
    return result.sliding_max_abs_error <= 0.00390625F &&
                   result.global_max_abs_error <= 0.00390625F
               ? 0 : 1;
  }
  if (command == "prefill-layer-bench") {
    if (argc > 4)
      throw std::runtime_error("prefill-layer-bench expects optional LAYER TOKENS");
    const bool all_layers = argc >= 3 && std::string(argv[2]) == "all";
    const int layer = all_layers ? 0 : (argc >= 3 ? std::stoi(argv[2]) : 0);
    const int tokens = argc >= 4 ? std::stoi(argv[3]) : 272;
    if (tokens < 8 || tokens > 512)
      throw std::runtime_error("prefill benchmark tokens must be in [8, 512]");
    print_gpu(g4::select_target_gpu());
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    std::vector<std::int32_t> groups(tokens, -1);
    for (int token = 1; token < tokens - 5; ++token) groups[token] = 0;
    for (int current_layer = layer;
         current_layer < (all_layers ? 30 : layer + 1); ++current_layer) {
      const auto result =
          g4::benchmark_prefill_layer(weights, current_layer, groups);
      std::cout << std::setprecision(8)
                << "prefill_layer.layer=" << current_layer << '\n'
                << "prefill_layer.tokens=" << tokens << '\n'
                << "prefill_layer.us=" << result.microseconds << '\n'
                << "prefill_layer.output_checksum="
                << result.output_checksum << '\n';
    }
    return 0;
  }
  if (command == "multimodal-layout") {
    if (argc < 4)
      throw std::runtime_error(
          "multimodal-layout expects TEXT and one soft-token count per image");
    std::vector<int> counts;
    counts.reserve(argc - 3);
    for (int index = 3; index < argc; ++index) counts.push_back(std::stoi(argv[index]));
    const g4::Tokenizer tokenizer(kDefaultModel);
    const auto prompt = g4::prepare_image_prompt(tokenizer, argv[2], counts);
    std::cout << "tokens=" << prompt.input_ids.size()
              << " image_offset=" << prompt.image_offsets.front()
              << " image_tokens=" << prompt.image_token_counts.front()
              << " head=";
    const std::size_t head = std::min<std::size_t>(2, prompt.input_ids.size());
    for (std::size_t index = 0; index < head; ++index) {
      if (index) std::cout << ',';
      std::cout << prompt.input_ids[index];
    }
    std::cout << " tail=";
    const std::size_t tail = std::min<std::size_t>(5, prompt.input_ids.size());
    for (std::size_t index = prompt.input_ids.size() - tail;
         index < prompt.input_ids.size(); ++index) {
      if (index != prompt.input_ids.size() - tail) std::cout << ',';
      std::cout << prompt.input_ids[index];
    }
    std::cout << '\n';
    return 0;
  }
  if (command == "image-preprocess") {
    if (argc < 3 || argc > 5)
      throw std::runtime_error(
          "image-preprocess expects IMAGE, optional SOFT_TOKENS and ORACLE_DIR");
    const int soft_tokens = argc >= 4 ? std::stoi(argv[3]) : 280;
    const auto image = g4::load_rgb_image(argv[2]);
    const auto input = g4::preprocess_gemma4_image(image, soft_tokens);
    std::uint64_t source_checksum = 0;
    for (const auto value : image.pixels) source_checksum += value;
    double checksum = 0.0;
    for (const float value : input.pixel_values) checksum += value;
    std::cout << "image.source_width=" << input.source_width << '\n'
              << "image.source_height=" << input.source_height << '\n'
              << "image.source_checksum=" << source_checksum << '\n'
              << "image.resized_width=" << input.resized_width << '\n'
              << "image.resized_height=" << input.resized_height << '\n'
              << "image.patch_count=" << input.patch_count << '\n'
              << "image.soft_token_count=" << input.soft_token_count << '\n'
              << std::fixed << std::setprecision(8)
              << "image.pixel_checksum=" << checksum << '\n';
    if (argc == 5) {
      const auto oracle = std::filesystem::path(argv[4]) / "pixel_values.f32";
      std::ifstream file(oracle, std::ios::binary | std::ios::ate);
      const auto expected_bytes = static_cast<std::streamoff>(
          input.pixel_values.size() * sizeof(float));
      if (!file || file.tellg() != expected_bytes)
        throw std::runtime_error("invalid image preprocessing oracle: " +
                                 oracle.string());
      std::vector<float> expected(input.pixel_values.size());
      file.seekg(0);
      file.read(reinterpret_cast<char*>(expected.data()), expected_bytes);
      float maximum = 0.0F;
      double mean = 0.0;
      std::size_t maximum_index = 0;
      std::size_t above_one_pixel = 0;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(input.pixel_values[i] - expected[i]);
        if (error > maximum) {
          maximum = error;
          maximum_index = i;
        }
        above_one_pixel += error > (1.01F / 255.0F);
        mean += error;
      }
      mean /= expected.size();
      std::cout << "image.oracle_max_abs_error=" << maximum << '\n'
                << "image.oracle_mean_abs_error=" << mean << '\n'
                << "image.oracle_max_index=" << maximum_index << '\n'
                << "image.oracle_actual_at_max="
                << input.pixel_values[maximum_index] << '\n'
                << "image.oracle_expected_at_max=" << expected[maximum_index] << '\n'
                << "image.oracle_above_one_pixel=" << above_one_pixel << '\n';
      if (maximum > 0.02F || mean > 0.001F) return 1;
    }
    return 0;
  }
  if (command == "multimodal-frontend") {
    if (argc != 4)
      throw std::runtime_error("multimodal-frontend expects IMAGE and TEXT");
    print_gpu(g4::select_target_gpu());
    const auto image = g4::load_rgb_image(argv[2]);
    const auto vision_input = g4::preprocess_gemma4_image(image);
    const g4::Tokenizer tokenizer(kDefaultModel);
    const std::array counts{vision_input.soft_token_count};
    const auto prompt = g4::prepare_image_prompt(tokenizer, argv[3], counts);
    const g4::DeviceModel model(kDefaultModel);
    const auto vision = g4::benchmark_vision_encoder(
        model, vision_input.pixel_values, vision_input.position_ids,
        vision_input.patch_count);
    const auto embedding = g4::benchmark_multimodal_embedding(
        model, prompt.input_ids, prompt.multimodal_types, vision.projected);
    std::cout << std::setprecision(8)
              << "multimodal.tokens=" << embedding.tokens << '\n'
              << "multimodal.image_tokens=" << embedding.image_tokens << '\n'
              << "multimodal.vision_us=" << vision.microseconds << '\n'
              << "multimodal.embedding_us=" << embedding.microseconds << '\n'
              << "multimodal.embedding_checksum=" << embedding.checksum << '\n'
              << "multimodal.embedding_max_abs_error="
              << embedding.max_abs_error << '\n';
    return embedding.max_abs_error == 0.0F ? 0 : 1;
  }
  if (command == "multimodal-prefill") {
    if (argc < 4 || argc > 5)
      throw std::runtime_error("multimodal-prefill expects IMAGE TEXT [LAYERS]");
    const int layer_count = argc == 5 ? std::stoi(argv[4]) : 30;
    print_gpu(g4::select_target_gpu());
    const auto image = g4::load_rgb_image(argv[2]);
    const auto vision_input = g4::preprocess_gemma4_image(image);
    const g4::Tokenizer tokenizer(kDefaultModel);
    const std::array counts{vision_input.soft_token_count};
    const auto prompt = g4::prepare_image_prompt(tokenizer, argv[3], counts);
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const g4::DeviceModel compressed_vocab(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    g4::KvCache cache(std::max<int>(4096, prompt.input_ids.size() + 8));
    const auto vision = g4::benchmark_vision_encoder(
        weights.model(), vision_input.pixel_values, vision_input.position_ids,
        vision_input.patch_count);
    const auto prefill = g4::benchmark_target_prefill(
        weights, compressed_vocab, cache, prompt.input_ids,
        prompt.multimodal_types, vision.projected, layer_count);
    std::cout << std::setprecision(8)
              << "multimodal_prefill.tokens=" << prefill.tokens << '\n'
              << "multimodal_prefill.vision_us=" << vision.microseconds << '\n'
              << "multimodal_prefill.language_us=" << prefill.microseconds << '\n'
              << "multimodal_prefill.total_us="
              << vision.microseconds + prefill.microseconds << '\n'
              << "multimodal_prefill.selected_token=" << prefill.selected_token << '\n'
              << "multimodal_prefill.decoded="
              << tokenizer.decode({static_cast<std::uint32_t>(prefill.selected_token)})
              << '\n'
              << "multimodal_prefill.hidden_checksum="
              << prefill.hidden_checksum << '\n';
    const auto oracle_path =
        std::filesystem::path("goldens/generated/multimodal_prefill.json");
    if (layer_count == 30 && std::filesystem::exists(oracle_path)) {
      std::ifstream oracle_file(oracle_path);
      nlohmann::json oracle;
      oracle_file >> oracle;
      const bool match = prefill.selected_token == oracle.at("selected_token").get<int>();
      std::cout << "multimodal_prefill.oracle_token="
                << oracle.at("selected_token").get<int>() << '\n'
                << "multimodal_prefill.oracle_token_match="
                << std::boolalpha << match << '\n';
      return match ? 0 : 1;
    }
    return 0;
  }
  if (command == "batch-serve-test") {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("batch-serve-test expects IMAGE [BATCH]");
    const int batch_size = argc == 4 ? std::stoi(argv[3]) : 2;
    if (batch_size < 1 || batch_size > 8)
      throw std::runtime_error("batch-serve-test BATCH must be in [1, 8]");
    const bool text_only = std::getenv("G4_BATCH_SERVE_TEXT_ONLY") != nullptr;
    std::vector<std::uint8_t> image;
    if (!text_only) {
      std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
      if (!file) throw std::runtime_error("cannot open batch test image");
      const auto bytes = file.tellg();
      if (bytes <= 0) throw std::runtime_error("batch test image is empty");
      image.resize(static_cast<std::size_t>(bytes));
      file.seekg(0);
      file.read(reinterpret_cast<char*>(image.data()), bytes);
    }
    const g4::MultimodalRuntimePaths paths{
        kDefaultModel, kDefaultExperts, kDefaultAssistant,
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-fp8"};
    g4::MultimodalRuntime runtime(paths);
    std::vector<g4::BatchGenerationRequest> requests(batch_size);
    std::vector<std::string> request_prompts(batch_size);
    const char* repeat_text = std::getenv("G4_BATCH_SERVE_PROMPT_REPEATS");
    const int prompt_repeats = repeat_text ? std::stoi(repeat_text) : 0;
    if (prompt_repeats < 0 || prompt_repeats > 10000)
      throw std::runtime_error("invalid batch serving prompt repeat count");
    const std::string context_paragraph =
        "The harbor district changed gradually as workshops, homes, gardens, "
        "and narrow streets adapted to new technology while residents "
        "preserved local customs and recorded practical lessons for later "
        "generations. ";
    for (int request_index = 0; request_index < batch_size; ++request_index) {
      auto& request = requests[request_index];
      auto& prompt = request_prompts[request_index];
      request.encoded_image = image;
      // BatchGenerationRequest uses OpenAI-compatible message semantics: the
      // runtime inserts one image marker for encoded_image.
      prompt = text_only
          ? "Write a concise paragraph explaining why batching improves transformer inference."
          : "Describe this image.";
      if (prompt_repeats) {
        prompt.clear();
        for (int repeat = 0; repeat < prompt_repeats; ++repeat)
          prompt += context_paragraph;
        prompt += text_only
            ? "Give a structured analysis with at least twenty concise "
              "numbered observations."
            : "Analyze the supplied manga page with at least twenty concise "
              "numbered observations.";
      }
      request.prompt = prompt;
      // By default this is an isolated packed-prefill symmetry test.  The
      // environment override extends it through multiple MTP cycles when
      // diagnosing serving-batch state transitions.
      const char* test_tokens = std::getenv("G4_BATCH_SERVE_TEST_TOKENS");
      request.maximum_tokens = test_tokens ? std::stoi(test_tokens) : 1;
    }
    if (const char* setting = std::getenv("G4_BATCH_SERVE_SWEEP_ENV")) {
      if (std::getenv("G4_DECODE_GRAPHS"))
        throw std::runtime_error("serving A/B sweep requires uncaptured decode");
      for (int width : {1, batch_size}) {
        auto cohort = std::span(requests).first(width);
        std::vector<g4::GenerationResult> reference;
        for (int round = -1; round < 3; ++round) {
          for (int variant = 0; variant < 2; ++variant) {
            const bool enabled = (variant ^ (round >= 0 ? round & 1 : 0)) != 0;
            if (std::string_view(setting) == "G4_FUSE_ATTENTION_ROUTER" ||
                std::string_view(setting) == "G4_EXACT_ATTENTION_OPT")
              setenv(setting, enabled ? "1" : "0", 1);
            else if (enabled) setenv(setting, "1", 1);
            else unsetenv(setting);
            std::vector<g4::GenerationResult> results;
            const char* profile_variant = std::getenv("G4_PROFILE_SWEEP_VARIANT");
            const bool capture = profile_variant && round == 0 && width == batch_size &&
                enabled == (std::string_view(profile_variant) == "1");
            if (capture && cudaProfilerStart() != cudaSuccess)
              throw std::runtime_error("start serving sweep profile failed");
            const bool continuous =
                std::getenv("G4_BATCH_SERVE_SWEEP_CONTINUOUS") != nullptr;
            if (continuous) {
              results.resize(width);
              auto wave = std::vector<g4::BatchGenerationRequest>(cohort.begin(), cohort.end());
              for (int row = 0; row < width; ++row)
                wave[row].on_complete = [&, row](g4::GenerationResult result) {
                  results[row] = std::move(result);
                };
              runtime.generate_continuous(wave, [] {
                return std::optional<g4::BatchGenerationRequest>{};
              });
            } else {
              results = runtime.generate_batch(cohort);
            }
            if (capture && cudaProfilerStop() != cudaSuccess)
              throw std::runtime_error("stop serving sweep profile failed");
            if (round < 0) continue;
            if (variant == 0) reference = results;
            bool matches = results.size() == reference.size();
            double aggregate_tps = 0.0, verifier_ms = 0.0, assistant_ms = 0.0;
            double acceptance = 0.0, first_token_ms = 0.0, prefill_ms = 0.0;
            int mismatched_requests = 0;
            std::size_t matching_prefix_tokens = 0;
            int output_tokens = 0, accepted = 0;
            std::uint64_t hash = 1469598103934665603ULL;
            for (std::size_t row = 0; row < results.size(); ++row) {
              const auto& result = results[row];
              matches &= result.token_ids == reference[row].token_ids;
              mismatched_requests += result.token_ids != reference[row].token_ids;
              std::size_t prefix = 0;
              while (prefix < result.token_ids.size() && prefix < reference[row].token_ids.size() &&
                     result.token_ids[prefix] == reference[row].token_ids[prefix]) ++prefix;
              matching_prefix_tokens += prefix;
              acceptance += result.draft_acceptance();
              first_token_ms += result.first_token_milliseconds;
              prefill_ms += result.prefill_microseconds / 1000.0;
              for (auto token : result.token_ids) {
                hash ^= token;
                hash *= 1099511628211ULL;
              }
              aggregate_tps += result.decode_postprefill_tokens_per_second();
              verifier_ms += result.verifier_milliseconds;
              assistant_ms += result.assistant_milliseconds;
              output_tokens += result.token_ids.size();
              accepted += result.accepted_drafts;
            }
            std::cout << nlohmann::json{
                {"batch", width}, {"round", round}, {"enabled", enabled},
                {"setting", setting}, {"text_only", text_only},
                {"continuous", continuous},
                {"prompt_tokens", results.front().prompt_tokens},
                {"output_tokens", output_tokens}, {"accepted_drafts", accepted},
                {"aggregate_decode_tps", aggregate_tps},
                {"verifier_ms_per_request", verifier_ms / width},
                {"assistant_ms_per_request", assistant_ms / width},
                {"first_token_ms", results.front().first_token_milliseconds},
                {"mean_first_token_ms", first_token_ms / width},
                {"mean_prefill_ms", prefill_ms / width},
                {"mean_draft_acceptance", acceptance / width},
                {"mismatched_requests", mismatched_requests},
                {"mean_matching_prefix_tokens", static_cast<double>(matching_prefix_tokens) / width},
                {"matches_pair", matches}, {"sequence_hash", hash}}.dump()
                << std::endl;
            if (!matches && !std::getenv("G4_SWEEP_ALLOW_TOKEN_MISMATCH"))
              throw std::runtime_error("serving sweep token mismatch");
          }
        }
        if (batch_size == 1) break;
      }
      return 0;
    }
    const bool profile_batch =
        std::getenv("G4_PROFILE_BATCH_SERVE") != nullptr;
    if (std::getenv("G4_BATCH_SERVE_WARMUP") != nullptr) {
      auto warmup_requests = requests;
      auto warmup_prompts = request_prompts;
      if (std::getenv("G4_BATCH_SERVE_WARMUP_SAME_PREFIX") == nullptr) {
        for (int request_index = 0; request_index < batch_size;
             ++request_index) {
          warmup_prompts[request_index].insert(
              0, "Kernel warmup with unrelated prefix. ");
          warmup_requests[request_index].prompt = warmup_prompts[request_index];
        }
      }
      (void)runtime.generate_batch(warmup_requests);
    }
    if (std::getenv("G4_BATCH_SERVE_B1_WARMUP") != nullptr) {
      auto warmup = requests.front();
      std::string warmup_prompt = request_prompts.front();
      if (std::getenv("G4_BATCH_SERVE_WARMUP_SAME_PREFIX") == nullptr)
        warmup_prompt.insert(0, "Single-slot cache reuse warmup. ");
      warmup.prompt = warmup_prompt;
      (void)runtime.generate_batch(std::span(&warmup, 1));
    }
    if (profile_batch && cudaProfilerStart() != cudaSuccess)
      throw std::runtime_error("cudaProfilerStart batch serve failed");
    std::vector<g4::GenerationResult> results;
    try {
      results = runtime.generate_batch(requests);
    } catch (...) {
      if (profile_batch) cudaProfilerStop();
      throw;
    }
    if (profile_batch && cudaProfilerStop() != cudaSuccess)
      throw std::runtime_error("cudaProfilerStop batch serve failed");
    bool repeatable = results.size() == requests.size();
    for (std::size_t index = 0; index < results.size(); ++index) {
      const bool matches =
          index == 0 ||
          (results[index].token_ids == results[0].token_ids &&
           results[index].text == results[0].text &&
           results[index].prefill_token == results[0].prefill_token &&
           results[index].prefill_checksum == results[0].prefill_checksum);
      repeatable &= matches;
      std::cout << "batch_serve.session[" << index
                << "].prefill_token=" << results[index].prefill_token << '\n'
                << "batch_serve.session[" << index
                << "].prefill_checksum=" << results[index].prefill_checksum
                << '\n'
                << "batch_serve.session[" << index
                << "].accepted_drafts=" << results[index].accepted_drafts
                << '\n'
                << "batch_serve.session[" << index
                << "].draft_acceptance=" << results[index].draft_acceptance()
                << '\n'
                << "batch_serve.session[" << index
                << "].assistant_ms=" << results[index].assistant_milliseconds
                << '\n'
                << "batch_serve.session[" << index
                << "].verifier_ms=" << results[index].verifier_milliseconds
                << '\n'
                << "batch_serve.session[" << index
                << "].first_token_ms="
                << results[index].first_token_milliseconds << '\n'
                << "batch_serve.session[" << index
                << "].image_decode_ms="
                << results[index].image_decode_milliseconds << '\n'
                << "batch_serve.session[" << index
                << "].image_preprocess_ms="
                << results[index].image_preprocess_milliseconds << '\n'
                << "batch_serve.session[" << index
                << "].vision_ms=" << results[index].vision_microseconds / 1000.0
                << '\n'
                << "batch_serve.session[" << index
                << "].prefill_ms=" << results[index].prefill_microseconds / 1000.0
                << '\n'
                << "batch_serve.session[" << index
                << "].decode_tps="
                << results[index].decode_postprefill_tokens_per_second()
                << '\n'
                << "batch_serve.session[" << index << "].token_ids=";
      for (const auto token : results[index].token_ids) std::cout << token << ',';
      std::cout << '\n';
      for (std::size_t cycle = 0; cycle < results[index].cycles.size(); ++cycle) {
        const auto& value = results[index].cycles[cycle];
        std::cout << "batch_serve.session[" << index << "].cycle[" << cycle
                  << "].drafts=";
        for (int draft = 0; draft < value.draft_count; ++draft)
          std::cout << value.drafts[draft] << ',';
        std::cout << " targets=";
        for (int token = 0; token < value.draft_count + 1; ++token)
          std::cout << value.target_tokens[token] << ',';
        std::cout << " matched=" << value.matched_drafts << '\n';
      }
    }
    std::cout << "batch_serve.sessions=" << results.size() << '\n'
              << "batch_serve.prompt_tokens=" << results[0].prompt_tokens << '\n'
              << "batch_serve.prefill_token=" << results[0].prefill_token << '\n'
              << "batch_serve.repeatable=" << std::boolalpha << repeatable
              << '\n';
    return repeatable ? 0 : 1;
  }
  if (command == "multimodal-serve") {
    if (argc != 2)
      throw std::runtime_error("multimodal-serve takes no arguments");
    const auto gpu = g4::select_target_gpu();
    const g4::MultimodalRuntimePaths paths{
        kDefaultModel,
        kDefaultExperts,
        kDefaultAssistant,
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-fp8"};
    g4::MultimodalRuntime runtime(paths);
    std::cout << nlohmann::json({
                     {"event", "ready"},
                     {"protocol", "g4-jsonl-v1"},
                     {"target_representation", "modelopt_nvfp4"},
                     {"gpu", gpu.name},
                     {"gpu_pci", gpu.pci_bus_id},
                     {"gpu_uuid", "GPU-" + gpu.uuid},
                     {"maximum_context", runtime.maximum_context()},
                     {"startup_ms", runtime.startup_milliseconds()}})
                     .dump()
              << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      nlohmann::json response;
      try {
        const auto request = nlohmann::json::parse(line);
        if (request.contains("id")) response["id"] = request.at("id");
        const bool profile = request.value("profile", false);
        if (profile && cudaProfilerStart() != cudaSuccess)
          throw std::runtime_error("cudaProfilerStart failed");
        g4::GenerationResult result;
        try {
          const auto prompt = request.at("prompt").get<std::string>();
          const int maximum_tokens = request.value("max_tokens", 32);
          if (request.contains("image"))
            result = runtime.generate(request.at("image").get<std::string>(),
                                      prompt, maximum_tokens);
          else
            result = runtime.generate(prompt, maximum_tokens);
        } catch (...) {
          if (profile) cudaProfilerStop();
          throw;
        }
        if (profile && cudaProfilerStop() != cudaSuccess)
          throw std::runtime_error("cudaProfilerStop failed");
        response["ok"] = true;
        response["result"] = generation_json(result);
      } catch (const std::exception& error) {
        response["ok"] = false;
        response["error"] = error.what();
      }
      std::cout << response.dump() << std::endl;
    }
    return 0;
  }
  if (command == "oai-serve") {
    if (argc > 3)
      throw std::runtime_error("oai-serve expects optional PORT");
    const auto gpu = g4::select_target_gpu();
    const g4::MultimodalRuntimePaths paths{
        kDefaultModel, kDefaultExperts, kDefaultAssistant,
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8",
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-fp8"};
    g4::MultimodalRuntime runtime(paths);
    std::cerr << "g4 OpenAI server ready on 127.0.0.1:"
              << (argc == 3 ? std::stoi(argv[2]) : 8080)
              << " using " << gpu.name << '\n';
    g4::OpenAiServer server(runtime, 8);
    server.run(static_cast<std::uint16_t>(argc == 3 ? std::stoi(argv[2]) : 8080));
  }
  if (command == "multimodal-generate") {
    if (argc < 4 || argc > 5)
      throw std::runtime_error(
          "multimodal-generate expects IMAGE TEXT [MAX_TOKENS]");
    const int maximum_tokens = argc == 5 ? std::stoi(argv[4]) : 32;
    if (maximum_tokens < 1 || maximum_tokens > 262144)
      throw std::runtime_error("MAX_TOKENS must be in [1, 262144]");
    print_gpu(g4::select_target_gpu());
    const auto wall_begin = std::chrono::steady_clock::now();
    const auto image = g4::load_rgb_image(argv[2]);
    const auto vision_input = g4::preprocess_gemma4_image(image);
    const g4::Tokenizer tokenizer(kDefaultModel);
    const std::array counts{vision_input.soft_token_count};
    const auto chat_prompt = g4::format_gemma4_user_turn(argv[3]);
    const auto prompt = g4::prepare_image_prompt(tokenizer, chat_prompt, counts);
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const g4::AssistantWeights assistant(kDefaultAssistant);
    const g4::DeviceModel target_vocab(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    const g4::DeviceModel assistant_vocab(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8");
    g4::CompressedVocabRunner persistent_target_vocab(
        weights.model(), "model.language_model.embed_tokens.weight",
        target_vocab, nullptr);
    g4::CompressedVocabRunner persistent_assistant_vocab(
        assistant.model(), "model.embed_tokens.weight", assistant_vocab,
        nullptr);
    std::vector<std::unique_ptr<g4::Nvfp4ExpertRunner>> verifier_expert_owners;
    std::vector<std::unique_ptr<g4::Nvfp4ExpertRunner>> prefill_expert_owners;
    std::array<g4::Nvfp4ExpertRunner*, 30> verifier_experts{};
    std::array<g4::Nvfp4ExpertRunner*, 30> prefill_experts{};
    verifier_expert_owners.reserve(30);
    prefill_expert_owners.reserve(30);
    for (int layer = 0; layer < 30; ++layer) {
      verifier_expert_owners.push_back(
          std::make_unique<g4::Nvfp4ExpertRunner>(weights.experts(), layer, 5));
      verifier_experts[layer] = verifier_expert_owners.back().get();
      prefill_expert_owners.push_back(
          std::make_unique<g4::Nvfp4ExpertRunner>(
              weights.experts(), layer,
              std::min<int>(prompt.input_ids.size(), 4608)));
      prefill_experts[layer] = prefill_expert_owners.back().get();
    }
    const auto loaded_at = std::chrono::steady_clock::now();
    const int maximum_context = static_cast<int>(prompt.input_ids.size()) +
                                maximum_tokens + 8;
    g4::KvCache cache(maximum_context);
    g4::CandidateKvCache candidate_cache;
    g4::AttentionWorkspace workspace(maximum_context, 5);
    g4::DeviceScratchPool assistant_scratch;
    g4::DeviceScratchPool verifier_scratch;
    g4::DeviceScratchPool vision_scratch;
    g4::DeviceScratchPool prefill_scratch;
    g4::GpuExecutionContext execution;
    const auto vision = g4::benchmark_vision_encoder(
        weights.model(), vision_input.pixel_values, vision_input.position_ids,
        vision_input.patch_count, {}, false, &vision_scratch, &execution);
    auto prefill = g4::benchmark_target_prefill(
        weights, target_vocab, cache, prompt.input_ids,
        prompt.multimodal_types, vision.projected, 30, false, prefill_experts,
        &prefill_scratch, &persistent_target_vocab, &execution);

    std::vector<std::uint32_t> generated;
    generated.reserve(maximum_tokens + 4);
    generated.push_back(static_cast<std::uint32_t>(prefill.selected_token));
    auto continuation_state = std::move(prefill.final_state);
    int cache_tokens = static_cast<int>(prompt.input_ids.size());
    int cycles = 0;
    int accepted_drafts = 0;
    std::vector<int> accepted_per_cycle;
    double assistant_ms = 0.0;
    double verifier_ms = 0.0;
    auto is_stop = [](std::uint32_t token) {
      return token == 1 || token == 50 || token == 106;
    };
    while (static_cast<int>(generated.size()) < maximum_tokens &&
           !is_stop(generated.back())) {
      const int previous_token = static_cast<int>(generated.back());
      const auto assistant_begin = std::chrono::steady_clock::now();
      const auto drafts = g4::benchmark_assistant_step(
          assistant, cache, workspace, cache_tokens, &assistant_vocab,
          &weights.model(), 4, continuation_state, previous_token,
          &assistant_scratch, &persistent_assistant_vocab, &execution);
      const auto assistant_end = std::chrono::steady_clock::now();
      std::array<std::uint32_t, 5> candidates{
          static_cast<std::uint32_t>(previous_token),
          static_cast<std::uint32_t>(drafts.drafted_tokens[0]),
          static_cast<std::uint32_t>(drafts.drafted_tokens[1]),
          static_cast<std::uint32_t>(drafts.drafted_tokens[2]),
          static_cast<std::uint32_t>(drafts.drafted_tokens[3])};
      const auto verified = g4::benchmark_target_verifier(
          weights, target_vocab, cache, workspace, cache_tokens + 1, 5,
          candidates,
          std::span<const int>(drafts.drafted_tokens.data(), 4),
          verifier_experts, &verifier_scratch, &persistent_target_vocab,
          &candidate_cache, &execution);
      const auto verifier_end = std::chrono::steady_clock::now();
      std::cout << "generation.cycle." << cycles << ".drafts=";
      for (int index = 0; index < 4; ++index)
        std::cout << (index ? "," : "") << drafts.drafted_tokens[index];
      std::cout << " target=";
      for (int index = 0; index < 5; ++index)
        std::cout << (index ? "," : "") << verified.selected_tokens[index];
      std::cout << " matched=" << verified.matched_drafts << '\n';
      assistant_ms += std::chrono::duration<double, std::milli>(
                          assistant_end - assistant_begin).count();
      verifier_ms += std::chrono::duration<double, std::milli>(
                         verifier_end - assistant_end).count();
      ++cycles;
      accepted_drafts += verified.matched_drafts;
      accepted_per_cycle.push_back(verified.matched_drafts);
      cache_tokens += verified.output_count;
      continuation_state = verified.continuation_state;
      for (int index = 0; index < verified.output_count; ++index) {
        generated.push_back(
            static_cast<std::uint32_t>(verified.output_tokens[index]));
        if (is_stop(generated.back()) ||
            static_cast<int>(generated.size()) >= maximum_tokens)
          break;
      }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(
                               wall_end - wall_begin).count();
    std::cout << "generation.tokens=" << generated.size() << '\n'
              << "generation.prefill_token=" << prefill.selected_token << '\n'
              << "generation.prefill_checksum=" << prefill.hidden_checksum << '\n'
              << "generation.vision_us=" << vision.microseconds << '\n'
              << "generation.prefill_us=" << prefill.microseconds << '\n'
              << "generation.cycles=" << cycles << '\n'
              << "generation.accepted_drafts=" << accepted_drafts << '\n'
              << "generation.draft_acceptance="
              << (cycles ? static_cast<double>(accepted_drafts) /
                                (4.0 * cycles)
                         : 0.0)
              << '\n'
              << "generation.accepted_per_cycle=";
    for (std::size_t index = 0; index < accepted_per_cycle.size(); ++index)
      std::cout << (index ? "," : "") << accepted_per_cycle[index];
    std::cout << '\n'
              << "generation.token_ids=";
    for (std::size_t index = 0; index < generated.size(); ++index)
      std::cout << (index ? "," : "") << generated[index];
    std::cout << '\n'
              << "generation.load_ms=" << std::fixed << std::setprecision(3)
              << std::chrono::duration<double, std::milli>(loaded_at - wall_begin).count()
              << '\n'
              << "generation.frontend_prefill_ms="
              << std::chrono::duration<double, std::milli>(
                     wall_end - loaded_at).count() -
                     assistant_ms - verifier_ms
              << '\n'
              << "generation.assistant_ms=" << assistant_ms << '\n'
              << "generation.verifier_ms=" << verifier_ms << '\n'
              << "generation.decode_postprefill_tps="
              << ((assistant_ms + verifier_ms) > 0.0
                      ? 1000.0 * (generated.size() - 1) /
                            (assistant_ms + verifier_ms)
                      : 0.0)
              << '\n'
              << "generation.wall_ms=" << std::fixed << std::setprecision(3)
              << wall_ms << '\n'
              << "generation.text=" << tokenizer.decode(generated) << '\n';
    return 0;
  }
  if (command == "vision-patch-bench") {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("vision-patch-bench expects IMAGE and optional ORACLE_DIR");
    print_gpu(g4::select_target_gpu());
    const auto image = g4::load_rgb_image(argv[2]);
    const auto input = g4::preprocess_gemma4_image(image);
    std::vector<float> expected;
    if (argc == 4) {
      const auto path = std::filesystem::path(argv[3]) / "patch_hidden.f32";
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      constexpr std::size_t kExpected = 2520ULL * 1152;
      if (!file || file.tellg() !=
                       static_cast<std::streamoff>(kExpected * sizeof(float)))
        throw std::runtime_error("invalid vision patch oracle: " + path.string());
      expected.resize(kExpected);
      file.seekg(0);
      file.read(reinterpret_cast<char*>(expected.data()),
                static_cast<std::streamsize>(expected.size() * sizeof(float)));
    }
    const g4::DeviceModel model(kDefaultModel);
    const auto result = g4::test_vision_patch_embedding(
        model, input.pixel_values, input.position_ids, expected);
    std::cout << "vision_patch.patches=2520\n"
              << "vision_patch.hidden=1152\n"
              << "vision_patch.us=" << result.microseconds << '\n'
              << "vision_patch.checksum=" << result.checksum << '\n';
    if (!expected.empty()) {
      std::cout << "vision_patch.max_abs_error=" << result.max_abs_error << '\n'
                << "vision_patch.mean_abs_error=" << result.mean_abs_error << '\n';
      if (result.max_abs_error > 0.0625F ||
          result.mean_abs_error > 0.002F)
        return 1;
    }
    return 0;
  }
  if (command == "vision-layer-bench") {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("vision-layer-bench expects ORACLE_DIR [full]");
    print_gpu(g4::select_target_gpu());
    const bool full = argc == 4 && std::string_view(argv[3]) == "full";
    if (argc == 4 && !full)
      throw std::runtime_error("vision-layer-bench mode must be 'full'");
    const std::size_t kTokens = full ? 2520 : 2394;
    constexpr std::size_t kHidden = 1152;
    auto read_f32 = [](const std::filesystem::path& path,
                       std::size_t elements) {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file || file.tellg() != static_cast<std::streamoff>(elements * 4))
        throw std::runtime_error("invalid vision oracle: " + path.string());
      std::vector<float> values(elements);
      file.seekg(0);
      file.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(elements * 4));
      return values;
    };
    const std::filesystem::path oracle = argv[2];
    auto input = read_f32(oracle / "patch_hidden.f32", 2520 * kHidden);
    input.resize(kTokens * kHidden);
    const auto expected = read_f32(
        oracle / (full ? "vision_full_layer0_hidden.f32"
                       : "vision_layer0_hidden.f32"),
        kTokens * kHidden);
    std::ifstream position_file(oracle / "position_ids.i32", std::ios::binary);
    std::vector<std::int32_t> positions(kTokens * 2);
    if (!position_file.read(reinterpret_cast<char*>(positions.data()),
                            static_cast<std::streamsize>(positions.size() * 4)))
      throw std::runtime_error("invalid vision position oracle");
    const g4::DeviceModel model(kDefaultModel);
    const auto result = g4::test_vision_layer0(
        model, input, positions, expected);
    std::cout << std::setprecision(8)
              << "vision_layer.tokens=" << kTokens << '\n'
              << "vision_layer.us=" << result.microseconds << '\n'
              << "vision_layer.checksum=" << result.checksum << '\n'
              << "vision_layer.max_abs_error=" << result.max_abs_error << '\n'
              << "vision_layer.mean_abs_error=" << result.mean_abs_error << '\n';
    return result.max_abs_error <= 1.0F && result.mean_abs_error <= 0.03F ? 0 : 1;
  }
  if (command == "vision-encoder-bench") {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("vision-encoder-bench expects IMAGE and optional ORACLE_DIR");
    print_gpu(g4::select_target_gpu());
    const auto image = g4::load_rgb_image(argv[2]);
    const auto input = g4::preprocess_gemma4_image(image);
    std::vector<float> expected;
    if (argc == 4) {
      const auto path = std::filesystem::path(argv[3]) / "vision_projected.f32";
      const std::size_t elements =
          static_cast<std::size_t>(input.soft_token_count) * 2816;
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file || file.tellg() != static_cast<std::streamoff>(elements * 4))
        throw std::runtime_error("invalid projected vision oracle: " + path.string());
      expected.resize(elements);
      file.seekg(0);
      file.read(reinterpret_cast<char*>(expected.data()),
                static_cast<std::streamsize>(elements * 4));
    }
    const g4::DeviceModel model(kDefaultModel);
    std::unique_ptr<g4::DeviceModel> vision_fp8_model;
    std::unique_ptr<g4::Fp8LinearRunner> vision_fp8;
    std::unique_ptr<g4::VisionFusedWeights> vision_fused_weights;
    if (std::getenv("G4_VISION_FP8") != nullptr) {
      vision_fp8_model = std::make_unique<g4::DeviceModel>(
          "/mnt/SSD/g4-models/gemma4-26b-a4b-vision-fp8");
      vision_fp8 = std::make_unique<g4::Fp8LinearRunner>(
          model, *vision_fp8_model, input.patch_count, false,
          (input.patch_count + 15) & ~15, 1, true);
    } else if (std::getenv("G4_DISABLE_FUSED_VISION_PROJECTIONS") == nullptr)
      vision_fused_weights = std::make_unique<g4::VisionFusedWeights>(model);
    const auto result = g4::benchmark_vision_encoder(
        model, input.pixel_values, input.position_ids, input.patch_count,
        expected, true, nullptr, nullptr, 1, nullptr, true,
        vision_fp8.get(), {}, {}, vision_fused_weights.get());
    std::cout << std::setprecision(8)
              << "vision_encoder.patches=" << input.patch_count << '\n'
              << "vision_encoder.soft_tokens=" << input.soft_token_count << '\n'
              << "vision_encoder.us=" << result.microseconds << '\n'
              << "vision_encoder.projected_checksum="
              << result.projected_checksum << '\n';
    if (!expected.empty()) {
      std::cout << "vision_encoder.projected_max_abs_error="
                << result.projected_max_abs_error << '\n'
                << "vision_encoder.projected_mean_abs_error="
                << result.projected_mean_abs_error << '\n';
      if (result.projected_max_abs_error > 8.0F ||
          result.projected_mean_abs_error > 0.06F)
        return 1;
    }
    return 0;
  }
  if (command == "inspect") {
    const auto model_dir = model_arg(argc, argv);
    print_gpu(g4::select_target_gpu());
    const auto config = g4::load_model_config(model_dir);
    g4::validate_target_model(config);
    const auto shards = g4::inspect_model_shards(model_dir);
    std::uint64_t bytes = 0;
    std::size_t tensors = 0;
    std::set<std::string> dtypes;
    for (const auto& shard : shards) {
      bytes += shard.file_size;
      tensors += shard.tensors.size();
      for (const auto& [_, tensor] : shard.tensors) dtypes.insert(tensor.dtype);
    }
    const auto& c = config.text;
    std::cout << "model.path=" << std::filesystem::canonical(model_dir) << '\n'
              << "model.architecture=" << config.architecture << '\n'
              << "model.quantized=" << std::boolalpha << config.quantized << '\n'
              << "model.quant_method=" << (config.quantized ? config.quant_method : "none") << '\n'
              << "model.layers=" << c.num_layers << '\n'
              << "model.hidden=" << c.hidden_size << '\n'
              << "model.experts=" << c.num_experts << '\n'
              << "model.active_experts=" << c.top_k_experts << '\n'
              << "model.sliding_head_dim=" << c.layer_head_dims.front() << '\n'
              << "model.global_head_dim=" << c.layer_head_dims[5] << '\n'
              << "model.max_context=" << c.max_positions << '\n'
              << "weights.shards=" << shards.size() << '\n'
              << "weights.tensors=" << tensors << '\n'
              << "weights.gib=" << std::fixed << std::setprecision(2)
              << static_cast<double>(bytes) / (1ULL << 30) << '\n'
              << "weights.dtypes=";
    bool first = true;
    for (const auto& dtype : dtypes) {
      std::cout << (first ? "" : ",") << dtype;
      first = false;
    }
    std::cout << '\n';
    return 0;
  }
  if (command == "tensor-info") {
    if (argc < 3) throw std::runtime_error("tensor-info requires NAME");
    const auto model_dir = tensor_model_arg(argc, argv);
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(model_dir);
    const auto view = weights.tensor(argv[2]);
    std::cout << "tensor.name=" << argv[2] << '\n'
              << "tensor.dtype=" << view.info->dtype << '\n'
              << "tensor.shape=";
    for (std::size_t i = 0; i < view.info->shape.size(); ++i) {
      std::cout << (i ? "x" : "") << view.info->shape[i];
    }
    // Touch one byte to prove the returned view points into file-backed data.
    std::cout << "\ntensor.bytes=" << view.bytes.size()
              << "\ntensor.first_byte="
              << static_cast<unsigned>(std::to_integer<unsigned char>(view.bytes.front())) << '\n'
              << "weights.mapped_tensors=" << weights.tensor_count() << '\n';
    return 0;
  }
  if (command == "upload-bench") {
    const auto model_dir = model_arg(argc, argv);
    print_gpu(g4::select_target_gpu());
    const auto config = g4::load_model_config(model_dir);
    g4::validate_target_model(config);
    const g4::ModelWeights weights(model_dir);
    const auto result = g4::benchmark_weight_upload(weights.shard_data());
    const double gib = static_cast<double>(result.bytes) / (1ULL << 30);
    std::cout << "upload.gib=" << std::fixed << std::setprecision(2) << gib << '\n'
              << "upload.device_ms=" << std::setprecision(3) << result.device_milliseconds << '\n'
              << "upload.wall_ms=" << result.wall_milliseconds << '\n'
              << "upload.device_gib_s=" << std::setprecision(2)
              << gib * 1000.0 / result.device_milliseconds << '\n';
    return 0;
  }
  if (command == "runtime-load-bench") {
    if (argc != 2) throw std::runtime_error("runtime-load-bench takes no arguments");
    print_gpu(g4::select_target_gpu());
    const auto wall_begin = std::chrono::steady_clock::now();
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const auto wall_end = std::chrono::steady_clock::now();
    const double gib = static_cast<double>(weights.device_bytes()) / (1ULL << 30);
    const double milliseconds =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    std::cout << "runtime_load.gib=" << std::fixed << std::setprecision(2) << gib << '\n'
              << "runtime_load.wall_ms=" << std::setprecision(3) << milliseconds << '\n'
              << "runtime_load.gib_s=" << std::setprecision(2) << gib * 1000.0 / milliseconds
              << '\n';
    return 0;
  }
  if (command == "assistant-batch-bench") {
    if (argc > 4)
      throw std::runtime_error(
          "assistant-batch-bench expects optional BATCH CONTEXT");
    const int batch = argc >= 3 ? std::stoi(argv[2]) : 8;
    const int context = argc >= 4 ? std::stoi(argv[3]) : 128;
    print_gpu(g4::select_target_gpu());
    const g4::AssistantWeights assistant(kDefaultAssistant);
    const g4::DeviceModel target(kDefaultModel);
    const g4::DeviceModel vocabulary(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8");
    std::unique_ptr<g4::DeviceModel> nvfp4_vocabulary;
    std::vector<int> hot_token_map;
    if (std::getenv("G4_ASSISTANT_BATCH_NVFP4") != nullptr) {
      nvfp4_vocabulary = std::make_unique<g4::DeviceModel>(
          "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4");
      if (const char* path = std::getenv("G4_ASSISTANT_HOT_VOCAB")) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open hot-vocabulary map");
        const auto bytes = input.tellg();
        if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int)))
          throw std::runtime_error("invalid hot-vocabulary map size");
        hot_token_map.resize(static_cast<std::size_t>(bytes) / sizeof(int));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(hot_token_map.data()), bytes);
        if (!input) throw std::runtime_error("cannot read hot-vocabulary map");
      }
    }
    const auto result = g4::test_assistant_batch(
        assistant, vocabulary, target, batch, context,
        nvfp4_vocabulary.get(), hot_token_map);
    std::cout << std::boolalpha
              << "assistant_batch.batch=" << result.batch << '\n'
              << "assistant_batch.us=" << result.microseconds << '\n'
              << "assistant_batch.rows_repeatable="
              << result.rows_repeatable << '\n'
              << "assistant_batch.first_row_matches_scalar="
              << result.first_row_matches_scalar << '\n'
              << "assistant_batch.first_row_drafts=";
    for (std::size_t index = 0; index < result.first_row_drafts.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << result.first_row_drafts[index];
    }
    std::cout << '\n'
              << "assistant_batch.scalar_drafts=";
    for (std::size_t index = 0; index < result.scalar_drafts.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << result.scalar_drafts[index];
    }
    std::cout << '\n';
    return result.rows_repeatable &&
                   (result.batch > 1 || result.first_row_matches_scalar)
               ? 0
               : 1;
  }
  if (command == "assistant-load-bench") {
    if (argc != 2) throw std::runtime_error("assistant-load-bench takes no arguments");
    print_gpu(g4::select_target_gpu());
    const auto wall_begin = std::chrono::steady_clock::now();
    const g4::AssistantWeights weights(kDefaultAssistant);
    const auto wall_end = std::chrono::steady_clock::now();
    const double mib = static_cast<double>(weights.device_bytes()) / (1ULL << 20);
    const double milliseconds =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    std::cout << "assistant_load.mib=" << std::fixed << std::setprecision(2) << mib << '\n'
              << "assistant_load.wall_ms=" << std::setprecision(3) << milliseconds << '\n'
              << "assistant_load.tensors=48\n";
    return 0;
  }
  if (command == "assistant-step-bench" || command == "assistant-cycle-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 4096;
    print_gpu(g4::select_target_gpu());
    const g4::AssistantWeights weights(kDefaultAssistant);
    const g4::DeviceModel compressed_vocab(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8");
    std::unique_ptr<g4::DeviceModel> target_model;
    const int drafts = command == "assistant-cycle-bench" ? 4 : 1;
    if (drafts == 4) target_model = std::make_unique<g4::DeviceModel>(kDefaultModel);
    g4::KvCache cache(context);
    g4::AttentionWorkspace workspace(context);
    const auto result =
        g4::benchmark_assistant_step(
            weights, cache, workspace, context, &compressed_vocab,
            target_model.get(), drafts);
    std::cout << "assistant_step.context=" << context << '\n'
              << "assistant_step.eager_us=" << result.eager_microseconds << '\n'
              << "assistant_step.graph_us=" << result.graph_microseconds << '\n'
              << "assistant_step.logits_us=" << result.logits_microseconds << '\n'
              << "assistant_step.cycle_us=" << result.cycle_microseconds << '\n'
              << "assistant_step.selected_token=" << result.selected_token << '\n'
              << "assistant_step.drafted_tokens=";
    for (int draft = 0; draft < drafts; ++draft)
      std::cout << (draft ? "," : "") << result.drafted_tokens[draft];
    std::cout << '\n'
              << "assistant_step.state_checksum=" << result.state_checksum << '\n'
              << "assistant_step.state_max_abs_error="
              << result.state_max_abs_error << '\n'
              << "assistant_step.state_mean_abs_error="
              << result.state_mean_abs_error << '\n'
              << "assistant_step.oracle_available=" << std::boolalpha
              << result.oracle_available << '\n'
              << "assistant_step.oracle_token_match=" << std::boolalpha
              << result.oracle_token_match << '\n';
    return 0;
  }
  if (command == "small-batch-bench") {
    if (argc != 2) throw std::runtime_error("small-batch-bench takes no arguments");
    print_gpu(g4::select_target_gpu());
    const g4::DeviceModel model(kDefaultModel);
    const auto result = g4::benchmark_target_small_batch(model);
    for (int tokens = 1; tokens <= 5; ++tokens) {
      const int index = tokens - 1;
      std::cout << "small_batch.tokens=" << tokens << '\n'
                << "small_batch.q_batched_us="
                << result.q_batched_microseconds[index] << '\n'
                << "small_batch.q_repeated_us="
                << result.q_repeated_microseconds[index] << '\n'
                << "small_batch.mlp_batched_us="
                << result.mlp_batched_microseconds[index] << '\n'
                << "small_batch.mlp_repeated_us="
                << result.mlp_repeated_microseconds[index] << '\n';
    }
    return 0;
  }
  if (command == "target-verifier-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 4096;
    print_gpu(g4::select_target_gpu());
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const g4::DeviceModel compressed_vocab(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    g4::KvCache cache(context + 4);
    g4::AttentionWorkspace workspace(context + 4, 5);
    const auto result = g4::benchmark_target_verifier(
        weights, compressed_vocab, cache, workspace, context, 5);
    std::cout << "target_verifier.context=" << context << '\n'
              << "target_verifier.tokens=5\n"
              << "target_verifier.eager_us=" << result.eager_microseconds << '\n'
              << "target_verifier.graph_us=" << result.graph_microseconds << '\n'
              << "target_verifier.selected_tokens=";
    for (int token = 0; token < 5; ++token)
      std::cout << (token ? "," : "") << result.selected_tokens[token];
    std::cout << '\n'
              << "target_verifier.hidden_checksum=" << result.hidden_checksum << '\n';
    return 0;
  }
  if (command == "tiled-attention-bench") {
    if (argc < 3 || argc > 6) throw std::runtime_error("expected FIXTURE [BATCH] [RAGGED] [QUERY_TOKENS]");
    print_gpu(g4::select_target_gpu());
    g4::benchmark_tiled_attention(argv[2], argc >= 4 ? std::stoi(argv[3]) : 1,
                                 argc >= 5 && std::stoi(argv[4]) != 0,
                                 argc >= 6 ? std::stoi(argv[5]) : 0);
    return 0;
  }
  if (command == "attention-router-test") {
    print_gpu(g4::select_target_gpu());
    std::cout << "attention_router.exact_cases="
              << g4::test_attention_router_fusion() << '\n';
    return 0;
  }
  if (command == "attention-softmax-test") {
    print_gpu(g4::select_target_gpu());
    std::cout << "attention_softmax.cases=" << g4::test_attention_softmax() << '\n';
    return 0;
  }
  if (command == "fast-softmax-test") {
    print_gpu(g4::select_target_gpu());
    std::cout << "fast_softmax.exact_cases=" << g4::test_fast_attention_softmax() << '\n';
    return 0;
  }
  if (command == "attention-unpack-test") {
    print_gpu(g4::select_target_gpu());
    std::cout << "attention_unpack.exact_cases=" << g4::test_attention_unpack() << '\n';
    return 0;
  }
  if (command == "target-batch-bench") {
    if (argc > 5)
      throw std::runtime_error("expected optional BATCH CONTEXT DRAFTS");
    const int batch = argc >= 3 ? std::stoi(argv[2]) : 8;
    const int context = argc >= 4 ? std::stoi(argv[3]) : 4096;
    const int drafts = argc >= 5 ? std::stoi(argv[4]) : 4;
    print_gpu(g4::select_target_gpu());
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const g4::DeviceModel vocabulary(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    const g4::DeviceModel nvfp4_vocabulary(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4");
    const g4::DeviceModel fp8(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8");
    const auto result = g4::benchmark_target_verifier_batch_fixed(
        weights, vocabulary, fp8, batch, context, 20, drafts,
        &nvfp4_vocabulary);
    std::cout << "target_batch.batch=" << batch << '\n'
              << "target_batch.context=" << context << '\n'
              << "target_batch.drafts=" << drafts << '\n'
              << "target_batch.us=" << result.microseconds << '\n'
              << "target_batch.output_count=" << result.output_count << '\n'
              << "target_batch.matched_drafts=" << result.matched_drafts << '\n'
              << "target_batch.sessions_repeatable=" << std::boolalpha
              << result.sessions_repeatable << '\n'
              << "target_batch.sequence_hash=" << result.sequence_hash << '\n'
              << "target_batch.targets=";
    for (int token = 0; token < drafts + 1; ++token)
      std::cout << (token ? "," : "") << result.target_tokens[token];
    std::cout << '\n' << "target_batch.outputs=";
    for (int token = 0; token < result.output_count; ++token)
      std::cout << (token ? "," : "") << result.output_tokens[token];
    std::cout << '\n';
    return 0;
  }
  if (command == "target-mtp-sweep" || command == "target-throughput-sweep") {
    if (argc > 4)
      throw std::runtime_error("expected optional CONTEXT REPETITIONS");
    const int context = argc >= 3 ? std::stoi(argv[2]) : 4096;
    const int repetitions = argc >= 4 ? std::stoi(argv[3]) : 20;
    print_gpu(g4::select_target_gpu());
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const g4::DeviceModel vocabulary(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    const g4::DeviceModel nvfp4_vocabulary(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4");
    const g4::DeviceModel fp8(
        "/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8");
    if (command == "target-throughput-sweep") {
      const char* setting = std::getenv("G4_THROUGHPUT_SWEEP_ENV");
      if (!setting) setting = "G4_FUSE_ATTENTION_ROUTER";
      for (int round = 0; round < 3; ++round) {
        for (int batch : {1, 8}) {
          std::uint64_t reference_hash = 0;
          for (int variant = 0; variant < 2; ++variant) {
            const bool enabled = (variant ^ (round & 1)) != 0;
            if (std::string_view(setting) == "G4_FUSE_ATTENTION_ROUTER" ||
                std::string_view(setting) == "G4_EXACT_ATTENTION_OPT")
              setenv(setting, enabled ? "1" : "0", 1);
            else if (enabled) setenv(setting, "1", 1);
            else unsetenv(setting);
            const int drafts = batch == 1 ? 3 : 4;
            const auto result = g4::benchmark_target_verifier_batch_fixed(
                weights, vocabulary, fp8, batch, context, repetitions,
                drafts, &nvfp4_vocabulary);
            if (variant == 0) reference_hash = result.sequence_hash;
            std::cout << nlohmann::json{
                {"round", round}, {"batch", batch}, {"context", context},
                {"drafts", drafts}, {"setting", setting}, {"enabled", enabled},
                {"verifier_us", result.microseconds},
                {"sequence_hash", result.sequence_hash},
                {"matches_pair", result.sequence_hash == reference_hash},
                {"sessions_repeatable", result.sessions_repeatable}}
                .dump() << std::endl;
          }
        }
      }
      return 0;
    }
    for (int batch = 1; batch <= 8; ++batch)
      for (int drafts = 0; drafts <= 4; ++drafts) {
        const auto result = g4::benchmark_target_verifier_batch_fixed(
            weights, vocabulary, fp8, batch, context, repetitions, drafts,
            &nvfp4_vocabulary);
        const double candidate_tokens_per_second =
            1.0e6 * batch * (drafts + 1) / result.microseconds;
        std::cout << "target_mtp.batch=" << batch
                  << " drafts=" << drafts
                  << " verifier_us=" << result.microseconds
                  << " candidate_tps=" << candidate_tokens_per_second
                  << " sequence_hash=" << result.sequence_hash << '\n';
      }
    return 0;
  }
  if (command == "vocab-head-bench") {
    if (argc > 3) throw std::runtime_error("expected assistant or target");
    const bool target = argc == 3 && std::string(argv[2]) == "target";
    if (argc == 3 && !target && std::string(argv[2]) != "assistant")
      throw std::runtime_error("expected assistant or target");
    print_gpu(g4::select_target_gpu());
    const std::filesystem::path exact_path = target ? kDefaultModel : kDefaultAssistant;
    const std::filesystem::path packed_path = target
        ? "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8"
        : "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8";
    const std::string tensor = target
        ? "model.language_model.embed_tokens.weight"
        : "model.embed_tokens.weight";
    const g4::DeviceModel exact(exact_path);
    const g4::DeviceModel packed(packed_path);
    const auto result = g4::benchmark_vocab_head(exact, tensor, packed);
    std::cout << "vocab_head.kind=" << (target ? "target" : "assistant") << '\n'
              << "vocab_head.exact_us=" << result.exact_microseconds << '\n'
              << "vocab_head.int8_us=" << result.int8_microseconds << '\n'
              << "vocab_head.exact_token=" << result.exact_token << '\n'
              << "vocab_head.approximate_token=" << result.approximate_token << '\n'
              << "vocab_head.int8_token=" << result.int8_token << '\n'
              << "vocab_head.token_match=" << std::boolalpha
              << (result.exact_token == result.int8_token) << '\n'
              << "vocab_head.validation_trials=" << result.validation_trials << '\n'
              << "vocab_head.approximate_matches=" << result.approximate_matches << '\n'
              << "vocab_head.corrected_matches=" << result.corrected_matches << '\n';
    std::cout << "vocab_head.exact_batch5_us="
              << result.exact_batch5_microseconds << '\n'
              << "vocab_head.int8_batch5_us="
              << result.int8_batch5_microseconds << '\n'
              << "vocab_head.batch5_matches=" << result.batch5_matches << "/5\n";
    return 0;
  }
  if (command == "nvfp4-vocab-bench") {
    if (argc > 5)
      throw std::runtime_error("expected optional ROWS assistant HOT_MAP");
    const int rows = argc >= 3 ? std::stoi(argv[2]) : 40;
    const bool assistant = argc >= 4 && std::string(argv[3]) == "assistant";
    if (argc >= 4 && !assistant) throw std::runtime_error("expected assistant");
    std::vector<int> hot_token_map;
    if (argc == 5) {
      std::ifstream input(argv[4], std::ios::binary | std::ios::ate);
      if (!input) throw std::runtime_error("cannot open hot-vocabulary map");
      const auto bytes = input.tellg();
      if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(int)))
        throw std::runtime_error("invalid hot-vocabulary map size");
      hot_token_map.resize(static_cast<std::size_t>(bytes) / sizeof(int));
      input.seekg(0);
      input.read(reinterpret_cast<char*>(hot_token_map.data()), bytes);
      if (!input) throw std::runtime_error("cannot read hot-vocabulary map");
    }
    print_gpu(g4::select_target_gpu());
    const g4::DeviceModel exact(assistant ? kDefaultAssistant : kDefaultModel);
    const g4::DeviceModel int8(
        assistant
            ? "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8"
            : "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-int8");
    const g4::DeviceModel nvfp4(
        assistant
            ? "/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4"
            : "/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4");
    const auto result =
        g4::benchmark_nvfp4_vocab(
            exact, int8, nvfp4, rows,
            assistant ? "model.embed_tokens.weight"
                      : "model.language_model.embed_tokens.weight",
            hot_token_map);
    std::cout << "nvfp4_vocab.kind="
              << (assistant ? "assistant" : "target") << '\n';
    std::cout << "nvfp4_vocab.rows=" << result.rows << '\n'
              << "nvfp4_vocab.int8_us=" << result.int8_microseconds << '\n'
              << "nvfp4_vocab.nvfp4_us=" << result.nvfp4_microseconds << '\n'
              << "nvfp4_vocab.hot_nvfp4_us="
              << result.hot_nvfp4_microseconds << '\n'
              << "nvfp4_vocab.mismatched_tokens="
              << result.mismatched_tokens << '\n'
              << "nvfp4_vocab.hot_mismatched_tokens="
              << result.hot_mismatched_tokens << '\n';
    return result.mismatched_tokens == 0 ? 0 : 1;
  }
  if (command == "cutlass-expert-bench" || command == "cutlass-w13-bench") {
    if (argc > 3) throw std::runtime_error("expected optional LAYER");
    const int layer = argc == 3 ? std::stoi(argv[2]) : 0;
    print_gpu(g4::select_target_gpu());
    const g4::DeviceModel experts(kDefaultExperts);
    const auto result = g4::benchmark_cutlass_grouped_w13(experts, layer);
    std::cout << "cutlass_w13.layer=" << layer << '\n'
              << "cutlass_w13.routes=8\n"
              << "cutlass_w13.shape=1x1408x2816\n"
              << "cutlass_w13.gemm_us=" << result.gemm_microseconds << '\n'
              << "cutlass_w13.quantized_us=" << result.quantized_microseconds << '\n'
              << "cutlass_expert_block.us=" << result.expert_block_microseconds << '\n'
              << "cutlass_expert_block.dynamic_runner_us="
              << result.dynamic_runner_microseconds << '\n'
              << "cutlass_expert_block.output_abs_checksum=" << result.output_checksum << '\n';
    const std::filesystem::path oracle = "goldens/generated/nvfp4_layer0_output.f32";
    if (layer == 0 && std::filesystem::exists(oracle)) {
      std::ifstream file(oracle, std::ios::binary | std::ios::ate);
      if (!file || file.tellg() != static_cast<std::streamoff>(result.output.size() * sizeof(float)))
        throw std::runtime_error("invalid NVFP4 expert oracle file");
      std::vector<float> expected(result.output.size());
      file.seekg(0);
      file.read(reinterpret_cast<char*>(expected.data()),
                static_cast<std::streamsize>(expected.size() * sizeof(float)));
      float maximum = 0.0F;
      double mean = 0.0;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(expected[i] - result.output[i]);
        maximum = std::max(maximum, error);
        mean += error;
      }
      mean /= expected.size();
      std::cout << std::setprecision(6)
                << "cutlass_expert_block.oracle_max_abs_error=" << maximum << '\n'
                << "cutlass_expert_block.oracle_mean_abs_error=" << mean << '\n';
      return maximum <= 0.0625F && mean <= 0.01 ? 0 : 1;
    }
    return 0;
  }
  if (command == "expert-repeat-test") {
    if (argc > 5) throw std::runtime_error(
        "expected optional LAYER, TOKENS, and REPETITIONS");
    const int layer = argc >= 3 ? std::stoi(argv[2]) : 0;
    const int tokens = argc >= 4 ? std::stoi(argv[3]) : 272;
    const int repetitions = argc >= 5 ? std::stoi(argv[4]) : 8;
    print_gpu(g4::select_target_gpu());
    const g4::DeviceModel experts(kDefaultExperts);
    const auto result = g4::test_nvfp4_expert_repeatability(
        experts, layer, tokens, repetitions);
    std::cout << "expert_repeat.layer=" << layer << '\n'
              << "expert_repeat.tokens=" << tokens << '\n'
              << "expert_repeat.repetitions=" << result.repetitions << '\n'
              << "expert_repeat.us=" << result.microseconds << '\n'
              << "expert_repeat.output_abs_checksum="
              << result.output_abs_checksum << '\n'
              << "expert_repeat.max_difference="
              << result.maximum_difference << '\n'
              << "expert_repeat.different_values="
              << result.different_values << '\n'
              << "expert_repeat.bitwise_repeatable=" << std::boolalpha
              << result.bitwise_repeatable << '\n';
    return result.bitwise_repeatable ? 0 : 2;
  }
  if (command == "decoder-layer-bench" ||
      command == "decoder-layer-batch-bench") {
    if (argc > 4) throw std::runtime_error("expected optional LAYER and CONTEXT");
    const bool both = argc >= 3 && std::string(argv[2]) == "both";
    const int layer = argc >= 3 && !both ? std::stoi(argv[2]) : 0;
    const int context = argc == 4 ? std::stoi(argv[3]) : 4096;
    print_gpu(g4::select_target_gpu());
    const g4::RuntimeWeights weights(kDefaultModel, kDefaultExperts);
    const int tokens = command == "decoder-layer-batch-bench" ? 5 : 1;
    g4::KvCache cache(context + tokens - 1);
    g4::AttentionWorkspace workspace(context + tokens - 1, tokens);
    auto run = [&](int selected_layer) {
      const auto result = g4::benchmark_decoder_layer(
          weights, cache, workspace, selected_layer, context, tokens);
      std::cout << "decoder_layer.layer=" << selected_layer << '\n'
                << "decoder_layer.context=" << context << '\n'
                << "decoder_layer.tokens=" << tokens << '\n'
                << "decoder_layer.eager_us=" << result.eager_microseconds << '\n'
                << "decoder_layer.graph_us=" << result.graph_microseconds << '\n'
                << "decoder_layer.output_checksum=" << result.output_checksum << '\n';
    };
    run(layer);
    if (both) run(5);
    return 0;
  }
  if (command == "kv-cache-info") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 32768;
    print_gpu(g4::select_target_gpu());
    const g4::KvCache cache(context);
    const g4::AttentionWorkspace workspace(context);
    const auto& sliding = cache.layer(0);
    const auto& global = cache.layer(5);
    std::cout << "kv.maximum_context=" << cache.maximum_context() << '\n'
              << "kv.gib=" << std::fixed << std::setprecision(3)
              << static_cast<double>(cache.bytes()) / (1ULL << 30) << '\n'
              << "kv.resident_context=" << cache.resident_context() << '\n'
              << "kv.sliding_capacity=" << sliding.capacity << '\n'
              << "kv.sliding_geometry=" << sliding.kv_heads << 'x' << sliding.head_dim << '\n'
              << "kv.global_capacity=" << global.capacity << '\n'
              << "kv.global_geometry=" << global.kv_heads << 'x' << global.head_dim << '\n'
              << "kv.attention_workspace_mib="
              << static_cast<double>(workspace.bytes()) / (1ULL << 20) << '\n';
    return 0;
  }
  if (command == "primitive-bench") {
    print_gpu(g4::select_target_gpu());
    const auto result = g4::test_and_benchmark_primitives();
    std::cout << "rmsnorm.max_abs_error=" << result.rmsnorm_max_abs_error << '\n'
              << "rmsnorm.us=" << result.rmsnorm_microseconds << '\n'
              << "routing.max_abs_error=" << result.routing_max_abs_error << '\n'
              << "routing.ids_match=" << std::boolalpha << result.routing_ids_match << '\n'
              << "routing.us=" << result.routing_microseconds << '\n';
    return result.routing_ids_match && result.rmsnorm_max_abs_error <= 0.015F &&
                   result.routing_max_abs_error <= 1e-5F ? 0 : 1;
  }
  if (command == "matvec-bench") {
    const auto args = matvec_args(argc, argv);
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(args.model);
    const auto weight = weights.tensor(args.tensor);
    if (weight.info->dtype != "BF16" || weight.info->shape.size() != 2) {
      throw std::runtime_error("matvec-bench requires a rank-2 BF16 tensor");
    }
    const int outputs = static_cast<int>(weight.info->shape[0]);
    const int inputs = static_cast<int>(weight.info->shape[1]);
    const auto result = g4::test_bf16_matvec(weight.bytes, outputs, inputs);
    std::cout << "matvec.tensor=" << args.tensor << '\n'
              << "matvec.shape=" << outputs << 'x' << inputs << '\n'
              << "matvec.max_abs_error=" << result.max_abs_error << '\n'
              << "matvec.us=" << result.microseconds << '\n';
    return result.max_abs_error <= 0.25F ? 0 : 1;
  }
  if (command == "nvfp4-bench") {
    auto args = matvec_args(argc, argv);
    if (argc == 2 || (argc == 4 && std::string(argv[2]) == "--model")) {
      args.tensor = "model.language_model.layers.0.experts.0.gate_proj";
    }
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(args.model);
    const auto packed = weights.tensor(args.tensor + ".weight");
    const auto scales = weights.tensor(args.tensor + ".weight_scale");
    const auto global = weights.tensor(args.tensor + ".weight_scale_2");
    if (packed.info->dtype != "U8" || packed.info->shape.size() != 2 ||
        scales.info->dtype != "F8_E4M3" || global.info->dtype != "F32") {
      throw std::runtime_error("nvfp4-bench requires ModelOpt NVFP4 weight tensors");
    }
    const int outputs = static_cast<int>(packed.info->shape[0]);
    const int inputs = static_cast<int>(packed.info->shape[1] * 2);
    const auto result = g4::test_nvfp4_matvec(
        packed.bytes, scales.bytes, global.bytes, outputs, inputs);
    std::cout << "nvfp4.tensor=" << args.tensor << '\n'
              << "nvfp4.shape=" << outputs << 'x' << inputs << '\n'
              << "nvfp4.max_abs_error=" << result.max_abs_error << '\n'
              << "nvfp4.us=" << result.microseconds << '\n';
    return result.max_abs_error <= 0.125F ? 0 : 1;
  }
  if (command == "nvfp4-grouped-bench") {
    const auto model_dir = model_arg(argc, argv);
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(model_dir);
    std::vector<std::span<const std::byte>> packed, scales, globals;
    for (int expert = 0; expert < 8; ++expert) {
      const std::string base = "model.language_model.layers.0.experts." +
                               std::to_string(expert) + ".gate_proj";
      packed.push_back(weights.tensor(base + ".weight").bytes);
      scales.push_back(weights.tensor(base + ".weight_scale").bytes);
      globals.push_back(weights.tensor(base + ".weight_scale_2").bytes);
    }
    const auto result = g4::test_nvfp4_grouped_matvec(packed, scales, globals, 704, 2816);
    std::cout << std::setprecision(8)
              << "nvfp4_grouped.groups=8\n"
              << "nvfp4_grouped.shape=704x2816\n"
              << "nvfp4_grouped.max_abs_error=" << result.max_abs_error << '\n'
              << "nvfp4_grouped.us=" << result.microseconds << '\n';
    return result.max_abs_error <= 0.125F ? 0 : 1;
  }
  if (command == "nvfp4-expert-block-bench") {
    const auto model_dir = model_arg(argc, argv);
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(model_dir);
    std::vector<std::span<const std::byte>> gate_w, gate_s, gate_g;
    std::vector<std::span<const std::byte>> up_w, up_s, up_g;
    std::vector<std::span<const std::byte>> down_w, down_s, down_g;
    auto append = [&](const std::string& base,
                      std::vector<std::span<const std::byte>>& packed,
                      std::vector<std::span<const std::byte>>& scales,
                      std::vector<std::span<const std::byte>>& globals) {
      packed.push_back(weights.tensor(base + ".weight").bytes);
      scales.push_back(weights.tensor(base + ".weight_scale").bytes);
      globals.push_back(weights.tensor(base + ".weight_scale_2").bytes);
    };
    for (int expert = 0; expert < 128; ++expert) {
      const std::string prefix = "model.language_model.layers.0.experts." +
                                 std::to_string(expert) + ".";
      append(prefix + "gate_proj", gate_w, gate_s, gate_g);
      append(prefix + "up_proj", up_w, up_s, up_g);
      append(prefix + "down_proj", down_w, down_s, down_g);
    }
    const auto result = g4::test_nvfp4_expert_block(
        gate_w, gate_s, gate_g, up_w, up_s, up_g, down_w, down_s, down_g);
    std::cout << std::setprecision(8)
              << "nvfp4_expert_block.routes=8\n"
              << "nvfp4_expert_block.max_abs_error=" << result.max_abs_error << '\n'
              << "nvfp4_expert_block.eager_us=" << result.eager_microseconds << '\n'
              << "nvfp4_expert_block.graph_us=" << result.graph_microseconds << '\n';
    return result.max_abs_error <= 0.5F ? 0 : 1;
  }
  if (command == "differential") {
    const auto args = differential_args(argc, argv);
    print_gpu(g4::select_target_gpu());
    const g4::ModelWeights weights(args.model);
    const g4::RawOracle oracle(args.oracle);
    const auto embedding = weights.tensor("model.language_model.embed_tokens.weight");
    const auto norm = weights.tensor("model.language_model.layers.0.input_layernorm.weight");
    const auto ids = oracle.read_i32("input_ids");
    const auto expected_embedding = oracle.read_f32("hidden.0");
    const auto expected_norm = oracle.read_f32("layer0.input_layernorm");
    const auto result = g4::test_bf16_frontend(embedding.bytes, norm.bytes, ids,
                                               expected_embedding, expected_norm);
    const auto q = weights.tensor("model.language_model.layers.0.self_attn.q_proj.weight");
    const auto k = weights.tensor("model.language_model.layers.0.self_attn.k_proj.weight");
    const auto v = weights.tensor("model.language_model.layers.0.self_attn.v_proj.weight");
    const auto o = weights.tensor("model.language_model.layers.0.self_attn.o_proj.weight");
    const auto q_norm = weights.tensor("model.language_model.layers.0.self_attn.q_norm.weight");
    const auto k_norm = weights.tensor("model.language_model.layers.0.self_attn.k_norm.weight");
    const auto expected_attention = oracle.read_f32("layer0.self_attn.0");
    const auto expected_probabilities = oracle.read_f32("layer0.self_attn.1");
    const auto attention = g4::test_bf16_layer0_attention(
        q.bytes, k.bytes, v.bytes, o.bytes, q_norm.bytes, k_norm.bytes, expected_norm,
        expected_attention, expected_probabilities);
    const std::string global_prefix = "model.language_model.layers.5.self_attn.";
    const auto global_q = weights.tensor(global_prefix + "q_proj.weight");
    const auto global_k = weights.tensor(global_prefix + "k_proj.weight");
    const auto global_o = weights.tensor(global_prefix + "o_proj.weight");
    const auto global_q_norm = weights.tensor(global_prefix + "q_norm.weight");
    const auto global_k_norm = weights.tensor(global_prefix + "k_norm.weight");
    const auto global_input = oracle.read_f32("layer5.input_layernorm");
    const auto expected_global_attention = oracle.read_f32("layer5.self_attn.0");
    const auto expected_global_probabilities = oracle.read_f32("layer5.self_attn.1");
    const auto global_attention = g4::test_bf16_layer5_attention(
        global_q.bytes, global_k.bytes, global_o.bytes, global_q_norm.bytes,
        global_k_norm.bytes, global_input, expected_global_attention,
        expected_global_probabilities);
    const auto gate = weights.tensor("model.language_model.layers.0.mlp.gate_proj.weight");
    const auto up = weights.tensor("model.language_model.layers.0.mlp.up_proj.weight");
    const auto down = weights.tensor("model.language_model.layers.0.mlp.down_proj.weight");
    const auto mlp_input = oracle.read_f32("layer0.pre_feedforward_layernorm");
    const auto expected_mlp = oracle.read_f32("layer0.mlp");
    const float mlp_error = g4::test_bf16_layer0_dense_mlp(
        gate.bytes, up.bytes, down.bytes, mlp_input, expected_mlp);
    const auto expert_gate_up = weights.tensor("model.language_model.layers.0.experts.gate_up_proj");
    const auto expert_down = weights.tensor("model.language_model.layers.0.experts.down_proj");
    const auto expert_input = oracle.read_f32("layer0.pre_feedforward_layernorm_2");
    const auto top_weights = oracle.read_f32("layer0.router.1");
    const auto top_ids = oracle.read_f32("layer0.router.2");
    const auto expected_experts = oracle.read_f32("layer0.experts");
    const float expert_error = g4::test_bf16_layer0_experts(
        expert_gate_up.bytes, expert_down.bytes, expert_input, top_weights, top_ids,
        expected_experts);
    const auto post_attention_norm =
        weights.tensor("model.language_model.layers.0.post_attention_layernorm.weight");
    const auto pre_ff_norm =
        weights.tensor("model.language_model.layers.0.pre_feedforward_layernorm.weight");
    const auto router_scale = weights.tensor("model.language_model.layers.0.router.scale");
    const auto router_weight = weights.tensor("model.language_model.layers.0.router.proj.weight");
    const auto per_expert_scale =
        weights.tensor("model.language_model.layers.0.router.per_expert_scale");
    const auto router_probabilities = oracle.read_f32("layer0.router.0");
    const auto expected_router_input = oracle.read_f32("layer0.router_input");
    const auto expected_router_logits = oracle.read_f32("layer0.router_logits");
    const auto router = g4::test_bf16_layer0_router(
        post_attention_norm.bytes, pre_ff_norm.bytes, router_scale.bytes, router_weight.bytes,
        per_expert_scale.bytes, expected_embedding, expected_attention, mlp_input,
        expected_router_input, expected_router_logits, router_probabilities, top_weights, top_ids);
    const auto dense_norm_weight =
        weights.tensor("model.language_model.layers.0.post_feedforward_layernorm_1.weight");
    const auto expert_norm_weight =
        weights.tensor("model.language_model.layers.0.post_feedforward_layernorm_2.weight");
    const auto combined_norm_weight =
        weights.tensor("model.language_model.layers.0.post_feedforward_layernorm.weight");
    const auto layer_scalar = weights.tensor("model.language_model.layers.0.layer_scalar");
    const auto expected_dense_norm = oracle.read_f32("layer0.post_feedforward_layernorm_1");
    const auto expected_expert_norm = oracle.read_f32("layer0.post_feedforward_layernorm_2");
    const auto expected_combined_norm = oracle.read_f32("layer0.post_feedforward_layernorm");
    const auto expected_layer_output = oracle.read_f32("hidden.1");
    const auto tail = g4::test_bf16_layer0_tail(
        dense_norm_weight.bytes, expert_norm_weight.bytes, combined_norm_weight.bytes,
        post_attention_norm.bytes, layer_scalar.bytes, expected_embedding, expected_attention, expected_mlp,
        expected_experts, expected_dense_norm, expected_expert_norm, expected_combined_norm,
        expected_layer_output);
    std::cout << std::setprecision(8)
              << "differential.embedding.max_abs_error=" << result.embedding_max_abs_error << '\n'
              << "differential.layer0_input_norm.max_abs_error="
              << result.input_norm_max_abs_error << '\n'
              << "differential.layer0_attention.max_abs_error="
              << attention.output_max_abs_error << '\n'
              << "differential.layer0_attention_prob.max_abs_error="
              << attention.probability_max_abs_error << '\n'
              << "differential.layer5_global_attention.max_abs_error="
              << global_attention.output_max_abs_error << '\n'
              << "differential.layer5_global_attention_prob.max_abs_error="
              << global_attention.probability_max_abs_error << '\n'
              << "differential.layer0_dense_mlp.max_abs_error=" << mlp_error << '\n'
              << "differential.layer0_experts.max_abs_error=" << expert_error << '\n'
              << "differential.layer0_pre_ff_norm.max_abs_error="
              << router.pre_feedforward_norm_max_abs_error << '\n'
              << "differential.layer0_router_input.max_abs_error="
              << router.input_max_abs_error << '\n'
              << "differential.layer0_router_logits.max_abs_error="
              << router.logits_max_abs_error << '\n'
              << "differential.layer0_router_prob.max_abs_error="
              << router.probability_max_abs_error << '\n'
              << "differential.layer0_router_weight.max_abs_error="
              << router.top_weight_max_abs_error << '\n'
              << "differential.layer0_router_fused.max_abs_error="
              << router.fused_max_abs_error << '\n'
              << "differential.layer0_router_ids_match=" << std::boolalpha
              << router.top_ids_match << '\n'
              << "differential.layer0_dense_post_norm.max_abs_error="
              << tail.dense_norm_max_abs_error << '\n'
              << "differential.layer0_expert_post_norm.max_abs_error="
              << tail.expert_norm_max_abs_error << '\n'
              << "differential.layer0_combined_post_norm.max_abs_error="
              << tail.combined_norm_max_abs_error << '\n'
              << "differential.layer0_output.max_abs_error="
              << tail.layer_output_max_abs_error << '\n'
              << "differential.layer0_tail_fused.max_abs_error="
              << tail.fused_max_abs_error << '\n';
    return result.embedding_max_abs_error <= 0.02F && result.input_norm_max_abs_error <= 0.02F &&
                   attention.output_max_abs_error <= 0.25F &&
                   attention.probability_max_abs_error <= 0.005F && mlp_error <= 0.25F &&
                   global_attention.output_max_abs_error <= 0.25F &&
                   global_attention.probability_max_abs_error <= 0.005F &&
                   expert_error <= 1.0F && router.pre_feedforward_norm_max_abs_error <= 0.02F &&
                   router.input_max_abs_error <= 0.0001F && router.logits_max_abs_error <= 0.02F &&
                   router.probability_max_abs_error <= 0.002F &&
                   router.top_weight_max_abs_error <= 0.002F &&
                   router.fused_max_abs_error == 0.0F && router.top_ids_match &&
                   tail.dense_norm_max_abs_error <= 0.02F && tail.expert_norm_max_abs_error <= 0.02F &&
                   tail.combined_norm_max_abs_error <= 0.02F &&
                   tail.layer_output_max_abs_error <= 0.02F &&
                   tail.fused_max_abs_error == 0.0F
               ? 0 : 1;
  }
  if (command == "tokenize") {
    if (argc < 3) throw std::runtime_error("tokenize requires TEXT");
    const auto model_dir = tensor_model_arg(argc, argv);
    const g4::Tokenizer tokenizer(model_dir);
    const auto ids = tokenizer.encode(argv[2]);
    std::cout << "ids=";
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::cout << (i ? "," : "") << ids[i];
    }
    std::cout << "\ndecoded=" << tokenizer.decode(ids) << '\n';
    return 0;
  }
  if (command == "decode-attention-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 4096;
    print_gpu(g4::select_target_gpu());
    const auto result = g4::test_decode_attention(context);
    std::cout << "attention.context=" << context << '\n'
              << "attention.sliding_max_abs_error=" << result.sliding_max_abs_error << '\n'
              << "attention.sliding_us=" << result.sliding_microseconds << '\n'
              << "attention.global_max_abs_error=" << result.global_max_abs_error << '\n'
              << "attention.global_us=" << result.global_microseconds << '\n';
    return 0;
  }
  if (command == "turboquant-kv-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 16384;
    print_gpu(g4::select_target_gpu());
    const auto result = g4::benchmark_turboquant_global_kv(context);
    std::cout << std::setprecision(6)
              << "turboquant.context=" << context << '\n'
              << "turboquant.bf16_us=" << result.bf16_microseconds << '\n'
              << "turboquant.tq4_us=" << result.tq4_microseconds << '\n'
              << "turboquant.speedup="
              << result.bf16_microseconds / result.tq4_microseconds << '\n'
              << "turboquant.quantize_us_per_token="
              << result.quantize_microseconds_per_token << '\n'
              << "turboquant.max_abs_error=" << result.max_abs_error << '\n'
              << "turboquant.mean_abs_error=" << result.mean_abs_error << '\n'
              << "turboquant.reference_rms=" << result.reference_rms << '\n'
              << "turboquant.normalized_rmse=" << result.normalized_rmse << '\n'
              << "turboquant.cosine_similarity=" << result.cosine_similarity << '\n'
              << "turboquant.compression_ratio=" << result.compression_ratio << '\n';
    return std::isfinite(result.normalized_rmse) &&
                   result.normalized_rmse < 0.25F &&
                   result.cosine_similarity > 0.98F &&
                   result.compression_ratio > 3.9F
               ? 0
               : 1;
  }
  if (command == "decode-attention-batch-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 4096;
    print_gpu(g4::select_target_gpu());
    const auto result = g4::test_decode_attention_batch(context);
    std::cout << "attention_batch.context=" << context << '\n'
              << "attention_batch.tokens=5\n"
              << "attention_batch.sliding_max_abs_error="
              << result.sliding_max_abs_error << '\n'
              << "attention_batch.sliding_us=" << result.sliding_microseconds << '\n'
              << "attention_batch.global_max_abs_error="
              << result.global_max_abs_error << '\n'
              << "attention_batch.global_us=" << result.global_microseconds << '\n';
    return 0;
  }
  if (command == "sliding-kv-append-test") {
    if (argc != 2) throw std::runtime_error("sliding-kv-append-test takes no arguments");
    print_gpu(g4::select_target_gpu());
    const auto result = g4::test_sliding_kv_large_append();
    std::cout << "sliding_kv_append.checked_tokens=" << result.checked_tokens << '\n'
              << "sliding_kv_append.key_max_abs_error="
              << result.key_max_abs_error << '\n'
              << "sliding_kv_append.value_max_abs_error="
              << result.value_max_abs_error << '\n';
    return result.checked_tokens == 1024 && result.key_max_abs_error == 0.0F &&
                   result.value_max_abs_error == 0.0F
               ? 0
               : 1;
  }
  if (command == "attention-sublayers-bench") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 4096;
    print_gpu(g4::select_target_gpu());
    const g4::DeviceModel model(kDefaultModel);
    g4::KvCache cache(context);
    g4::AttentionWorkspace workspace(context);
    const auto sliding = g4::benchmark_attention_sublayer(
        model, cache, workspace, 0, context);
    const auto global = g4::benchmark_attention_sublayer(
        model, cache, workspace, 5, context);
    std::cout << "attention_sublayer.context=" << context << '\n'
              << "attention_sublayer.sliding_eager_us=" << sliding.eager_microseconds << '\n'
              << "attention_sublayer.sliding_graph_us=" << sliding.graph_microseconds << '\n'
              << "attention_sublayer.sliding_checksum=" << sliding.output_checksum << '\n'
              << "attention_sublayer.global_eager_us=" << global.eager_microseconds << '\n'
              << "attention_sublayer.global_graph_us=" << global.graph_microseconds << '\n'
              << "attention_sublayer.global_checksum=" << global.output_checksum << '\n';
    return 0;
  }
  usage();
  return 2;
} catch (const std::exception& error) {
  std::cerr << "g4: " << error.what() << '\n';
  return 1;
}
