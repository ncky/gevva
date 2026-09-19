#include "gevva/decisions.hpp"
#include "gevva/gpu.hpp"
#include "gevva/image_gpu.hpp"
#include "gevva/runtime.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void print_gpu(const gevva::GpuInfo& gpu) {
  std::cout << "gpu.name=" << gpu.name << '\n'
            << "gpu.ordinal=" << gpu.ordinal << '\n'
            << "gpu.pci=" << gpu.pci_bus_id << '\n'
            << "gpu.uuid=GPU-" << gpu.uuid << '\n'
            << "gpu.compute=" << gpu.compute_major << '.' << gpu.compute_minor << '\n'
            << "gpu.vram_gib=" << std::fixed << std::setprecision(2)
            << static_cast<double>(gpu.total_memory) / (1ULL << 30) << '\n';
}
void usage() {
  std::cerr << "Gevva native decision worker (SM120)\n"
            << "Launch the HTTP API with: python -m gevva --config gevva.toml\n"
            << "Native commands: decision-serve, gpu-info, decision-attention-test,\n"
            << "decision-kv-test, text-prefill-attention-test [TOKENS],\n"
            << "prefill-attention-test, sliding-kv-append-test, kv-cache-info [CONTEXT],\n"
            << "image-preprocess-gpu-test IMAGE\n";
}
}
int main(int argc, char** argv) try {
  if (argc < 2) { usage(); return 2; }
  const std::string command = argv[1];
  if (command == "--help" || command == "help") { usage(); return 0; }
  if (command == "decision-attention-test") {
    if (argc != 2) throw std::runtime_error("decision-attention-test takes no arguments");
    return gevva::test_decision_attention();
  }
  if (command == "decision-kv-test") {
    if (argc != 2) throw std::runtime_error("decision-kv-test takes no arguments");
    return gevva::test_decision_kv_fork();
  }
  if (command == "decision-serve") {
    if (argc != 2) throw std::runtime_error("decision-serve takes no arguments");
    return gevva::run_decision_worker();
  }
  if (command == "gpu-info") {
    print_gpu(gevva::select_target_gpu());
    return 0;
  }
  if (command == "prefill-attention-test" || command == "text-prefill-attention-test") {
    if (argc > (command == "text-prefill-attention-test" ? 3 : 2))
      throw std::runtime_error("text-prefill-attention-test accepts optional TOKENS");
    const int text_tokens = argc == 3 ? std::stoi(argv[2]) : 257;
    if (text_tokens < 2 || text_tokens > 4608)
      throw std::runtime_error("text prefill test TOKENS must be in [2, 4608]");
    print_gpu(gevva::select_target_gpu());
    const std::vector<std::int32_t> blocks = command == "text-prefill-attention-test"
        ? std::vector<std::int32_t>(text_tokens, -1)
        : std::vector<std::int32_t>{
        -1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1};
    const auto result = gevva::test_prefill_attention(blocks);
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
  if (command == "image-preprocess-gpu-test") {
    if (argc != 3) throw std::runtime_error("image-preprocess-gpu-test expects IMAGE");
    return gevva::test_gpu_image_preprocessing(argv[2]);
  }
  if (command == "kv-cache-info") {
    if (argc > 3) throw std::runtime_error("expected optional CONTEXT");
    const int context = argc == 3 ? std::stoi(argv[2]) : 32768;
    print_gpu(gevva::select_target_gpu());
    const gevva::KvCache cache(context);
    const gevva::AttentionWorkspace workspace(context);
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
  if (command == "sliding-kv-append-test") {
    if (argc != 2) throw std::runtime_error("sliding-kv-append-test takes no arguments");
    print_gpu(gevva::select_target_gpu());
    const auto result = gevva::test_sliding_kv_large_append();
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
  usage();
  throw std::runtime_error("Unknown command: " + command);
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
