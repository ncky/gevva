// Isolated native-vs-cuDNN attention correctness, graph, and timing fixture.
#include "g4/cudnn_attention.hpp"
#include "g4/gpu.hpp"
#include "g4/sm120_attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}
struct Buffers {
  std::vector<void *> pointers;
  ~Buffers() {
    for (auto ptr : pointers)
      cudaFree(ptr);
  }
  void *allocate(size_t bytes) {
    void *ptr{};
    check(cudaMalloc(&ptr, bytes));
    pointers.push_back(ptr);
    return ptr;
  }
  void *upload(const std::vector<__nv_bfloat16> &data) {
    void *ptr = allocate(data.size() * 2);
    check(
        cudaMemcpy(ptr, data.data(), data.size() * 2, cudaMemcpyHostToDevice));
    return ptr;
  }
};
} // namespace

int main(int argc, char **argv) try {
  const int b = argc > 1 ? std::stoi(argv[1]) : 1;
  const int sq = argc > 2 ? std::stoi(argv[2]) : 1057;
  const int sk = argc > 3 ? std::stoi(argv[3]) : sq;
  if (b < 1 || b > 8 || sq < 1 || sq > 4608 || sk < sq || sk > 5632)
    throw std::runtime_error(
        "usage: g4-sm120-attention-bench [B=1..8] [Sq=1..4608] [Sk=Sq..5632]");
  g4::select_target_gpu();
  // Always compare the actual cuDNN backend against the native backend.
  unsetenv("G4_SM120_TEXT_PREFILL");
  std::mt19937 rng(71237);
  auto random = [&](size_t n, float scale) {
    std::vector<__nv_bfloat16> result(n);
    for (auto &x : result)
      x = __float2bfloat16_rn((static_cast<float>(rng() % 20001) / 10000 - 1) *
                              scale);
    return result;
  };
  const auto hq = random(size_t(b) * sq * 4096, 0.5F);
  const auto hk = random(size_t(b) * sk * 2048, 0.5F);
  const auto hv = random(hk.size(), 1.0F);
  Buffers memory;
  const auto q = memory.upload(hq), k = memory.upload(hk),
             v = memory.upload(hv);
  const auto output = memory.allocate(hq.size() * 2);
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto native = [&] {
    g4::launch_sm120_text_prefill(q, k, v, output, b, sq, sk, stream);
  };
  auto cudnn = [&] {
    g4::launch_cudnn_text_prefill(q, k, v, output, b, sq, stream, sk);
  };
  auto download = [&] {
    check(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> result(hq.size());
    check(cudaMemcpy(result.data(), output, result.size() * 2,
                     cudaMemcpyDeviceToHost));
    return result;
  };
  cudnn();
  const auto reference = download();
  native();
  const auto actual = download();
  const char *override_tile = std::getenv("G4_SM120_ATTENTION_Q_TILE");
  const bool had_override = override_tile != nullptr;
  const std::string saved_tile = override_tile ? override_tile : "";
  setenv("G4_SM120_ATTENTION_Q_TILE", "64", 1);
  native();
  const auto tile64 = download();
  setenv("G4_SM120_ATTENTION_Q_TILE", "128", 1);
  native();
  const auto tile128 = download();
  if (had_override)
    setenv("G4_SM120_ATTENTION_Q_TILE", saved_tile.c_str(), 1);
  else
    unsetenv("G4_SM120_ATTENTION_Q_TILE");
  const bool exact_tiles =
      std::memcmp(tile64.data(), tile128.data(), actual.size() * 2) == 0;
  float max_diff = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float a = __bfloat162float(actual[i]),
                r = __bfloat162float(reference[i]);
    if (!std::isfinite(a) || !std::isfinite(r))
      throw std::runtime_error("nonfinite output");
    max_diff = std::max(max_diff, std::abs(a - r));
  }
  // Independent full-FP32 softmax oracle: batch/head mapping, tails, and both
  // sides of the sliding-window boundary. Unlike the old oracle, no rounded QK.
  float max_oracle_error = 0;
  for (int batch = 0; batch < b; ++batch)
    for (int row : {0, std::min(sq - 1, 1), std::min(sq - 1, 1023),
                    std::min(sq - 1, 1024), sq - 1})
      for (int head : {0, 1, 7, 15}) {
        const int end = sk - sq + row, begin = std::max(0, end - 1023);
        std::vector<double> logits(end - begin + 1);
        double maximum = -INFINITY, denominator = 0;
        for (int key = begin; key <= end; ++key) {
          double dot = 0;
          for (int d = 0; d < 256; ++d)
            dot +=
                double(__bfloat162float(
                    hq[(size_t(batch) * sq + row) * 4096 + head * 256 + d])) *
                __bfloat162float(hk[(size_t(batch) * sk + key) * 2048 +
                                    (head / 2) * 256 + d]);
          logits[key - begin] = dot;
          maximum = std::max(maximum, dot);
        }
        for (auto &p : logits) {
          p = std::exp(p - maximum);
          denominator += p;
        }
        for (int d = 0; d < 256; ++d) {
          double expected = 0;
          for (int key = begin; key <= end; ++key)
            expected += logits[key - begin] / denominator *
                        __bfloat162float(hv[(size_t(batch) * sk + key) * 2048 +
                                            (head / 2) * 256 + d]);
          max_oracle_error = std::max(
              max_oracle_error,
              float(std::abs(
                  expected -
                  __bfloat162float(actual[(size_t(batch) * sq + row) * 4096 +
                                          head * 256 + d]))));
        }
      }
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  native();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  check(cudaGraphLaunch(executable, stream));
  const auto replay = download();
  const bool exact_replay =
      std::memcmp(actual.data(), replay.data(), actual.size() * 2) == 0;
  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  auto time = [&](auto launch) {
    for (int i = 0; i < 10; ++i)
      launch();
    check(cudaEventRecord(start, stream));
    for (int i = 0; i < 100; ++i)
      launch();
    check(cudaEventRecord(stop, stream));
    check(cudaEventSynchronize(stop));
    float ms{};
    check(cudaEventElapsedTime(&ms, start, stop));
    return ms * 10;
  };
  std::vector<float> cudnn_times, native_times, graph_times;
  for (int round = 0; round < 5; ++round) {
    if (round % 2 == 0)
      cudnn_times.push_back(time(cudnn));
    if (round == 0 && std::getenv("G4_PROFILE_NATIVE_ATTENTION"))
      check(cudaProfilerStart());
    native_times.push_back(time(native));
    if (round == 0 && std::getenv("G4_PROFILE_NATIVE_ATTENTION"))
      check(cudaProfilerStop());
    if (round % 2 == 1)
      cudnn_times.push_back(time(cudnn));
    graph_times.push_back(
        time([&] { check(cudaGraphLaunch(executable, stream)); }));
  }
  auto median = [](std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const float cudnn_us = median(cudnn_times), native_us = median(native_times),
              graph_us = median(graph_times);
  std::cout << nlohmann::json{{"batch", b},
                              {"query_tokens", sq},
                              {"key_tokens", sk},
                              {"cudnn_us", cudnn_us},
                              {"native_us", native_us},
                              {"native_graph_us", graph_us},
                              {"speedup", cudnn_us / native_us},
                              {"max_cudnn_difference", max_diff},
                              {"cudnn_samples_us", cudnn_times},
                              {"native_samples_us", native_times},
                              {"native_graph_samples_us", graph_times},
                              {"exact_tile_variants", exact_tiles},
                              {"max_fp32_oracle_error", max_oracle_error},
                              {"exact_graph_replay", exact_replay}}
                   .dump()
            << '\n';
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  return max_oracle_error < 0.0078125F && max_diff < 0.015625F &&
                 exact_replay && exact_tiles
             ? 0
             : 1;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
