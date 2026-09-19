#include "gevva/image_gpu.hpp"
#include "gevva/gpu.hpp"
#include "gevva/runtime.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gevva {
namespace {
void check_image(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
__global__ void resize_horizontal(const unsigned char* rgb, unsigned char* out,
    int source_width, int source_height, int width, const ResizeAxisEntry* entries,
    const short* weights, int precision) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < source_height * width * 3;
       i += blockDim.x * gridDim.x) {
    const int channel = i % 3, x = (i / 3) % width, y = i / (width * 3);
    const auto row = entries[x];
    int sum = 1 << (precision - 1);
    for (int tap = 0; tap < row.count; ++tap)
      sum += weights[row.offset + tap] * rgb[(static_cast<std::size_t>(y) * source_width + row.first + tap) * 3 + channel];
    out[i] = max(0, min(255, sum >> precision));
  }
}
__global__ void resize_vertical_pack(const unsigned char* rgb, float* out, int* positions,
    int width, int height, const ResizeAxisEntry* entries, const short* weights, int precision) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < height * width * 3;
       i += blockDim.x * gridDim.x) {
    const int channel = i % 3, x = (i / 3) % width, y = i / (width * 3);
    const auto row = entries[y];
    int sum = 1 << (precision - 1);
    for (int tap = 0; tap < row.count; ++tap)
      sum += weights[row.offset + tap] * rgb[(static_cast<std::size_t>(row.first + tap) * width + x) * 3 + channel];
    const int patch = (y / 16) * (width / 16) + x / 16;
    const int within = ((y % 16) * 16 + x % 16) * 3 + channel;
    // Match the reference float division, even under --use_fast_math.
    out[static_cast<std::size_t>(patch) * 768 + within] = __fdiv_rn(float(max(0, min(255, sum >> precision))), 255.0F);
    if (within == 0) { positions[patch * 2] = x / 16; positions[patch * 2 + 1] = y / 16; }
  }
}
}
struct GpuImagePreprocessor::Impl {
  nvjpegHandle_t jpeg{};
  nvjpegJpegState_t state{};
  DeviceScratchPool storage;
  std::array<int,3> previous{};
  ResizeAxis horizontal, vertical;
  VisionInput geometry;
  Impl() {
    if (!std::getenv("GEVVA_CPU_JPEG_DECODE") &&
        nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION, &jpeg) == NVJPEG_STATUS_SUCCESS) {
      if (nvjpegJpegStateCreate(jpeg, &state) != NVJPEG_STATUS_SUCCESS ||
          nvjpegDecodeBatchedInitialize(jpeg, state, 1, 1, NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS) {
        if (state) nvjpegJpegStateDestroy(state);
        nvjpegDestroy(jpeg); state = nullptr; jpeg = nullptr;
      }
    }
  }
  ~Impl() { if (state) nvjpegJpegStateDestroy(state); if (jpeg) nvjpegDestroy(jpeg); }
};
GpuImagePreprocessor::GpuImagePreprocessor() : impl_(std::make_unique<Impl>()) {}
GpuImagePreprocessor::~GpuImagePreprocessor() = default;
DeviceVisionInput GpuImagePreprocessor::prepare(std::span<const std::uint8_t> encoded,
    int soft_tokens, GpuExecutionContext& context) {
  auto& p = *impl_; auto stream = context.stream();
  bool hardware = false;
  int width = 0, height = 0;
  if (p.jpeg && encoded.size() >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8) {
    int channels, widths[4]{}, heights[4]{}; nvjpegChromaSubsampling_t sub;
    if (nvjpegGetImageInfo(p.jpeg, encoded.data(), encoded.size(), &channels, &sub, widths, heights) == NVJPEG_STATUS_SUCCESS) {
      width = widths[0]; height = heights[0]; hardware = width > 0 && height > 0;
    }
  }
  auto decode_cpu = [&] {
    try { return load_rgb_image(encoded); }
    catch (const std::runtime_error& error) { throw ImageInputError(error.what()); }
  };
  RgbImage cpu;
  if (!hardware) { cpu = decode_cpu(); width = cpu.width; height = cpu.height; }
  p.storage.reset();
  auto* rgb = static_cast<unsigned char*>(p.storage.allocate(static_cast<std::size_t>(width) * height * 3));
  if (hardware) {
    nvjpegImage_t out{}; out.channel[0] = rgb; out.pitch[0] = width * 3;
    const auto* data = encoded.data(); const auto size = encoded.size();
    const auto status = nvjpegDecodeBatched(p.jpeg, p.state, &data, &size, &out, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
      // Progressive/unsupported JPEGs retain the existing CPU decoder. Reset
      // the reusable decoder after an unsuccessful batch before another use.
      check_image(cudaStreamSynchronize(stream), "finish unsupported JPEG decode");
      if (nvjpegDecodeBatchedInitialize(p.jpeg, p.state, 1, 1, NVJPEG_OUTPUT_RGBI) != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error("reset hardware JPEG decoder failed");
      cpu = decode_cpu(); hardware = false;
      if (cpu.width != width || cpu.height != height) throw std::runtime_error("JPEG decoder geometry mismatch");
    }
  }
  if (!hardware) {
    auto* stage = p.storage.host_staging(cpu.pixels.size());
    std::memcpy(stage, cpu.pixels.data(), cpu.pixels.size());
    check_image(cudaMemcpyAsync(rgb, stage, cpu.pixels.size(), cudaMemcpyHostToDevice, stream), "upload RGB bytes");
  }
  const std::array shape{width, height, soft_tokens};
  const bool changed = shape != p.previous;
  if (changed) {
    p.previous = {}; // Failed allocation/upload must not leave old cache identity valid.
    try { p.geometry = gemma4_image_geometry(width, height, soft_tokens); }
    catch (const std::runtime_error& error) { throw ImageInputError(error.what()); }
    p.horizontal = gemma4_resize_axis(width, p.geometry.resized_width);
    p.vertical = gemma4_resize_axis(height, p.geometry.resized_height);
  }
  const auto& g = p.geometry;
  auto* temporary = static_cast<unsigned char*>(p.storage.allocate(static_cast<std::size_t>(height) * g.resized_width * 3));
  auto* pixels = static_cast<float*>(p.storage.allocate(static_cast<std::size_t>(g.patch_count) * 768 * sizeof(float)));
  auto* positions = static_cast<int*>(p.storage.allocate(static_cast<std::size_t>(g.patch_count) * 2 * sizeof(int)));
  auto table = [&](const ResizeAxis& axis) {
    auto* entries = static_cast<ResizeAxisEntry*>(p.storage.allocate(axis.entries.size() * sizeof(ResizeAxisEntry)));
    auto* weights = static_cast<short*>(p.storage.allocate(axis.weights.size() * sizeof(short)));
    if (changed) {
      check_image(cudaMemcpyAsync(entries, axis.entries.data(), axis.entries.size() * sizeof(ResizeAxisEntry), cudaMemcpyHostToDevice, stream), "upload resize indices");
      check_image(cudaMemcpyAsync(weights, axis.weights.data(), axis.weights.size() * sizeof(short), cudaMemcpyHostToDevice, stream), "upload resize coefficients");
    }
    return std::pair(entries, weights);
  };
  const auto h = table(p.horizontal), v = table(p.vertical);
  resize_horizontal<<<(height * g.resized_width * 3 + 255) / 256, 256, 0, stream>>>(rgb, temporary,
      width, height, g.resized_width, h.first, h.second, p.horizontal.precision);
  resize_vertical_pack<<<(g.resized_height * g.resized_width * 3 + 255) / 256, 256, 0, stream>>>(temporary,
      pixels, positions, g.resized_width, g.resized_height, v.first, v.second, p.vertical.precision);
  check_image(cudaGetLastError(), "resize and pack image");
  // Callers may destroy or replace CPU coefficient/image buffers after return.
  check_image(cudaStreamSynchronize(stream), "finish image preprocessing");
  p.previous = shape;
  return {g, pixels, positions, hardware};
}
int test_gpu_image_preprocessing(const std::filesystem::path& path) {
  select_target_gpu();
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open preprocessing fixture");
  const std::vector<std::uint8_t> encoded(std::istreambuf_iterator<char>(file), {});
  const auto rgb = load_rgb_image(encoded);
  GpuExecutionContext context;
  GpuImagePreprocessor processor;
  double maximum = 0, error_sum = 0; std::size_t values = 0; int hardware = 0;
  for (int budget : {70, 140, 280, 560, 1120}) {
    const auto reference = preprocess_gemma4_image(rgb, budget, true);
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto actual = processor.prepare(encoded, budget, context);
      if (actual.geometry.resized_width != reference.resized_width ||
          actual.geometry.resized_height != reference.resized_height ||
          actual.geometry.position_ids != reference.position_ids)
        throw std::runtime_error("GPU image preprocessing geometry mismatch");
      std::vector<float> pixels(reference.pixel_values.size());
      std::vector<int> positions(reference.position_ids.size());
      check_image(cudaMemcpy(pixels.data(), actual.pixels, pixels.size() * sizeof(float), cudaMemcpyDeviceToHost), "check image pixels");
      check_image(cudaMemcpy(positions.data(), actual.positions, positions.size() * sizeof(int), cudaMemcpyDeviceToHost), "check image positions");
      if (positions != reference.position_ids) throw std::runtime_error("GPU image patch positions mismatch");
      for (std::size_t i = 0; i < pixels.size(); ++i) {
        if (!std::isfinite(pixels[i])) throw std::runtime_error("non-finite GPU image pixel");
        const double error = std::abs(pixels[i] - reference.pixel_values[i]);
        maximum = std::max(maximum, error); error_sum += error;
      }
      values += pixels.size(); hardware += actual.hardware_jpeg;
    }
  }
  const double tolerance = hardware ? 1.01 / 255.0 : 1e-7;
  const bool passed = maximum <= tolerance;
  std::cout << nlohmann::json({{"passed", passed}, {"exact", maximum == 0},
      {"mean_absolute_error", error_sum / values}, {"max_absolute_error", maximum},
      {"values_checked", values}, {"cases", 10}, {"hardware_jpeg_cases", hardware}}).dump() << '\n';
  return passed ? 0 : 1;
}

}
