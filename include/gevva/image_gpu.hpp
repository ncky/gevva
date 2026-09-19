#pragma once
#include "gevva/image.hpp"
#include <memory>
#include <stdexcept>

namespace gevva {
class GpuExecutionContext;
struct ImageInputError : std::runtime_error { using std::runtime_error::runtime_error; };
int test_gpu_image_preprocessing(const std::filesystem::path& path);
struct DeviceVisionInput {
  VisionInput geometry;
  const float* pixels{};
  const std::int32_t* positions{};
  bool hardware_jpeg{};
};
// Owns one reusable image's scratch and coefficient tables. Returned pointers
// remain valid until the next prepare call; consume on the supplied stream.
class GpuImagePreprocessor {
 public:
  GpuImagePreprocessor();
  ~GpuImagePreprocessor();
  DeviceVisionInput prepare(std::span<const std::uint8_t> encoded, int soft_tokens,
                            GpuExecutionContext& context);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
