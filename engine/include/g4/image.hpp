#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace g4 {

struct RgbImage {
  int width{};
  int height{};
  std::vector<std::uint8_t> pixels;
};

struct VisionInput {
  int source_width{};
  int source_height{};
  int resized_width{};
  int resized_height{};
  int patch_count{};
  int soft_token_count{};
  // Fixed Gemma 4 input shape: max_patches x (16 x 16 x RGB), row-major
  // patches. Values are rescaled to [0, 1]; padded rows are zero.
  std::vector<float> pixel_values;
  // Fixed max_patches x 2 array of (x, y), padded with (-1, -1).
  std::vector<std::int32_t> position_ids;
};

RgbImage load_rgb_image(const std::filesystem::path& path);
RgbImage load_rgb_image(std::span<const std::uint8_t> encoded);
VisionInput preprocess_gemma4_image(const RgbImage& image,
                                    int maximum_soft_tokens = 280,
                                    bool compact = false,
                                    int worker_threads = 8);

}  // namespace g4
