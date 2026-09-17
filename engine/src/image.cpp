#include "g4/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <setjmp.h>
#include <stdexcept>
#include <string>

#include <jpeglib.h>
#include <png.h>

namespace {

struct JpegError {
  jpeg_error_mgr base;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX]{};
};

void jpeg_error_exit(j_common_ptr context) {
  auto* error = reinterpret_cast<JpegError*>(context->err);
  (*context->err->format_message)(context, error->message);
  longjmp(error->jump, 1);
}

g4::RgbImage decode_jpeg(jpeg_decompress_struct& decoder) {
  jpeg_read_header(&decoder, TRUE);
  decoder.out_color_space = JCS_RGB;
  jpeg_start_decompress(&decoder);
  g4::RgbImage result;
  result.width = static_cast<int>(decoder.output_width);
  result.height = static_cast<int>(decoder.output_height);
  result.pixels.resize(static_cast<std::size_t>(result.width) * result.height * 3);
  while (decoder.output_scanline < decoder.output_height) {
    auto* row = result.pixels.data() +
        static_cast<std::size_t>(decoder.output_scanline) * result.width * 3;
    JSAMPROW rows[] = {row};
    jpeg_read_scanlines(&decoder, rows, 1);
  }
  jpeg_finish_decompress(&decoder);
  return result;
}

g4::RgbImage load_jpeg(const std::filesystem::path& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open JPEG: " + path.string());
  jpeg_decompress_struct decoder{};
  JpegError error{};
  decoder.err = jpeg_std_error(&error.base);
  error.base.error_exit = jpeg_error_exit;
  if (setjmp(error.jump)) {
    jpeg_destroy_decompress(&decoder);
    std::fclose(file);
    throw std::runtime_error("JPEG decode failed: " + std::string(error.message));
  }
  jpeg_create_decompress(&decoder);
  jpeg_stdio_src(&decoder, file);
  auto result = decode_jpeg(decoder);
  jpeg_destroy_decompress(&decoder);
  std::fclose(file);
  return result;
}

g4::RgbImage load_png(const std::filesystem::path& path) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, path.c_str()))
    throw std::runtime_error("PNG header decode failed: " + path.string());
  image.format = PNG_FORMAT_RGB;
  g4::RgbImage result;
  result.width = static_cast<int>(image.width);
  result.height = static_cast<int>(image.height);
  result.pixels.resize(PNG_IMAGE_SIZE(image));
  if (!png_image_finish_read(&image, nullptr, result.pixels.data(), 0, nullptr)) {
    const std::string message = image.message;
    png_image_free(&image);
    throw std::runtime_error("PNG decode failed: " + message);
  }
  png_image_free(&image);
  return result;
}

g4::RgbImage load_jpeg(std::span<const std::uint8_t> encoded) {
  jpeg_decompress_struct decoder{};
  JpegError error{};
  decoder.err = jpeg_std_error(&error.base);
  error.base.error_exit = jpeg_error_exit;
  if (setjmp(error.jump)) {
    jpeg_destroy_decompress(&decoder);
    throw std::runtime_error("JPEG decode failed: " + std::string(error.message));
  }
  jpeg_create_decompress(&decoder);
  jpeg_mem_src(&decoder, encoded.data(), encoded.size());
  auto result = decode_jpeg(decoder);
  jpeg_destroy_decompress(&decoder);
  return result;
}

g4::RgbImage load_png(std::span<const std::uint8_t> encoded) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&image, encoded.data(), encoded.size()))
    throw std::runtime_error("PNG header decode failed from memory");
  image.format = PNG_FORMAT_RGB;
  g4::RgbImage result;
  result.width = static_cast<int>(image.width);
  result.height = static_cast<int>(image.height);
  result.pixels.resize(PNG_IMAGE_SIZE(image));
  if (!png_image_finish_read(&image, nullptr, result.pixels.data(), 0, nullptr)) {
    const std::string message = image.message;
    png_image_free(&image);
    throw std::runtime_error("PNG decode failed: " + message);
  }
  png_image_free(&image);
  return result;
}

double cubic(double x) {
  // The antialiased path follows Pillow and Torchvision's AA functor.
  constexpr double a = -0.5;
  x = std::abs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  return 0.0;
}

struct Contribution {
  int first{};
  std::vector<std::int16_t> weights;
};

struct ContributionTable {
  std::vector<Contribution> rows;
  unsigned precision{};
};

ContributionTable resize_contributions(int source, int destination) {
  const double scale = static_cast<double>(source) / destination;
  const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
  const double support = 2.0 * std::max(1.0, scale);
  struct FloatContribution {
    int first{};
    std::vector<double> weights;
  };
  std::vector<FloatContribution> floating(destination);
  double maximum_weight = 0.0;
  for (int output = 0; output < destination; ++output) {
    const double center = scale * (output + 0.5);
    const int first = std::max(static_cast<int>(center - support + 0.5), 0);
    const int last = std::min(static_cast<int>(center + support + 0.5), source);
    auto& contribution = floating[output];
    contribution.first = first;
    contribution.weights.resize(last - first);
    double sum = 0.0;
    for (int input = first; input < last; ++input) {
      const double weight = cubic((input - center + 0.5) * inverse_scale);
      contribution.weights[input - first] = weight;
      sum += weight;
    }
    for (double& weight : contribution.weights) {
      weight /= sum;
      maximum_weight = std::max(maximum_weight, weight);
    }
  }
  ContributionTable result;
  for (result.precision = 0; result.precision < 22; ++result.precision) {
    const int next = static_cast<int>(
        0.5 + maximum_weight * (1 << (result.precision + 1)));
    if (next >= (1 << 15)) break;
  }
  result.rows.resize(destination);
  for (int output = 0; output < destination; ++output) {
    result.rows[output].first = floating[output].first;
    for (const double weight : floating[output].weights) {
      const double scaled = weight * (1 << result.precision);
      result.rows[output].weights.push_back(static_cast<std::int16_t>(
          scaled < 0.0 ? static_cast<int>(scaled - 0.5)
                       : static_cast<int>(scaled + 0.5)));
    }
  }
  return result;
}

std::vector<float> resize_bicubic_antialias_patches(
    const g4::RgbImage& image, int width, int height, int worker_threads) {
  constexpr int kPatch = 16;
  constexpr int kPatchValues = kPatch * kPatch * 3;
  const auto horizontal = resize_contributions(image.width, width);
  const auto vertical = resize_contributions(image.height, height);
  std::vector<std::uint8_t> temporary(
      static_cast<std::size_t>(image.height) * width * 3);
#pragma omp parallel for schedule(static) num_threads(worker_threads)
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto& c = horizontal.rows[x];
      for (int channel = 0; channel < 3; ++channel) {
        int value = 1 << (horizontal.precision - 1);
        for (std::size_t tap = 0; tap < c.weights.size(); ++tap) {
          const int source_x = c.first + static_cast<int>(tap);
          value += c.weights[tap] * image.pixels[
              (static_cast<std::size_t>(y) * image.width + source_x) * 3 +
              channel];
        }
        temporary[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
            static_cast<std::uint8_t>(
                std::clamp(value >> horizontal.precision, 0, 255));
      }
    }
  }
  std::vector<float> output(static_cast<std::size_t>(height) * width * 3);
  const int patch_width = width / kPatch;
#pragma omp parallel for schedule(static) num_threads(worker_threads)
  for (int y = 0; y < height; ++y) {
    const auto& c = vertical.rows[y];
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        int value = 1 << (vertical.precision - 1);
        for (std::size_t tap = 0; tap < c.weights.size(); ++tap) {
          const int source_y = c.first + static_cast<int>(tap);
          value += c.weights[tap] * temporary[
              (static_cast<std::size_t>(source_y) * width + x) * 3 + channel];
        }
        const int patch = (y / kPatch) * patch_width + x / kPatch;
        const int within_patch =
            ((y % kPatch) * kPatch + x % kPatch) * 3 + channel;
        output[static_cast<std::size_t>(patch) * kPatchValues + within_patch] =
            static_cast<float>(
                std::clamp(value >> vertical.precision, 0, 255)) / 255.0F;
      }
    }
  }
  return output;
}

}  // namespace

namespace g4 {

RgbImage load_rgb_image(const std::filesystem::path& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open image: " + path.string());
  unsigned char signature[8]{};
  const auto bytes = std::fread(signature, 1, sizeof(signature), file);
  std::fclose(file);
  if (bytes >= 2 && signature[0] == 0xff && signature[1] == 0xd8)
    return load_jpeg(path);
  if (bytes == 8 && png_sig_cmp(signature, 0, 8) == 0) return load_png(path);
  throw std::runtime_error("unsupported image format (expected JPEG or PNG): " +
                           path.string());
}

RgbImage load_rgb_image(std::span<const std::uint8_t> encoded) {
  if (encoded.size() >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8)
    return load_jpeg(encoded);
  if (encoded.size() >= 8 && png_sig_cmp(encoded.data(), 0, 8) == 0)
    return load_png(encoded);
  throw std::runtime_error("unsupported in-memory image format (expected JPEG or PNG)");
}

VisionInput preprocess_gemma4_image(const RgbImage& image,
                                    int maximum_soft_tokens, bool compact,
                                    int worker_threads) {
  constexpr int kPatch = 16;
  constexpr int kPool = 3;
  constexpr int kPatchValues = kPatch * kPatch * 3;
  if (image.width < 1 || image.height < 1 ||
      image.pixels.size() !=
          static_cast<std::size_t>(image.width) * image.height * 3)
    throw std::runtime_error("invalid RGB image geometry");
  if (worker_threads < 1 || worker_threads > 8)
    throw std::runtime_error("image preprocessing workers must be in [1, 8]");
  if (maximum_soft_tokens != 70 && maximum_soft_tokens != 140 &&
      maximum_soft_tokens != 280 && maximum_soft_tokens != 560 &&
      maximum_soft_tokens != 1120)
    throw std::runtime_error("Gemma 4 soft tokens must be 70, 140, 280, 560, or 1120");
  const int maximum_patches = maximum_soft_tokens * kPool * kPool;
  const double factor = std::sqrt(
      static_cast<double>(maximum_patches) * kPatch * kPatch /
      (static_cast<double>(image.width) * image.height));
  constexpr int kSideMultiple = kPatch * kPool;
  int height = static_cast<int>(std::floor(image.height * factor /
                                           kSideMultiple)) * kSideMultiple;
  int width = static_cast<int>(std::floor(image.width * factor /
                                          kSideMultiple)) * kSideMultiple;
  const int maximum_side = (maximum_patches / (kPool * kPool)) * kSideMultiple;
  if (height == 0 && width == 0)
    throw std::runtime_error("image aspect ratio produces a zero-sized resize");
  if (height == 0) {
    height = kSideMultiple;
    width = std::min((image.width / image.height) * kSideMultiple, maximum_side);
  } else if (width == 0) {
    width = kSideMultiple;
    height = std::min((image.height / image.width) * kSideMultiple, maximum_side);
  }
  const int patch_height = height / kPatch;
  const int patch_width = width / kPatch;
  const int patches = patch_height * patch_width;
  if (patches > maximum_patches)
    throw std::runtime_error("Gemma 4 resized image exceeds its patch budget");
  VisionInput result;
  result.source_width = image.width;
  result.source_height = image.height;
  result.resized_width = width;
  result.resized_height = height;
  result.patch_count = patches;
  result.soft_token_count = patches / (kPool * kPool);
  const int stored_patches = compact ? patches : maximum_patches;
  result.pixel_values = resize_bicubic_antialias_patches(
      image, width, height, worker_threads);
  result.pixel_values.resize(
      static_cast<std::size_t>(stored_patches) * kPatchValues, 0.0F);
  result.position_ids.assign(static_cast<std::size_t>(stored_patches) * 2, -1);
#pragma omp parallel for schedule(static) num_threads(worker_threads)
  for (int patch_y = 0; patch_y < patch_height; ++patch_y) {
    for (int patch_x = 0; patch_x < patch_width; ++patch_x) {
      const int patch = patch_y * patch_width + patch_x;
      result.position_ids[patch * 2] = patch_x;
      result.position_ids[patch * 2 + 1] = patch_y;
    }
  }
  return result;
}

}  // namespace g4
