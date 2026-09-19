#include "gevva/multimodal.hpp"

#include <stdexcept>

namespace gevva {

std::string format_gemma4_user_turn(std::string_view content) {
  std::string result;
  result.reserve(content.size() + 80);
  result += "<bos><|turn>user\n";
  result += content;
  result += "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  return result;
}

std::string format_gemma4_system_user_turn(std::string_view system,
                                           std::string_view content) {
  std::string result;
  result.reserve(system.size() + content.size() + 120);
  result += "<bos><|turn>system\n";
  result += system;
  result += "<turn|>\n<|turn>user\n";
  result += content;
  result += "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  return result;
}

MultimodalPrompt prepare_image_prompt(const Tokenizer& tokenizer,
                                      std::string_view text,
                                      std::span<const int> image_token_counts) {
  const auto source = tokenizer.encode(text);
  MultimodalPrompt result;
  result.image_token_counts.assign(image_token_counts.begin(),
                                   image_token_counts.end());
  std::size_t image_index = 0;
  std::size_t expanded_size = source.size();
  for (const auto id : source) {
    if (id != kImagePlaceholderToken) continue;
    if (image_index >= image_token_counts.size())
      throw std::runtime_error("prompt has more image markers than image inputs");
    const int count = image_token_counts[image_index++];
    if (count < 1 || count > 1120)
      throw std::runtime_error("Gemma 4 image soft-token count must be in [1, 1120]");
    expanded_size += static_cast<std::size_t>(count) + 1;
  }
  if (image_index != image_token_counts.size())
    throw std::runtime_error("prompt has fewer image markers than image inputs");

  result.input_ids.reserve(expanded_size);
  result.multimodal_types.reserve(expanded_size);
  image_index = 0;
  auto append = [&](std::uint32_t id, std::uint8_t type) {
    result.input_ids.push_back(id);
    result.multimodal_types.push_back(type);
  };
  for (const auto id : source) {
    if (id != kImagePlaceholderToken) {
      append(id, 0);
      continue;
    }
    const int count = image_token_counts[image_index++];
    append(kBeginImageToken, 0);
    result.image_offsets.push_back(result.input_ids.size());
    for (int token = 0; token < count; ++token)
      append(kImagePlaceholderToken, 1);
    append(kEndImageToken, 0);
  }
  return result;
}

}  // namespace gevva
