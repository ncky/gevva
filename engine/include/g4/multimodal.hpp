#pragma once

#include "g4/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace g4 {

inline constexpr std::uint32_t kBeginImageToken = 255999;
inline constexpr std::uint32_t kImagePlaceholderToken = 258880;
inline constexpr std::uint32_t kEndImageToken = 258882;

// Exact no-tools, no-thinking, single-user-turn rendering of the checkpoint's
// chat_template.jinja with add_generation_prompt=true.
std::string format_gemma4_user_turn(std::string_view content);
std::string format_gemma4_system_user_turn(std::string_view system,
                                           std::string_view content);

// Tokenized Gemma 4 input after each user-facing <|image|> marker has been
// expanded to <|image>, N soft-token slots, <image|>. Image slots are marked 1
// so the language embedding frontend can replace them with projected vision
// features and construct the bidirectional image-attention block mask.
struct MultimodalPrompt {
  std::vector<std::uint32_t> input_ids;
  std::vector<std::uint8_t> multimodal_types;
  std::vector<std::size_t> image_offsets;
  std::vector<int> image_token_counts;
};

MultimodalPrompt prepare_image_prompt(const Tokenizer& tokenizer,
                                      std::string_view text,
                                      std::span<const int> image_token_counts);

}  // namespace g4
