#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gevva {

class Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path& model_dir);
  std::vector<std::uint32_t> encode(std::string_view text) const;
  std::string decode_token(std::uint32_t id,
                           bool skip_special_tokens = false) const;
  std::string decode(const std::vector<std::uint32_t>& ids,
                     bool skip_special_tokens = false) const;

 private:
  std::vector<std::uint32_t> encode_ordinary(std::string_view text) const;
  std::unordered_map<std::string, std::uint32_t> vocab_;
  std::vector<std::string> inverse_vocab_;
  std::unordered_map<std::string, std::uint32_t> merge_rank_;
  std::vector<std::pair<std::string, std::uint32_t>> specials_;
};

}  // namespace gevva
