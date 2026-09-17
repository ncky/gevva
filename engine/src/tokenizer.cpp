#include "g4/tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

std::string pair_key(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back('\0');
  key.append(right);
  return key;
}

std::vector<std::string> utf8_symbols(std::string_view text) {
  std::vector<std::string> result;
  for (std::size_t i = 0; i < text.size();) {
    const auto c = static_cast<unsigned char>(text[i]);
    std::size_t width = 1;
    if ((c & 0xE0) == 0xC0) width = 2;
    else if ((c & 0xF0) == 0xE0) width = 3;
    else if ((c & 0xF8) == 0xF0) width = 4;
    if (i + width > text.size()) width = 1;
    result.emplace_back(text.substr(i, width));
    i += width;
  }
  return result;
}

void replace_all(std::string& text, std::string_view from, std::string_view to) {
  for (std::size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos;) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

}  // namespace

namespace g4 {

Tokenizer::Tokenizer(const std::filesystem::path& model_dir) {
  const auto path = model_dir / "tokenizer.json";
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  nlohmann::json root;
  input >> root;
  const auto& model = root.at("model");
  if (model.at("type") != "BPE" || !model.value("byte_fallback", false)) {
    throw std::runtime_error("unsupported tokenizer: expected byte-fallback BPE");
  }

  std::uint32_t max_id = 0;
  for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
    const auto id = it.value().get<std::uint32_t>();
    vocab_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }
  inverse_vocab_.resize(static_cast<std::size_t>(max_id) + 1);
  for (const auto& [token, id] : vocab_) inverse_vocab_[id] = token;

  const auto& merges = model.at("merges");
  merge_rank_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto& merge = merges[rank];
    merge_rank_.emplace(pair_key(merge[0].get_ref<const std::string&>(),
                                 merge[1].get_ref<const std::string&>()), rank);
  }
  for (const auto& added : root.at("added_tokens")) {
    if (!added.value("special", false)) continue;
    specials_.emplace_back(added.at("content").get<std::string>(),
                           added.at("id").get<std::uint32_t>());
  }
  std::sort(specials_.begin(), specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

std::vector<std::uint32_t> Tokenizer::encode_ordinary(std::string_view input) const {
  std::string normalized;
  normalized.reserve(input.size() + input.size() / 4);
  for (const char c : input) {
    if (c == ' ') normalized += "▁";
    else normalized.push_back(c);
  }
  auto symbols = utf8_symbols(normalized);
  if (symbols.empty()) return {};

  struct Node {
    std::string text;
    int prev{-1};
    int next{-1};
    std::uint32_t generation{};
    bool alive{true};
  };
  struct Candidate {
    std::uint32_t rank{};
    int left{};
    int right{};
    std::uint32_t left_generation{};
    std::uint32_t right_generation{};
    bool operator>(const Candidate& other) const {
      if (rank != other.rank) return rank > other.rank;
      return left > other.left;
    }
  };

  std::vector<Node> nodes;
  nodes.reserve(symbols.size());
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    nodes.push_back({std::move(symbols[i]), static_cast<int>(i) - 1,
                     i + 1 < symbols.size() ? static_cast<int>(i) + 1 : -1});
  }
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
  auto offer = [&](int left) {
    if (left < 0 || !nodes[left].alive || nodes[left].next < 0) return;
    const int right = nodes[left].next;
    const auto it = merge_rank_.find(pair_key(nodes[left].text, nodes[right].text));
    if (it != merge_rank_.end()) {
      queue.push({it->second, left, right, nodes[left].generation, nodes[right].generation});
    }
  };
  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) offer(static_cast<int>(i));

  while (!queue.empty()) {
    const auto c = queue.top();
    queue.pop();
    auto& left = nodes[c.left];
    auto& right = nodes[c.right];
    if (!left.alive || !right.alive || left.next != c.right ||
        left.generation != c.left_generation || right.generation != c.right_generation) continue;
    left.text += right.text;
    ++left.generation;
    right.alive = false;
    ++right.generation;
    left.next = right.next;
    if (right.next >= 0) nodes[right.next].prev = c.left;
    offer(left.prev);
    offer(c.left);
  }

  std::vector<std::uint32_t> ids;
  for (int i = 0; i >= 0; i = nodes[i].next) {
    const auto found = vocab_.find(nodes[i].text);
    if (found != vocab_.end()) {
      ids.push_back(found->second);
      continue;
    }
    // Hugging Face byte fallback spells raw bytes as uppercase <0xXX> tokens.
    for (const unsigned char byte : nodes[i].text) {
      constexpr char hex[] = "0123456789ABCDEF";
      std::string fallback = "<0x00>";
      fallback[3] = hex[byte >> 4];
      fallback[4] = hex[byte & 15];
      const auto byte_token = vocab_.find(fallback);
      ids.push_back(byte_token == vocab_.end() ? 3 : byte_token->second);
    }
  }
  return ids;
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<std::uint32_t> result;
  std::size_t ordinary_begin = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    const std::pair<std::string, std::uint32_t>* matched = nullptr;
    for (const auto& special : specials_) {
      if (text.substr(pos).starts_with(special.first)) {
        matched = &special;
        break;
      }
    }
    if (!matched) {
      ++pos;
      continue;
    }
    auto ordinary = encode_ordinary(text.substr(ordinary_begin, pos - ordinary_begin));
    result.insert(result.end(), ordinary.begin(), ordinary.end());
    result.push_back(matched->second);
    pos += matched->first.size();
    ordinary_begin = pos;
  }
  auto tail = encode_ordinary(text.substr(ordinary_begin));
  result.insert(result.end(), tail.begin(), tail.end());
  return result;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& ids,
                              bool skip_special_tokens) const {
  std::string result;
  for (const auto id : ids) result += decode_token(id, skip_special_tokens);
  return result;
}

std::string Tokenizer::decode_token(std::uint32_t id,
                                    bool skip_special_tokens) const {
  if (id >= inverse_vocab_.size() || inverse_vocab_[id].empty()) return {};
  const auto& token = inverse_vocab_[id];
  const bool special = std::any_of(specials_.begin(), specials_.end(),
      [&](const auto& item) { return item.second == id; });
  if (skip_special_tokens && special) return {};
  std::string result;
  if (token.size() == 6 && token.starts_with("<0x") && token.ends_with('>')) {
    unsigned byte = 0;
    const auto parsed = std::from_chars(token.data() + 3, token.data() + 5,
                                       byte, 16);
    if (parsed.ec == std::errc{}) result.push_back(static_cast<char>(byte));
  } else {
    result = token;
  }
  replace_all(result, "▁", " ");
  return result;
}

}  // namespace g4
