#include "gevva/safetensors.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

using nlohmann::json;

struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::array<char, 40> name{};
    std::string pattern = "/tmp/gevva-safetensors-XXXXXX";
    std::copy(pattern.begin(), pattern.end(), name.begin());
    if (!mkdtemp(name.data())) throw std::runtime_error("mkdtemp failed");
    path = name.data();
  }
  ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

json tensor(std::string dtype, json shape, json offsets) {
  return {{"dtype", dtype}, {"shape", shape}, {"data_offsets", offsets}};
}

int main() try {
  TempDirectory directory;
  const auto path = directory.path / "test.safetensors";
  auto write = [&](const json& header, std::size_t bytes) {
    std::string text = header.dump();
    while (text.size() % 8) text += ' ';
    std::ofstream file(path, std::ios::binary);
    for (int i = 0; i < 8; ++i) file.put(static_cast<char>((std::uint64_t(text.size()) >> (8*i)) & 255));
    file << text << std::string(bytes, '\0');
  };
  int rejected = 0;
  auto invalid = [&](const char* label, const json& header, std::size_t bytes) {
    write(header, bytes);
    try { (void)gevva::inspect_safetensors(path); }
    catch (const std::exception&) { ++rejected; return; }
    throw std::runtime_error(std::string("accepted invalid tensor: ") + label);
  };
  invalid("undersized", {{"x", tensor("BF16", {2816}, {0, 2})}}, 2);
  invalid("oversized", {{"x", tensor("BF16", {1}, {0, 4})}}, 4);
  invalid("shape overflow", {{"x", tensor("U8", {std::numeric_limits<std::uint64_t>::max(), 2}, {0, 0})}}, 0);
  invalid("byte overflow", {{"x", tensor("BF16", {std::numeric_limits<std::uint64_t>::max()}, {0, 0})}}, 0);
  invalid("unknown dtype", {{"x", tensor("CUSTOM", {1}, {0, 1})}}, 1);
  invalid("unsupported subbyte dtype", {{"x", tensor("F4", {2}, {0, 1})}}, 1);
  invalid("negative shape", {{"x", tensor("U8", {-1}, {0, 0})}}, 0);
  invalid("fractional shape", {{"x", tensor("U8", {1.0}, {0, 1})}}, 1);
  invalid("negative offset", {{"x", tensor("U8", {1}, {-1, 0})}}, 1);
  invalid("fractional offset", {{"x", tensor("U8", {1}, {0.0, 1})}}, 1);
  invalid("overlap", {{"x", tensor("U8", {2}, {0, 2})}, {"y", tensor("U8", {2}, {1, 3})}}, 3);
  invalid("gap", {{"x", tensor("U8", {1}, {0, 1})}, {"y", tensor("U8", {1}, {2, 3})}}, 3);
  invalid("unclaimed bytes", {{"x", tensor("U8", {1}, {0, 1})}}, 2);
  invalid("reversed offsets", {{"x", tensor("U8", {1}, {1, 0})}}, 1);
  invalid("out of file", {{"x", tensor("U8", {2}, {0, 2})}}, 1);
  invalid("extra offset", {{"x", tensor("U8", {1}, {0, 1, 1})}}, 1);
  invalid("shape not array", {{"x", tensor("U8", 1, {0, 1})}}, 1);
  invalid("header not object", json::array(), 0);
  invalid("metadata not strings", {{"__metadata__", {{"value", 5}}}}, 0);

  write({{"scalar", tensor("F32", json::array(), {0, 4})},
         {"matrix", tensor("BF16", {2, 2}, {4, 12})},
         {"empty", tensor("U8", {0, 99}, {12, 12})},
         {"__metadata__", {{"format", "test"}}}}, 12);
  gevva::MappedShard shard(gevva::inspect_safetensors(path));
  if (shard.tensor("scalar").bytes.size() != 4 || shard.tensor("matrix").bytes.size() != 8 ||
      !shard.tensor("empty").bytes.empty()) throw std::runtime_error("valid tensor mapping failed");
  write(json::object(), 0);
  if (!gevva::inspect_safetensors(path).tensors.empty()) throw std::runtime_error("empty shard failed");
  std::cout << "Rejected " << rejected << " malformed files; valid scalar/matrix/empty mappings passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
