#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace g4 {

struct OracleArray {
  std::filesystem::path path;
  std::string dtype;
  std::vector<std::uint64_t> shape;
};

class RawOracle {
 public:
  explicit RawOracle(const std::filesystem::path& directory);
  const OracleArray& array(const std::string& name) const;
  std::vector<float> read_f32(const std::string& name) const;
  std::vector<std::int32_t> read_i32(const std::string& name) const;

 private:
  std::unordered_map<std::string, OracleArray> arrays_;
};

}  // namespace g4
