#include "g4/oracle.hpp"

#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace g4 {
namespace {

std::uint64_t elements(const OracleArray& array) {
  return std::accumulate(array.shape.begin(), array.shape.end(), std::uint64_t{1},
                         std::multiplies<>());
}

template <typename T>
std::vector<T> read_array(const OracleArray& array, const char* expected_dtype) {
  if (array.dtype != expected_dtype) {
    throw std::runtime_error("oracle dtype mismatch for " + array.path.string());
  }
  std::vector<T> result(elements(array));
  std::ifstream input(array.path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(result.data()),
                  static_cast<std::streamsize>(result.size() * sizeof(T)))) {
    throw std::runtime_error("could not read oracle array " + array.path.string());
  }
  if (input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("oracle array is larger than its manifest shape: " + array.path.string());
  }
  return result;
}

}  // namespace

RawOracle::RawOracle(const std::filesystem::path& directory) {
  std::ifstream input(directory / "manifest.json");
  if (!input) throw std::runtime_error("could not open oracle manifest");
  nlohmann::json manifest;
  input >> manifest;
  for (const auto& [name, value] : manifest.at("arrays").items()) {
    arrays_.emplace(name, OracleArray{directory / value.at("file").get<std::string>(),
                                     value.at("dtype").get<std::string>(),
                                     value.at("shape").get<std::vector<std::uint64_t>>()});
  }
}

const OracleArray& RawOracle::array(const std::string& name) const {
  const auto found = arrays_.find(name);
  if (found == arrays_.end()) throw std::runtime_error("oracle array not found: " + name);
  return found->second;
}

std::vector<float> RawOracle::read_f32(const std::string& name) const {
  return read_array<float>(array(name), "float32");
}

std::vector<std::int32_t> RawOracle::read_i32(const std::string& name) const {
  return read_array<std::int32_t>(array(name), "int32");
}

}  // namespace g4
