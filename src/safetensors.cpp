#include "gevva/safetensors.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace gevva {
namespace {
std::uint64_t dtype_bytes(std::string_view dtype) {
  if (dtype == "BOOL" || dtype == "I8" || dtype == "U8" || dtype == "F8_E4M3" ||
      dtype == "F8_E5M2" || dtype == "F8_E8M0") return 1;
  if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") return 2;
  if (dtype == "I32" || dtype == "U32" || dtype == "F32") return 4;
  if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
  throw std::runtime_error("unsupported safetensors dtype: " + std::string(dtype));
}

std::uint64_t unsigned_integer(const nlohmann::json& value) {
  if (!value.is_number_unsigned())
    throw std::runtime_error("safetensors dimensions and offsets must be nonnegative integers");
  return value.get<std::uint64_t>();
}
}  // namespace

ShardInfo inspect_safetensors(const std::filesystem::path& path) {
  ShardInfo out;
  out.path = path;
  out.file_size = std::filesystem::file_size(path);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());

  std::array<unsigned char, 8> size_bytes{};
  input.read(reinterpret_cast<char*>(size_bytes.data()), size_bytes.size());
  if (!input) throw std::runtime_error("truncated safetensors prefix: " + path.string());
  std::uint64_t header_size = 0;
  for (unsigned i = 0; i < size_bytes.size(); ++i) {
    header_size |= std::uint64_t{size_bytes[i]} << (8 * i);
  }
  if (header_size > out.file_size - 8 || header_size > (256ULL << 20)) {
    throw std::runtime_error("invalid safetensors header size: " + path.string());
  }
  out.header_size = header_size;
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input) throw std::runtime_error("truncated safetensors header: " + path.string());

  const auto json = nlohmann::json::parse(header);
  const std::uint64_t data_size = out.file_size - 8 - header_size;
  if (!json.is_object()) throw std::runtime_error("safetensors header must be an object");
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (auto it = json.begin(); it != json.end(); ++it) {
    if (it.key() == "__metadata__") {
      if (!it.value().is_object()) throw std::runtime_error("invalid safetensors metadata");
      for (const auto& value : it.value())
        if (!value.is_string()) throw std::runtime_error("safetensors metadata values must be strings");
      continue;
    }
    TensorInfo tensor;
    tensor.dtype = it.value().at("dtype").get<std::string>();
    const auto& shape = it.value().at("shape");
    const auto& offsets = it.value().at("data_offsets");
    if (!shape.is_array() || !offsets.is_array() || offsets.size() != 2)
      throw std::runtime_error("invalid safetensors shape/offsets for " + it.key());
    std::uint64_t elements = 1;
    for (const auto& dimension : shape) {
      const auto size = unsigned_integer(dimension);
      if (size && elements > std::numeric_limits<std::uint64_t>::max() / size)
        throw std::runtime_error("safetensors shape overflow for " + it.key());
      elements *= size;
      tensor.shape.push_back(size);
    }
    const auto element_bytes = dtype_bytes(tensor.dtype);
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes)
      throw std::runtime_error("safetensors payload size overflow for " + it.key());
    tensor.begin = unsigned_integer(offsets[0]);
    tensor.end = unsigned_integer(offsets[1]);
    if (tensor.begin > tensor.end || tensor.end > data_size)
      throw std::runtime_error("out-of-range tensor " + it.key() + " in " + path.string());
    if (elements * element_bytes != tensor.end - tensor.begin)
      throw std::runtime_error("safetensors payload size mismatch for " + it.key());
    ranges.emplace_back(tensor.begin, tensor.end);
    out.tensors.emplace(it.key(), std::move(tensor));
  }
  std::sort(ranges.begin(), ranges.end());
  std::uint64_t end = 0;
  for (const auto& [begin, next] : ranges) {
    if (begin != end)
      throw std::runtime_error("safetensors payload has overlapping tensors or gaps: " + path.string());
    end = next;
  }
  if (end != data_size)
    throw std::runtime_error("safetensors payload has unclaimed bytes: " + path.string());
  return out;
}

std::vector<ShardInfo> inspect_model_shards(const std::filesystem::path& model_dir) {
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw std::runtime_error("no .safetensors files in " + model_dir.string());

  std::vector<ShardInfo> result;
  result.reserve(paths.size());
  for (const auto& path : paths) result.push_back(inspect_safetensors(path));
  return result;
}

MappedShard::MappedShard(ShardInfo info) : info_(std::move(info)) {
  fd_ = ::open(info_.path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) throw std::runtime_error("open failed for " + info_.path.string());
  void* ptr = ::mmap(nullptr, info_.file_size, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (ptr == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("mmap failed for " + info_.path.string());
  }
  mapping_ = static_cast<const std::byte*>(ptr);
  // Weight copies are shard-sequential during startup. This is only a kernel
  // readahead hint; pages remain demand-loaded and are never copied by mmap.
  ::madvise(const_cast<std::byte*>(mapping_), info_.file_size, MADV_SEQUENTIAL);
}

MappedShard::~MappedShard() { close(); }

MappedShard::MappedShard(MappedShard&& other) noexcept
    : info_(std::move(other.info_)), fd_(other.fd_), mapping_(other.mapping_) {
  other.fd_ = -1;
  other.mapping_ = nullptr;
}

MappedShard& MappedShard::operator=(MappedShard&& other) noexcept {
  if (this == &other) return *this;
  close();
  info_ = std::move(other.info_);
  fd_ = other.fd_;
  mapping_ = other.mapping_;
  other.fd_ = -1;
  other.mapping_ = nullptr;
  return *this;
}

void MappedShard::close() noexcept {
  if (mapping_) ::munmap(const_cast<std::byte*>(mapping_), info_.file_size);
  if (fd_ >= 0) ::close(fd_);
  mapping_ = nullptr;
  fd_ = -1;
}

TensorView MappedShard::tensor(const std::string& name) const {
  const auto it = info_.tensors.find(name);
  if (it == info_.tensors.end()) throw std::runtime_error("tensor not found: " + name);
  const auto& t = it->second;
  const auto offset = 8 + info_.header_size + t.begin;
  return {&t, {mapping_ + offset, static_cast<std::size_t>(t.end - t.begin)}, 0};
}

std::span<const std::byte> MappedShard::data() const {
  const auto offset = 8 + info_.header_size;
  return {mapping_ + offset, static_cast<std::size_t>(info_.file_size - offset)};
}

ModelWeights::ModelWeights(const std::filesystem::path& model_dir) {
  auto infos = inspect_model_shards(model_dir);
  shards_.reserve(infos.size());
  for (auto& info : infos) shards_.emplace_back(std::move(info));
  for (std::size_t shard = 0; shard < shards_.size(); ++shard) {
    for (const auto& [name, _] : shards_[shard].info().tensors) {
      const auto [__, inserted] = locations_.emplace(name, Location{shard});
      if (!inserted) throw std::runtime_error("duplicate tensor across shards: " + name);
    }
  }

  const auto index_path = model_dir / "model.safetensors.index.json";
  if (std::filesystem::exists(index_path)) {
    std::ifstream input(index_path);
    nlohmann::json index;
    input >> index;
    const auto& weight_map = index.at("weight_map");
    if (weight_map.size() != locations_.size()) {
      throw std::runtime_error("safetensors index/tensor count mismatch");
    }
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      const auto location = locations_.find(it.key());
      if (location == locations_.end()) {
        throw std::runtime_error("indexed tensor is missing: " + it.key());
      }
      const auto expected_shard = it.value().get<std::string>();
      if (shards_[location->second.shard].info().path.filename() != expected_shard) {
        throw std::runtime_error("tensor is in the wrong shard: " + it.key());
      }
    }
  }
}

TensorView ModelWeights::tensor(const std::string& name) const {
  const auto it = locations_.find(name);
  if (it == locations_.end()) throw std::runtime_error("tensor not found: " + name);
  auto view = shards_[it->second.shard].tensor(name);
  view.shard_index = it->second.shard;
  return view;
}

std::vector<std::span<const std::byte>> ModelWeights::shard_data() const {
  std::vector<std::span<const std::byte>> result;
  result.reserve(shards_.size());
  for (const auto& shard : shards_) result.push_back(shard.data());
  return result;
}

std::vector<std::string> ModelWeights::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(locations_.size());
  for (const auto& [name, _] : locations_) names.push_back(name);
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace gevva
