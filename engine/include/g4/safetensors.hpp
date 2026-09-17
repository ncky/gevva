#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace g4 {

struct TensorInfo {
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{};
  std::uint64_t end{};
};

struct ShardInfo {
  std::filesystem::path path;
  std::uint64_t file_size{};
  std::uint64_t header_size{};
  std::map<std::string, TensorInfo> tensors;
};

ShardInfo inspect_safetensors(const std::filesystem::path& path);
std::vector<ShardInfo> inspect_model_shards(const std::filesystem::path& model_dir);

struct TensorView {
  const TensorInfo* info{};
  std::span<const std::byte> bytes;
  std::size_t shard_index{};
};

class MappedShard {
 public:
  explicit MappedShard(ShardInfo info);
  ~MappedShard();
  MappedShard(const MappedShard&) = delete;
  MappedShard& operator=(const MappedShard&) = delete;
  MappedShard(MappedShard&& other) noexcept;
  MappedShard& operator=(MappedShard&& other) noexcept;

  const ShardInfo& info() const { return info_; }
  std::span<const std::byte> data() const;
  TensorView tensor(const std::string& name) const;

 private:
  void close() noexcept;
  ShardInfo info_;
  int fd_{-1};
  const std::byte* mapping_{};
};

class ModelWeights {
 public:
  explicit ModelWeights(const std::filesystem::path& model_dir);
  TensorView tensor(const std::string& name) const;
  std::size_t tensor_count() const { return locations_.size(); }
  std::vector<std::span<const std::byte>> shard_data() const;

 private:
  struct Location {
    std::size_t shard{};
  };
  std::vector<MappedShard> shards_;
  std::unordered_map<std::string, Location> locations_;
};

}  // namespace g4
