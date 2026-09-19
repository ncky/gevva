#include "gevva/gpu.hpp"
#include "cuda_errors.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>
#include <nvtx3/nvToolsExt.h>

namespace gevva {
using detail::check;

DeviceArena::DeviceArena(const std::vector<std::span<const std::byte>>& shard_data) {
  constexpr std::size_t kAlignment = 256;
  auto align_up = [](std::size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
  };
  std::size_t arena_bytes = 0;
  std::size_t payload_bytes = 0;
  for (const auto shard : shard_data) {
    arena_bytes = align_up(arena_bytes);
    shard_offsets_.push_back(arena_bytes);
    arena_bytes += shard.size();
    payload_bytes += shard.size();
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  constexpr std::size_t kHeadroom = 4ULL << 30;
  if (arena_bytes + kHeadroom > free_bytes) {
    throw std::runtime_error("insufficient target GPU memory for weight arena plus 4 GiB headroom");
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  cudaEvent_t buffer_done[2]{};
  void* staging[2]{};
  bool buffer_issued[2]{};
  constexpr std::size_t kStagingBytes = 64ULL << 20;
  check(cudaMalloc(&arena_, arena_bytes), "cudaMalloc(weight arena)");
  arena_bytes_ = arena_bytes;
  try {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check(cudaEventCreate(&begin), "cudaEventCreate(begin)");
    check(cudaEventCreate(&end), "cudaEventCreate(end)");
    for (int i = 0; i < 2; ++i) {
      check(cudaEventCreateWithFlags(&buffer_done[i], cudaEventDisableTiming), "cudaEventCreate(staging)");
      check(cudaHostAlloc(&staging[i], kStagingBytes, cudaHostAllocPortable), "cudaHostAlloc(staging)");
    }
    nvtxRangePushA("weights/upload_shards");
    const auto wall_begin = std::chrono::steady_clock::now();
    check(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
    std::size_t chunk_index = 0;
    for (std::size_t i = 0; i < shard_data.size(); ++i) {
      const auto shard = shard_data[i];
      for (std::size_t position = 0; position < shard.size(); position += kStagingBytes, ++chunk_index) {
        const int buffer = static_cast<int>(chunk_index & 1);
        if (buffer_issued[buffer]) {
          check(cudaEventSynchronize(buffer_done[buffer]), "cudaEventSynchronize(staging)");
        }
        const auto bytes = std::min(kStagingBytes, shard.size() - position);
        std::memcpy(staging[buffer], shard.data() + position, bytes);
        check(cudaMemcpyAsync(arena_ + shard_offsets_[i] + position, staging[buffer], bytes,
                              cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(weight chunk)");
        check(cudaEventRecord(buffer_done[buffer], stream), "cudaEventRecord(staging)");
        buffer_issued[buffer] = true;
      }
    }
    check(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    check(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    const auto wall_end = std::chrono::steady_clock::now();
    nvtxRangePop();
    float device_ms = 0.0F;
    check(cudaEventElapsedTime(&device_ms, begin, end), "cudaEventElapsedTime");
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    for (int i = 0; i < 2; ++i) {
      cudaEventDestroy(buffer_done[i]);
      cudaFreeHost(staging[i]);
    }
    cudaStreamDestroy(stream);
    upload_ = {payload_bytes, device_ms, wall_ms};
  } catch (...) {
    if (end) cudaEventDestroy(end);
    if (begin) cudaEventDestroy(begin);
    for (int i = 0; i < 2; ++i) {
      if (buffer_done[i]) cudaEventDestroy(buffer_done[i]);
      if (staging[i]) cudaFreeHost(staging[i]);
    }
    if (stream) cudaStreamDestroy(stream);
    release();
    throw;
  }
}

DeviceArena::~DeviceArena() { release(); }

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : arena_(other.arena_), arena_bytes_(other.arena_bytes_),
      shard_offsets_(std::move(other.shard_offsets_)), upload_(other.upload_) {
  other.arena_ = nullptr;
  other.arena_bytes_ = 0;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
  if (this == &other) return *this;
  release();
  arena_ = other.arena_;
  arena_bytes_ = other.arena_bytes_;
  shard_offsets_ = std::move(other.shard_offsets_);
  upload_ = other.upload_;
  other.arena_ = nullptr;
  other.arena_bytes_ = 0;
  return *this;
}

void DeviceArena::release() noexcept {
  if (arena_) cudaFree(arena_);
  arena_ = nullptr;
  arena_bytes_ = 0;
}

const std::byte* DeviceArena::shard_base(std::size_t index) const {
  if (index >= shard_offsets_.size()) throw std::out_of_range("invalid device shard index");
  return arena_ + shard_offsets_[index];
}

UploadResult benchmark_weight_upload(
    const std::vector<std::span<const std::byte>>& shard_data) {
  DeviceArena arena(shard_data);
  return arena.upload_result();
}

}  // namespace gevva
