#include "gevva/runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_driver(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) return;
  const char* detail = nullptr;
  cuGetErrorString(status, &detail);
  throw std::runtime_error(std::string(operation) + ": " +
                           (detail ? detail : "unknown CUDA driver error"));
}

}  // namespace

namespace gevva {

struct KvCache::VirtualRegion {
  struct PhysicalPage {
    CUmemGenericAllocationHandle handle{};
    ~PhysicalPage() { if (handle) cuMemRelease(handle); }
  };
  VirtualRegion(std::size_t requested_bytes, std::size_t initial_bytes,
                int device) {
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    check_driver(cuMemGetAllocationGranularity(
                     &granularity, &properties,
                     CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                 "cuMemGetAllocationGranularity(KV)");
    auto align = [&](std::size_t bytes) {
      return (bytes + granularity - 1) / granularity * granularity;
    };
    reserved_bytes = align(requested_bytes);
    check_driver(cuMemAddressReserve(&address, reserved_bytes, granularity, 0, 0),
                 "cuMemAddressReserve(KV)");
    try {
      map(align(initial_bytes), properties);
    } catch (...) {
      for (std::size_t offset = 0; offset < mapped_bytes; offset += granularity)
        cuMemUnmap(address + offset, granularity);
      mapped_bytes = 0;
      pages.clear();
      cuMemAddressFree(address, reserved_bytes);
      address = 0;
      throw;
    }
  }

  ~VirtualRegion() {
    for (std::size_t offset = 0; offset < mapped_bytes; offset += granularity)
      cuMemUnmap(address + offset, granularity);
    if (address) cuMemAddressFree(address, reserved_bytes);
  }

  void ensure(std::size_t requested_bytes, int device, cudaStream_t stream) {
    const std::size_t target = std::min(
        reserved_bytes,
        (requested_bytes + granularity - 1) / granularity * granularity);
    if (target <= mapped_bytes) return;
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    map(target, properties, stream);
  }

  void map(std::size_t target, const CUmemAllocationProp& properties,
           cudaStream_t stream = nullptr) {
    if (target <= mapped_bytes) return;
    while (mapped_bytes < target) {
      auto page = std::make_shared<PhysicalPage>();
      check_driver(cuMemCreate(&page->handle, granularity, &properties, 0),
                   "cuMemCreate(KV page)");
      check_driver(cuMemMap(address + mapped_bytes, granularity, 0, page->handle, 0),
                   "cuMemMap(KV page)");
      try {
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check_driver(cuMemSetAccess(address + mapped_bytes, granularity, &access, 1),
                     "cuMemSetAccess(KV page)");
        check_driver(cuMemsetD8Async(address + mapped_bytes, 0, granularity, stream),
                     "zero newly committed KV page");
        pages.push_back(std::move(page));
      } catch (...) {
        cuMemUnmap(address + mapped_bytes, granularity);
        throw;
      }
      mapped_bytes += granularity;
    }
    if (!stream) check_driver(cuStreamSynchronize(nullptr), "initialize KV pages");
  }

  // All streams must be idle before remapping. Keeping the virtual addresses
  // unchanged preserves the existing attention kernels and device descriptors.
  std::size_t share_prefix(VirtualRegion& source, std::size_t bytes, int device) {
    if (granularity != source.granularity || bytes > mapped_bytes || bytes > source.mapped_bytes)
      throw std::runtime_error("incompatible shared KV regions");
    const auto whole_pages = bytes / granularity;
    CUmemAccessDesc read_only{};
    read_only.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    read_only.location.id = device;
    read_only.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    for (std::size_t i = 0; i < whole_pages; ++i) {
      check_driver(cuMemUnmap(address + i * granularity, granularity), "unmap private KV page");
      const auto old_page = pages[i];
      const auto status = cuMemMap(address + i * granularity, granularity, 0, source.pages[i]->handle, 0);
      if (status != CUDA_SUCCESS) {
        cuMemMap(address + i * granularity, granularity, 0, old_page->handle, 0);
        throw std::runtime_error("map shared KV page failed");
      }
      pages[i] = source.pages[i];
      check_driver(cuMemSetAccess(address + i * granularity, granularity, &read_only, 1),
                   "protect branch prefix KV page");
      check_driver(cuMemSetAccess(source.address + i * granularity, granularity, &read_only, 1),
                   "protect source prefix KV page");
    }
    const std::size_t shared = whole_pages * granularity;
    if (bytes > shared)
      check_driver(cuMemcpyDtoD(address + shared, source.address + shared, bytes - shared),
                   "copy private boundary KV page");
    return shared;
  }

  CUdeviceptr address{};
  std::size_t reserved_bytes{};
  std::size_t mapped_bytes{};
  std::size_t granularity{};
  std::vector<std::shared_ptr<PhysicalPage>> pages;
};

MtpAcceptance resolve_greedy_mtp(std::span<const int> draft_tokens,
                                 std::span<const int> target_tokens) {
  if (draft_tokens.empty() || draft_tokens.size() > 4 ||
      target_tokens.size() != draft_tokens.size() + 1)
    throw std::runtime_error("MTP acceptance expects 1..4 drafts and one extra target token");
  MtpAcceptance result;
  while (result.matched_drafts < static_cast<int>(draft_tokens.size()) &&
         draft_tokens[result.matched_drafts] == target_tokens[result.matched_drafts]) {
    result.output_tokens[result.output_count++] =
        draft_tokens[result.matched_drafts++];
  }
  if (result.matched_drafts == static_cast<int>(draft_tokens.size())) {
    result.output_tokens[result.output_count++] = target_tokens[draft_tokens.size()];
  } else {
    result.output_tokens[result.output_count++] = target_tokens[result.matched_drafts];
  }
  return result;
}

DeviceModel::DeviceModel(const std::filesystem::path& model_directory, bool omit_original_experts)
    : weights_(model_directory), selective_upload_(omit_original_experts), arena_([&] {
        if (!omit_original_experts) return weights_.shard_data();
        std::vector<std::span<const std::byte>> spans;
        const auto shards = weights_.shard_data();
        for (const auto& name : weights_.tensor_names()) {
          // RuntimeWeights uses the separately packed expert checkpoint. Keep
          // original expert metadata/host mappings, but do not upload a second
          // unused 12 GiB copy of those weights to the GPU.
          if (name.find(".experts.") != std::string::npos) continue;
          tensor_spans_.emplace(name, spans.size());
          const auto view = weights_.tensor(name);
          const auto leading = view.info->begin % 256;
          // Preserve each tensor's original alignment so cuBLAS does not
          // silently select a different numerical kernel after repacking.
          spans.push_back(shards[view.shard_index].subspan(
              view.info->begin - leading, view.bytes.size() + leading));
        }
        return spans;
      }()) {}

DeviceTensorView DeviceModel::tensor(const std::string& name) const {
  const auto host = weights_.tensor(name);
  if (selective_upload_)
    return {host.info, arena_.shard_base(tensor_spans_.at(name)) + host.info->begin % 256, host.bytes.size()};
  return {host.info, arena_.shard_base(host.shard_index) + host.info->begin,
          host.bytes.size()};
}

RuntimeWeights::RuntimeWeights(const std::filesystem::path& model_directory,
                               const std::filesystem::path& expert_directory, bool omit_original_experts)
    : model_(model_directory, omit_original_experts), experts_(expert_directory) {
  embedding_ = model_.tensor("model.language_model.embed_tokens.weight");
  final_norm_ = model_.tensor("model.language_model.norm.weight").data;
  for (int layer = 0; layer < 30; ++layer) {
    const bool global = layer % 6 == 5;
    const auto prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    auto tensor = [&](const char* suffix) -> const void* {
      return model_.tensor(prefix + suffix).data;
    };
    layers_[layer] = {
        tensor("input_layernorm.weight"), tensor("self_attn.q_proj.weight"),
        tensor("self_attn.k_proj.weight"),
        global ? nullptr : tensor("self_attn.v_proj.weight"),
        tensor("self_attn.o_proj.weight"), tensor("self_attn.q_norm.weight"),
        tensor("self_attn.k_norm.weight"),
        tensor("post_attention_layernorm.weight"),
        tensor("pre_feedforward_layernorm.weight"),
        tensor("pre_feedforward_layernorm_2.weight"), tensor("router.scale"),
        tensor("router.proj.weight"), tensor("router.per_expert_scale"),
        tensor("mlp.gate_proj.weight"), tensor("mlp.up_proj.weight"),
        tensor("mlp.down_proj.weight"),
        tensor("post_feedforward_layernorm_1.weight"),
        tensor("post_feedforward_layernorm_2.weight"),
        tensor("post_feedforward_layernorm.weight"), tensor("layer_scalar")};
  }
  constexpr std::uint64_t kExperts = 128;
  constexpr std::uint64_t kHidden = 2816;
  constexpr std::uint64_t kIntermediate = 704;
  auto require = [&](const std::string& name, const std::string& dtype,
                     std::initializer_list<std::uint64_t> shape) {
    const auto tensor = experts_.tensor(name);
    if (tensor.info->dtype != dtype ||
        tensor.info->shape != std::vector<std::uint64_t>(shape)) {
      throw std::runtime_error("invalid packed expert tensor: " + name);
    }
  };
  for (int layer = 0; layer < 30; ++layer) {
    const auto prefix = "layers." + std::to_string(layer) + ".";
    require(prefix + "w13.weight", "U8",
            {kExperts, 2 * kIntermediate, kHidden / 2});
    require(prefix + "w13.weight_scale", "U8",
            {kExperts, 2 * kIntermediate, kHidden / 16});
    require(prefix + "w2.weight", "U8",
            {kExperts, kHidden, kIntermediate / 2});
    require(prefix + "w2.weight_scale", "U8",
            {kExperts, kHidden, kIntermediate / 16});
    require(prefix + "w13.input_scale", "F32", {kExperts});
    require(prefix + "w2.input_scale", "F32", {kExperts});
    require(prefix + "w13.weight_scale_2", "F32", {kExperts, 2});
    require(prefix + "w2.weight_scale_2", "F32", {kExperts});
    require(prefix + "g1.alpha", "F32", {kExperts});
    require(prefix + "g1.alpha_up", "F32", {kExperts});
    require(prefix + "g2.alpha", "F32", {kExperts});
    require(prefix + "g1.input_scale_quant", "F32", {kExperts});
    require(prefix + "g2.input_scale_quant", "F32", {kExperts});
  }
  if (experts_.tensor_count() != 30 * 13)
    throw std::runtime_error("packed expert sidecar has unexpected tensors");
  device_bytes_ = model_.upload_result().bytes + experts_.upload_result().bytes;
}

AssistantWeights::AssistantWeights(const std::filesystem::path& model_directory)
    : model_(model_directory) {
  auto require = [&](const std::string& name, const std::string& dtype,
                     std::initializer_list<std::uint64_t> shape) {
    const auto tensor = model_.tensor(name);
    if (tensor.info->dtype != dtype ||
        tensor.info->shape != std::vector<std::uint64_t>(shape)) {
      throw std::runtime_error("invalid Gemma 4 assistant tensor: " + name);
    }
  };

  require("model.embed_tokens.weight", "BF16", {262144, 1024});
  require("pre_projection.weight", "BF16", {1024, 5632});
  require("post_projection.weight", "BF16", {2816, 1024});
  require("model.norm.weight", "BF16", {1024});
  for (int layer = 0; layer < 4; ++layer) {
    const bool global = layer == 3;
    const std::uint64_t attention_width = global ? 8192 : 4096;
    const std::uint64_t head_dimension = global ? 512 : 256;
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    require(prefix + "input_layernorm.weight", "BF16", {1024});
    require(prefix + "post_attention_layernorm.weight", "BF16", {1024});
    require(prefix + "pre_feedforward_layernorm.weight", "BF16", {1024});
    require(prefix + "post_feedforward_layernorm.weight", "BF16", {1024});
    require(prefix + "layer_scalar", "BF16", {1});
    require(prefix + "self_attn.q_proj.weight", "BF16",
            {attention_width, 1024});
    require(prefix + "self_attn.q_norm.weight", "BF16", {head_dimension});
    require(prefix + "self_attn.o_proj.weight", "BF16",
            {1024, attention_width});
    require(prefix + "mlp.gate_proj.weight", "BF16", {8192, 1024});
    require(prefix + "mlp.up_proj.weight", "BF16", {8192, 1024});
    require(prefix + "mlp.down_proj.weight", "BF16", {1024, 8192});
  }
  if (model_.tensor_count() != 48)
    throw std::runtime_error("Gemma 4 assistant has unexpected tensors");

  auto pointer = [&](const std::string& name) -> const void* {
    return model_.tensor(name).data;
  };
  embedding_ = pointer("model.embed_tokens.weight");
  pre_projection_ = pointer("pre_projection.weight");
  post_projection_ = pointer("post_projection.weight");
  final_norm_ = pointer("model.norm.weight");
  for (int layer = 0; layer < 4; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    layers_[layer] = {
        pointer(prefix + "input_layernorm.weight"),
        pointer(prefix + "self_attn.q_proj.weight"),
        pointer(prefix + "self_attn.q_norm.weight"),
        pointer(prefix + "self_attn.o_proj.weight"),
        pointer(prefix + "post_attention_layernorm.weight"),
        pointer(prefix + "pre_feedforward_layernorm.weight"),
        pointer(prefix + "mlp.gate_proj.weight"),
        pointer(prefix + "mlp.up_proj.weight"),
        pointer(prefix + "mlp.down_proj.weight"),
        pointer(prefix + "post_feedforward_layernorm.weight"),
        pointer(prefix + "layer_scalar")};
  }
  if (std::getenv("GEVVA_DISABLE_ASSISTANT_FUSED_GATE_UP") == nullptr) {
    constexpr std::size_t kProjectionBytes = 8192ULL * 1024 * 2;
    check_cuda(cudaMalloc(&fused_gate_up_storage_, 8 * kProjectionBytes),
               "allocate fused assistant gate/up weights");
    try {
      for (int layer = 0; layer < 4; ++layer) {
        auto* destination =
            static_cast<std::byte*>(fused_gate_up_storage_) +
            static_cast<std::size_t>(layer) * 2 * kProjectionBytes;
        check_cuda(cudaMemcpy(destination, layers_[layer].gate,
                              kProjectionBytes, cudaMemcpyDeviceToDevice),
                   "pack assistant gate weights");
        check_cuda(cudaMemcpy(destination + kProjectionBytes,
                              layers_[layer].up, kProjectionBytes,
                              cudaMemcpyDeviceToDevice),
                   "pack assistant up weights");
        fused_gate_up_[layer] = destination;
      }
    } catch (...) {
      cudaFree(fused_gate_up_storage_);
      fused_gate_up_storage_ = nullptr;
      throw;
    }
  }
}

AssistantWeights::~AssistantWeights() {
  if (fused_gate_up_storage_) cudaFree(fused_gate_up_storage_);
}

KvCache::KvCache(int maximum_context) : maximum_context_(maximum_context) {
  if (maximum_context < 1 || maximum_context > 262144)
    throw std::runtime_error("KV maximum context must be in [1, 262144]");
  constexpr std::size_t kAlignment = 256;
  auto align_up = [](std::size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
  };
  struct Layout { std::size_t key_offset; std::size_t value_offset; int capacity; int heads; int dim; };
  std::vector<Layout> layout;
  layout.reserve(30);
  std::size_t offset = 0;
  for (int layer = 0; layer < 30; ++layer) {
    const bool global = layer % 6 == 5;
    const int capacity = global ? maximum_context : 1024;
    const int heads = global ? 2 : 8;
    const int dim = global ? 512 : 256;
    // Sliding KV keeps three mirrored rings plus four tail rows. A direct
    // token-major batched GEMM can then read the 1023 historical rows and up
    // to five speculative candidates as one contiguous interval without
    // materializing a head-packed copy on every decode layer. The first ring
    // remains canonical; speculative tails only touch the mirrors.
    const int storage_tokens = global ? capacity : 3 * capacity + 4;
    const std::size_t plane_bytes =
        static_cast<std::size_t>(storage_tokens) * heads * dim *
        sizeof(std::uint16_t);
    std::size_t key_offset = 0, value_offset = 0;
    if (!global) {
      offset = align_up(offset);
      key_offset = offset;
      offset += plane_bytes;
      offset = align_up(offset);
      value_offset = offset;
      offset += plane_bytes;
    }
    layout.push_back({key_offset, value_offset, capacity, heads, dim});
  }
  const std::size_t sliding_bytes = align_up(offset);
  sliding_bytes_ = sliding_bytes;
  int device = 0;
  check_cuda(cudaFree(nullptr), "initialize CUDA context for KV");
  check_cuda(cudaGetDevice(&device), "cudaGetDevice(KV)");
  resident_context_ = std::min(maximum_context, 8192);
  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&arena_), sliding_bytes),
               "cudaMalloc(sliding KV cache)");
    layers_.reserve(layout.size());
    virtual_regions_.reserve(10);
    for (int layer = 0; layer < 30; ++layer) {
      const auto& item = layout[layer];
      if (layer % 6 != 5) {
        layers_.push_back({arena_ + item.key_offset, arena_ + item.value_offset,
                           item.capacity, item.heads, item.dim});
        continue;
      }
      const std::size_t bytes_per_token =
          static_cast<std::size_t>(item.heads) * item.dim * 2;
      auto keys = std::make_unique<VirtualRegion>(
          static_cast<std::size_t>(maximum_context) * bytes_per_token,
          static_cast<std::size_t>(resident_context_) * bytes_per_token,
          device);
      auto values = std::make_unique<VirtualRegion>(
          static_cast<std::size_t>(maximum_context) * bytes_per_token,
          static_cast<std::size_t>(resident_context_) * bytes_per_token,
          device);
      auto* key_pointer = reinterpret_cast<std::byte*>(keys->address);
      auto* value_pointer = reinterpret_cast<std::byte*>(values->address);
      bytes_ += keys->mapped_bytes + values->mapped_bytes;
      virtual_regions_.push_back(std::move(keys));
      virtual_regions_.push_back(std::move(values));
      layers_.push_back({key_pointer, value_pointer, item.capacity,
                         item.heads, item.dim});
    }
    bytes_ += sliding_bytes;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_layers_),
                          layers_.size() * sizeof(DeviceKvView)),
               "cudaMalloc(KV cache views)");
    check_cuda(cudaMemcpy(device_layers_, layers_.data(),
                          layers_.size() * sizeof(DeviceKvView),
                          cudaMemcpyHostToDevice),
               "copy KV cache views");
  } catch (...) {
    release();
    throw;
  }
}

KvCache::~KvCache() { release(); }

KvCache::KvCache(KvCache&& other) noexcept
    : arena_(other.arena_), bytes_(other.bytes_), sliding_bytes_(other.sliding_bytes_),
      maximum_context_(other.maximum_context_),
      resident_context_(other.resident_context_), layers_(std::move(other.layers_)),
      device_layers_(other.device_layers_),
      virtual_regions_(std::move(other.virtual_regions_)) {
  shared_prefix_tokens_ = other.shared_prefix_tokens_;
  shared_global_bytes_ = other.shared_global_bytes_;
  sealed_ = other.sealed_;
  other.arena_ = nullptr;
  other.device_layers_ = nullptr;
  other.bytes_ = 0;
  other.sliding_bytes_ = 0;
}

KvCache& KvCache::operator=(KvCache&& other) noexcept {
  if (this == &other) return *this;
  release();
  arena_ = other.arena_;
  bytes_ = other.bytes_;
  sliding_bytes_ = other.sliding_bytes_;
  maximum_context_ = other.maximum_context_;
  resident_context_ = other.resident_context_;
  shared_prefix_tokens_ = other.shared_prefix_tokens_;
  shared_global_bytes_ = other.shared_global_bytes_;
  sealed_ = other.sealed_;
  layers_ = std::move(other.layers_);
  device_layers_ = other.device_layers_;
  virtual_regions_ = std::move(other.virtual_regions_);
  other.arena_ = nullptr;
  other.device_layers_ = nullptr;
  other.bytes_ = 0;
  other.sliding_bytes_ = 0;
  return *this;
}

const DeviceKvView& KvCache::layer(int index) const {
  if (index < 0 || index >= static_cast<int>(layers_.size()))
    throw std::runtime_error("KV layer index out of range");
  return layers_[index];
}

void KvCache::ensure_context(int context_tokens, cudaStream_t stream) {
  if (sealed_) throw std::runtime_error("shared prefix source is immutable");
  if (context_tokens < 1 || context_tokens > maximum_context_)
    throw std::runtime_error("KV requested context exceeds reservation");
  if (context_tokens <= resident_context_) return;
  int device = 0;
  check_cuda(cudaFree(nullptr), "initialize CUDA context for KV growth");
  check_cuda(cudaGetDevice(&device), "cudaGetDevice(KV grow)");
  const std::size_t old_bytes = bytes_;
  for (auto& region : virtual_regions_) {
    const auto before = region->mapped_bytes;
    region->ensure(static_cast<std::size_t>(context_tokens) * 2 * 512 * 2,
                   device, stream);
    bytes_ += region->mapped_bytes - before;
  }
  (void)old_bytes;
  // Track the physical page coverage, not the last requested token. Otherwise
  // every decode step after 8192 re-enters driver bookkeeping even when no new
  // page is required (the allocator commits at a much coarser granularity).
  resident_context_ = maximum_context_;
  for (const auto& region : virtual_regions_)
    resident_context_ = std::min(resident_context_, static_cast<int>(
        region->mapped_bytes / (2 * 512 * 2)));
}

void KvCache::share_global_prefix_from(KvCache& source, int prefix_tokens) {
  if (&source == this || shared_prefix_tokens_ || sealed_ || prefix_tokens < 2 ||
      prefix_tokens > source.resident_context_ || prefix_tokens > maximum_context_)
    throw std::runtime_error("invalid immutable KV fork");
  ensure_context(prefix_tokens);
  check_cuda(cudaDeviceSynchronize(), "synchronize before KV fork");
  int device = 0;
  check_cuda(cudaGetDevice(&device), "get KV fork device");
  // Seal before any page becomes read-only, including on an exceptional path.
  source.sealed_ = true;
  for (std::size_t i = 0; i < virtual_regions_.size(); ++i)
    shared_global_bytes_ += virtual_regions_[i]->share_prefix(
        *source.virtual_regions_[i], static_cast<std::size_t>(prefix_tokens) * 2048, device);
  shared_prefix_tokens_ = prefix_tokens;
}

void KvCache::release() noexcept {
  if (device_layers_) cudaFree(device_layers_);
  virtual_regions_.clear();
  if (arena_) cudaFree(arena_);
  device_layers_ = nullptr;
  arena_ = nullptr;
  bytes_ = 0;
  sliding_bytes_ = 0;
  resident_context_ = 0;
  shared_prefix_tokens_ = 0;
  shared_global_bytes_ = 0;
  sealed_ = false;
  layers_.clear();
}

DeviceScratchPool::~DeviceScratchPool() {
  for (const auto& block : blocks_) cudaFree(block.pointer);
  if (host_staging_) cudaFreeHost(host_staging_);
  if (assistant_global_views_) cudaFree(assistant_global_views_);
  if (assistant_sliding_views_) cudaFree(assistant_sliding_views_);
  if (verifier_candidate_views_) cudaFree(verifier_candidate_views_);
  if (verifier_live_views_) cudaFree(verifier_live_views_);
}

void* DeviceScratchPool::host_staging(std::size_t bytes) {
  if (!bytes) throw std::runtime_error("host staging allocation cannot be empty");
  if (host_staging_bytes_ < bytes) {
    if (host_staging_) cudaFreeHost(host_staging_);
    host_staging_ = nullptr;
    host_staging_bytes_ = 0;
    check_cuda(cudaHostAlloc(&host_staging_, bytes, cudaHostAllocPortable),
               "cudaHostAlloc(generation staging)");
    host_staging_bytes_ = bytes;
  }
  return host_staging_;
}

void* DeviceScratchPool::allocate(std::size_t bytes) {
  if (!bytes) throw std::runtime_error("scratch allocation cannot be empty");
  if (cursor_ == blocks_.size()) {
    void* pointer{};
    check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc(generation scratch)");
    blocks_.push_back({pointer, bytes});
    bytes_ += bytes;
    ++generation_;
  } else if (blocks_[cursor_].bytes < bytes) {
    const std::size_t capacity = std::max(bytes, blocks_[cursor_].bytes * 2);
    check_cuda(cudaFree(blocks_[cursor_].pointer), "cudaFree(generation scratch resize)");
    bytes_ -= blocks_[cursor_].bytes;
    blocks_[cursor_].pointer = nullptr;
    blocks_[cursor_].bytes = 0;
    ++generation_; // Invalidate captured addresses even if the new allocation fails.
    check_cuda(cudaMalloc(&blocks_[cursor_].pointer, capacity),
               "cudaMalloc(generation scratch resize)");
    blocks_[cursor_].bytes = capacity;
    bytes_ += capacity;
  }
  return blocks_[cursor_++].pointer;
}

void DeviceScratchPool::prepare_verifier_views(
    std::span<KvCache* const> caches,
    std::span<CandidateKvCache* const> candidates,
    void* stream_pointer, const DeviceKvView*& live,
    const DeviceKvView*& candidate) {
  if (caches.empty() || caches.size() > verifier_caches_.size() ||
      candidates.size() != caches.size())
    throw std::runtime_error("invalid verifier cache-view cohort");
  if (!verifier_live_views_) {
    constexpr std::size_t kViews = 30 * 8;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&verifier_live_views_),
                          kViews * sizeof(DeviceKvView)),
               "cudaMalloc(verifier live views)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&verifier_candidate_views_),
                          kViews * sizeof(DeviceKvView)),
               "cudaMalloc(verifier candidate views)");
  }
  bool changed = verifier_sessions_ != static_cast<int>(caches.size());
  for (std::size_t session = 0; session < caches.size(); ++session)
    changed = changed || verifier_caches_[session] != caches[session] ||
        verifier_candidates_[session] != candidates[session];
  if (changed) {
    std::array<DeviceKvView, 30 * 8> host_live{};
    std::array<DeviceKvView, 30 * 8> host_candidates{};
    const int sessions = static_cast<int>(caches.size());
    for (int layer = 0; layer < 30; ++layer)
      for (int session = 0; session < sessions; ++session) {
        host_live[layer * sessions + session] = caches[session]->layer(layer);
        host_candidates[layer * sessions + session] =
            candidates[session]->layer(layer);
      }
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    const std::size_t bytes = 30 * caches.size() * sizeof(DeviceKvView);
    check_cuda(cudaMemcpyAsync(verifier_live_views_, host_live.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "copy verifier live views");
    check_cuda(cudaMemcpyAsync(verifier_candidate_views_,
                               host_candidates.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "copy verifier candidate views");
    verifier_sessions_ = sessions;
    for (int session = 0; session < sessions; ++session) {
      verifier_caches_[session] = caches[session];
      verifier_candidates_[session] = candidates[session];
    }
  }
  live = verifier_live_views_;
  candidate = verifier_candidate_views_;
}

void DeviceScratchPool::prepare_assistant_views(
    std::span<KvCache* const> caches, void* stream_pointer,
    const DeviceKvView*& sliding, const DeviceKvView*& global) {
  if (caches.empty() || caches.size() > assistant_caches_.size())
    throw std::runtime_error("invalid assistant cache-view cohort");
  if (!assistant_sliding_views_) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&assistant_sliding_views_),
                          8 * sizeof(DeviceKvView)),
               "cudaMalloc(assistant sliding views)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&assistant_global_views_),
                          8 * sizeof(DeviceKvView)),
               "cudaMalloc(assistant global views)");
  }
  bool changed = assistant_sessions_ != static_cast<int>(caches.size());
  for (std::size_t session = 0; session < caches.size(); ++session)
    changed = changed || assistant_caches_[session] != caches[session];
  if (changed) {
    std::array<DeviceKvView, 8> host_sliding{}, host_global{};
    for (std::size_t session = 0; session < caches.size(); ++session) {
      host_sliding[session] = caches[session]->layer(28);
      host_global[session] = caches[session]->layer(29);
    }
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    const std::size_t bytes = caches.size() * sizeof(DeviceKvView);
    check_cuda(cudaMemcpyAsync(assistant_sliding_views_, host_sliding.data(),
                               bytes, cudaMemcpyHostToDevice, stream),
               "copy assistant sliding views");
    check_cuda(cudaMemcpyAsync(assistant_global_views_, host_global.data(),
                               bytes, cudaMemcpyHostToDevice, stream),
               "copy assistant global views");
    assistant_sessions_ = static_cast<int>(caches.size());
    for (int session = 0; session < assistant_sessions_; ++session)
      assistant_caches_[session] = caches[session];
  }
  sliding = assistant_sliding_views_;
  global = assistant_global_views_;
}

CandidateKvCache::CandidateKvCache() {
  constexpr std::size_t kAlignment = 256;
  auto align_up = [](std::size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
  };
  struct Layout {
    std::size_t key_offset;
    std::size_t value_offset;
    int heads;
    int dim;
  };
  std::vector<Layout> layout;
  layout.reserve(30);
  std::size_t offset = 0;
  for (int layer = 0; layer < 30; ++layer) {
    const bool global = layer % 6 == 5;
    const int heads = global ? 2 : 8;
    const int dim = global ? 512 : 256;
    const std::size_t plane_bytes = static_cast<std::size_t>(kMaximumTokens) *
                                    heads * dim * sizeof(std::uint16_t);
    offset = align_up(offset);
    const auto key_offset = offset;
    offset += plane_bytes;
    offset = align_up(offset);
    const auto value_offset = offset;
    offset += plane_bytes;
    layout.push_back({key_offset, value_offset, heads, dim});
  }
  bytes_ = align_up(offset);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&arena_), bytes_),
             "cudaMalloc(candidate KV cache)");
  layers_.reserve(layout.size());
  for (const auto& item : layout)
    layers_.push_back({arena_ + item.key_offset, arena_ + item.value_offset,
                       kMaximumTokens, item.heads, item.dim});
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_layers_),
                        layers_.size() * sizeof(DeviceKvView)),
             "cudaMalloc(candidate KV views)");
  check_cuda(cudaMemcpy(device_layers_, layers_.data(),
                        layers_.size() * sizeof(DeviceKvView),
                        cudaMemcpyHostToDevice),
             "copy candidate KV views");
}

CandidateKvCache::~CandidateKvCache() { release(); }

CandidateKvCache::CandidateKvCache(CandidateKvCache&& other) noexcept
    : arena_(other.arena_), bytes_(other.bytes_),
      layers_(std::move(other.layers_)), device_layers_(other.device_layers_) {
  other.arena_ = nullptr;
  other.device_layers_ = nullptr;
  other.bytes_ = 0;
}

CandidateKvCache& CandidateKvCache::operator=(CandidateKvCache&& other) noexcept {
  if (this == &other) return *this;
  release();
  arena_ = other.arena_;
  bytes_ = other.bytes_;
  layers_ = std::move(other.layers_);
  device_layers_ = other.device_layers_;
  other.arena_ = nullptr;
  other.device_layers_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

const DeviceKvView& CandidateKvCache::layer(int index) const {
  if (index < 0 || index >= static_cast<int>(layers_.size()))
    throw std::runtime_error("candidate KV layer index out of range");
  return layers_[index];
}

void CandidateKvCache::release() noexcept {
  if (device_layers_) cudaFree(device_layers_);
  if (arena_) cudaFree(arena_);
  device_layers_ = nullptr;
  arena_ = nullptr;
  bytes_ = 0;
  layers_.clear();
}

AttentionWorkspace::AttentionWorkspace(int maximum_context, int maximum_batch)
    : maximum_context_(maximum_context), maximum_batch_(maximum_batch) {
  if (maximum_context < 1 || maximum_context > 262144)
    throw std::runtime_error("attention workspace context must be in [1, 262144]");
  if (maximum_batch < 1 || maximum_batch > 5)
    throw std::runtime_error("attention workspace batch must be in [1, 5]");
  constexpr std::size_t kQueryHeads = 16;
  constexpr std::size_t kPartitions = 16;
  constexpr std::size_t kMaximumHeadDim = 512;
  const std::size_t score_bytes =
      maximum_batch * kQueryHeads * static_cast<std::size_t>(maximum_context) *
      sizeof(float);
  const std::size_t partial_bytes =
      maximum_batch * kPartitions * kQueryHeads * kMaximumHeadDim * sizeof(float);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&scores_), score_bytes),
             "cudaMalloc(attention scores)");
  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&partials_), partial_bytes),
               "cudaMalloc(attention partials)");
  } catch (...) {
    cudaFree(scores_);
    scores_ = nullptr;
    throw;
  }
  bytes_ = score_bytes + partial_bytes;
}

AttentionWorkspace::~AttentionWorkspace() {
  if (partials_) cudaFree(partials_);
  if (scores_) cudaFree(scores_);
}

}  // namespace gevva
