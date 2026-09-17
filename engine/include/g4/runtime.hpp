#pragma once

#include "g4/gpu.hpp"
#include "g4/safetensors.hpp"

#include <cstddef>
#include <array>
#include <filesystem>
#include <string>
#include <span>
#include <utility>
#include <vector>

namespace g4 {

struct MtpAcceptance {
  std::array<int, 5> output_tokens{};
  int output_count{};
  int matched_drafts{};
};

MtpAcceptance resolve_greedy_mtp(std::span<const int> draft_tokens,
                                 std::span<const int> target_tokens);

struct DeviceTensorView {
  const TensorInfo* info{};
  const std::byte* data{};
  std::size_t bytes{};
};

// Immutable device addresses used by every target-model layer invocation.
// Resolve these once while loading the checkpoint: constructing tensor names
// and searching the safetensors index does not belong on the decode path.
struct TargetLayerWeightsView {
  const void *input_norm{}, *q{}, *k{}, *v{}, *o{}, *q_norm{}, *k_norm{};
  const void *post_attention{}, *dense_pre{}, *expert_pre{};
  const void *router_scale{}, *router{}, *expert_scale{};
  const void *gate{}, *up{}, *down{}, *dense_post{}, *expert_post{};
  const void *combined_post{}, *layer_scalar{};
};

struct AssistantLayerWeightsView {
  const void *input_norm{}, *q{}, *q_norm{}, *o{};
  const void *post_attention{}, *pre_feedforward{};
  const void *gate{}, *up{}, *down{}, *post_feedforward{}, *layer_scalar{};
};

// Owns the file mappings before the device arena so host views remain valid
// for the complete upload. Tensor addresses stay fixed for CUDA Graph capture.
class DeviceModel {
 public:
  explicit DeviceModel(const std::filesystem::path& model_directory);

  DeviceTensorView tensor(const std::string& name) const;
  TensorView host_tensor(const std::string& name) const { return weights_.tensor(name); }
  const UploadResult& upload_result() const { return arena_.upload_result(); }
  std::size_t tensor_count() const { return weights_.tensor_count(); }

 private:
  ModelWeights weights_;
  DeviceArena arena_;
};

// Complete production weight ownership: the original checkpoint supplies all
// BF16/non-expert tensors while the sidecar supplies SM120 fused expert data.
class RuntimeWeights {
 public:
  RuntimeWeights(const std::filesystem::path& model_directory,
                 const std::filesystem::path& expert_directory);

  const DeviceModel& model() const { return model_; }
  const DeviceModel& experts() const { return experts_; }
  const DeviceTensorView& embedding() const { return embedding_; }
  const void* final_norm() const { return final_norm_; }
  const TargetLayerWeightsView& layer(int index) const {
    return layers_.at(index);
  }
  std::size_t device_bytes() const { return device_bytes_; }

 private:
  DeviceModel model_;
  DeviceModel experts_;
  DeviceTensorView embedding_{};
  const void* final_norm_{};
  std::array<TargetLayerWeightsView, 30> layers_{};
  std::size_t device_bytes_{};
};

// The official Gemma 4 MTP assistant is a four-layer BF16 model which borrows
// the target model's sliding/full KV states. It is deliberately kept in a
// separate arena so target-only and speculative measurements remain distinct.
class AssistantWeights {
 public:
  explicit AssistantWeights(const std::filesystem::path& model_directory);
  ~AssistantWeights();
  AssistantWeights(const AssistantWeights&) = delete;
  AssistantWeights& operator=(const AssistantWeights&) = delete;

  const DeviceModel& model() const { return model_; }
  const void* embedding() const { return embedding_; }
  const void* pre_projection() const { return pre_projection_; }
  const void* post_projection() const { return post_projection_; }
  const void* final_norm() const { return final_norm_; }
  const AssistantLayerWeightsView& layer(int index) const {
    return layers_.at(index);
  }
  const void* fused_gate_up(int index) const {
    return fused_gate_up_.at(index);
  }
  std::size_t device_bytes() const {
    return model_.upload_result().bytes +
        (fused_gate_up_storage_ ? 128ULL * 1024 * 1024 : 0);
  }

 private:
  DeviceModel model_;
  const void *embedding_{}, *pre_projection_{}, *post_projection_{},
      *final_norm_{};
  std::array<AssistantLayerWeightsView, 4> layers_{};
  void* fused_gate_up_storage_{};
  std::array<const void*, 4> fused_gate_up_{};
};

struct DeviceKvView {
  std::byte* keys{};
  std::byte* values{};
  int capacity{};
  int kv_heads{};
  int head_dim{};
};

class KvCache {
 public:
  explicit KvCache(int maximum_context);
  ~KvCache();
  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;
  KvCache(KvCache&& other) noexcept;
  KvCache& operator=(KvCache&& other) noexcept;

  const DeviceKvView& layer(int index) const;
  const DeviceKvView* device_layers() const { return device_layers_; }
  // Commits physical storage for global-attention KV through this context.
  // The full model address range is reserved once, keeping kernel pointers
  // stable while physical memory grows in coarse pages on demand.
  // Newly committed pages are zeroed on stream before use. A null stream
  // provides synchronous initialization for callers without an execution stream.
  void ensure_context(int context_tokens, cudaStream_t stream = nullptr);
  int maximum_context() const { return maximum_context_; }
  int resident_context() const { return resident_context_; }
  std::size_t bytes() const { return bytes_; }
  const void* sliding_arena() const { return arena_; }
  void* sliding_arena() { return arena_; }
  std::size_t sliding_bytes() const { return sliding_bytes_; }

 private:
  void release() noexcept;
  std::byte* arena_{};
  std::size_t bytes_{};
  std::size_t sliding_bytes_{};
  int maximum_context_{};
  int resident_context_{};
  std::vector<DeviceKvView> layers_;
  DeviceKvView* device_layers_{};
  struct VirtualRegion;
  std::vector<std::unique_ptr<VirtualRegion>> virtual_regions_;
};

// Transactional storage for one target verification group. Candidate rows are
// kept out of the live ring until MTP acceptance decides the committed prefix.
class CandidateKvCache {
 public:
  static constexpr int kMaximumTokens = 5;

  CandidateKvCache();
  ~CandidateKvCache();
  CandidateKvCache(const CandidateKvCache&) = delete;
  CandidateKvCache& operator=(const CandidateKvCache&) = delete;
  CandidateKvCache(CandidateKvCache&& other) noexcept;
  CandidateKvCache& operator=(CandidateKvCache&& other) noexcept;

  const DeviceKvView& layer(int index) const;
  const DeviceKvView* device_layers() const { return device_layers_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void release() noexcept;
  std::byte* arena_{};
  std::size_t bytes_{};
  std::vector<DeviceKvView> layers_;
  DeviceKvView* device_layers_{};
};

class AttentionWorkspace {
 public:
  explicit AttentionWorkspace(int maximum_context, int maximum_batch = 1);
  ~AttentionWorkspace();
  AttentionWorkspace(const AttentionWorkspace&) = delete;
  AttentionWorkspace& operator=(const AttentionWorkspace&) = delete;

  float* scores() const { return scores_; }
  float* partials() const { return partials_; }
  std::size_t bytes() const { return bytes_; }
  int maximum_context() const { return maximum_context_; }
  int maximum_batch() const { return maximum_batch_; }

 private:
  float* scores_{};
  float* partials_{};
  std::size_t bytes_{};
  int maximum_context_{};
  int maximum_batch_{};
};

// Fixed-session scratch reuse for generation stages whose temporary geometry
// is identical from one speculative cycle to the next.
class DeviceScratchPool {
 public:
  DeviceScratchPool() = default;
  ~DeviceScratchPool();
  DeviceScratchPool(const DeviceScratchPool&) = delete;
  DeviceScratchPool& operator=(const DeviceScratchPool&) = delete;

  void reset() { cursor_ = 0; }
  void* allocate(std::size_t bytes);
  // Persistent page-locked staging for the verifier's tiny host/device
  // control records. This keeps asynchronous copies truly asynchronous and
  // avoids a host allocation on every speculative cycle.
  void* host_staging(std::size_t bytes);
  // Layer-major cache descriptors only change when the active cohort changes.
  // Keep their device allocation and last uploaded cohort alongside the
  // verifier scratch instead of rebuilding and copying them every cycle.
  void prepare_verifier_views(
      std::span<KvCache* const> caches,
      std::span<CandidateKvCache* const> candidates,
      void* stream, const DeviceKvView*& live,
      const DeviceKvView*& candidate);
  void prepare_assistant_views(
      std::span<KvCache* const> caches, void* stream,
      const DeviceKvView*& sliding, const DeviceKvView*& global);
  std::size_t bytes() const { return bytes_; }
  std::uint64_t generation() const { return generation_; }

 private:
  struct Block { void* pointer{}; std::size_t bytes{}; };
  std::vector<Block> blocks_;
  DeviceKvView* verifier_live_views_{};
  DeviceKvView* verifier_candidate_views_{};
  std::array<const KvCache*, 8> verifier_caches_{};
  std::array<const CandidateKvCache*, 8> verifier_candidates_{};
  int verifier_sessions_{};
  DeviceKvView* assistant_sliding_views_{};
  DeviceKvView* assistant_global_views_{};
  std::array<const KvCache*, 8> assistant_caches_{};
  int assistant_sessions_{};
  void* host_staging_{};
  std::size_t host_staging_bytes_{};
  std::size_t cursor_{};
  std::size_t bytes_{};
  std::uint64_t generation_{};
};

}  // namespace g4
