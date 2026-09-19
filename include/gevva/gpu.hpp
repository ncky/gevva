#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <span>
#include <vector>

#include <cuda_runtime_api.h>

namespace gevva {

int test_precomputed_rope();
int test_attention_router_fusion();
int test_attention_softmax();
int test_fast_attention_softmax();
int test_attention_unpack();

class DeviceModel;
class RuntimeWeights;
class AssistantWeights;
class KvCache;
class CandidateKvCache;
class AttentionWorkspace;
class DeviceScratchPool;

class GpuExecutionContext {
 public:
  explicit GpuExecutionContext(bool high_priority = false);
  ~GpuExecutionContext();
  GpuExecutionContext(const GpuExecutionContext&) = delete;
  GpuExecutionContext& operator=(const GpuExecutionContext&) = delete;

  cudaStream_t stream() const;
  cudaStream_t auxiliary_stream() const;
  cudaStream_t tertiary_stream() const;
  void* blas_handle() const;
  void begin_auxiliary() const;
  void end_auxiliary() const;
  void begin_tertiary() const;
  void end_tertiary() const;
  void join_tertiary_to_auxiliary() const;
  // Exact signatures for allocation-sensitive prefill graphs. The cache is
  // bounded independently of legacy decode graphs.
  bool has_graph(std::span<const std::uint64_t> signature) const;
  void capture_graph(std::span<const std::uint64_t> signature,
                     const std::function<void()>& enqueue) const;
  void launch_graph(std::span<const std::uint64_t> signature) const;
  std::size_t prefill_graph_count() const;
  std::uint64_t prefill_graph_hits() const;
  std::uint64_t prefill_graph_builds() const;
  bool should_capture_graph(std::span<const std::uint64_t> signature) const;
  bool has_graph(std::uint64_t key) const;
  void capture_graph(std::uint64_t key,
                     const std::function<void()>& enqueue) const;
  void launch_graph(std::uint64_t key) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Copy the valid prefix rows of the fixed Gemma 4 sliding-KV arena and rebuild
// its mirrored rings in one launch. This is substantially cheaper than copying
// all 3076 physical rows for a short shared prefix.
void copy_sliding_prefix_arena(const void* source, void* destination,
                               int tokens, cudaStream_t stream);

class CompressedVocabRunner {
 public:
  CompressedVocabRunner(const DeviceModel& exact_model,
                        const std::string& exact_tensor,
                        const DeviceModel& int8_model,
                        cudaStream_t stream);
  ~CompressedVocabRunner();
  CompressedVocabRunner(const CompressedVocabRunner&) = delete;
  CompressedVocabRunner& operator=(const CompressedVocabRunner&) = delete;

  void launch(const void* activation, int* selected_token,
              cudaStream_t stream) const;
  void launch_batch(const void* activations, int* selected_tokens,
                    int tokens, cudaStream_t stream) const;
  void warmup(int maximum_tokens, cudaStream_t stream) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// SM120 block-scaled W4A4 vocabulary projection. The NVFP4 matrix is only a
// shortlist generator; finalist logits are recomputed from the exact BF16 tied
// embedding before argmax, matching the compressed INT8 runner's contract.
class Nvfp4VocabRunner {
 public:
  Nvfp4VocabRunner(const DeviceModel& exact_model,
                   const std::string& exact_tensor,
                   const DeviceModel& nvfp4_model, int maximum_tokens,
                   std::span<const int> token_map = {},
                   int alternate_vocab_rows = 0);
  ~Nvfp4VocabRunner();
  Nvfp4VocabRunner(const Nvfp4VocabRunner&) = delete;
  Nvfp4VocabRunner& operator=(const Nvfp4VocabRunner&) = delete;

  void launch_batch(const void* activations, int* selected_tokens,
                    int tokens, cudaStream_t stream,
                    int active_vocab_rows = 0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Nvfp4VocabBenchmark {
  float int8_microseconds{};
  float nvfp4_microseconds{};
  float hot_nvfp4_microseconds{};
  int rows{};
  int mismatched_tokens{};
  int hot_mismatched_tokens{};
};

Nvfp4VocabBenchmark benchmark_nvfp4_vocab(
    const DeviceModel& exact_model, const DeviceModel& int8_model,
    const DeviceModel& nvfp4_model, int rows,
    const std::string& exact_tensor =
        "model.language_model.embed_tokens.weight",
    std::span<const int> hot_token_map = {});

// SM120 FP8 projection runner for the target's non-expert linear layers.
// It dynamically quantizes each activation row and applies the offline
// per-tensor weight scale after FP8 tensor-core accumulation.
struct Fp8OutputScale {
  const float* activation_rows{};
  const float* weight{};
};

class Fp8LinearRunner {
 public:
  Fp8LinearRunner(const DeviceModel& exact_model,
                  const DeviceModel& fp8_model, int maximum_tokens = 5,
                  bool assistant = false, int projection_tile_tokens = 0,
                  int concurrent_workspaces = 3, bool vision = false,
                  Fp8LinearRunner* shared_workspace = nullptr);
  std::uint64_t instance_id() const;
  int padded_output_rows(int tokens) const;
  ~Fp8LinearRunner();
  Fp8LinearRunner(const Fp8LinearRunner&) = delete;
  Fp8LinearRunner& operator=(const Fp8LinearRunner&) = delete;

  bool launch(const void* exact_weight, int outputs, int inputs,
              const void* activation, void* output, int tokens,
              cudaStream_t stream, void* blas_handle,
              bool rescale_output = true,
              int workspace_index = 0,
              bool output_has_padded_capacity = false) const;
  bool launch_unscaled(const void* exact_weight, int outputs, int inputs,
                       const void* activation, void* output, int tokens,
                       cudaStream_t stream, void* blas_handle,
                       Fp8OutputScale& scale,
                       int workspace_index = 0,
                       bool output_has_padded_capacity = false) const;

  void prepare_rmsnorm_2816(const void* input, const void* norm_weight,
                            const void* activation_cache_key, int tokens,
                            cudaStream_t stream) const;
  void prepare_packed_attention(const void* packed, void* output,
      const void* cache_views, const int* contexts, int sessions, int tokens,
      int dimension, bool restore_ring, cudaStream_t stream) const;
  void prepare_dual_rmsnorm_router_2816(
      const void* input, const void* dense_weight, const void* expert_weight,
      const void* router_weight, void* dense_output, void* expert_output,
      const void* activation_cache_key, int tokens, cudaStream_t stream,
      int threads = 256, const void* attention = nullptr,
      const void* attention_weight = nullptr, void* residual_output = nullptr,
      const float* attention_scales = nullptr,
      const float* attention_projection_scale = nullptr) const;
  void use_secondary_activation_2816(const void* activation_cache_key,
                                     int tokens,
                                     cudaStream_t dependency_ordered_stream) const;
  void assume_activation_ready_on(cudaStream_t dependency_ordered_stream) const;
  void prepare_gelu_2112(const void* gate, const void* up,
                         Fp8OutputScale gate_scale,
                         Fp8OutputScale up_scale,
                         const void* activation_cache_key, int tokens,
                         cudaStream_t stream, bool packed = false) const;
  bool launch_fused_gate_up(const void* gate_weight, const void* up_weight,
      const void* input, void* output, int tokens, cudaStream_t stream,
      void* blas_handle, Fp8OutputScale& gate_scale, Fp8OutputScale& up_scale) const;
  void invalidate_activation_cache() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Persistent BF16 vision projection weights packed along the output axis.
// Gemma checkpoints store Q/K/V and gate/up separately; production executes
// each group as one GEMM without repacking weights in the request path.
class VisionFusedWeights {
 public:
  explicit VisionFusedWeights(const DeviceModel& model);
  ~VisionFusedWeights();
  VisionFusedWeights(const VisionFusedWeights&) = delete;
  VisionFusedWeights& operator=(const VisionFusedWeights&) = delete;

  const void* qkv(int layer) const;
  const void* gate_up(int layer) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Nvfp4ExpertRunner {
 public:
  Nvfp4ExpertRunner(const DeviceModel& experts, int layer,
                    int maximum_tokens = 1,
                    Nvfp4ExpertRunner* shared_scratch = nullptr);
  // Shared scratch requires serialized launches, including routed-output consumers.
  ~Nvfp4ExpertRunner();
  Nvfp4ExpertRunner(const Nvfp4ExpertRunner&) = delete;
  Nvfp4ExpertRunner& operator=(const Nvfp4ExpertRunner&) = delete;
  Nvfp4ExpertRunner(Nvfp4ExpertRunner&&) noexcept;
  Nvfp4ExpertRunner& operator=(Nvfp4ExpertRunner&&) noexcept;

  // All arguments are device pointers. expert_ids and route_weights contain
  // eight entries produced by the router; no host routing synchronization.
  void launch(const void* input, const int* expert_ids,
              const float* route_weights, void* output,
              cudaStream_t stream) const;
  void launch_batch(const void* input, const int* expert_ids,
                    const float* route_weights, void* output, int tokens,
                    cudaStream_t stream) const;
  // Prefill fusion hook: run both expert GEMMs but leave the eight routed
  // outputs in runner-owned storage for an immediately following GPU kernel.
  void launch_batch_unreduced(const void* input, const int* expert_ids,
                              int tokens, cudaStream_t stream) const;
  const void* routed_output() const;
  const int* inverse_routes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ExpertRepeatTestResult {
  float microseconds{};
  float output_abs_checksum{};
  float maximum_difference{};
  std::size_t different_values{};
  int repetitions{};
  bool bitwise_repeatable{};
};

// Exercises only the production packed expert sidecar. Fixed activations,
// routes, and weights make any output variation a kernel/runtime defect.
ExpertRepeatTestResult test_nvfp4_expert_repeatability(
    const DeviceModel& experts, int layer, int tokens, int repetitions = 8);

inline constexpr int kTargetComputeMajor = 12;
inline constexpr int kTargetComputeMinor = 0;

struct GpuInfo {
  int ordinal{};
  std::string name;
  std::string pci_bus_id;
  std::string uuid;
  int compute_major{};
  int compute_minor{};
  std::size_t total_memory{};
};

// Selects GEVVA_GPU=pro6000 (default) or 5090 by model name, then verifies SM 12.0
// and its memory class. CUDA_VISIBLE_DEVICES can restrict which cards are visible.
GpuInfo select_target_gpu();

struct UploadResult {
  std::size_t bytes{};
  float device_milliseconds{};
  double wall_milliseconds{};
};

class DeviceArena {
 public:
  explicit DeviceArena(const std::vector<std::span<const std::byte>>& shard_data);
  ~DeviceArena();
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&& other) noexcept;
  DeviceArena& operator=(DeviceArena&& other) noexcept;

  const std::byte* shard_base(std::size_t index) const;
  const UploadResult& upload_result() const { return upload_; }
  std::size_t size() const { return arena_bytes_; }

 private:
  void release() noexcept;
  std::byte* arena_{};
  std::size_t arena_bytes_{};
  std::vector<std::size_t> shard_offsets_;
  UploadResult upload_;
};

// Allocates one fixed-address device arena and uploads each shard's contiguous
// data section. The arena is released after timing; inference will retain it.
UploadResult benchmark_weight_upload(
    const std::vector<std::span<const std::byte>>& shard_data);

struct PrimitiveTestResult {
  float rmsnorm_max_abs_error{};
  float routing_max_abs_error{};
  bool routing_ids_match{};
  float rmsnorm_microseconds{};
  float routing_microseconds{};
};

PrimitiveTestResult test_and_benchmark_primitives();

struct DecodeAttentionTestResult {
  float sliding_max_abs_error{};
  float sliding_microseconds{};
  float global_max_abs_error{};
  float global_microseconds{};
};

DecodeAttentionTestResult test_decode_attention(int context_tokens);

// Isolated long-context global-KV experiment.  This is deliberately separate
// from the serving cache until both the numerical error and the SM120
// crossover against the production BF16 attention path are measured.
struct TurboQuantKvBenchmark {
  float bf16_microseconds{};
  float tq4_microseconds{};
  float quantize_microseconds_per_token{};
  float max_abs_error{};
  float mean_abs_error{};
  float reference_rms{};
  float normalized_rmse{};
  float cosine_similarity{};
  float compression_ratio{};
};

TurboQuantKvBenchmark benchmark_turboquant_global_kv(int context_tokens);

struct DecodeAttentionBatchTestResult {
  float sliding_max_abs_error{};
  float sliding_microseconds{};
  float global_max_abs_error{};
  float global_microseconds{};
};

struct SlidingKvAppendTestResult {
  float key_max_abs_error{};
  float value_max_abs_error{};
  int checked_tokens{};
};

// Exercises the production large-prefill append path at a non-zero absolute
// position and verifies that the newest 1024 rows land in the physical ring.
SlidingKvAppendTestResult test_sliding_kv_large_append();

DecodeAttentionBatchTestResult test_decode_attention_batch(int context_tokens,
                                                           int tokens = 5);

struct VisionPatchTestResult {
  float max_abs_error{};
  float mean_abs_error{};
  float microseconds{};
  float checksum{};
};

VisionPatchTestResult test_vision_patch_embedding(
    const DeviceModel& model, std::span<const float> pixel_values,
    std::span<const std::int32_t> position_ids,
    std::span<const float> expected = {});

struct VisionLayerTestResult {
  float max_abs_error{};
  float mean_abs_error{};
  float microseconds{};
  float checksum{};
};

VisionLayerTestResult test_vision_layer0(
    const DeviceModel& model, std::span<const float> input,
    std::span<const std::int32_t> position_ids,
    std::span<const float> expected = {});

struct VisionEncoderBenchmark {
  float projected_max_abs_error{};
  float projected_mean_abs_error{};
  float microseconds{};
  float projected_checksum{};
  std::vector<float> projected;
};

VisionEncoderBenchmark benchmark_vision_encoder(
    const DeviceModel& model, std::span<const float> pixel_values,
    std::span<const std::int32_t> position_ids, int valid_patches,
    std::span<const float> expected_projected = {}, bool benchmark = true,
    DeviceScratchPool* scratch = nullptr,
    GpuExecutionContext* context = nullptr, int image_batch = 1,
    void* device_projected_output = nullptr,
    bool copy_projected_to_host = true,
    Fp8LinearRunner* fp8_linears = nullptr,
    std::span<const float> second_pixel_values = {},
    std::span<const std::int32_t> second_position_ids = {},
    const VisionFusedWeights* fused_weights = nullptr,
    const float* device_pixel_values = nullptr,
    const std::int32_t* device_position_ids = nullptr);

struct MultimodalEmbeddingBenchmark {
  float microseconds{};
  float checksum{};
  float max_abs_error{};
  int tokens{};
  int image_tokens{};
};

MultimodalEmbeddingBenchmark benchmark_multimodal_embedding(
    const DeviceModel& model, std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images);

struct PrefillAttentionTestResult {
  float sliding_max_abs_error{};
  float sliding_microseconds{};
  float global_max_abs_error{};
  float global_microseconds{};
};

// Full-sequence attention used during prompt ingestion. Equal non-negative
// block ids form bidirectional image/video islands on top of the causal mask.
PrefillAttentionTestResult test_prefill_attention(
    std::span<const std::int32_t> block_sequence_ids);

struct PrefillLayerBenchmark {
  float microseconds{};
  float output_checksum{};
};

PrefillLayerBenchmark benchmark_prefill_layer(
    const RuntimeWeights& weights, int layer,
    std::span<const std::int32_t> block_sequence_ids);

struct TargetPrefillBenchmark {
  float microseconds{};
  float hidden_checksum{};
  int selected_token{};
  int tokens{};
  // Unnormalized last-layer state for the final prompt token. The MTP
  // assistant combines this with the selected token's target embedding.
  std::vector<float> final_state;
};

// Exact tied-embedding readout with on-device softcap and full-vocabulary
// normalization. Only the 26 alias logits and summary leave the GPU.
struct DecisionReadout {
  std::vector<float> option_logits;
  bool full_vocabulary{true};
  float log_normalizer{};
  std::uint32_t argmax_token{};
};
struct DecisionHeadView {
  std::vector<std::uint32_t> alias_ids;
  const std::uint32_t* device_aliases{};
  const void* embeddings{};
};
DecisionHeadView prepare_decision_head(const RuntimeWeights& weights,
    std::span<const std::uint32_t> alias_ids, GpuExecutionContext& context,
    DeviceScratchPool& storage);
std::vector<DecisionReadout> decision_scores(
    const RuntimeWeights& weights, std::span<const float> hidden_states,
    std::span<const std::uint32_t> alias_ids,
    GpuExecutionContext& context, DeviceScratchPool& scratch,
    const void* device_hidden_states = nullptr, int device_rows = 0,
    bool options_only = false, const DecisionHeadView* prepared_head = nullptr);

struct TargetPrefillBatchItem {
  KvCache* cache{};
  std::span<const std::uint32_t> input_ids;
  std::span<const std::uint8_t> multimodal_types;
  std::span<const float> projected_images;
  // Optional BF16 [image_tokens, 2816] device input. When present the host
  // span is intentionally empty and prefill keeps vision features on-device.
  const void* device_projected_images{};
  int prefix_tokens{};
  int final_state_row{};
};

TargetPrefillBenchmark benchmark_target_prefill(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    KvCache& cache, std::span<const std::uint32_t> input_ids,
    std::span<const std::uint8_t> multimodal_types,
    std::span<const float> projected_images, int layer_count = 30,
    bool benchmark = true,
    std::span<Nvfp4ExpertRunner* const> persistent_experts = {},
    DeviceScratchPool* scratch = nullptr,
    CompressedVocabRunner* persistent_vocab = nullptr,
    GpuExecutionContext* context = nullptr,
    void* device_final_state = nullptr,
    int prefix_tokens = 0,
    Fp8LinearRunner* fp8_linears = nullptr,
    const void* device_projected_images = nullptr, bool kv_only = false);

// Packs independent prompt rows through every dense and routed-expert layer.
// Attention remains ragged per session so RoPE positions, multimodal islands,
// and KV ownership are identical to separate prefills. The optional final
// states are written as consecutive [session, 2816] BF16 rows.
std::vector<TargetPrefillBenchmark> benchmark_target_prefill_batch(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    std::span<const TargetPrefillBatchItem> items, int layer_count = 30,
    std::span<Nvfp4ExpertRunner* const> persistent_experts = {},
    DeviceScratchPool* scratch = nullptr,
    CompressedVocabRunner* persistent_vocab = nullptr,
    GpuExecutionContext* context = nullptr,
    void* device_final_states = nullptr,
    Fp8LinearRunner* fp8_linears = nullptr,
    Nvfp4VocabRunner* nvfp4_vocab = nullptr,
    bool decision_only = false, bool read_only_prefix = false,
    bool multimodal_prefix = false);

void launch_decode_attention(bool global_layer,
                             const void* query,
                             const void* keys,
                             const void* values,
                             void* output,
                             int context_tokens,
                             float* score_workspace,
                             float* partial_workspace,
                             cudaStream_t stream);

void launch_decode_attention_batch(bool global_layer,
                                   const void* queries,
                                   const void* keys,
                                   const void* values,
                                   const void* candidate_keys,
                                   const void* candidate_values,
                                   void* outputs,
                                   int context_tokens,
                                   int tokens,
                                   float* score_workspace,
                                   float* partial_workspace,
                                   cudaStream_t stream);

void stage_candidate_kv(CandidateKvCache& candidates, int layer,
                        const void* keys, const void* values, int tokens,
                        cudaStream_t stream);

void commit_candidate_kv(const CandidateKvCache& candidates, KvCache& cache,
                         int position, int tokens, cudaStream_t stream);

void launch_qkv_transform(bool global_layer,
                          void* query,
                          void* key,
                          void* value,
                          const void* query_norm_weight,
                          const void* key_norm_weight,
                          int position,
                          void* key_cache,
                          void* value_cache,
                          cudaStream_t stream);

struct AttentionSublayerBenchmark {
  float eager_microseconds{};
  float graph_microseconds{};
  float output_checksum{};
};

AttentionSublayerBenchmark benchmark_attention_sublayer(
    const DeviceModel& model, KvCache& cache, AttentionWorkspace& workspace,
    int layer, int context_tokens);

struct DecoderLayerBenchmark {
  float eager_microseconds{};
  float graph_microseconds{};
  float output_checksum{};
};

struct AssistantStepBenchmark {
  float eager_microseconds{};
  float graph_microseconds{};
  float logits_microseconds{};
  float cycle_microseconds{};
  int selected_token{};
  std::array<int, 4> drafted_tokens{};
  float state_checksum{};
  float state_max_abs_error{};
  float state_mean_abs_error{};
  bool oracle_available{};
  bool oracle_token_match{};
};

// One exact autoregressive MTP draft: pre-projection, all four assistant
// layers against target-owned KV, final/post projections, full vocabulary
// head, and greedy token selection.
AssistantStepBenchmark benchmark_assistant_step(
    const AssistantWeights& weights, KvCache& target_cache,
    AttentionWorkspace& workspace, int context_tokens,
    const DeviceModel* compressed_vocab = nullptr,
    const DeviceModel* target_model = nullptr, int drafts = 1,
    std::span<const float> initial_target_state = {},
    int previous_token = -1, DeviceScratchPool* scratch = nullptr,
    CompressedVocabRunner* persistent_vocab = nullptr,
    GpuExecutionContext* context = nullptr,
    const void* initial_target_state_device = nullptr,
    int* drafted_tokens_device = nullptr,
    Fp8LinearRunner* fp8_linears = nullptr);

struct DecodeBatchInputs {
  const int* contexts{};
  const int* previous{};
};

// Shared inputs remain valid until the assistant scratch is reused. The
// consumer must run on the same stream and finish before the next upload.
// Runs the four-layer Gemma 4 assistant for several independent sessions as
// one matrix batch. Context lengths and KV arenas remain ragged per session;
// output drafts are laid out [session][draft].
void launch_assistant_batch(
    const AssistantWeights& weights,
    std::span<KvCache* const> target_caches,
    std::span<const int> context_tokens,
    const DeviceModel& compressed_vocab,
    const DeviceModel& target_model,
    const void* initial_target_states_device,
    std::span<const int> previous_tokens,
    int* drafted_tokens_device,
    DeviceScratchPool& scratch,
    CompressedVocabRunner& vocabulary,
    Nvfp4VocabRunner* nvfp4_vocabulary,
    GpuExecutionContext& context,
    Fp8LinearRunner* fp8_linears = nullptr,
    int drafts = 4,
    int minimum_rows = 1,
    int assistant_vocab_rows = 0,
    DecodeBatchInputs* shared_inputs = nullptr);

struct AssistantBatchTestResult {
  int batch{};
  float microseconds{};
  bool rows_repeatable{};
  bool first_row_matches_scalar{};
  std::array<int, 4> first_row_drafts{};
  std::array<int, 4> scalar_drafts{};
};

AssistantBatchTestResult test_assistant_batch(
    const AssistantWeights& weights, const DeviceModel& compressed_vocab,
    const DeviceModel& target_model, int batch, int context_tokens,
    const DeviceModel* nvfp4_vocab = nullptr,
    std::span<const int> hot_token_map = {});

struct SmallBatchBenchmark {
  std::array<float, 5> q_batched_microseconds{};
  std::array<float, 5> q_repeated_microseconds{};
  std::array<float, 5> mlp_batched_microseconds{};
  std::array<float, 5> mlp_repeated_microseconds{};
};

SmallBatchBenchmark benchmark_target_small_batch(const DeviceModel& model);

struct TargetVerifierBenchmark {
  float eager_microseconds{};
  float graph_microseconds{};
  std::array<int, 5> selected_tokens{};
  std::array<int, 4> draft_tokens{};
  float hidden_checksum{};
  std::array<int, 5> output_tokens{};
  int output_count{};
  int matched_drafts{};
  std::vector<float> continuation_state;
};

TargetVerifierBenchmark benchmark_target_verifier(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    KvCache& cache, AttentionWorkspace& workspace, int context_tokens,
    int tokens = 5, std::span<const std::uint32_t> candidate_input_ids = {},
    std::span<const int> draft_tokens = {},
    std::span<Nvfp4ExpertRunner* const> persistent_experts = {},
    DeviceScratchPool* scratch = nullptr,
    CompressedVocabRunner* persistent_vocab = nullptr,
    CandidateKvCache* persistent_candidates = nullptr,
    GpuExecutionContext* context = nullptr,
    void* continuation_state_device = nullptr,
    Fp8LinearRunner* fp8_linears = nullptr,
    const int* draft_tokens_device = nullptr,
    int previous_token = -1);

struct TargetVerifierBatchResult {
  std::vector<TargetVerifierBenchmark> sessions;
};

TargetVerifierBatchResult launch_target_verifier_batch(
    const RuntimeWeights& weights,
    std::span<KvCache* const> caches,
    std::span<CandidateKvCache* const> candidates,
    std::span<const int> context_tokens,
    std::span<const int> previous_tokens,
    const int* drafted_tokens_device,
    void* continuation_states_device,
    std::span<Nvfp4ExpertRunner* const> experts,
    DeviceScratchPool& scratch,
    CompressedVocabRunner& vocabulary,
    Nvfp4VocabRunner* nvfp4_vocabulary,
    Fp8LinearRunner& fp8_linears,
    GpuExecutionContext& context,
    int drafts = 4, const DecodeBatchInputs* shared_inputs = nullptr);

struct TargetVerifierBatchFixedBenchmark {
  float microseconds{};
  std::array<int, 5> target_tokens{};
  std::array<int, 5> output_tokens{};
  int output_count{};
  int matched_drafts{};
  bool sessions_repeatable{};
  std::uint64_t sequence_hash{};
};

TargetVerifierBatchFixedBenchmark benchmark_target_verifier_batch_fixed(
    const RuntimeWeights& weights, const DeviceModel& compressed_vocab,
    const DeviceModel& fp8_model, int batch, int context_tokens,
    int repetitions = 20, int drafts = 4,
    const DeviceModel* nvfp4_vocab = nullptr);

struct VocabHeadBenchmark {
  float exact_microseconds{};
  float int8_microseconds{};
  int exact_token{};
  int approximate_token{};
  int int8_token{};
  int validation_trials{};
  int approximate_matches{};
  int corrected_matches{};
  float exact_batch5_microseconds{};
  float int8_batch5_microseconds{};
  int batch5_matches{};
};

VocabHeadBenchmark benchmark_vocab_head(
    const DeviceModel& exact_model, const std::string& exact_tensor,
    const DeviceModel& int8_model);

DecoderLayerBenchmark benchmark_decoder_layer(
    const RuntimeWeights& weights, KvCache& cache,
    AttentionWorkspace& workspace, int layer, int context_tokens,
    int tokens = 1);

struct MatvecTestResult {
  float max_abs_error{};
  float microseconds{};
};

MatvecTestResult test_bf16_matvec(std::span<const std::byte> row_major_weight,
                                 int output_features, int input_features);

MatvecTestResult test_nvfp4_matvec(std::span<const std::byte> packed_weight,
                                  std::span<const std::byte> block_scale,
                                  std::span<const std::byte> global_scale,
                                  int output_features, int input_features);

MatvecTestResult test_nvfp4_grouped_matvec(
    const std::vector<std::span<const std::byte>>& packed_weights,
    const std::vector<std::span<const std::byte>>& block_scales,
    const std::vector<std::span<const std::byte>>& global_scales,
    int output_features, int input_features);

struct ExpertBlockTestResult {
  float max_abs_error{};
  float eager_microseconds{};
  float graph_microseconds{};
};

struct CutlassGroupedTestResult {
  float gemm_microseconds{};
  float quantized_microseconds{};
  float expert_block_microseconds{};
  float dynamic_runner_microseconds{};
  float output_checksum{};
  std::vector<float> output;
};

// Runs the first fused expert projection using the production packed sidecar
// and CUTLASS's native SM120 block-scaled tensor-core grouped GEMM.
CutlassGroupedTestResult benchmark_cutlass_grouped_w13(
    const DeviceModel& experts, int layer = 0);

ExpertBlockTestResult test_nvfp4_expert_block(
    const std::vector<std::span<const std::byte>>& gate_weights,
    const std::vector<std::span<const std::byte>>& gate_scales,
    const std::vector<std::span<const std::byte>>& gate_globals,
    const std::vector<std::span<const std::byte>>& up_weights,
    const std::vector<std::span<const std::byte>>& up_scales,
    const std::vector<std::span<const std::byte>>& up_globals,
    const std::vector<std::span<const std::byte>>& down_weights,
    const std::vector<std::span<const std::byte>>& down_scales,
    const std::vector<std::span<const std::byte>>& down_globals);

struct FrontendTestResult {
  float embedding_max_abs_error{};
  float input_norm_max_abs_error{};
};

FrontendTestResult test_bf16_frontend(std::span<const std::byte> embedding,
                                     std::span<const std::byte> norm_weight,
                                     std::span<const std::int32_t> input_ids,
                                     std::span<const float> expected_embedding,
                                     std::span<const float> expected_norm);

struct AttentionTestResult {
  float output_max_abs_error{};
  float probability_max_abs_error{};
};

AttentionTestResult test_bf16_layer0_attention(
    std::span<const std::byte> q_weight, std::span<const std::byte> k_weight,
    std::span<const std::byte> v_weight, std::span<const std::byte> o_weight,
    std::span<const std::byte> q_norm_weight, std::span<const std::byte> k_norm_weight,
    std::span<const float> normalized_input, std::span<const float> expected_output,
    std::span<const float> expected_probabilities);

AttentionTestResult test_bf16_layer5_attention(
    std::span<const std::byte> q_weight, std::span<const std::byte> k_weight,
    std::span<const std::byte> o_weight, std::span<const std::byte> q_norm_weight,
    std::span<const std::byte> k_norm_weight, std::span<const float> normalized_input,
    std::span<const float> expected_output, std::span<const float> expected_probabilities);

float test_bf16_layer0_dense_mlp(std::span<const std::byte> gate_weight,
                                 std::span<const std::byte> up_weight,
                                 std::span<const std::byte> down_weight,
                                 std::span<const float> input,
                                 std::span<const float> expected_output);

float test_bf16_layer0_experts(std::span<const std::byte> gate_up_weights,
                               std::span<const std::byte> down_weights,
                               std::span<const float> input,
                               std::span<const float> top_weights,
                               std::span<const float> top_ids,
                               std::span<const float> expected_output);

struct RouterTestResult {
  float pre_feedforward_norm_max_abs_error{};
  float input_max_abs_error{};
  float logits_max_abs_error{};
  float probability_max_abs_error{};
  float top_weight_max_abs_error{};
  float fused_max_abs_error{};
  bool top_ids_match{};
};

RouterTestResult test_bf16_layer0_router(
    std::span<const std::byte> post_attention_norm_weight,
    std::span<const std::byte> pre_feedforward_norm_weight,
    std::span<const std::byte> router_scale, std::span<const std::byte> router_weight,
    std::span<const std::byte> per_expert_scale, std::span<const float> residual,
    std::span<const float> attention_output, std::span<const float> expected_pre_ff_norm,
    std::span<const float> expected_router_input, std::span<const float> expected_logits,
    std::span<const float> expected_probabilities, std::span<const float> expected_top_weights,
    std::span<const float> expected_top_ids);

struct LayerTailTestResult {
  float dense_norm_max_abs_error{};
  float expert_norm_max_abs_error{};
  float combined_norm_max_abs_error{};
  float layer_output_max_abs_error{};
  float fused_max_abs_error{};
};

LayerTailTestResult test_bf16_layer0_tail(
    std::span<const std::byte> dense_norm_weight,
    std::span<const std::byte> expert_norm_weight,
    std::span<const std::byte> combined_norm_weight,
    std::span<const std::byte> post_attention_norm_weight,
    std::span<const std::byte> layer_scalar,
    std::span<const float> embedding, std::span<const float> attention_output,
    std::span<const float> dense_output, std::span<const float> expert_output,
    std::span<const float> expected_dense_norm, std::span<const float> expected_expert_norm,
    std::span<const float> expected_combined_norm, std::span<const float> expected_layer_output);

}  // namespace gevva
