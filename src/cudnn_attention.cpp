#include "gevva/cudnn_attention.hpp"
#include <cstdlib>
#include <algorithm>
#include <string>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <cudnn.h>
#include <cudnn_frontend.h>

namespace gevva {
namespace {

namespace fe = cudnn_frontend;

struct SharedCudnnHandle {
  cudnnHandle_t value{};
  SharedCudnnHandle() {
    if (cudnnCreate(&value) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("create shared cuDNN handle failed");
  }
  ~SharedCudnnHandle() { if (value) cudnnDestroy(value); }
};
std::shared_ptr<SharedCudnnHandle> shared_cudnn_handle() {
  // All build/launch operations are serialized by plan_mutex below.
  static const auto handle = std::make_shared<SharedCudnnHandle>();
  return handle;
}

struct SdpaPlan {
  static constexpr std::int64_t kQuery = 1;
  static constexpr std::int64_t kKey = 2;
  static constexpr std::int64_t kValue = 3;
  static constexpr std::int64_t kOutput = 4;

  std::shared_ptr<SharedCudnnHandle> handle_owner = shared_cudnn_handle();
  cudnnHandle_t handle = handle_owner->value;
  std::shared_ptr<fe::graph::Graph> graph;
  void* workspace{};

  SdpaPlan(int batch, int heads, int sequence, int head_dim, float scale,
           bool token_major, int input_token_stride, bool text_prefill = false,
           int key_sequence = 0) {
    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const std::int64_t b = batch, h = heads, s = sequence, d = head_dim;
    const std::vector<std::int64_t> dimensions{b, h, s, d};
    // The vision projections already produce token-major [B, S, H, D].
    // Describe that physical layout directly to cuDNN while retaining SDPA's
    // logical [B, H, S, D] dimensions. This avoids two full-tensor layout
    // kernels around every encoder attention layer.
    const std::int64_t token_stride = input_token_stride > 0
        ? input_token_stride : h * d;
    const std::vector<std::int64_t> input_strides = token_major
        ? std::vector<std::int64_t>{s * token_stride, d, token_stride, 1}
        : std::vector<std::int64_t>{h * s * d, s * d, d, 1};
    const std::vector<std::int64_t> output_strides = token_major
        ? std::vector<std::int64_t>{s * h * d, d, h * d, 1}
        : input_strides;
    auto tensor = [&](const char* name, std::int64_t uid) {
      return graph->tensor(fe::graph::Tensor_attributes()
                               .set_name(name)
                               .set_uid(uid)
                               .set_dim(dimensions)
                               .set_stride(input_strides));
    };
    auto query = tensor("Q", kQuery);
    auto key = tensor("K", kKey);
    auto value = tensor("V", kValue);
    if (text_prefill) {
      const std::int64_t sk = key_sequence > 0 ? key_sequence : s;
      const std::vector<std::int64_t> kv_dimensions{b, 8, sk, d};
      const std::vector<std::int64_t> kv_strides{sk * 8 * d, d, 8 * d, 1};
      key->set_dim(kv_dimensions).set_stride(kv_strides);
      value->set_dim(kv_dimensions).set_stride(kv_strides);
    }
    auto attributes = fe::graph::SDPA_attributes()
                          .set_name("vision_sdpa")
                          .set_generate_stats(false)
                          .set_attn_scale(scale)
                          .set_unfuse_fma(true);
    if (text_prefill)
      attributes.set_causal_mask_bottom_right(true).set_diagonal_band_left_bound(1024);
    auto [output, stats] = graph->sdpa(query, key, value, attributes);
    (void)stats;
    output->set_output(true)
        .set_uid(kOutput)
        .set_dim(dimensions)
        .set_stride(output_strides);
    auto built = graph->build(handle, {fe::HeurMode_t::A});
    if (built.is_bad())
      throw std::runtime_error("build cuDNN SDPA graph failed: " + built.get_message());
    std::int64_t workspace_bytes = 0;
    if (graph->get_workspace_size(workspace_bytes).is_bad())
      throw std::runtime_error("query cuDNN vision SDPA workspace failed");
    if (workspace_bytes > 0 &&
        cudaMalloc(&workspace, static_cast<std::size_t>(workspace_bytes)) !=
            cudaSuccess)
      throw std::runtime_error("allocate cuDNN vision SDPA workspace failed");
  }

  ~SdpaPlan() {
    if (workspace) cudaFree(workspace);
    graph.reset();
  }

  void launch(const void* query, const void* key, const void* value,
              void* output, cudaStream_t stream) {
    if (cudnnSetStream(handle, stream) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("set cuDNN vision SDPA stream failed");
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pointers{
        {kQuery, const_cast<void*>(query)},
        {kKey, const_cast<void*>(key)},
        {kValue, const_cast<void*>(value)},
        {kOutput, output}};
    if (graph->execute(handle, pointers, workspace).is_bad())
      throw std::runtime_error("execute cuDNN vision SDPA failed");
  }
};

std::mutex plan_mutex;
struct CachedPlan {
  std::unique_ptr<SdpaPlan> plan;
  std::uint64_t last_used{};
};
using PlanCache = std::unordered_map<std::uint64_t, CachedPlan>;
PlanCache plans, text_plans;
std::uint64_t plan_clock{}, plan_builds{}, plan_evictions{}, plan_fallbacks{};

std::size_t cache_limit(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (!value) return fallback;
  char* end = nullptr;
  const auto parsed = std::strtoul(value, &end, 10);
  if (end == value || *end || parsed < 1 || parsed > 512)
    throw std::runtime_error(std::string(name) + " must be an integer in 1..512");
  return parsed;
}
std::size_t text_limit() {
  static const auto value = cache_limit("GEVVA_CUDNN_TEXT_PLAN_LIMIT", 64);
  return value;
}
std::size_t vision_limit() {
  static const auto value = cache_limit("GEVVA_CUDNN_VISION_PLAN_LIMIT", 16);
  return value;
}

template <class Factory>
SdpaPlan& cached_plan(PlanCache& cache, std::uint64_t key,
                      std::size_t limit, Factory factory) {
  if (auto found = cache.find(key); found != cache.end()) {
    found->second.last_used = ++plan_clock;
    return *found->second.plan;
  }
  if (cache.size() >= limit) {
    if (!std::getenv("GEVVA_CUDNN_EVICT_PLANS"))
      throw std::runtime_error("uncached cuDNN shape must use bounded fallback");
    const auto oldest = std::min_element(cache.begin(), cache.end(),
        [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
    // Launches are asynchronous. Ensure no evicted graph/workspace remains in
    // flight before releasing it. Hits do not introduce a synchronization.
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("synchronize before cuDNN plan eviction failed");
    cache.erase(oldest);
    ++plan_evictions;
  }
  auto plan = factory();
  auto& inserted = cache.emplace(key, CachedPlan{std::move(plan), ++plan_clock}).first->second;
  ++plan_builds;
  return *inserted.plan;
}

}  // namespace

// Rebuilding an evicted plan retains compiler allocations inside cuDNN on
// this backend. Bound total plan builds, not just the graph map. Existing
// shapes remain fast; callers use their native attention path for other shapes.
bool cudnn_text_plan_available(int batch, int sequence, int key_sequence) {
  if (!key_sequence) key_sequence = sequence;
  const auto key = std::uint64_t(batch) | (std::uint64_t(sequence) << 16) |
                   (std::uint64_t(key_sequence) << 32);
  std::lock_guard lock(plan_mutex);
  if (text_plans.contains(key) || text_plans.size() < text_limit() || std::getenv("GEVVA_CUDNN_EVICT_PLANS")) return true;
  ++plan_fallbacks; return false;
}
bool cudnn_vision_plan_available(int batch, int heads, int sequence, int head_dim,
                                 bool token_major, int input_token_stride) {
  const int stride = input_token_stride > 0 ? input_token_stride : heads * head_dim;
  const auto key = std::uint64_t(batch) | (std::uint64_t(heads) << 8) |
      (std::uint64_t(sequence) << 16) | (std::uint64_t(head_dim) << 40) |
      (std::uint64_t(token_major) << 56) | (std::uint64_t(stride != heads * head_dim) << 57);
  std::lock_guard lock(plan_mutex);
  if (plans.contains(key) || plans.size() < vision_limit() || std::getenv("GEVVA_CUDNN_EVICT_PLANS")) return true;
  ++plan_fallbacks; return false;
}

void launch_cudnn_text_prefill(const void* query, const void* key,
                               const void* value, void* output,
                               int batch, int sequence, cudaStream_t stream,
                               int key_sequence) {
  if (!key_sequence) key_sequence = sequence;
  if (batch < 1 || batch > 256 || sequence < 1 || sequence > 4608 ||
      key_sequence < sequence || key_sequence > 5632)
    throw std::runtime_error("invalid cuDNN text prefill geometry");
  const std::uint64_t geometry = static_cast<std::uint64_t>(batch) |
      (static_cast<std::uint64_t>(sequence) << 16) |
      (static_cast<std::uint64_t>(key_sequence) << 32);
  std::lock_guard lock(plan_mutex);
  auto& plan = cached_plan(text_plans, geometry, text_limit(), [&] {
    return std::make_unique<SdpaPlan>(batch, 16, sequence, 256, 1.0F,
                                     true, 0, true, key_sequence);
  });
  plan.launch(query, key, value, output, stream);
}

void launch_cudnn_sdpa_bshd(const void* query, const void* key,
                            const void* value, void* output, int batch,
                            int heads, int sequence, int head_dim,
                            float scale, cudaStream_t stream,
                            bool token_major, int input_token_stride) {
  const int effective_token_stride = input_token_stride > 0
      ? input_token_stride : heads * head_dim;
  const std::uint64_t key_value =
      static_cast<std::uint64_t>(batch) |
      (static_cast<std::uint64_t>(heads) << 8) |
      (static_cast<std::uint64_t>(sequence) << 16) |
      (static_cast<std::uint64_t>(head_dim) << 40) |
      (static_cast<std::uint64_t>(token_major) << 56) |
      (static_cast<std::uint64_t>(effective_token_stride != heads * head_dim)
       << 57);
  std::lock_guard lock(plan_mutex);
  auto& plan = cached_plan(plans, key_value, vision_limit(), [&] {
    return std::make_unique<SdpaPlan>(batch, heads, sequence, head_dim, scale,
                                     token_major, effective_token_stride);
  });
  plan.launch(query, key, value, output, stream);
}

CudnnPlanCacheStats cudnn_plan_cache_stats() {
  std::lock_guard lock(plan_mutex);
  return {text_plans.size(), plans.size(), text_limit(), vision_limit(), plan_builds, plan_evictions, plan_fallbacks};
}

}  // namespace gevva
