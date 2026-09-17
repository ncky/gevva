#include "g4/cudnn_attention.hpp"
#include "g4/sm120_attention.hpp"
#include <cstdlib>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <cudnn.h>
#include <cudnn_frontend.h>

namespace g4 {
namespace {

namespace fe = cudnn_frontend;

struct SdpaPlan {
  static constexpr std::int64_t kQuery = 1;
  static constexpr std::int64_t kKey = 2;
  static constexpr std::int64_t kValue = 3;
  static constexpr std::int64_t kOutput = 4;

  cudnnHandle_t handle{};
  std::shared_ptr<fe::graph::Graph> graph;
  void* workspace{};

  SdpaPlan(int batch, int heads, int sequence, int head_dim, float scale,
           bool token_major, int input_token_stride, bool text_prefill = false,
           int key_sequence = 0) {
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("cudnnCreate vision SDPA failed");
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
    if (handle) cudnnDestroy(handle);
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
std::unordered_map<std::uint64_t, std::unique_ptr<SdpaPlan>> plans;
std::unordered_map<std::uint64_t, std::unique_ptr<SdpaPlan>> text_plans;

}  // namespace

void launch_cudnn_text_prefill(const void* query, const void* key,
                               const void* value, void* output,
                               int batch, int sequence, cudaStream_t stream,
                               int key_sequence) {
  if (!key_sequence) key_sequence = sequence;
  if (std::getenv("G4_SM120_TEXT_PREFILL")) {
    launch_sm120_text_prefill(query, key, value, output, batch, sequence,
                             key_sequence, stream);
    return;
  }
  if (batch < 1 || batch > 8 || sequence < 1 || sequence > 4608 ||
      key_sequence < sequence || key_sequence > 5632)
    throw std::runtime_error("invalid cuDNN text prefill geometry");
  const std::uint64_t geometry = static_cast<std::uint64_t>(batch) |
      (static_cast<std::uint64_t>(sequence) << 8) |
      (static_cast<std::uint64_t>(key_sequence) << 24);
  std::lock_guard lock(plan_mutex);
  auto& plan = text_plans[geometry];
  if (!plan)
    plan = std::make_unique<SdpaPlan>(batch, 16, sequence, 256, 1.0F,
                                     true, 0, true, key_sequence);
  plan->launch(query, key, value, output, stream);
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
  auto& plan = plans[key_value];
  if (!plan)
    plan = std::make_unique<SdpaPlan>(batch, heads, sequence, head_dim, scale,
                                     token_major, effective_token_stride);
  plan->launch(query, key, value, output, stream);
}

}  // namespace g4
