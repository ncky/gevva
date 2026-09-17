#include "g4/sm120_attention.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifdef G4_HAVE_SM120_ATTENTION
namespace {
void check_sm120(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("SM120 attention: ") +
                             cudaGetErrorString(status));
}
} // namespace
#define CUTE_DSL_CUDA_ERROR_CHECK(err) check_sm120(err)
#include "attention128.h"
#include "attention64.h"

namespace {
// AOT exports use distinct C ABI names. Keep their naming glue here; all
// resource ownership and dispatch below is shared between tile variants.
#define G4_NATIVE_BACKEND(N)                                                   \
  struct Backend##N {                                                          \
    using Module = g4_sm120_attention##N##_Kernel_Module_t;                    \
    using Q = g4_sm120_attention##N##_Tensor_q_t;                              \
    using K = g4_sm120_attention##N##_Tensor_k_t;                              \
    using V = g4_sm120_attention##N##_Tensor_v_t;                              \
    using O = g4_sm120_attention##N##_Tensor_o_t;                              \
    using Lse = g4_sm120_attention##N##_Tensor_lse_t;                          \
    using Sinks = g4_sm120_attention##N##_Tensor_sinks_t;                      \
    using Qlens = g4_sm120_attention##N##_Tensor_seq_q_lens_t;                 \
    using Klens = g4_sm120_attention##N##_Tensor_seq_kv_lens_t;                \
    static constexpr auto init = _mlir_g4_sm120_attention##N##_cuda_init;      \
    static constexpr auto load =                                               \
        _mlir_g4_sm120_attention##N##_cuda_load_to_device;                     \
    static constexpr auto launch = cute_dsl_g4_sm120_attention##N##_wrapper;   \
  };
G4_NATIVE_BACKEND(64)
G4_NATIVE_BACKEND(128)
#undef G4_NATIVE_BACKEND

template <class Backend> struct NativeAttention {
  typename Backend::Module module{};
  std::mutex mutex;
  // Separate LSE storage for independent streams. Allocate maximum chunk size
  // once; no allocation on subsequent shapes or graph replays.
  std::unordered_map<cudaStream_t, void *> lse;
  NativeAttention() {
    int device{};
    check_sm120(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    check_sm120(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 0)
      throw std::runtime_error("native attention requires SM120");
    // The generated convenience loader visits every visible GPU; deliberately
    // load only the caller's selected device instead.
    cudaLibrary_t *library = &module.module;
    cudaError_t status = cudaSuccess;
    struct {
      cudaLibrary_t **library;
      cudaError_t *status;
    } init{&library, &status};
    Backend::init(reinterpret_cast<void **>(&init));
    check_sm120(status);
    struct {
      cudaLibrary_t **library;
      int *device;
      cudaError_t *status;
    } load{&library, &device, &status};
    Backend::load(reinterpret_cast<void **>(&load));
    check_sm120(status);
  }
  ~NativeAttention() {
    for (const auto &[stream, pointer] : lse)
      cudaFree(pointer);
    cudaLibraryUnload(module.module);
  }
};
template <class Backend>
void launch_native(const void *query, const void *key, const void *value,
                   void *output, int batch, int sequence, int key_sequence,
                   cudaStream_t stream) {
  static NativeAttention<Backend> native;
  std::lock_guard lock(native.mutex);
  auto &scratch = native.lse[stream];
  if (!scratch)
    check_sm120(cudaMalloc(&scratch, 8 * 16 * 4608 * sizeof(float)));
  typename Backend::Q q{const_cast<void *>(query),
                        {batch, sequence},
                        {static_cast<std::int64_t>(sequence) * 4096}};
  typename Backend::K k{const_cast<void *>(key),
                        {batch, key_sequence},
                        {static_cast<std::int64_t>(key_sequence) * 2048}};
  typename Backend::V v{const_cast<void *>(value),
                        {batch, key_sequence},
                        {static_cast<std::int64_t>(key_sequence) * 2048}};
  typename Backend::O o{
      output, {batch, sequence}, {static_cast<std::int64_t>(sequence) * 4096}};
  typename Backend::Lse lse{
      scratch, {batch, sequence}, {16LL * sequence, sequence}};
  typename Backend::Sinks sinks{};
  typename Backend::Qlens qlens{nullptr, {batch}};
  typename Backend::Klens klens{nullptr, {batch}};
  const auto status =
      Backend::launch(&native.module, &q, &k, &v, &o, &lse, &sinks, &qlens,
                      &klens, 1.4426950408889634F, stream);
  if (status)
    throw std::runtime_error("SM120 attention AOT launch error " +
                             std::to_string(status));
}
} // namespace
#endif

namespace g4 {
void launch_sm120_text_prefill(const void *query, const void *key,
                               const void *value, void *output, int batch,
                               int sequence, int key_sequence,
                               cudaStream_t stream) {
#ifdef G4_HAVE_SM120_ATTENTION
  if (batch < 1 || batch > 8 || sequence < 1 || sequence > 4608 ||
      key_sequence < sequence || key_sequence > 5632)
    throw std::runtime_error("invalid SM120 text prefill geometry");
  // 128-row tiles amortize KV loads if there are enough CTAs to occupy the
  // workstation. Otherwise 64 rows avoids underfilling it. No runtime tuning.
  bool large = sequence >= 128 && batch * ((sequence + 127) / 128) * 16 >= 256;
  if (const char *tile = std::getenv("G4_SM120_ATTENTION_Q_TILE")) {
    if (std::strcmp(tile, "64") && std::strcmp(tile, "128"))
      throw std::runtime_error("G4_SM120_ATTENTION_Q_TILE must be 64 or 128");
    large = !std::strcmp(tile, "128");
  }
  if (large)
    launch_native<Backend128>(query, key, value, output, batch, sequence,
                              key_sequence, stream);
  else
    launch_native<Backend64>(query, key, value, output, batch, sequence,
                             key_sequence, stream);
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)batch;
  (void)sequence;
  (void)key_sequence;
  (void)stream;
  throw std::runtime_error(
      "SM120 attention not built: configure G4_SM120_DSL_PYTHON");
#endif
}
} // namespace g4
