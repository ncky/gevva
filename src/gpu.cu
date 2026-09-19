#include "gevva/gpu.hpp"
#include "gevva/tiled_attention.hpp"
#include "gevva/runtime.hpp"
#include "gevva/cudnn_attention.hpp"
#include "gpu/fp8_resources.hpp"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cub/block/block_radix_sort.cuh>
#include <nvtx3/nvToolsExt.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <mutex>
#include <unordered_map>
#include <vector>

// Private CUDA implementation fragments are included in dependency order.
// Keep one CUDA translation unit: no device-link/inlining change in this refactor.
namespace {
#include "cuda/common.cuh"
#include "cuda/configuration.cuh"
#include "cuda/kernels/normalization.cuh"
#include "cuda/kernels/embedding_metadata.cuh"
#include "cuda/kernels/fp8.cuh"
#include "cuda/kernels/vision_attention.cuh"
#include "cuda/kernels/prefill_attention.cuh"
#include "cuda/kernels/decision_attention.cuh"
#include "cuda/kernels/vision_pooling.cuh"
#include "cuda/kernels/routing_sampling.cuh"
#include "cuda/kernels/reference_ops.cuh"
#include "cuda/kernels/decode_attention.cuh"
#include "cuda/experiments/turboquant.cuh"
#include "cuda/kernels/decode_attention_batch.cuh"
#include "cuda/kernels/nvfp4_reference.cuh"
#include "cuda/timing.cuh"
#include "cuda/diagnostics/decode_attention_helpers.cuh"
}  // namespace

namespace gevva {
// Runtime building blocks, then model execution paths.
#include "cuda/diagnostics/decision_attention.cuh"
#include "cuda/runtime/prefix_copy.cuh"
#include "cuda/runtime/vision_weights.cuh"
#include "cuda/runtime/fp8_linear.cuh"
#include "cuda/runtime/fp8_preparation.cuh"
#include "cuda/runtime/decode_dispatch.cuh"
#include "cuda/kernels/qkv_transform.cuh"
#include "cuda/kernels/ragged_attention.cuh"
#include "cuda/runtime/kv_append.cuh"
#include "cuda/runtime/kv_commit.cuh"
#include "cuda/runtime/vocabulary.cuh"
#include "cuda/runtime/decision_logits.cuh"
#include "cuda/runtime/assistant.cuh"
#include "cuda/runtime/prefill.cuh"
#include "cuda/runtime/verifier.cuh"
#include "cuda/runtime/vision_encoder.cuh"

// CLI diagnostics are kept separate from the serving implementation.
#include "cuda/diagnostics/attention_router.cuh"
#include "cuda/diagnostics/rotary.cuh"
#include "cuda/diagnostics/kv_append.cuh"
#include "cuda/diagnostics/decoder_layers.cuh"
#include "cuda/diagnostics/assistant_prefill_layers.cuh"
#include "cuda/diagnostics/verifier_vocabulary_attention.cuh"
#include "cuda/diagnostics/vision_layers.cuh"
#include "cuda/diagnostics/multimodal_prefill.cuh"
#include "cuda/diagnostics/primitives_layers.cuh"
#include "cuda/diagnostics/attention.cuh"
}  // namespace gevva
