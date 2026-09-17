#pragma once
#include <cuda_runtime_api.h>

namespace g4 {
// Optional AOT CuTeDSL experiment. Same compact D256 GQA/window contract as
// launch_cudnn_text_prefill; no Python, torch, or compilation at runtime.
void launch_sm120_text_prefill(const void *query, const void *key,
                               const void *value, void *output, int batch,
                               int sequence, int key_sequence,
                               cudaStream_t stream);
} // namespace g4
