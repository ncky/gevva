#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <string>

namespace g4 {
void prepare_serving_tiled_attention();
void launch_serving_tiled_attention(const void* const* queries,
    const void* const* keys, const void* const* values, const int* contexts,
    void* output, float* partial, int batch, int tokens, int dim, int key_tokens,
    cudaStream_t stream);
void capture_attention(const std::string& path, const void* const* queries,
                       const void* const* keys, const void* const* values,
                       const int* contexts, const void* reference,
                       int sessions, int tokens, int dimension, int kv_heads,
                       int key_tokens, bool ring, cudaStream_t stream);
void benchmark_tiled_attention(const std::string& path, int batch, bool ragged,
                               int query_tokens = 0);
}
