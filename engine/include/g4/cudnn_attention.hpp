#pragma once

#include <cuda_runtime_api.h>

namespace g4 {

// Experimental Gemma sliding text prefill: Q=[B,S,16,256], KV=[B,S,8,256].
// KV may include a chronological prefix; queries align to its right edge.
// No image islands; causal 1024-token window, scale already in Q.
void launch_cudnn_text_prefill(const void* query, const void* key,
                               const void* value, void* output,
                               int batch, int sequence, cudaStream_t stream,
                               int key_sequence = 0);

// Executes non-causal BF16 SDPA. token_major selects contiguous physical
// [B, S, H, D]; the diagnostic fallback uses contiguous [B, H, S, D].
// Plans are cached by geometry and reused across all vision layers.
void launch_cudnn_sdpa_bshd(const void* query, const void* key,
                            const void* value, void* output, int batch,
                            int heads, int sequence, int head_dim,
                            float scale, cudaStream_t stream,
                            bool token_major = true,
                            int input_token_stride = 0);

}  // namespace g4
