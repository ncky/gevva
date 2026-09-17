# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Dense dynamic-shape AOT adapter for cuDNN frontend's SM120 kernel.

Adapted from prefill_f16_sm120.py's __call__. Geometry is checked by the C++
caller rather than the original static-shape Python validation. Device math
and masks are unchanged. This module is used at build time only.
"""
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda


@cute.jit
def launch(kernel: cutlass.Constexpr, q: cute.Tensor, k: cute.Tensor,
           v: cute.Tensor, o: cute.Tensor, lse: cute.Tensor,
           sinks: cute.Tensor, seq_q_lens: cute.Tensor,
           seq_kv_lens: cute.Tensor, softmax_scale_log2: cutlass.Float32,
           stream: cuda_driver.CUstream):
    layout = cute.make_layout(
        (k.shape[0], 8, kernel.tma_swizzle_chunks, k.shape[1],
         kernel.tma_swizzle_chunk_elems),
        stride=(k.shape[1] * 8 * 256, 256,
                kernel.tma_swizzle_chunk_elems, 8 * 256, 1))
    box = (1, 1, kernel.tma_swizzle_chunks, kernel.kv_tile,
           kernel.tma_swizzle_chunk_elems)
    tk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, layout), box_dims=box,
        stride_order=(4, 3, 2, 1, 0), swizzle=kernel.tma_swizzle)
    tv = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, layout), box_dims=box,
        stride_order=(4, 3, 2, 1, 0), swizzle=kernel.tma_swizzle)
    kernel.kernel(q, k, v, o, lse, sinks, seq_q_lens, seq_kv_lens,
                  tk, tv, softmax_scale_log2).launch(
        grid=((q.shape[1] + kernel.q_tile - 1) // kernel.q_tile, q.shape[0], 16),
        block=(kernel.threads_per_cta, 1, 1), stream=stream,
        min_blocks_per_mp=1)
