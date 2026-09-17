# Code layout

The engine targets one Gemma 4 configuration and one GPU. Keep that specificity
explicit: prefer small, named execution paths over a general backend framework.

## Entry points and serving

- `src/main.cpp`: CLI dispatch, diagnostics and benchmark commands.
- `src/openai_server.cpp`: HTTP/OpenAI-compatible requests and SSE responses.
- `src/service.cpp`: request preparation, continuous batching, prefix reuse,
  prefill/decode orchestration and per-request state.
- `src/runtime.cpp`: model/weight ownership, KV caches and scratch storage.
- `src/image.cpp`, `src/multimodal.cpp`, `src/tokenizer.cpp`: input preparation.
- `src/model_config.cpp`, `src/safetensors.cpp`: checkpoint configuration/I/O.

## GPU implementation

`src/gpu.cu` is the dependency-ordered CUDA assembly file, not the place to
add new kernel or benchmark bodies. Its private implementation fragments live
under `src/cuda/`:

| Directory/file | Responsibility |
| --- | --- |
| `common.cuh`, `configuration.cuh` | Internal helpers and dispatch controls |
| `kernels/` | Normalization, quantization, routing, QKV, attention and vision kernels |
| `runtime/fp8_*.cuh`, `runtime/vocabulary.cuh` | Projection plans and activation preparation |
| `runtime/kv_*.cuh`, `runtime/prefix_copy.cuh` | KV append/commit and prefix operations |
| `runtime/assistant.cuh` | Draft-model execution |
| `runtime/prefill.cuh` | Single/batched target prefill |
| `runtime/verifier.cuh` | Target speculative verification |
| `runtime/vision_*.cuh` | Vision weight packing and encoder execution |
| `diagnostics/` | Correctness oracles and isolated benchmark implementations |
| `experiments/` | Standalone experimental representations, currently TurboQuant |
| `timing.cuh` | CUDA event timing used by execution and benchmarks |

These `.cuh` files are **implementation fragments, not public or standalone
headers**. Include them only from `src/gpu.cu`, in its indicated namespace.
Shared includes belong above those namespaces. Kernel definitions precede
launchers, model execution follows its building blocks, and public diagnostics
come last. The anonymous-namespace diagnostic templates need early visibility
to their public wrappers.

CUDA remains one translation unit for now, preserving kernel visibility and
avoiding a simultaneous change to device linking/inlining. This split improves
navigation, not CUDA rebuild granularity. Ninja tracks changes in the fragments
through compiler-generated dependencies. A future separate-compilation change
should introduce explicit launch interfaces and benchmark its effect rather
than including these files independently or enabling relocatable device code
incidentally.

Host-only GPU responsibilities are independently compiled:

- `src/gpu/device.cpp`: PCI/architecture guard and device selection.
- `src/gpu/execution_context.cpp`: streams, events and graph ownership/replay.
- `src/gpu/weight_arena.cpp`: pinned-staging uploads and weight-arena lifetime.
- `src/gpu/cuda_errors.hpp`: private host-side CUDA/cuBLAS error translation.

Other GPU backends remain self-contained: `src/nvfp4_cutlass.cu` owns expert
and vocabulary GEMM plans; `src/cudnn_attention.cpp` owns cuDNN plans;
`src/tiled_attention.cu` owns the staged-attention experiment;
`src/sm120_attention.cpp` owns the optional [AOT native-prefill backend](sm120-attention.md).
Its offline exporter/launch adapter and isolated benchmark live in `tools/`.
Public API
declarations remain under `include/g4/`.

Some execution entry points still have historical `benchmark_*` names despite
being used by serving (notably target prefill and vision encoding). Classify
them by callers and responsibility, not by the name alone. Numerical experiments
that plug directly into a production kernel stay alongside that kernel behind
their existing switches; directory placement does not enable an experiment.

## Development checks

Use `rg` over `src/` to find implementations, not just `src/gpu.cu`. Keep GPU
tests sequential and select the RTX PRO 6000 UUID as documented in the profiling
scripts. A structural change should preserve correctness/sequence checks;
benchmark improvements must be established separately. No speedup is implied
by splitting source files.

`tools/` contains offline preparation, differential and benchmarking helpers;
`scripts/` contains repeatable profiling entry points. Local generated results
belong in `benchmarks/` and `profiles/`, not among implementation files.
