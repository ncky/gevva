# Reference implementations

The runtime is independent; these versions are numerical, kernel, and design
references used during development.

| Project | Version inspected | Used for |
|---|---|---|
| Transformers | 5.15.0 inspected; 5.12.1 local oracle | Gemma 4 equations, dtype semantics, tokenizer/image/full-prefill oracle |
| SGLang | `627980596254` | Gemma 4 integration, ModelOpt loading, Marlin, fused MoE |
| vLLM | 0.24.0 and source `95dc96d1d012` | Scheduling, paged KV, baseline behavior |
| NVIDIA CUTLASS | `59e3a3338d51` (4.8.0) | SM120 block-scaled grouped FP4 GEMM |
| antirez/ds4 | `599e49d25397` | Small explicit native runtime structure |
| PufferLib | `42f70d` | Torch-free CUDA binding and allocator/profile patterns |
| TurboQuant paper | arXiv `2504.19874` / ICLR 2026 | Randomized Hadamard KV transforms, Lloyd-Max scalar quantization, QJL residual correction |
| AmesianX/TurboQuant | `570f5e790b` (MIT) | D=512 Gemma 4 cache format and numerical constants; kernels were independently specialized for SM120 |
| LMDeploy TurboQuant | docs inspected 2026-09-04 | Current production constraints and H200 end-to-end comparison |
| cuBLAS/cuBLASLt | CUDA 13.2/13.3 docs | BF16 and block-scaled FP4 contracts |

The SGLang Marlin benchmark is isolated in
`tools/benchmark_sglang_marlin.py`; it is a development comparison and never
enters the native dependency graph. Adapted code must retain the upstream
license notice. Current g4 kernels were written against documented formats and
validated numerically rather than copied from those implementations.

The experimental D=512 TQ4 benchmark uses the public TurboQuant sign pattern
and Gaussian Lloyd-Max boundaries/centroids from the MIT-licensed AmesianX
implementation. Its CUDA layout and direct attention kernels are native g4
code. It is not yet a serving representation and should not be confused with
the paper's lower-bit production construction, which adds a QJL residual term.

The CPU uint8 bicubic-antialias frontend follows the algorithm in PyTorch's
BSD-licensed `aten/src/ATen/native/cpu/UpSampleKernel.cpp` (Pillow cubic
coefficient, float64 filter generation, fixed-point weights, and two-pass uint8
rounding). It is independently checked against the installed Torchvision image
processor on the local page sample.
