# Experimental SM120 fused text prefill

The optional native backend AOT-compiles NVIDIA's SM120 CuTeDSL attention
kernel and links it, plus the static CUDA dialect support library, into `g4`.
Serving does not import Python/torch, JIT-compile kernels, or convert layouts.
The ordinary build does not need CuTeDSL. This is **off by default**.

## Build and run

The tested build-only environment uses Python 3.12, `nvidia-cutlass-dsl[cu13]`
4.7.1 and `cuda-python` 13.3.1:

```sh
uv venv --python 3.12 build/sm120-venv
uv pip install --python build/sm120-venv/bin/python \
  'nvidia-cutlass-dsl[cu13]==4.7.1' 'cuda-python==13.3.1'
cmake -S . -B build \
  -DG4_SM120_DSL_PYTHON="$PWD/build/sm120-venv/bin/python"
cmake --build build -j 4
```

Enable **both** `G4_CUDNN_TEXT_PREFILL=1` (the experimental fused-prefill
dispatch) and `G4_SM120_TEXT_PREFILL=1` (select its native backend). Omitting
the second flag selects cuDNN for A/B comparisons. Omitting both retains the
original attention. Requesting the native backend in a build without its AOT
objects throws an explicit error rather than silently falling back.

All GPU commands must use:

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
./build/g4-sm120-attention-bench 8 513 1537
ctest --test-dir build -R experimental_sm120 --output-on-failure
bash scripts/profile_text_prefill.sh 8 120 native
```

The profiling script captures only warmed `cudaProfilerApi` ranges, never
checkpoint loading. `G4_PROFILE_NATIVE_ATTENTION=1` enables a similarly scoped
capture in the isolated executable. Sanitizer timings are not benchmarks.

## Contract and selection

- Compact BF16 Q/O `[B,Sq,16,256]`, K/V `[B,Sk,8,256]`; B=1..8.
- Sq=1..4608, Sq<=Sk<=5632, matching the existing prefill chunk capacities.
  These are **chunk** limits, not a new model context/output-length limit.
- Scale is already in Q. The kernel receives `log2(e)` for its base-2 softmax.
- Bottom-right causal mask, 1024 tokens including the current token. The
  kernel's inclusive left offset is **1023**, unlike cuDNN's bound of 1024.
- Image-query islands retain the original mask-aware path. Global D512 layers
  and decode attention are unchanged. Text chunks following cached images can
  use the normal causal path, as in the existing cuDNN experiment.
- Both 64x64 and 128x64 Q/KV tilings are built for `sm_120a`. D256's two KV
  shared-memory buffers make a 128-row KV tile too large for this architecture.
  Automatic selection uses 128 Q rows when Sq>=128 and its grid has at least
  256 CTAs; otherwise 64. `G4_SM120_ATTENTION_Q_TILE=64|128` overrides this for
  measurements. This is a simple workload heuristic, not an autotuner.

The adapter checks geometry in C++, allowing one dynamic-shape object per
tile instead of compiling every batch/chunk shape. Module initialization and
per-stream LSE scratch allocation occur on first use; subsequent calls reuse
them. Warm a shape using each selected tile/stream before CUDA graph capture.
TMA descriptors and launch arguments are still constructed on the host, but
there is no per-call allocation, device synchronization, or D2H transfer after
initialization. Separate streams have separate scratch storage.

## Correctness and attribution

The independent isolated oracle uses unrounded QK and double-precision
softmax/accumulation over BF16 inputs on sampled rows, heads, and every batch
entry. It covers window boundaries, tails, and cached prefixes. All output
elements are also checked for finiteness and compared with cuDNN. CUDA graph
replay and agreement between both native tile variants are checked bitwise.

This does **not** establish generated-output equivalence or task quality.
Fused attention changes the original intermediate rounding. Serving A/Bs
produce different tokens; decode tok/s and speculative acceptance changes
must not be attributed to faster decode kernels. No decode kernel changes
are part of this experiment.

Reference: cuDNN frontend `f77fbc3d21be3f24cd0286b9b368105f7c518b8a`,
`python/cudnn/sdpa/fwd/kernels/prefill_f16_sm120.py` (MIT). The device kernel is
used unchanged; `tools/sm120_attention_launch.py` adapts its launch/layout
construction for dynamic shapes. Preserve NVIDIA's copyright and
`external/cudnn-frontend/LICENSE-MIT.txt` when distributing it.

The kernel uses SM120-specific scheduling, TMA loads and `stmatrix`, but still
uses `mma.sync.m16n8k16` tensor-core operations. An `sm80` name alone does not
prove a kernel is slow: use the actual attention and serving A/B measurements
in [benchmarks.md](benchmarks.md), not architecture labels as performance evidence.

The later [Infiyomi bundle evaluation](infiyomi-sm120-evaluation.md) compares
full-corpus serving throughput and quality. It does not isolate this attention
kernel, and image-mask attention retains its fallback.
