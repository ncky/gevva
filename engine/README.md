# g4

Hardware-specific, torch-free Gemma 4 26B-A4B inference for the RTX PRO 6000
Blackwell in this workstation.

See [the code layout guide](docs/code-layout.md) for implementation ownership,
CUDA fragment ordering, and where to add kernels, runtime paths or diagnostics.

Current milestone: exact GPU guard, native JPEG/PNG preprocessing and Gemma 4
vision tower, direct BF16 and NVIDIA ModelOpt NVFP4 checkpoint validation, and
a native tokenizer. A framework-free SM120 CUTLASS expert block and
correctness-checked sliding/global attention support full multimodal prefill.
The official four-layer MTP assistant runs a native four-draft autoregressive
chain, and the target has an M=5 30-layer verifier with transactional candidate
KV staging and greedy acceptance. See
[docs/architecture.md](docs/architecture.md)
and [docs/benchmarks.md](docs/benchmarks.md). Exact upstream versions are in
[docs/references.md](docs/references.md).

The serving path loads one NVFP4 target checkpoint and its packed expert
sidecar. The original BF16 checkpoint is an offline differential oracle only;
it is never co-resident with the target during serving.

```bash
git clone https://github.com/NVIDIA/cutlass.git external/cutlass
git -C external/cutlass checkout 59e3a3338d516ca6ce0e073af8da65289678a35c
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/g4 gpu-info
./build/g4 image-preprocess /path/to/image.jpg
./build/g4 multimodal-frontend /path/to/image.jpg '<|image|>Describe this image.'
./build/g4 multimodal-prefill /path/to/image.jpg '<|image|>Describe this image.'
./build/g4 multimodal-generate /path/to/image.jpg '<|image|>Describe this image.' 64
./build/g4 multimodal-serve
./build/g4 oai-serve 8080
./build/g4 inspect
./build/g4 inspect --model /mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model
./build/g4 tensor-info model.language_model.layers.0.router.proj.weight
./build/g4 upload-bench
./build/g4 runtime-load-bench
./build/g4 assistant-load-bench
./build/g4 assistant-step-bench 4096
./build/g4 target-batch-bench 8 4096
./build/g4 decoder-layer-batch-bench both 4096
./build/g4 vocab-head-bench assistant
./build/g4 vocab-head-bench target
./build/g4 kv-cache-info 32768
./build/g4 tokenize "Hello world!"
./build/g4 primitive-bench
./build/g4 decode-attention-bench 4096
./build/g4 attention-sublayers-bench 4096
./build/g4 differential
./build/g4 nvfp4-bench --model /mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4
./build/g4 nvfp4-grouped-bench --model /mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4
./build/g4 nvfp4-expert-block-bench
./build/g4 cutlass-expert-bench 0
./build/g4 expert-repeat-test 0 272 8
```

An optional frequency-restricted MTP proposal head can be enabled with a
generated token map. The target verifier remains full-vocabulary; only the
assistant's proposals are restricted:

```bash
G4_ASSISTANT_HOT_VOCAB=assets/gemma4-assistant-hot-vocab-32768.bin \
  ./build/g4 oai-serve 8080
```

Set `G4_ASSISTANT_HOT_VOCAB_MIN_BATCH=8` to use it only for B8. The full and
compact heads are both prepared at startup when a smaller-batch fallback is
possible, so continuous-batching transitions do not reload weights.

One server can select workload-specific maps with
`G4_ASSISTANT_HOT_VOCAB_TEXT` and
`G4_ASSISTANT_HOT_VOCAB_MULTIMODAL`. Current measurements favor the bundled
32K map for text and 64K map for Infiyomi images. A mixed cohort uses the
multimodal map; `G4_ASSISTANT_HOT_VOCAB` remains the fallback for either kind.
When the text IDs are a subset of the multimodal IDs, the server packs one
sidecar with the text set as its prefix and prepares two views over the same
GPU buffers. This avoids duplicate weights and performs no dynamic gathering.

The assistant's gate/up projection pairs are copied once into an aligned
128 MiB sidecar at load and run as one BF16 projection per layer. Set
`G4_DISABLE_ASSISTANT_FUSED_GATE_UP=1` only for unfused differential or
profiling comparisons.

Verifier decode fuses attention post-normalization/residual addition with the
feed-forward/router normalization and FP8 preparation, removing 30 launches
per MTP cycle. `G4_FUSE_ATTENTION_ROUTER=0` selects the unfused comparison.
`./build/g4 attention-router-test` checks the intermediate tensors byte-for-byte;
`./build/g4 target-throughput-sweep 4096 40` runs alternating BS1/BS8 A/B trials
using the production three-/four-draft geometries.

Steady serving decode shares one pinned metadata upload between the assistant
and verifier, reads continuation states directly, and compacts states only
when the active batch changes. The unused final assistant continuation
projection/input preparation is skipped. `G4_DISABLE_DECODE_TRANSFER_CULL=1`
restores the comparison path. `python3 tools/test_oai_refill.py` checks a running
server with 24 mixed-length requests at concurrency eight plus SSE streaming.

The serving verifier now uses parity-tested attention preparation optimizations
by default: short softmax rows retain the original reduction/rounding order,
and attention unpacking is fused with FP8 output-projection preparation.
Ring restoration remains widely parallel inside the fused launch. This removes
30 launches per verifier cycle and improved measured BS1/BS8 decode throughput
about 1–1.6% on the tested text/image workloads without changing output tokens
or draft acceptance. `G4_EXACT_ATTENTION_OPT=0` selects the previous path
(also unset `G4_FAST_GEMM_SOFTMAX` and `G4_FUSED_ATTENTION_UNPACK` if you used
those development overrides). `fast-softmax-test` and `attention-unpack-test`
check 36 and 19 byte-exact cases, including mirrored ring-cache restoration.
This is separate from the numerically different staged-attention experiment.

Structural experiments (all off by default):

- `G4_CUDNN_TEXT_PREFILL=1` replaces sliding text-prefill attention with
  fused cuDNN GQA, including chronological cached-prefix chunks. Global
  D512 attention and any prefill containing image islands keep the existing
  implementation. It changes floating-point rounding and generated tokens;
  quality equivalence is **not established**. At 3,865 prompt tokens, paired
  no-prefix-reuse serving tests reduced prefill time about 40% BS1 and 8% BS8.
  Decode-rate changes are confounded by changed outputs/acceptance.
- `G4_SM120_TEXT_PREFILL=1`, together with `G4_CUDNN_TEXT_PREFILL=1`, selects
  the optional AOT SM120 backend instead of cuDNN. It reduced measured sliding
  attention GPU time by 36% and full BS1/BS8 prefill time by about 5% versus
  cuDNN, but also changes generated tokens. See [build and test instructions](docs/sm120-attention.md).
- `G4_PREPARE_EXPERT_TILES=1` prepares alternative expert GEMM plans sharing
  existing weights/routing buffers. Select `G4_EXPERT_NARROW_TILES`,
  `G4_EXPERT_WIDE_TILES`, or `G4_EXPERT_COOPERATIVE` to compare them. These
  trials have not justified replacing the default. Set
  `G4_EXPERT_COMPARE_BASELINE=1` with `expert-repeat-test` to compare output
  tensors byte-for-byte against the original plan.
- `G4_TUNE_FP8_PREFILL=1` extends initialization-time, byte-checked cuBLASLt
  tuning to larger prefill shapes. It has not demonstrated a serving gain
  and can make first use of a new shape slow. `tools/bench_prefill_plans.py`
  performs process-isolated A/Bs; toggling the flag after initialization
  would not retune existing plans.
- `G4_PRECOMPUTED_ROPE=1` reuses a 192 MiB GPU table across prefill heads and
  layers, covering all 262,144 native positions. Its 32 byte-exact tests and
  serving parity checks pass, but measured prefill benefit was negligible,
  so it remains off. Decode is not switched to the table by this flag.

`text-prefill-attention-test [TOKENS]` checks attention against a CPU oracle;
above 512 tokens it samples causal/window boundary rows. With the cuDNN flag,
`G4_TEST_TEXT_PREFILL_CHUNKS=1` also exercises truncated chronological prefixes.
`bash scripts/profile_text_prefill.sh 8 120 baseline` (or `fused`) profiles a
warmed serving prefill without prefix reuse or checkpoint-load profiling.

Attention experiments can capture real verifier Q/K/V with
`G4_CAPTURE_ATTENTION=/path/to/new/directory` and an optional
`G4_CAPTURE_ATTENTION_MIN_CONTEXT=3000` to exclude startup warmup. Captures
synchronize and copy tensors to disk; never enable them for throughput timing.
`./build/g4 tiled-attention-bench FIXTURE 8 0` replays one capture without loading
the model. `tools/bench_tiled_attention.py` runs sequential BS1/BS8 and ragged
comparisons and saves local JSONL. The tiled kernel is experimental, not used
by serving. See `docs/benchmarks.md` for its numerical and performance limits.
The sweep now selects four query positions for BS1 and five for BS8;
`--bs1-tokens 0` retains the full fixture for older comparisons. The replay
command also accepts a final `QUERY_TOKENS` argument. Asynchronous staging,
compact feature tiles, and query-group reuse are included in the sweep.
`G4_ATTENTION_CHECK_ONLY=1` runs one untimed pass for CUDA sanitizers.
`./build/g4 attention-softmax-test` checks the production ragged softmax
reductions against uniform-probability oracles, including partially filled warps.

`G4_STAGED_ATTENTION=1` opts into the serving experiment: staged sliding
attention at BS1/BS8, plus direct-cache staged global attention at BS1 with
roughly 1K–8K context. Other shapes retain their normal dispatch. This is
**not parity-approved**: paired generation tests changed output tokens and
acceptance, and BS8 short-text/image throughput regressed. Leave it unset
for normal serving (`0` still enables this presence-based switch).
`G4_BATCH_SERVE_SWEEP_ENV=G4_STAGED_ATTENTION` runs a paired comparison;
`G4_SWEEP_ALLOW_TOKEN_MISMATCH=1` lets that benchmark record all failing pairs
instead of stopping at the first mismatch. It does not bypass serving checks
or change which kernels normal serving uses. Detailed local results are in
`docs/benchmarks.md`.

`G4_FUSED_TARGET_GATE_UP=1` enables experimental target FP8 gate/up packing
(about 340 MiB, shared between runners). `G4_PREPARE_FUSED_TARGET_GATE_UP=1`
prepares the sidecar without enabling it, for resident serving A/B tests using
`G4_BATCH_SERVE_SWEEP_ENV=G4_FUSED_TARGET_GATE_UP`. Both switches are off by
default; unlike the assistant's fusion, this has no demonstrated broad serving
win. These switches use presence semantics: unset them to disable.

`differential` consumes small raw oracle tensors generated once with
Transformers. The executable itself remains C++/CUDA-only and never imports
Python or Torch.

`tools/generate_multimodal_generation_oracle.py` runs the original BF16 target
and official assistant offline to record an end-to-end greedy reference. It is
a validation tool only and is never part of the resident serving process.

`multimodal-serve` is a persistent JSON Lines worker: it loads the single NVFP4
target, packed experts, and official MTP assistant once, emits a `ready` record,
then serves requests from standard input without reloading weights. Each request
and response is one line, making the process easy to supervise from an HTTP or
RPC frontend without adding a framework to the GPU runtime:

```json
{"id":"demo","image":"/path/to/image.jpg","prompt":"<|image|>Describe this image.","max_tokens":64}
```

The response includes token IDs, decoded text, per-cycle drafts and target
tokens, MTP acceptance, prompt length, CPU image timing, vision/prefill time,
and post-prefill throughput.
Setting `"profile":true` brackets just that request with the CUDA profiler API.
`scripts/profile_server.sh` first warms the worker, then uses this flag to
capture a clean Nsight Systems trace of the second request.

`oai-serve` provides `/v1/chat/completions`, multimodal data URLs, true SSE
token streaming, continuous batch growth/refill up to eight live requests, and
correct `stop`/`length` finish
reasons. Generation accepts output requests up to the model's native 262144
tokens, bounded per request so prompt plus generated sequence fits the native
context. The final MTP cycle automatically narrows at the boundary. The KV
arena reserves that complete context virtually and commits global-attention
pages on demand.

For the current full-corpus throughput and quality comparison, see the
[2026-09-05 Infiyomi evaluation](docs/infiyomi-sm120-evaluation.md). It uses
strict region-coverage checks and the published five-attempt retry policy.
The quality-qualified Infiyomi profile is
`bash scripts/serve_infiyomi_fast.sh 8080`, with client `max_tokens: 8192`, after building the
[optional SM120 backend](docs/sm120-attention.md). It measured 1.1226 successful
pages/s and 8.371/10 translation quality, versus native default 0.9400 pages/s
and 8.283 at the same cap. This is a workload-specific promotion, not a claim
of identical outputs or universal quality equivalence.

A historical native batch-8 run reported 240/240 pages in 182.54 seconds
(1.3148 pages/s), but returned only 3282/3328 regions and 2993 translations
without retries; that success count is **not comparable to the current strict
coverage result**. A later direct-ring run completed 240/240 while doing
more generated work; normalized by completion tokens it reached 1230 tok/s
versus 1201 tok/s for the clean native reference. The latest 24-page
concurrency-8 smoke completed 24/24 in 12.389 seconds (1.9372 pages/s).
Decode uses direct global and mirrored sliding-KV GEMMs for batches of four or
more, fused QKV transform/attention preparation, fused FP8 scale/norm
boundaries, and a stable single-kernel MoE route sort. CUDA Graph decode is
available with `G4_DECODE_GRAPHS=1` for experiments but is intentionally not the
serving default because padded graph buckets changed long-run output behavior.
Prefill uses one pointer-batched QK and PV call for all 16 query heads instead
of 32 per-head cuBLAS submissions per layer. Compact routed-expert scale storage
also permits one unsplit 4608-token expert dispatch. Throughput peaks around
41.0k sustained target-prefill tok/s at 1536 tokens, and the 285-token
multimodal target pass takes about 14.0 ms after vision.

The production W4A4 expert sidecar is generated offline with the installed
reference environment, then read by the same native safetensors loader:

```bash
/mnt/SSD/qwen38/.venv/bin/python tools/pack_trtllm_experts.py --layers all
/mnt/SSD/qwen38/.venv/bin/python tools/validate_trtllm_experts.py --layer 0
/mnt/SSD/qwen38/.venv/bin/python tools/pack_vocab_int8.py \
  --model /mnt/SSD/g4-models/gemma4-26b-a4b-assistant \
  --tensor model.embed_tokens.weight \
  --output /mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-int8
/mnt/SSD/qwen38/.venv/bin/python tools/pack_nvfp4_vocab.py
/mnt/SSD/gemma4/.venvs/sglang-latest/bin/python tools/pack_nvfp4_vocab.py \
  --source /mnt/SSD/g4-models/gemma4-26b-a4b-assistant \
  --tensor model.embed_tokens.weight \
  --output /mnt/SSD/g4-models/gemma4-26b-a4b-assistant-vocab-nvfp4
```

The current artifact is
`/mnt/SSD/g4-models/gemma4-26b-a4b-trtllm` (30 layers, about 12 GiB). The
converter is not a runtime dependency. `cutlass-expert-bench` exercises the
production expert path: dynamic BF16 to NVFP4 quantization, grouped fused
gate/up, GeGLU and requantization, grouped down projection, and weighted top-8
reduction. CUTLASS is header-only here; none of the reference frameworks are
linked or loaded.

Profile a command with Nsight Systems:

```bash
scripts/profile.sh inspect
scripts/profile_fixed_verifier.sh 8 4096 4
scripts/profile_prefill_layer.sh 0 272
```
