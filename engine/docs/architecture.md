# g4 architecture

`g4` is a narrow, standalone Gemma 4 26B-A4B inference engine for one machine:
Ryzen 7 9800X3D plus the 96 GB RTX PRO 6000 Blackwell at PCI `0000:0f:00.0`.
It must never fall back to the 5090, CPU inference, or another model shape.
Serving loads only the NVFP4 target checkpoint plus its packed expert sidecar;
the 51.6 GB BF16 checkpoint is used solely by offline differential tests and is
not a second resident target or a prefill/decode split.

## Scope and execution plan

1. Directly inspect and memory-map Hugging Face safetensors. Keep tokenizer and
   chat-template compatibility testable against Transformers, but do not link
   Python or torch into the runtime.
2. Establish a BF16 correctness path: embeddings, alternating 1K sliding/full
   GQA (256-d sliding heads, 512-d full heads), QK norm and dual RoPE,
   top-8/128 MoE plus the parallel dense MLP, final norm and
   soft-capped logits. Compare per-layer tensors and greedy token IDs.
3. Add the production Blackwell path around the local ModelOpt NVFP4 weights.
   Focus profiling on fused router/top-k, grouped expert GEMMs, and the five
   global-attention layers whose cost grows with context.
4. Capture decode with CUDA Graphs, use fixed-address arenas, keep KV entirely
   on the PRO 6000, and overlap only transfers that Nsight proves material.
5. Add the official four-layer Gemma 4 MTP assistant and verify up to four
   sequential drafts in one small-batch target pass.
6. Build on the persistent JSONL serving worker with continuous batching,
   paged KV, prefix reuse, and an OpenAI-compatible network frontend.

## Multimodal input

The native image frontend accepts JPEG and PNG, converts to RGB, reproduces
Gemma 4's aspect-preserving resize in multiples of 48, and emits the fixed
patch/position tensors expected by the vision tower. Its bicubic antialias path
matches Torchvision's float64 filter construction, int16 fixed-point weights,
and uint8 rounding after each separable pass. Resize rows and patch output are
independent and run across the 9800X3D's eight physical cores. On the 1126x1600 local page
sample it produces 672x912 pixels, 2394 16x16 patches, and 266 pooled soft
tokens; no resized pixel differs from the Transformers oracle by one 8-bit
level. The native 27-layer BF16 vision tower implements full bidirectional
attention, two-dimensional RoPE, 3x3 spatial pooling/standardization, no-scale
RMSNorm, and the 1152-to-2816 language projection.

Prompt processing expands every user-facing `<|image|>` marker to begin-image,
the exact dynamic number of projected soft-token slots, and end-image. The GPU
embedding frontend substitutes projected image rows for those slots while
scaling ordinary tied token embeddings by BF16 `sqrt(2816)`. Contiguous image
slots receive a shared block id. During target prefill, the block id overlays
bidirectional attention within each image on the causal (and, where applicable,
sliding-window) text mask. The resulting K/V rows are persisted in the same
cache consumed by decode. The complete local sample currently reaches the
first target logit natively, and its greedy token matches the exact BF16
Transformers checkpoint. Video will reuse the image tower frame-wise; this
checkpoint has `audio_config=null`, so audio input is deliberately rejected
rather than silently mishandled.

The generation entry point renders the checkpoint's exact single-user-turn
chat framing (`chat_template.jinja`, no tools and thinking disabled) before
image expansion. It stops on any configured end token (1, 50, or 106), rather
than assuming a single EOS id.

The long-lived `multimodal-serve` worker owns one NVFP4 target representation,
its packed expert sidecar, and the official assistant. It retains the weights,
CUDA/cuBLAS context, 4608-token prefill plans, batched verifier plans, a
262144-token virtually reserved target KV arena, candidate KV, and scratch
pools across newline-delimited JSON
requests. Request-local prompt and generation state is overwritten before use;
two identical back-to-back requests are covered by a repeatability test. A
continuous-batching OpenAI-compatible HTTP/SSE server shares the same runtime.
Image towers remain independent request jobs. CPU decode/resize work may overlap,
but the default GPU path does not fuse unrelated images or force concurrent CUDA
streams: on this GPU, two-stream vision contention had no repeatable cohort-time
win. The throughput policy coalesces only language prefill, where sharing the
transformer weight reads improves the batch-8 makespan. `G4_CONCURRENT_VISION`,
`G4_PAIR_VISION`, `G4_BATCH_VISION`, and `G4_DISABLE_BATCH_PREFILL` are retained
as profiler controls rather than production defaults.
For a reusable cohort prefix, all 25 mirrored sliding layers occupy one
identically laid-out arena. Prefixes through 1024 tokens use one vectorized GPU
kernel per destination to copy only valid canonical rows and reconstruct both
mirrors and the four-row speculative tail. Longer prefixes retain a whole-arena
D2D copy; the five global layers always copy only resident prefix rows. A cache
miss for a lone request does not construct that
cache ahead of its response. It uses ordinary prefill and, when its generated
tail has not wrapped the 1024-row sliding ring, queues the reusable copy only
after the completion callback. A later hit is ordered behind the copy with a
CUDA event. Reusing the source slot is ordered by that event too, preventing a
new prefill from overwriting K/V while the low-priority snapshot reads it. Warm
hits require at least 128 shared tokens; a cold simultaneously admitted cohort
requires 256 before it computes a common prefix once and clones it. Warmed B8
tests with the former full copy found no benefit at 57 tokens and parity near
89; the sparse copy makes 57-token TTFT faster, but the 128-token production
gate remains because shorter cache-prefill geometry reduced MTP acceptance in
the end-to-end throughput workload. At 32025 prompt tokens and B8 this measured 2.08 seconds
to first token versus 11.99 seconds for eight packed recomputations. Set
`G4_DISABLE_SHARED_PREFIX_CACHE=1` for the diagnostic independent-prefill path.
The request that supplies a new cache entry completes before its low-priority
copy begins; neither a clone nor cache population blocks that originating
response. Verifier control/result records and packed-prefill metadata use
persistent pinned staging, while continuation states remain device-resident.
On a cold shared prefix of at least 4096 tokens, slot zero is cloned and
prefilled first, exposing its first streamed token before copies for the other
cohort members are submitted; all sessions then join the same decode batch.
Output requests may consume every context position left by the prompt: the
last MTP cycle narrows from four drafts through zero drafts at the native
boundary rather than reserving or discarding a speculative four-token tail.

Prompt attention preserves the checkpoint's token-major Q/K/V layout and uses
one device kernel to build 16 Q/K/V/output pointer tuples. One pointer-batched
BF16 QK call, the multimodal-aware causal softmax, and one pointer-batched PV
call replace 32 per-head cuBLAS submissions in every target layer. This also
preserves arbitrary prefix positions, the 1024-token sliding window, and
bidirectional image islands. The legacy per-head schedule remains available
only as the `G4_DISABLE_BATCHED_PREFILL_ATTENTION=1` profiling control.

Vision SDPA describes the projection's native physical `[B,S,H,D]` layout
directly to cuDNN, avoiding Q/K/V and output transpose kernels around every
encoder layer. The old contiguous `[B,H,S,D]` staging path remains available as
`G4_CUDNN_PACKED_LAYOUT=1` for A/B profiling.
At runtime, immutable BF16 Q/K/V and gate/up matrices are also packed along
their output axes once, outside the request path. Each vision layer consequently
uses one QKV GEMM and one gate/up GEMM; the following norm/RoPE and GeGLU kernels
consume the packed token-major outputs directly. Separate projections remain
available as `G4_DISABLE_FUSED_VISION_PROJECTIONS=1`.

## Quantized experts

The ModelOpt expert tensors are packed E2M1 with consecutive K values in the
low then high nibble. Each group of 16 weights uses a row-major E4M3 scale and
the projection has an FP32 `weight_scale_2`. A direct-layout W4A16 kernel is the
compact correctness path and beats SGLang Marlin at M=1. Production decode and
prefill offline-fuse the up/gate halves, swizzle scale blocks for the CUTLASS
ABI, dynamically quantize activations, and use SM120 W4A4 tensor cores. The
native implementation executes both grouped GEMMs, GeGLU/requantization, and
route reduction in 35.37 us with device-dynamic expert IDs (31.59 us with
static routes), versus 46.69 us for the framework reference and 96.4 us for the
earlier graph-captured W4A16 block.
The runtime remains independent of Torch, Python, SGLang, and FlashInfer.

For prompt batches, the router's eight choices per token are stably sorted by
expert on device. Up to 320 tokens use one resident block-radix kernel that
also creates the inverse map and all 128 grouped problem descriptors; larger
route sets retain the faster device-wide CUB path. CUB reads the router IDs
directly and uses an immutable route-index vector initialized during startup
warmup, avoiding a per-layer key copy and index-initialization kernel. Its
inverse-map stores are folded into the following activation-quantization
kernel. Two dynamic grouped
CUTLASS launches then process one
variable-height problem per expert, followed by an inverse-route weighted
reduction. In packed prefill, that reduction is fused into the feed-forward
finalizer: each token CTA preserves the exact FP32-to-BF16 reduction boundary
in shared memory and immediately performs the expert/dense normalization and
residual update. This avoids a global expert-output tensor and a separate
kernel per layer. The verifier uses the same exact-boundary fusion to avoid 30
route-reduction launches in every speculative cycle. CUTLASS's device-dynamic grouped scheduler reads the 128 problem
shapes and A/D pointers written by a preceding CUDA kernel; there is no route
count readback, host compaction, metadata re-upload, or GEMM reinitialization in
the request path. The corrected grouped schedule averages about 0.79 ms for a
complete 272-token layer. Each in-flight request must own its grouped
problem metadata, radix-sort workspace, and expert intermediates; those buffers
are intentionally reused only after that request has completed. Activation
scales are packed at each expert's 128-row-aligned route offset instead of
reserving the full route capacity independently for all 128 experts.
The exact-hardware serving session reserves one 4608-token variable-M plan per
layer at startup, so prompt-length changes do not rebuild CUTLASS plans and a
complete tile needs only one route sort and one W13/W2 pair. Longer prompts use
4608-token chunks without splitting a bidirectional image island. A separate
variable-M plan set remains resident for speculative verification.

## Decode attention

KV is laid out token-major, then KV head, then head dimension. Sliding layers
dispatch over only the newest 1024 tokens; global layers use the full prefix.
Sliding K/V uses three mirrored 1024-row rings plus four speculative tail rows.
This lets B4-B8 pointer-array GEMMs read a chronological window directly while
keeping future candidate rows outside canonical history until acceptance. The
existing unpack launch restores touched mirror-tail rows after attention.
For small continuous batches, ragged queries and sliding KV are packed eight
head rows per CTA with 16-byte transactions, followed by BF16 tensor-core QK/PV
and an FP32 softmax. At batch sizes four through eight, global and sliding
attention skip the historical-KV copy: pointer-array GEMMs read token-major
global cache or the chronological mirrored sliding ring directly. Q/K/V scale,
normalization, and RoPE write packed Q and transactional live/candidate K/V in
one kernel, so there is no separate attention-preparation launch. This path
remains active while
`batch * context <= 262144`, so it
covers 32K at batch 8 and nearly the full model context at batch 1 without an
unbounded workspace. Larger products use the bounded-memory tiled score/value
implementation. Both the 16Q/8KV/D256 and 16Q/2KV/D512 geometries are checked
against a CPU BF16 oracle.

TurboQuant is being evaluated only for the five D=512 global layers. The 25
sliding layers retain BF16 because they read at most 1024 rows, where transform
and dequantization overhead cannot amortize. An isolated TQ4 implementation
applies a randomized 512-point Hadamard transform, two 256-value Lloyd-Max
scale blocks, and direct packed KQ/PV consumption without reconstructing a
BF16 cache. It remains outside serving until real-model generation and MTP
transaction tests pass; synthetic timing indicates that it should dispatch
only above a roughly 20--30K context threshold.

The KV allocator uses a single fixed-address arena. Per session, the mirrored
layout occupies about 5.59 GiB at the model's full 262144-token limit and about
1.21 GiB at 32768 tokens. This includes 25 fixed mirrored sliding rings and five
virtually reserved, on-demand-committed full-prefix global caches. A new cache
commits only 8192 global rows and therefore starts at 0.743 GiB regardless of
its reserved maximum.

The production verifier keeps pointer-batched GEMM attention enabled while
`batch_size * maximum_context <= 1048576` rows (overridable with
`G4_GEMM_ATTENTION_MAX_CONTEXT_ROWS`). This covers B8 through 131K and B1
through the native 262144-token limit. At B8/64025, the previous scalar
fallback delivered 19.14 tok/s per request; GEMM attention delivered 72.18
tok/s and stayed repeatable across all sessions. The 96 GiB device's committed
KV and scratch capacity, rather than this dispatch guard, remains the ultimate
limit for multiple simultaneous near-native-length requests.

## Reference policy

- Transformers is the numerical oracle for layer equations, preprocessing,
  tokenizer, chat template, and logits.
- vLLM and SGLang are references for Gemma integration, scheduling, paged KV,
  fused MoE dispatch, and production benchmarking.
- TensorRT-LLM / FlashInfer kernels are references for Blackwell NVFP4 grouped
  GEMM and attention choices.
- antirez/ds4 and current PufferLib are references for a small inspectable native
  runtime, explicit ownership, direct file access, and profiling-first builds.

No reference source is copied blindly: licenses and numerical behavior must be
recorded before code is adapted.

## Baseline already observed on this machine

The existing vLLM run reports 276.7 accepted output tok/s, roughly 1227.5 prompt
tok/s (estimated from TTFT), and 27.7 ms TTFT for a 34-token prompt and 256-token
generation. Its launch used INT8 per-channel target weights, FP8 KV, and the
official Gemma 4 MTP assistant with four speculative tokens. Future comparisons
must also record context length, batch/concurrency, quantization, exact commit,
clocks, power state, assistant time, acceptance length, and verified tokens per
target pass.

## Multi-token prediction

The official assistant has four BF16 dense decoder layers (hidden 1024,
intermediate 8192), three sliding-attention layers and one full-attention layer.
It owns Q and O projections but deliberately owns no K or V projections and no
KV cache. Every layer attends to the target model's most recently exported KV
states of the matching attention type.

For each draft, the runtime concatenates the target embedding of the previous
token (2816 values) with the target-sized state from the previous accepted or
assistant step (2816 values), applies the 5632-to-1024 pre-projection, runs the
four assistant layers, and applies both the tied 262144-token vocabulary head
and the 1024-to-2816 post-projection. Four drafts are autoregressive, but all
use the constant position of the last target-visible token: the shared target
K/V does not advance during an assistant drafting round. The target
then evaluates the accepted prefix plus all candidates together and accepts the
matching run. Consequently, the production target path needs efficient
M=1..5 kernels; multiplying single-token target latency by five would erase the
benefit of speculative verification.

The tied vocabulary tables are unusually large (512 MiB for the assistant and
1.375 GiB for the target). The exact path retains their BF16 GEMVs. An optional
proposal path stores rowwise INT8 weights, obtains four candidates from each
1024-row vocabulary partition, and recomputes all 1024 shortlisted logits from
the original BF16 rows before choosing a token. This is a measured component
choice, not a global INT8 design constraint: it is accepted only while the
BF16-corrected choice matches the exact path on generation tests. FP8 and other
Blackwell-native representations remain candidates if they improve that
speed/acceptance tradeoff.

The assistant NVFP4 proposal head also accepts a frequency-ranked token map.
At load, the runtime gathers the mapped NVFP4 weight and scale rows into a
compact, GPU-resident sidecar. Draft argmax runs in compact index space, exact
BF16 correction uses the corresponding original embedding rows, and the final
IDs are remapped before target verification. The target vocabulary is never
trimmed. `G4_ASSISTANT_HOT_VOCAB` supplies the little-endian uint32 map;
`G4_ASSISTANT_HOT_VOCAB_MIN_BATCH` can restrict its use to wider live batches.
Both compact and full plans remain resident so switching at a batch tail is a
GPU-only policy decision.
Separate `G4_ASSISTANT_HOT_VOCAB_TEXT` and
`G4_ASSISTANT_HOT_VOCAB_MULTIMODAL` maps allow one resident runtime to apply
workload-specific proposal policies. A cohort containing any image uses the
multimodal map; otherwise it uses the text map. The generic map is the fallback
when a specialized map is absent. If the text map is a subset of the
multimodal map, load constructs one compact sidecar ordered with the text map
as a prefix. Two preinitialized CUTLASS plans share its input, logits, map, and
weight buffers, so switching policy is a pointer/row-count choice and never a
runtime gather or model reload.

Serving resolves the assistant's 48 device tensor addresses once at load and
retains the two shared-KV descriptor arrays for each active cache cohort. It no
longer constructs tensor names, searches the safetensors index, or uploads
unchanged layer-28/layer-29 descriptors between draft tokens. Draft length is
runtime policy rather than architecture: `G4_MTP_DRAFTS=0..4` overrides every
batch, while `G4_MTP_DRAFTS_B1` through `G4_MTP_DRAFTS_B8` tune individual
live-batch sizes. The measured default is three drafts at B1 and four at
B2--B8. The exact target still verifies every proposal and supplies the first
rejected token.

Uniform B1--B8 assistant cohorts read the target K/V arenas directly with two
pointer-batched GEMMs per attention layer. Queries are packed by grouped-query
head, while keys and values remain in their persistent global or mirrored
sliding-cache layout; no K/V repack is performed. Ragged cohorts retain the
grouped scalar path. `G4_DISABLE_ASSISTANT_GEMM_ATTENTION=1` is the profiling
fallback.

The OpenAI scheduler gives continuous mode eight reusable slots even when a
wave starts at B1. It polls the request queue at every MTP boundary, prepares a
late arrival asynchronously, prefills it behind the dense survivor rows, and
then grows the active cohort without waiting in the source request. Direct
`generate_batch` calls still allocate and return exactly their requested row
count.

The 22-projection assistant E4M3 sidecar is supported only as the opt-in
`G4_ASSISTANT_FP8=1` diagnostic. Dynamic activation quantization made its tiny
M projections slower on this GPU and reduced acceptance on tested text. BF16
therefore remains the production drafter representation. Likewise,
`G4_VOCAB_CANDIDATES_PER_TILE=1|2|4` can reduce the exact correction shortlist
from 1024 to 256/512 entries, and `G4_ASSISTANT_MIN_ROWS=1..8` can pad physical
assistant rows, but the accuracy/serving sweeps retain their defaults of four
candidates per tile and no padding.

The production BF16 path copies each layer's gate and up matrices once into an
aligned 16,384-row sidecar and writes an interleaved result consumed by a
packed GELU-multiply kernel. This removes four cuBLAS projection calls per
draft (sixteen per four-draft cycle) without runtime concatenation. A direct
view of the checkpoint rows was functionally correct but 5.5% slower because
their arena offsets are only 2--8-byte aligned; the 128 MiB aligned copy is a
measured throughput tradeoff. The original tensors remain available, and
`G4_DISABLE_ASSISTANT_FUSED_GATE_UP=1` selects the unfused diagnostic path.

Quantization policy is deliberately mixed. NVFP4 expert input multipliers are
offline calibrated and their conversions are fused into route packing and
GeGLU. Target FP8 uses live per-row activation scales, normally fused into
RMSNorm, router, and GeGLU kernels. Attention-output scale traces varied by
more than three orders of magnitude even within one short generation, so a
fixed load-time scale would either clip outliers or waste most E4M3 precision.
The INT8 vocabulary head has one dynamic activation conversion per verifier
pass; at about 2.9 microseconds it is too small to justify a less robust fixed
scale.

The target and assistant vocabulary projections additionally have static
offline NVFP4 sidecars. Dynamic block-16 activation quantization remains
necessary, but neither tied embedding is requantized while serving. The W4A4
projection generates four candidates from each native 2048-row GEMM group (512
total per row); every finalist is recomputed against the exact BF16 embedding
before argmax. This halves shortlist-extractor blocks relative to two finalists
per 1024 rows. A four-round hierarchical maximum reduction selects those
finalists without fully radix-sorting each group;
`G4_DISABLE_NVFP4_VOCAB_REDUCTION_TOP=1` restores the radix implementation and
`G4_DISABLE_NVFP4_VOCAB_GROUPED_TOP=1` restores the split groups for
differential testing. `G4_NVFP4_VOCAB_CANDIDATES_PER_TILE=1|2|4` retains the
earlier measured sweep: one candidate is faster but changed a real B8 text
continuation, so two is the production finalist density. The INT8 path stays
available for unsupported tail shapes or with `G4_DISABLE_NVFP4_VOCAB=1`.
The same exact-BF16 correction contract applies to the assistant's
262144-by-1024 head; `G4_DISABLE_ASSISTANT_NVFP4_VOCAB=1` restores its INT8
fallback for differential runs.

Speculative target K/V must be transactional. In particular, committing five
candidate rows to a 1024-entry sliding ring before acceptance can destroy up to
four old rows that are still live after an early rejection. The verifier reads
old cache plus separate candidate K/V during its M=5 pass. Production will
retain the per-layer candidate rows in a 1.07 MiB staging arena. The native
runtime now exposes staging and prefix-commit operations; the verifier benchmark
commits all five rows to measure the all-accepted path, while generation will
pass the acceptance result's output count and copy only that prefix into target
cache. Global-cache rows beyond the accepted length likewise remain uncommitted.

## Serving plans and admission

Checkpoint tensor names are resolved once by `RuntimeWeights`; production
prefill and decode consume an immutable table of device addresses. Verifier
scratch similarly retains its layer-major live/candidate KV descriptor tables
and uploads them only when the active cache cohort changes. All legal B1--B8
verification and shortened tail cycles map to three prebuilt FP8 projection
plans (16, 32, and 64 physical rows), avoiding duplicate per-row-count
descriptors and workspaces.

Fixed serving arenas, verifier runners, and expert dispatch objects are created
before the OpenAI server announces readiness. The target INT8 vocabulary paths
for B1--B8 and tiny benchmark-shaped B1/B8 text cohorts are also run against
isolated request caches at startup. They initialize the 34- and 272-row FP8
prefill plans, pointer-batched attention, assistant, and verifier geometry. This
moves library JIT and heuristic setup out of the first client query without
introducing a second resident model; the same cache slots are overwritten by
the first request.

The HTTP scheduler tracks sockets that are already parsing requests. An
isolated request receives a five-millisecond coalescing grace period, while a
known concurrent group may wait up to 49 ms for its request bodies to join the
initial cohort. This is especially important for inline base64 images: queue
length alone otherwise launches B1 while the other seven sockets are still
decoding JSON. Once admitted, continuous refill supports every active batch
size from one through eight as requests finish at different times. A completely
drained cohort returns admission to the HTTP scheduler instead of allowing its
last request to capture the first member of the next concurrent wave.

Refill polling is nonblocking. Completed slots remain vacant and are checked
again at each later MTP boundary, so a concurrency-limited client can enqueue a
replacement after receiving its streamed completion without imposing a host
sleep on requests that are still decoding. Prefix-cache construction follows
the same rule: the originating request's completion callback runs before the
low-priority device-to-device copy is submitted; only a later cache user or
reuse of the source slot waits for that copy's completion event.
