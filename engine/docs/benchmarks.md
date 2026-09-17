# Development benchmarks

Measurements below are from the target NVIDIA RTX PRO 6000 Blackwell
Workstation Edition (SM120), CUDA 13.2, on 2026-09-03. They are kernel event
times after warmup unless noted otherwise.

## Current serving latency by context

On 2026-09-04, the production eager-safe path was measured after explicit
text and image warmups. Each request generated 64 tokens from a normal prose
instruction. The B1 prompts were unique, so these are not prefix-cache hits.
Decode rate includes actual MTP acceptance and consequently varies with output
content rather than only with context length.

| B1 input | Prompt tokens | TTFT | Target prefill | Decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| Text | 137 | 14.24 ms | 12.89 ms | 344.8 | 51.2% |
| Text | 521 | 20.06 ms | 17.90 ms | 311.9 | 44.0% |
| Text | 1,961 | 57.08 ms | 52.34 ms | 282.3 | 41.5% |
| Text | 3,881 | 143.07 ms | 134.54 ms | 292.7 | 45.5% |
| Image + text | 615 | 77.60 ms | 21.34 ms | 360.9 | 57.3% |
| Image + text | 967 | 81.03 ms | 28.05 ms | 309.6 | 48.2% |
| Image + text | 2,023 | 103.96 ms | 49.81 ms | 328.5 | 52.5% |
| Image + text | 3,783 | 195.25 ms | 138.48 ms | 376.3 | 65.2% |

The multimodal rows use the same 1654x1170 Manga109 page and 560 visual soft
tokens. Its isolated vision encoder was stable at 36.22 ms; remaining TTFT
includes JPEG decode, preprocessing, and scheduling.

The controlled B8 run admitted all eight requests together and used an
identical text prefix, exercising the production compute-once-and-clone KV
path. Aggregate decode rate is eight times the reported per-request rate
because the identical requests retained a complete B8 cohort.

| B8 input | Prompt tokens/request | Cohort TTFT | Per-request tok/s | Aggregate tok/s |
|---|---:|---:|---:|---:|
| Text | 121 | 25.17 ms | 268.2 | 2,146 |
| Text | 505 | 34.46 ms | 283.7 | 2,269 |
| Text | 1,945 | 64.86 ms | 228.8 | 1,830 |
| Text | 3,865 | 127.17 ms | 183.9 | 1,471 |
| Image + text | 592 | 484.32 ms | 285.1 | 2,281 |
| Image + text | 944 | 503.93 ms | 227.6 | 1,821 |
| Image + text | 2,000 | 546.02 ms | 235.8 | 1,887 |
| Image + text | 3,760 | 618.56 ms | 232.3 | 1,859 |

Eight independent vision encodes dominate multimodal cohort TTFT. Previous
paired-stream and whole-batch experiments did not improve their makespan, so
the server retains serial independent vision work before packed language
prefill.

The same pass removed decode-time tensor-name resolution, caches layer-major
KV descriptor uploads until the cohort changes, and lets the result commit
kernel initialize its own output. Fixed five-token target verification at
context 4096 is now approximately 6.51 ms at B1 (direct sliding KV) and
8.86--8.87 ms at B8, with unchanged sequence hashes. Fresh traces are in
`profiles/invariant-b1-current` and `profiles/invariant-b8-current`.

## Independent-request scheduling check

On 2026-09-04, the 1126x1600 page sample was measured through the current
batch-8 serving frontend with one output token. Serial independent vision jobs
produced 506.9 and 521.2 ms cohort TTFT samples; two independent CUDA streams
produced 512.0 and 504.9 ms. The difference is noise-level, while each contended
vision job stretched from roughly 36--48 ms to as much as 96 ms, so concurrent
vision is not a default. Whole-batch and paired vision likewise remain profiler
experiments.

Language prefill has a different tradeoff. Eight independent prefills emitted
their first token at 441.4 ms and their last at 581.0 ms. Packed prefill emitted
all eight at roughly 507--521 ms, improving batch makespan while giving up the
earliest single response. The throughput server therefore retains packed
prefill; latency-oriented scheduling can use smaller ready microbatches.

Native-layout cuDNN vision SDPA removed 54 full-tensor transpose launches per
image without changing the projected checksum. At 2394 patches, isolated
encoder time improved from roughly 17.73 ms to 16.92 ms (about 4.6%); at the
larger 4788-patch serving shape the change was approximately neutral. The trace
is `profiles/vision-native-layout/g4.nsys-rep`.

Packing Q/K/V and gate/up weights once at runtime then reduced projection
submissions by another 81 per image. Three alternating 2394-patch A/B pairs
measured 14.36--14.43 ms fused versus 16.77--16.98 ms separate, with the exact
same projected checksum (`6245.99755859`). At the 4788-patch serving shape,
vision fell from 37.3--38.5 ms to 33.8--34.6 ms. Independent-image batch-8 TTFT
fell from 509--516 ms to 484--487 ms; no cross-request vision fusion was used.
The launch trace is `profiles/vision-fused-projections/g4.nsys-rep`: across its
seven measured iterations, kernel count fell from 2688 to 2121, exactly 81
launches per image.

## Infiyomi batch-8 result

The native OpenAI-compatible server was run from a clean process against all
240 Manga109 pages at concurrency 8 with four-token MTP and a 4096-token output
cap. After the Nsight-guided decode pass, the server completed 240/240 in
182.540 s, or 1.3148 successful pages/s. It returned 3282/3328 region IDs
(98.62%) and 2993 non-empty translations (89.93%) with no retries. The run is
stored under `profiles/infiyomi-fused-route-full240-reference`. The previous
native result was 203.598 s / 1.1788 pages/s, and the published reference
completed 239/240 in 203.015 s / 1.177 pages/s. The new result is 11.5% faster
than both rates. A separate run produced more text: 237756 completion
tokens in 207.140 s (1148 aggregate completion tok/s), versus the reference's
231159 in 203.015 s (1139 tok/s). Both comparisons are retained because greedy
quantized trajectories change total generated work.

The fixed-work verifier benchmark removes that trajectory variance. At batch
8 and context 4096, a complete five-token target verification pass now takes
9.458 ms with tensor attention, versus 23.62 ms with the scalar diagnostic
path. At context 8187 it takes 12.25 ms, and at context 12000 it takes 15.11 ms.
The tensor path is enabled while `batch * context <= 262144`; larger products
fall back to the bounded-memory attention implementation. A 249912-token
prompt plus generation completed successfully in 85.07 s.

| Operation | Shape | Time | Notes |
|---|---:|---:|---|
| Native JPEG decode | 1126x1600 RGB sample | 4.09 ms wall | Warm serving request |
| Native image preprocessing | 1126x1600 RGB -> 2394 x 768 patches | 4.43 ms wall | Eight-core exact resize/patchify; Torchvision uint8 differential unchanged |
| Native vision patch projection | 2520 x 768 -> 1152 | 28.2 us | BF16 position embedding included |
| Native vision encoder | 27 layers, 2520 patches -> 266 x 2816 | 30.25 ms | Canonical padded shape; full image projection |
| Multimodal embedding composition | 272 x 2816, 266 image slots | 2.47 us | Exact scaled-text/image-slot differential |
| Multimodal prefill attention | 12-token semantic test, both attention types | 84 us each | Causal plus bidirectional image islands; max error 1.2e-7 |
| Native target prefill layer | 272 tokens | about 0.79 ms average | Device-dynamic sorted W4A4 path; no per-layer host synchronization |
| Native vision + target first logit | 285 tokens / 266 image tokens | about 41 ms warm | about 27 ms vision + 14.0 ms target; oracle token match and bit-repeatable |
| Native multimodal speculative generation | 16 total / 15 post-prefill tokens, 6 MTP cycles | 79.92 ms decode components | 9 accepted drafts; 187.7 post-prefill tok/s |
| Persistent serving request (first-use) | same image, 16 total tokens | 246.86 ms wall | First request after startup; includes CUDA/CUTLASS first-use work |
| Persistent serving request (warm) | same image, 16 total tokens | 142.51 ms wall | Includes native image decode/preprocessing, vision, prefill, and MTP decode |
| BF16 RMSNorm | 64 x 2816 | 3.66 us | One block per row |
| BF16 layer-0 sliding Q projection | 1 x 2816 -> 4096 | 6.58 us | cuBLAS |
| Native sliding decode attention | context 1024, 16 Q / 8 KV, D=256 | 18.5 us | BF16 differential; physical 1024-entry ring remains flat beyond window |
| Native global decode attention | context 4096, 16 Q / 2 KV, D=512 | 47.3 us | Tiled QK/softmax and 16-way split PV reduction |
| Native global decode attention | context 16384, 16 Q / 2 KV, D=512 | 171.6 us | Same correctness-checked split reduction |
| Experimental TQ4 global attention | context 4096, 16 Q / 2 KV, D=512 | 48.5 us | 47.4 us BF16; no short-context gain |
| Experimental TQ4 global attention | context 32768, 16 Q / 2 KV, D=512 | 367.7 us | 493.2 us BF16; 1.34x faster |
| Experimental TQ4 global attention | context 65536, 16 Q / 2 KV, D=512 | 720.7 us | 1238.9 us BF16; 1.72x faster |
| Experimental TQ4 global attention | context 262144, 16 Q / 2 KV, D=512 | 4611.8 us | 7139.1 us BF16; 1.55x faster |
| Real sliding attention sublayer | context 4096 (1024 attended) | 51.2 us | Input norm + QKV + RoPE/cache + attention + O, CUDA Graph |
| Real global attention sublayer | context 4096 | 86.8 us | Same full path; eager device schedule was faster than isolated graph (90.6 us) |
| Native NVFP4 expert gate | 1 x 2816 -> 704 | 3.72 us | Checkpoint layout, W4A16 |
| Native NVFP4 expert down | 1 x 704 -> 2816 | 3.82 us | Checkpoint layout, W4A16 |
| Native NVFP4 eight static gates | 8 independent projections | 29.7 us | Eight locality-preserving launches |
| Native NVFP4 eight routed gates | Device-selected from contiguous pack | 31.7 us | No CPU routing round trip |
| Native NVFP4 expert block | Top-8 gate, up, GELU, down, reduce | 98.9 us | 96.4 us with CUDA Graph |
| Native SM120 CUTLASS fused W13 | 8 x (1 x 2816 -> 1408) | 14.59 us | Real layer-0 sidecar, grouped W4A4 |
| Native activation quant + fused W13 | BF16 token, eight expert-specific scales | 15.86 us | Quantization adds 1.27 us |
| Native SM120 CUTLASS expert block | Top-8 quant, W13, exact GeGLU/requant, W2, reduce | 31.59 us | Static-route lower bound, real weights |
| Native SM120 dynamic expert runner | Same block, router-selected device IDs | 35.37 us | Graph-safe; no CPU routing round trip |
| Native exact MTP assistant draft | context 128, four layers + full logits/argmax | 667.7 us | CUDA Graph; target-owned zero KV benchmark |
| Native exact MTP assistant draft | context 4096, four layers + BF16 logits/argmax | 768.0 us | Greedy token matches Transformers oracle |
| Native corrected MTP assistant draft | context 4096, four layers + compressed proposal head | 620.5 us | BF16-corrected token matches Transformers oracle |
| Native exact MTP assistant draft | context 16384, four layers + full logits/argmax | 975.0 us | Global shared-KV attention scales with context |
| MTP vocabulary head + greedy argmax | 1 x 1024 -> 262144 | 335.0 us | Exact BF16 tied embedding; two-stage argmax |
| Corrected assistant vocabulary head | 1 x 1024 -> 262144 | 191.2 us | Rowwise INT8 proposal, 1024 BF16 corrections; 32/32 exact-token trials |
| Corrected target vocabulary head | 1 x 2816 -> 262144 | 477.4 us | 32/32 after correction; raw proposal was 31/32 |
| Corrected target vocabulary head | 5 x 2816 -> 262144 | 565.1 us | One batched INT8 proposal GEMM; five BF16-corrected choices |
| Native M=5 decode attention | sliding, context 4096 | 36.3 us | Separate candidate KV; bit-exact to CPU BF16 oracle across ring wrap |
| Native M=5 decode attention | global, context 4096 | 180.5 us | Separate candidate KV; bit-exact to CPU BF16 oracle |
| Native target layer, M=5 | sliding, context 4096 | 392.6 us | Batched dense, attention, and one 40-route grouped MoE; checksum unchanged |
| Native target layer, M=5 | global, context 4096 | 522.2 us | Batched dense, attention, and one 40-route grouped MoE; checksum unchanged |
| Native four-draft assistant chain | context 4096 | 2462.8 us | Real autoregressive dependencies and target embedding gathers; four-token Transformers match |
| Native target verifier, batch 8 | M=40, context 4096 | 9458.1 us | Direct global KV, mirrored direct sliding KV, fused QKV preparation/scale/norms, stable counting-sort MoE, transactional KV |
| SGLang Marlin reference | 1 x 2816 -> 704 | 9.58 us | Repacked W4A16 JIT kernel |
| SGLang CUTLASS NVFP4 MoE | M=1, E=128, top-8, N=704, K=2816 | 55.31 us | CUDA Graph; synthetic valid FP4 tensors |
| FlashInfer/TensorRT-LLM fused NVFP4 MoE | M=1, E=128, top-8, N=704, K=2816 | 46.69 us | CUDA Graph; GeGLU, real packed layer-0 weights |
| CUTLASS NVFP4 grouped | 8 x (16 x 2816 -> 704) | 16.43 us | W4A4, 30.9 TFLOP/s |
| CUTLASS NVFP4 grouped | 8 x (64 x 2816 -> 704) | 16.25 us | W4A4, 125.0 TFLOP/s |

The native W4A16 path beats the Marlin W4A16 reference at M=1. The native
tensor-core W4A4 implementation is faster again: its production dynamic-route
block takes 35.37 us, 24% faster than the 46.69 us FlashInfer reference, and
projects to 1.06 ms for
all 30 expert blocks before surrounding dense work. On the reference activation,
the final native absolute checksum is 800.31 versus 800.23 from FlashInfer. The
direct W4A16 kernel remains the simple arithmetic oracle.

The measured four-draft assistant chain plus the five-token verifier totals
about 12.71 ms before request-scheduler overhead. If all four drafts are
accepted, assisted greedy decoding emits the four drafts plus the verifier's
bonus token, an upper-bound projection of roughly 393 output tok/s. Real prompt
acceptance length, rollback/commit work, and sampling overhead still have to be
measured before treating that projection as an end-to-end result.

The first integrated greedy run carries real prefill state and KV into the
assistant, verifies `[previous token + four drafts]`, commits only the accepted
candidate prefix, and feeds the selected verifier state into the next cycle.
The assistant uses the target-visible position unchanged for all four drafts,
as required by Transformers' `SinglePositionMultiTokenCandidateGenerator`;
the independent four-draft oracle now records and enforces this invariant.
After fixing the shared-memory softmax race, two independent integrated runs
produced identical cycle decisions, token IDs, and text: `A black-and-white
manga page consisting of several panels depicting a dramatic confrontation`.
They retained 15 post-prefill tokens in six cycles with nine accepted drafts.
The latest warm assistant and verifier wall totals are 79.92 ms, or 187.7
post-prefill tok/s. This is not compared as equivalent to the 355 tok/s
all-drafts-accepted kernel upper bound.

The persistent JSONL serving worker was also exercised with two requests in one
process. Profiled startup varied from 36.22 to 69.74 s with filesystem cache
state and includes construction of both resident expert plan sets and the
reusable 8K KV arena. The latest first 16-token request took 246.86 ms; the
identical warm request took 142.51 ms and returned identical token IDs, text,
and MTP acceptance. The warm request spent 4.09 ms decoding JPEG, 4.43 ms in
exact resize/patchify, 26.62 ms in vision, 23.86 ms in target prefill, and
79.86 ms in six MTP cycles. Startup is reported separately and occurs once per
worker, not once per request.

An offline end-to-end BF16 Transformers oracle uses the same 285-token exact
chat/image prompt and official four-layer assistant. Its greedy continuation is
`A black and white manga page featuring several panels with various characters.\n\nThe top`.
The serving NVFP4 target agrees for the first two tokens, then takes the also
coherent quantized continuation `A black-and-white manga page consisting of
several panels depicting a dramatic confrontation`. This is reported as a
target-representation difference: component equations and the assistant are
differentially checked, but the NVFP4 output is not claimed token-identical to
the original BF16 checkpoint.

Checkpoint startup is storage-bound. The original Crucial P3 path took 320 s
for a cold 17.49 GiB upload during one degraded read episode. The staged SATA
copy measured 29.1 s cold and 6.83 s warm (2.56 GiB/s) using double-buffered
pinned host staging. These are wall times including file page faults.
The combined production arenas (17.5 GiB base plus the 12.0 GiB fused-expert
sidecar) loaded and exhaustively validated in 20.0 s warm, or 1.47 GiB/s, for
29.46 GiB of device-resident weights.

The earlier vLLM baseline reported approximately 276.7 accepted output tokens/s
and 1227.5 prefill tokens/s. It was not a plain one-token decode run. The
recoverable launch configuration in `/mnt/SSD/localmaxxing/benchmark.json` used
an INT8 per-channel weight-only target, FP8 KV cache, and the official
`26b-a4b-it-assistant` with `method=mtp` and four speculative tokens. It is the
speculative end-to-end target. Native target-only throughput, assistant cost,
draft acceptance, verified tokens per target pass, and accepted output tok/s
must all be reported separately.

Nsight Compute counter collection is currently disabled by the driver's
`ERR_NVGPUCTRPERM` policy for this user. Nsight Systems tracing works and is
kept under `profiles/`; low-level occupancy/cache-counter tuning will resume if
the workstation enables non-admin performance counters.

The TQ4 figures use a standalone D=512 global-KV benchmark, not the serving
cache. Packed K and V plus BF16 per-half scales occupy 260 bytes per 512-value
row versus 1024 bytes in BF16 (3.94x compression). At 65K, Systems attributes
209 us to packed KQ versus 414 us for BF16 KQ, and 392 us to packed PV versus
656 us for BF16 PV. Query transform plus inverse output transform cost about
3.6 us. Full-cache quantization costs roughly 0.02 us per token for both K and
V. The synthetic differential measured max/mean BF16-output error near
0.01/0.00 at 65K, but this is only a kernel sanity check; content quality and
MTP acceptance remain production gates. The trace is
`profiles/turboquant-64k/g4.nsys-rep`.

The latest scoped batch-8 Nsight Systems trace is in
`profiles/fixed-b8-fused-direct-qkv-scoped/g4.nsys-rep`. A verifier pass now
launches 980 GPU kernels, down from roughly 1545 before this work. The removed
launches comprise partial FP8 output copies, expert-output clears,
candidate-KV staging, decode route-sort/configuration stages, standalone FP8
scale kernels, separate Q/K/V transforms, and direct-attention preparation.
The exact fixed sequence hash remains `4722933180576200099`; advancing tests
also retain their hashes across the 1024-token ring boundary. CUDA Graph replay
remains opt-in because its padded buckets changed long-run output behavior.

A 2026-09-04 graph retest measured a fixed verifier pass at 6.164 ms for batch
1 and 8.363 ms for batch 8, versus 6.665 and 8.985 ms without replay. The
fixed-work hashes matched. It is still not a serving win: over a real advancing
64-token batch-8 generation, the bucketed GEMM reduction changed the trajectory
after four cycles, reduced accepted drafts from 40 to 39, added a verifier
cycle, and increased verifier time from about 269 to 288 ms. Softmax masks the
padded columns correctly; the difference comes from BF16 tensor-core rounding
under the wider QK/PV reduction shape. Full replay therefore remains a
diagnostic until attention can retain the eager reduction geometry.

### Drafter policy and precision sweep (2026-09-04)

Real 128-token generations show that proposal throughput alone is not a valid
MTP objective; acceptance and the number of resulting target passes dominate.
On representative text, B1 draft lengths 0--4 produced 193, 297, 362, 398,
and 346 accepted tok/s respectively. On a manga page they produced 192, 287,
290, 322, and 278 tok/s. Three drafts is therefore the B1 default. B8 remains
content-sensitive: four drafts reached 2145 aggregate tok/s on the manga page
versus 1912 for three, while a second text prompt favored three by 2112 versus
2064 aggregate tok/s. Per-batch environment overrides are retained for
workload-specific tuning.

The assistant E4M3 experiment was not competitive. On the B1 text case its
cumulative assistant time rose from about 19.9 to 38.3 ms, draft acceptance
fell from 41.7% to 36.5%, and decode fell from 290.5 to 266.4 tok/s. It is not
loaded by default. Halving the exact vocabulary correction set from 1024 to
512 candidates preserved the sampled sequence but did not reduce measured
assistant time; 256 candidates changed the draft trajectory and was slower
end-to-end. Padding B1 assistant computation to the unusually fast physical B2
GEMM shape also failed the serving test, reducing decode from about 289.4 to
276.8 tok/s despite a faster isolated GPU interval.

The post-fusion Systems trace is
`profiles/target-verifier-fused/g4.nsys-rep`. Its GPU-kernel time is led by the
two grouped W4A4 expert GEMMs per layer (25.2%), small-N cuBLAS projections
(13.1%), sliding/global attention (about 19%), and the three new fused
normalization stages (about 8%). The fusion removed four launches per layer and
reduced the verifier by another 0.38 ms without changing its hidden checksum or
selected token IDs.

The current end-to-end trace is
`profiles/multimodal-serve-warm/g4.nsys-rep`. It captures only the second warm
request via the CUDA profiler API and contains one
vision pass, one target prefill, and six four-draft/M=5 MTP cycles. There are
exactly 420 grouped expert GEMMs (60 in prefill and 360 in verification), and
they account for 19.3% of GPU kernel time. Device-dynamic CUTLASS scheduling
removed 720 synchronous metadata copies and 60 stream synchronizations per
request. Two-segment sliding-ring writes then reduced D2D copies from 15,495 to
845 and warm language prefill from 52.78 to 23.84 ms. The temporary
one-expert-per-launch
diagnostic emitted 5,108 expert GEMMs and consumed 41.3%, so it is no longer
used by serving.

The first complete multimodal baseline trace is
`profiles/multimodal-prefill/g4.nsys-rep`. For the language prefill, the two
route-per-token W4A4 expert GEMMs consume about 94 ms of 123 ms (62.5% of all
GPU kernel time in the combined multi-iteration trace). Language prefill
softmax is only 0.2%.

The earlier experimental optimized trace is
`profiles/multimodal-prefill-sorted/g4.nsys-rep`. Routes are radix-sorted by
expert entirely on device and dispatched as 128 variable-M grouped problems.
That implementation measured 31.12 ms for language prefill, but the trace
predated the softmax-race fix and used a grouped launch schedule that was not
reliably repeatable in the full stack. It is preserved as profiling evidence,
not as the current serving result. After fixing the actual softmax race and
restoring one grouped expert launch per projection, the current schedule
measured 43.93 ms and was bit-repeatable. Device-dynamic grouped scheduling and
bulk sliding-ring writes subsequently reduced the same target prefill to
23.84 ms without changing its output. Three-stream Q/K/V and dense/expert
overlap, consumer-side FP8 scale fusion, a one-block stable route sort for up to
320 prompt tokens, and a combined scale/Q/K/V transform reduced the warm
285-token target prefill to 17.1-17.9 ms. Pointer-batching the 16 original head
matrices then replaced 16 QK plus 16 PV cuBLAS submissions with one call each;
the production pass now takes 13.98 ms. The scoped trace is
`profiles/multimodal-serve-prefill-pointer-batched/g4.nsys-rep`; its complete
vision, prefill, and 16-token MTP capture has 1263 kernels versus 3094 in the
preceding trace. Compacting per-expert activation-scale storage then made a
resident 4608-token route plan practical and removed the old five-way expert
split at 2064 tokens. The power-stable prefill sweep measured 39.8k tok/s at
1024 tokens, 41.0k at 1536, 40.3k at 2048, 38.8k at 2560, 37.5k at 3072, and
32.4k at 4104 as global-attention cost grows. Thus the measured sustained
ceiling is about 41.0k tok/s (42.3k best sample), up from 32.8k. The isolated
12-token attention semantic test fell from about 100
us to 9.9 us while retaining its 1.2e-7 maximum error. Pointer batching changes
permitted BF16 reduction rounding, so its hidden checksum differs from the
legacy schedule, but the Transformers differential, first-token oracle, and
repeatability checks pass. `G4_DISABLE_BATCHED_PREFILL_ATTENTION=1` retains an
A/B diagnostic. A future packed continuous-prefill scheduler must assign
separate sort/GEMM scratch to every in-flight execution lane.

A zero-copy BF16 dense gate/up fusion was measured because every target
checkpoint gate tensor is immediately followed by its up tensor. It reduced
the isolated layer-5 512-token benchmark from 675.0 to 658.5 us, but production
prefill uses separately calibrated FP8 projections and overlaps that branch
under NVFP4 expert work. The BF16 diagnostic was removed rather than bypassing
the calibrated serving path for an isolated 2.4% layer result.

Shared-prefix cloning was also collapsed from 50 sliding-layer K/V copy
submissions per session to one contiguous mirrored-arena copy plus ten global
copies. On the same 24-page concurrency-8 workload, measured prefix-clone host
time fell from 3.28 ms to 1.14 ms total. With pointer-batched attention and
unsplit expert prefill, the follow-up run completed 24/24 in 12.389 s (1.9372
pages/s), returned 174/174 regions, and produced 170 non-empty translations.
Physical prefill wall time fell from 1294 to 1032 ms;
`profiles/infiyomi-unsplit-prefill-smoke24` retains the run.

## Localmaxxing OpenAI-compatible serving check

The local, unsubmitted localmaxxing payloads exercise the actual streaming
`/v1/chat/completions` endpoint with a 34-token text prompt, 256 requested
output tokens, greedy sampling, one warmup, and three timed iterations. At B1,
`benchmarks/localmaxxing-g4-20260904-streaming.json` reports 313.8 output tok/s
median, 6.33 ms median TTFT, and 354.1 total tok/s. The comparable recovered
reference payload reports 276.7 output tok/s and 27.7 ms TTFT, making the native
path 13.4% faster on accepted output throughput for this prompt.

At concurrency eight,
`benchmarks/localmaxxing-g4-b8-final-20260904.json` reports 2257.8 aggregate
output tok/s median and 13.61 ms median TTFT across 24 requests. Its two steady
timed waves measured 2257.8 and 2259.5 aggregate output tok/s (about 281.2 and
281.4 tok/s per request). The first timed wave measured 1596.0 aggregate tok/s
because one client-observed request was an outlier; the payload retains that
sample rather than filtering it. Runtime telemetry verifies one eight-row
packed prefill wave and B8 verifier cycles for each timed cohort.

Localmaxxing labels 5374 tok/s at B1 and 2499 tok/s at B8 as prefill throughput,
but these values are estimates from client-observed TTFT after the identical
warmup prompt populated the shared-prefix cache. They measure the desired warm
prefix-serving behavior, not the isolated target-prefill kernel ceiling. The
payloads were written locally and were neither API-validated nor uploaded.

The later static NVFP4 vocabulary sidecar initially reduced isolated target
vocabulary selection from 906 to 555 microseconds at 40 rows and from 474 to
377 microseconds at one row, including dynamic activation quantization,
shortlist, exact BF16 correction, and argmax. Reducing the shortlist from four
to two candidates per 1024-row tile lowered the 40-row interval again, from 557
to 516 microseconds. One candidate reached 492 microseconds but changed the
complete B8 text continuation and was rejected. Two candidates matched INT8
byte-for-byte on the original 256-token B8 workload, three additional
128-token text prompts, and 64-token multimodal B1/B8 checks.

Aligning extraction to the NVFP4 projection's native 2048-row groups and
retaining four finalists per group preserves the same 512-entry correction
budget while halving extractor blocks. It reduced the complete 40-row stage
again from 516 to 474 microseconds and matched every text and multimodal output
above. Steady localmaxxing B8 waves reached 2404.8 and 2405.2 aggregate output
tok/s. The first timed wave split seven requests at B8 plus a short B1 tail and
is retained as a 1629.1 tok/s scheduler outlier in the local payload
`benchmarks/localmaxxing-g4-b8-nvfp4-vocab-k2-grouped-20260904.json`.
The promoted default's matching B1 payload reports 319.1 output tok/s and
11.31 ms median client-observed TTFT in
`benchmarks/localmaxxing-g4-b1-final-20260904.json`.

Replacing the complete 2048-element radix sort with four hierarchical argmax
rounds reduced the 40-row stage further from 474.7 to 454.5 microseconds. Two
paired fixed-verifier measurements improved from 8316/8320 to 8304/8294
microseconds with identical sequence hashes. Its localmaxxing payload retained
the exact output hash, although end-to-end throughput remained within ordinary
run variance; `benchmarks/localmaxxing-g4-b8-nvfp4-reduction-top-20260904.json`
therefore remains supporting correctness evidence rather than the headline
throughput result.

With the production two-candidate shortlist, localmaxxing B8 reports 2394.3
aggregate output tok/s (steady waves 2394.3 and 2396.8), 6.0% above the original
2257.8 INT8 baseline. The paired multimodal checks improved B1 from 285.1 to
295.3 tok/s and B8 from 218.0 to 229.5 tok/s per request. The four-candidate
payloads remain
`benchmarks/localmaxxing-g4-b1-nvfp4-vocab-20260904.json` and
`benchmarks/localmaxxing-g4-b8-nvfp4-vocab-20260904.json`; the local,
unsubmitted two-candidate B8 payload is
`benchmarks/localmaxxing-g4-b8-nvfp4-vocab-k2-20260904.json`.

Generalizing the static sidecar to the assistant's 1024-wide tied embedding
reduced one-row vocabulary selection from 182.8 to 152.1 microseconds and the
eight-row case from 226.5 to 178.8 microseconds. It runs once for each of four
drafts. The complete multimodal B1/B8 sequences and three additional text
continuations stayed exact. In localmaxxing, B1 improved from 319.1 to 323.0
output tok/s and B8 from 2404.8 to 2429.0 aggregate output tok/s, with steady
B8 waves of 2429.0 and 2430.4. The local, unsubmitted payloads are
`benchmarks/localmaxxing-g4-b1-assistant-nvfp4-20260904.json` and
`benchmarks/localmaxxing-g4-b8-assistant-nvfp4-20260904.json`.
An assistant-only 256-finalist sweep reduced its isolated B8 vocabulary stage
again to 153.6 microseconds, but introduced uneven B5/B7 tail cycles in one
serving wave and did not improve median B8 throughput. The stable 512-finalist
assistant setting remains the default.

After making single-request cache population post-response, the same local,
streaming localmaxxing B8 workload reported 2380.8 aggregate output tok/s
median (2334.8, 2383.0, and 2380.8), 13.59 ms median TTFT, and 2502.1 tok/s
warm-prefix prefill estimate. The locally validated, unsubmitted payload is
`benchmarks/localmaxxing-g4-b8-deferred-prefix-20260904.json`. The warmup
request now takes its ordinary full-prompt path; its reusable short-prefix KV
copy is queued only after the response completion callback.

Subsequent warmed crossover measurements tightened cache admission. At B8, a
57-token warm hit measured 15.04 ms TTFT versus 14.44 ms for packed prefill;
89 tokens was effectively tied at 17.39 versus 17.45 ms. Cold construct-and-
clone measured 28.66 versus 29.65 ms at 153 tokens, but improved to 31.80
versus 40.64 ms at 281 and 36.11 versus 96.88 ms at 537. Production therefore
requires 128 tokens for an existing warm hit and 256 tokens to construct a
cold cohort cache. A source-slot event also prevents refill from overwriting KV
during the deferred low-priority copy without delaying the completed request.

The former verifier dispatch guard switched B8 to scalar attention above about
32768 tokens. A real B8 generation at 64025 prompt tokens measured only 19.14
decode tok/s per request on that fallback. Raising the default GEMM-attention
budget to 1048576 batch-context rows produced 72.18 tok/s per request (about
577 aggregate), a 3.77x gain, with identical results across all eight sessions.
The lower-order BF16 reduction changed one near-tied punctuation token and one
accepted draft over the 16-token continuation; both paths remained stable and
the GEMM path is covered by the attention differential tests.

The assistant now also uses direct-cache pointer-batched GEMM attention for
uniform B4--B8 cohorts. It avoids repacking target K/V and preserves every
draft in the B4, B6, and B8 sweeps. Four-draft B8 latency improved from
2940.1 to 2766.8 microseconds at context 128, 3857.6 to 3183.6 at context
1024, and 4241.7 to 3434.2 at context 4096 after fusing query norm, RoPE, and
grouped-query packing. At B4 the corresponding context
128 and 4096 changes were 2951.3 to 2803.9 and 3549.7 to 3112.0 microseconds.
A 128-token multimodal B8 serving check retained identical tokens and 0.3427
draft acceptance while aggregate decode rose from 1664.2 to 1698.1 tok/s.
At B1, the isolated zero-KV component improved from 2216.8 to 2045.8
microseconds at context 128 and 2421.6 to 2140.0 at context 4096. That result
did not survive the live-KV serving gate: a 128-token multimodal generation
dropped acceptance from 0.4303 to 0.4012 and decode from 274.0 to 266.3 tok/s.
B1--B3 therefore retain scalar attention by default.

The combined B8 serving path was then measured with LocalMaxxing streaming and
the profitability-gated prefix policy. The locally validated, unsubmitted
payload `benchmarks/localmaxxing-g4-b8-direct-assistant-prefix-gated-20260904.json`
reports 2507.0 aggregate output tok/s median (2448.1, 2507.0, 2511.9), 12.97
ms median TTFT, and a 2621.7 tok/s client-derived prefill estimate. This is
3.2% above the prior 2429.0 best and 11.0% above the original 2257.8 baseline.
Server telemetry confirms every timed wave used one 272-row packed prefill and
86 B8 verifier cycles, with no prefix cloning for the 34-token prompt.

The paired B1 payload
`benchmarks/localmaxxing-g4-b1-prefix-gated-20260904.json` reports 364.9 output
tok/s median (365.1, 364.9, 364.9), 14.87 ms median TTFT, and a 2286.2 tok/s
client-derived prefill estimate. This is 13.0% above the earlier 323.0 B1
payload. Each warmed request used 93 verifier cycles for 255 post-prefill
tokens, with about 34.7 ms attributed to the assistant and 663.0 ms to the
verifier; the 34-token prefix remained below the cache threshold.

Increasing the three 2816-wide fused normalization boundaries from 256 to 512
threads improved the fixed B1 verifier from 6374.4 to 6029.9 microseconds and
the fixed B8 verifier from 8340.0 to 8021.2 microseconds. Both retained their
exact sequence hashes. The 128-thread alternative regressed B1 to 7241.7
microseconds. A naive version changed the reduction tree and failed the live
gate: B1 acceptance fell from 0.4303 to 0.3196, and a verifier-only B8 change
fell from 0.3427 to 0.3097. Isolating those drifting reductions did not fix the
long-run trajectory either.

The promoted implementation instead retains the original 256-thread column
ownership and summation tree, using the second 256 threads only for independent
elementwise output passes. In paired builds it improved B1 from 6438.9 to
6141.2 microseconds and B8 from 8381.3 to 8109.1 while preserving fixed hashes.
The 128-token live checks were also byte-identical: B1 retained token hash
`bba1f580...`, 0.4303 acceptance, and reached 283.4 tok/s; B8 retained hash
`3e3df7e...`, 0.3427 acceptance, and improved from 209.8 to 215.8 tok/s per
request. Verifier boundaries therefore default to 512 threads, while prefill
retains 256. `G4_FUSED_NORM_THREADS=128|256|512` and the individual
`G4_ATTENTION_NORM_THREADS`, `G4_ROUTER_NORM_THREADS`, and
`G4_FINAL_NORM_THREADS` overrides remain available for profiling.

With the exact wide-elementwise schedule promoted, the locally validated B1
payload `benchmarks/localmaxxing-g4-b1-exact-wide-elementwise-20260904.json`
reports 376.5 output tok/s median (376.3, 376.5, 376.6), 3.2% above its paired
364.9 baseline, with 15.06 ms TTFT. The B8 payload
`benchmarks/localmaxxing-g4-b8-exact-wide-elementwise-20260904.json` reports
2507.8 tok/s median; its first full steady wave reached 2567.9, while the last
wave fell to 2118.1 when HTTP telemetry recorded six B7 tail cycles. Fresh
256-token B8 text and multimodal cohorts, plus B1-then-B8 cache-reuse tests,
remained exactly symmetric, so that tail is retained as scheduler/arrival
evidence rather than filtered from the payload. Neither payload was uploaded.

After replacing the HTTP scheduler's 50 ms refill wait with nonblocking polling
at MTP boundaries, the locally validated, unsubmitted payload
`benchmarks/localmaxxing-g4-b8-nonblocking-refill-20260904.json` reports a
2506.8 aggregate tok/s median (2506.8, 2570.0, 2198.6) and 12.85 ms median
TTFT. The final wave no longer contained a host-blocked B7 phase: seven
sessions completed together and the last emitted 18 tokens in eight B1 cycles.
That residual tail is genuine per-session MTP acceptance skew. The slower wave
also accumulated 819 ms in its B8 verifier cycles versus 736--742 ms in the
other waves, so it is not attributable to refill sleeping.

Fresh scoped Nsight Systems captures are in
`profiles/fixed-b1-c4096-exact-wide/g4.nsys-rep` and
`profiles/fixed-b8-c4096-exact-wide/g4.nsys-rep`. The eager fixed verifier took
6246 microseconds at B1 and 8092 microseconds at B8. CUDA Graph replay reduced
those fixed-input figures to 5754 and 7523 microseconds with matching hashes,
but remains disabled: in live B1 generation the 256-row attention bucket first
diverged after 22 output tokens, reducing acceptance from 0.4012 to 0.3503 and
throughput from 281.6 to 272.5 tok/s. Exact masking cannot preserve the eager
trajectory when cuBLAS sees a different padded K dimension and therefore uses a
different accumulation geometry.

Short shared-prefix clones no longer copy the complete 630 MiB sliding-KV
arena. A single vectorized GPU kernel copies only valid canonical rows and
reconstructs both mirrors plus the four speculative tail rows. At a warmed
153-token B8 prefix this reduced TTFT from 30.39 to 22.30 ms; at 281 tokens it
reduced 28.36 to 22.25 ms. A 32-token decode comparison against the former full
copy retained identical prefill checksum, token IDs, and 0.3673 draft
acceptance. `G4_FULL_SLIDING_PREFIX_COPY=1` keeps the old path as a diagnostic.

The cheaper copy made a 57-token warm hit faster than packed prefill in a TTFT
isolation (22.19 versus 27.09 ms), while 25 tokens remained tied. Nevertheless,
the production warm threshold remains 128 tokens. An actual 34-token
Localmaxxing B8 run with the diagnostic threshold lowered to 32 reached 7.11 ms
median TTFT but required 89 verifier cycles instead of 86--87 because the
different prefill geometry changed MTP acceptance; median decode fell to 2457.2
tok/s. The unsubmitted diagnostic payload is
`benchmarks/localmaxxing-g4-b8-sparse-prefix-20260904.json`.

A production packed-prefill capture at eight 537-token text prompts is in
`profiles/prefill-b8-t537-production/g4.nsys-rep`; its matching warmed capture
is `profiles/prefill-b8-t537-warm/g4.nsys-rep`. The cold pass took 133.2 ms,
while the same 4296-row shape took 74.2 ms after initialization, or about 57.9k
aggregate prompt tok/s. Nsight attributes the cold gap to first-use cuBLAS
planning/JIT, including the INT8 tied-vocabulary GEMM, rather than idle CPU
work in steady state. The server now warms vocabulary row counts 1--8 and the
exact 34- and 272-row FP8 prefill buckets before announcing readiness. A true
first-query Localmaxxing B1 probe consequently reduced TTFT from 86.67 to 17.36
ms; its measured prefill wall time was 11.92 ms. Set
`G4_DISABLE_STARTUP_PREFILL_WARMUP=1` only for cold-start diagnostics.
Extending that startup probe through tiny real B1 and B8 MTP cohorts also
improved a 32-token first-query B8 run from 1622 to 2168 aggregate tok/s. Its
19.29 ms TTFT still included roughly 6 ms of GPU clock ramp after an idle
interval; a back-to-back prefill remains near 12 ms.

The large-prefill route pipeline now passes router IDs directly to CUB,
initializes its immutable route-index vector once, and writes the inverse map
inside the following activation-quantization kernel. At eight independent
505-token prompts this reduced the profiled `batch/prefill` operation count
from 1330 to 1240 and its GPU range from 76.06 to 75.15 ms (about 1.2%). The
matching reports are `profiles/prefill-b8-t505-cub/g4.nsys-rep` and
`profiles/prefill-b8-t505-direct-fused-inverse/g4.nsys-rep`. A 32-output B8
comparison retained identical prefill checksums, token IDs, and 21 accepted
drafts; the complete 28-test suite also passed. A one-block atomic counting
sort was measured and rejected: despite fewer launches, the sorter itself took
2.28 ms across the 30 layers and was slower than the CUB route-management work.

Packed prefill now also leaves the eight W2 route rows in runner-owned storage
and reduces them inside the feed-forward finalizer. A per-token shared BF16 row
retains the former reduction's exact accumulation and rounding boundary while
avoiding the materialized expert output. At the same B8 x 505 shape, the old
reduction plus finalizer consumed 6.114 ms across 30 layers; the fused kernel
consumed 5.362 ms, saving 0.752 ms and 30 launches. The report is
`profiles/prefill-b8-t505-fused-expert-finalize/g4.nsys-rep`. The fused run
matched all 32 comparison tokens and 21 accepted drafts. B2 and B4 prefill
checksums also matched the materialized path; `G4_DISABLE_PREFILL_FUSED_EXPERT_FINALIZE=1`
retains that path for diagnostics.

The same fused expert finalizer is enabled for single-request prefill. With a
505-token B1 text prompt it reduced language prefill from 21.80 to 21.21 ms and
TTFT from 24.43 to 23.75 ms while retaining the identical checksum, 32 output
tokens, and 20 accepted drafts. The multimodal B1 serving repeatability test
also passes under this default.

The verifier now uses the same shared-row reduction/finalizer fusion. At
context 4096 with four drafts, fixed-cycle latency improved from 6244 to 6132
us at B1 and from 8114 to 8067 us at B8. Both established sequence hashes are
unchanged (`14667575765401665723` and `13571370454237472131`). Live 32-token
B1/B8 runs retained every output token and the same 20/21 accepted drafts;
accumulated verifier time fell by 0.49/0.48 ms respectively. Set
`G4_DISABLE_VERIFIER_FUSED_EXPERT_FINALIZE=1` for the materialized diagnostic.
Updated fixed-cycle traces are in
`profiles/fixed-b1-c4096-fused-expert-finalize/g4.nsys-rep` and
`profiles/fixed-b8-c4096-fused-expert-finalize/g4.nsys-rep`. Across their 20
cycles the fusion reduces kernel launches by exactly 600 (30 per cycle) and
GPU kernel time by 2.05 ms at B1 and 2.97 ms at B8, or about 102/149 us per
cycle. Profiled wall results were 6133 us at B1 and 7961 us at B8 with the
same established hashes.

`target-mtp-sweep` covers every batch width from 1 through 8 (and draft
lengths 0 through 4), rather than sampling only the endpoints. A fixed B4
diagnostic with identical cache/state rows exposed slot-dependent greedy ties
in the optimized sliding GEMM attention; the same split occurs with the old
prefill finalizer, exact 20-row FP8 geometry, serialized gate/up projections,
and CUB expert sorting. The scalar sliding-attention oracle restores bitwise
session symmetry but increases a context-153 verifier cycle from about 5.96 to
6.66 ms and follows a different valid quantized trajectory. Independent
requests do not require bit-identical tie outcomes, so production retains the
faster GEMM implementation.

The expanded all-width context-4096 MTP sweep is recorded in
`benchmarks/target-mtp-b1-b8-c4096-20260904.txt` (10 repetitions per point).
With the production four-draft verifier, fixed-cycle latency scales from
6201 us at B1 through 6299, 6621, 7014, 7254, 7468, and 7823 us at B2--B7 to
7989 us at B8. Corresponding candidate throughput is 806, 1588, 2266, 2851,
3446, 4017, 4474, and 5007 candidate tok/s. This verifies every intermediate
batch geometry and confirms that four drafts maximize raw verifier work per
second at each width; emitted throughput still depends on measured acceptance.

Verifier control traffic now uses persistent page-locked staging and one
combined context/previous-token upload. The fixed context-4096 hashes remain
unchanged, with 6137 us at B1 and 7985 us in the scoped B8 trace. Across 20 B8
cycles, pageable `cudaMemcpyAsync` API blocking fell from 109.0 ms to 0.267 ms;
the actual dependency is now reported at the final stream wait. The trace is
`profiles/fixed-b8-c4096-pinned-staging/g4.nsys-rep`, and the three live
multimodal serving regressions pass.

Packed prefill likewise combines eight metadata arrays into one pinned upload
and does not copy final hidden rows back to the CPU when serving has already
retained them on device. A forced-independent 8x505-token text pass completed
4040 prompt tokens in 67.91 ms (59.5k aggregate prompt tok/s), with 70.3 ms
text TTFT, 1202 GPU operations, and identical selected tokens across sessions.
Its report is `profiles/prefill-b8-t505-packed-metadata/g4.nsys-rep`. With the
normal shared-prefix policy, this deliberately identical workload instead used
one B1 prefill plus seven sparse clones and reached about 30.7 ms TTFT, so the
cache decision remains profitable here without delaying the source response.

Nsight Compute's dominant B1 routed-expert sample is a one-wave kernel: 188
CTAs on 188 SMs, 52% DRAM throughput, 20% tensor utilization, 90 KiB shared
memory per CTA, and 17% achieved occupancy. Output-width 64, K-depth 128/64,
cooperative scheduling, and N-major raster variants all regressed B1 and/or B8;
K=64 also changed the exact trajectory. The native 128x128x256 ping-pong kernel
therefore remains the production expert shape. A finalizer-to-next-layer FP8
fusion was also exact but slower (6347/8249 us at B1/B8) and was removed.

## Frequency-restricted MTP vocabulary

The draft-only NVFP4 head now supports an opt-in FR-Spec-style token map. A
reproducible 32,768-ID map was generated with `tools/build_hot_vocab.py` from
the repository's code/docs plus the two local Infiyomi quality corpora. The
compact sidecar is gathered once on GPU at load; target verification continues
to use the full 262,144-token vocabulary.

Isolated assistant-head latency fell from 153.2 to 32.8 us at B1 (4.7x) and
from 180.1 to 33.6 us at B8 (5.4x). A 65,536-ID alternative measured 53.1 us
at B8 and covered 96.7% of observed multimodal proposals versus 94.6% for the
32K map, which was not enough extra coverage to offset its slower projection.
Using one contiguous CUTLASS problem for the mapped matrix instead of sixteen
2048-row groups further reduced the 32K head to 31.4 us at B1 and 32.0 us at
B8 while retaining every proposal in a 64-token B8 multimodal trajectory. A
32-finalist correction shortlist reached 29.3 us but changed rejected proposal
rows and showed no end-to-end gain, so the 64-finalist setting remains default.

On an ordinary 128-token text check, 32K improved B1 from 310.3 to 332.9 tok/s
and B8 aggregate from 1938 to 2214 tok/s. The paired 64-token multimodal B8
check improved aggregate decode from 2178 to 2325 tok/s and retained identical
output. Persistent OpenAI serving measured 358.4 tok/s at B1 versus a fresh
331.6 full-head control. Localmaxxing B8 reached 2520.2 tok/s median, with
steady waves of 2520.2 and 2520.4, compared with the earlier 2429.0 assistant-
NVFP4 result. The unsubmitted payloads are
`benchmarks/localmaxxing-g4-b1-hot-vocab-32k-20260904.json` and
`benchmarks/localmaxxing-g4-b8-hot-vocab-32k-20260904.json`.

A live B1 draft-length sweep on that same localmaxxing prompt measured 322.0,
349.6, 358.4, and 336.2 tok/s for one through four drafts respectively. Three
drafts therefore remains the B1 optimum after vocabulary restriction. B8's
production four-draft setting produced the 2520.2 tok/s result above. The
additional local payloads are named `localmaxxing-g4-b1-hot32k-mtp{1,2,4}-
20260904.json` under `benchmarks/`; none were uploaded.

This remains opt-in because the quantized verifier is not token-trajectory
invariant across speculative row positions. On the same text prompt, an MTP=0
run first differed from the full-vocabulary speculative path at token 54 and
from the 32K path at token 86. Scalar attention moved but did not remove the
boundary. This is numerical row-geometry sensitivity in the approximate target
path rather than an unsafe ID remap, but it prevents claiming byte-for-byte
losslessness for this runtime today. The compact map itself is therefore a
measured throughput/acceptance policy, not a default correctness assumption.

A same-binary 24-page Infiyomi comparison reinforces that decision. Full vocab
completed in 10.287 s (2.333 pages/s, 12,465 completion tokens, 1211.7
completion tok/s) and returned 174/174 regions. The 32K map completed in
10.244 s (2.343 pages/s, 12,466 tokens, 1216.9 tok/s) but returned 173/174.
That is only a 0.4% normalized throughput improvement on the target image
workload and one fewer region, so the map is not enabled by default. The paired
runs are `profiles/infiyomi-full-vocab-paired-smoke24` and
`profiles/infiyomi-hot32k-smoke24`.

The 65,536-ID map is a better image-specific tradeoff. Two 24-page Infiyomi
runs completed in 9.576 s and 9.966 s, reporting 1293.1 and 1257.4 aggregate
completion tok/s respectively, versus 10.287 s and 1211.7 tok/s for the paired
full-vocabulary control. The first returned all 174 regions; the repeat missed
the same trajectory-sensitive `p019` region seen in the 32K run. With the
normal retry policy enabled, it completed 174/174 regions in 10.080 s without
needing a retry. The local artifacts are
`profiles/infiyomi-hot64k-smoke24`,
`profiles/infiyomi-hot64k-smoke24-repeat`, and
`profiles/infiyomi-hot64k-retry-smoke24`.

That larger map is slower on the text benchmark: warmed Localmaxxing measured
353.5 tok/s at B1 and 2408.6 aggregate tok/s at B8, compared with 358.4 and
2520.2 tok/s for 32K. Consequently one resident server may use 32K for
text-only cohorts and 64K for cohorts containing images. The two maps are
packed into one 64K sidecar with a 32K prefix and share GPU work buffers; a B8
multimodal MTP repeatability test passed through the 64K view, and a text-only
persistent-server generation exercised the 32K view. This workload routing is
still opt-in for the numerical trajectory reason above.

An isolated Nsight capture of the hot-32K assistant measured a complete
four-draft cycle at 1839 us for B1 and 2709 us for B8. The trace contained
roughly 278 launches per B1 cycle, but GPU kernels occupied about 94--96% of
the interval; after vocabulary restriction, the dominant cost was the 22 BF16
assistant projections rather than idle launch gaps. Packing each layer's
shared-input gate/up pair into one projection reduced the isolated cycle to
1768 us at B1 (-3.8%) and 2549 us at B8 (-5.9%). The traces are
`profiles/assistant-hot32-b1/g4.nsys-rep` and
`profiles/assistant-hot32-b8/g4.nsys-rep`.

On the persistent server, a paired 256-token B1 prompt retained exactly the
same output IDs and 53.4% draft acceptance while assistant time fell from
about 195 ms to 179 ms; median decode improved from 233.1 to 235.9 tok/s on
that trajectory. Localmaxxing's standard text prompt measured 360.3 tok/s,
versus the previous 358.4 hot-32K result. B8 improved more: the repeat produced
2733.4, 2734.5, and 2735.8 aggregate tok/s, versus the prior 2520.2 median,
with median TTFT 12.93 ms. A fused 64K multimodal B8 MTP repeatability test also
passed. Local, unsubmitted payloads are
`benchmarks/localmaxxing-g4-b1-hot32-fused-gate-up-20260905.json` and
`benchmarks/localmaxxing-g4-b8-hot32-fused-gate-up-repeat-20260905.json`.

Re-sweeping attention after that fusion exposed an obsolete B4 threshold.
At context 4096, enabling the existing direct pointer-batched attention path
reduced the four-draft assistant cycle from 1768 to 1487 us at B1, from 2596
to 2218 us at B2, and from 2802 to 2290 us at B3; draft rows were identical in
all cases. Uniform-context cohorts now use direct attention at every B1--B8
width, while genuinely ragged continuous-batch cohorts retain the descriptor
kernel path.

On Localmaxxing's standard prompt, the promoted B1 path measured 362.0--362.4
tok/s (362.1 median) with 14.90 ms median TTFT. More importantly, a scheduler
audit found that refill could replace completed rows but could not grow beyond
the initial cohort size. After reserving all eight logical slots in continuous
mode, a request arriving 120 ms into a B1 decode joined after 45 B1 cycles,
ran for 55 B2 cycles, and completed in 0.50 s while the original 256-token
request continued to 0.83 s. The same test before the fix formed two serial B1
waves and the late request took 0.95 s. Telemetry reported one successful
refill, and the non-refill B8 multimodal repeatability test remained green.

## Attention residual / router fusion (2026-09-05)

The verifier now combines attention post-normalization, BF16 residual addition,
dual feed-forward normalization, router scaling, and FP8 activation preparation
in one kernel. The residual stays in shared memory for the consuming operations;
the stored residual and all original BF16 rounding boundaries remain unchanged.
Warp reductions also replace the block-wide maximum-reduction barrier tree.
`G4_FUSE_ATTENTION_ROUTER=0` restores the original pair of kernels. Differing
attention/router thread overrides retain the unfused path so their reduction
orders are preserved.

An 18-case differential test checks every output byte: residual, dense/expert
BF16 inputs, router/dense FP8 inputs, and both scale arrays. It covers 1, 4,
and 40 rows, 128/256/512 threads, and scaled/unscaled attention projections:
`./build/g4 attention-router-test`.

Three alternating-order warmed A/B pairs on one resident runtime measured:

| Workload | Batch | Baseline decode tok/s | Fused decode tok/s | Gain |
|---|---:|---:|---:|---:|
| Text, 25 prompt tokens | 1 | 344.8 | 349.4 | 1.3% |
| Text, 25 prompt tokens | 8 | 2315.3 | 2340.1 | 1.1% |
| Image, 551 expanded prompt tokens | 1 | 376.8 | 383.2 | 1.7% |
| Image, 551 expanded prompt tokens | 8 | 2240.3 | 2264.2 | 1.1% |

Rates are medians of emitted post-prefill tokens, aggregated across the batch,
with a 128-token output cap and the existing three-/four-draft policies. The
text prompt asks why batching improves transformer inference; the image prompt
is `Describe this image.` using the rendered-browser-page1 JPEG. The 32K text
and 64K image assistant maps were enabled. Every paired output sequence and
accepted-draft count matched. These are isolated serving tests, not a new full
Infiyomi run or the different Localmaxxing prompt. Median text TTFT remained
about 8.8 ms / 10.6 ms at B1/B8; this change targets decode.

Raw samples are `benchmarks/throughput-router-text-ab-20260905.jsonl` and
`benchmarks/throughput-router-image-ab-20260905.jsonl`. Reproduce by setting
`G4_BATCH_SERVE_SWEEP_ENV=G4_FUSE_ATTENTION_ROUTER` and
`G4_BATCH_SERVE_TEST_TOKENS=128` on `batch-serve-test IMAGE 8`, adding
`G4_BATCH_SERVE_TEXT_ONLY=1` for text. This mode warms both variants, compares
B1 and B8 in the same process, alternates order, and fails on token mismatches.

At fixed 4096-token context, the final Nsight capture reduced verifier launches
from 754 to 724 per cycle. The replaced kernel pair used 14.71 us per layer;
the fused kernel uses 12.13 us. Profiled B8 wall latency fell from 7967 to
7896 us with the identical sequence hash `13571370454237472131`.
Reports: `profiles/throughput-next-b8-{baseline,fused}/g4.nsys-rep`.
The same-process 40-cycle sweep recorded B1 medians of 6001 versus 5919 us and
B8 medians of 8037 versus 7950 us, with matching hashes in all pairs, in
`benchmarks/throughput-attention-router-warp-ab-20260905.jsonl`.
Expert/dense projections and attention GEMMs remain the main GPU costs; the
launch-count reduction produces a measured incremental gain, not a large
change in the bottleneck.

The final build also passed an advancing-context A/B sweep starting at 32768
tokens (`G4_ADVANCE_FIXED=1 ./build/g4 target-throughput-sweep 32768 40`).
All B1/B8 hashes and session repeatability checks matched. Median verifier
latency was 9289 → 9203 us at B1 and 12368 → 12250 us at B8. The raw artifact
is `benchmarks/throughput-router-32k-advancing-ab-20260905.jsonl`; this validates
the changed decode boundary at 32K, not a new native-limit context benchmark.

## Decode control transfers and projection experiments (2026-09-05)

Serving now uploads the assistant's context lengths and previous tokens in
one pinned block and passes the GPU pointers to the verifier. Logical rows
remain separate from optional padded assistant rows. Both consumers run on
the same stream, and the verifier completes before the staging block is
reused. Zero-draft cycles use the verifier's existing pinned upload. Optional
CUDA-graph mode retains independent uploads.

The assistant's first input-preparation kernel reads the target continuation
states directly, including padded-row mapping. Continuous batching leaves
states in place when every row survives. Actual cohort changes still compact
the survivors. The final assistant draft also skips its unused continuation
projection and preparation of a nonexistent next draft. The full comparison
path is available with `G4_DISABLE_DECODE_TRANSFER_CULL=1`.

Nsight `profiles/transfer-cull-image-b8.nsys-rep` captured 48 continuous B8 MTP
cycles. Runtime/GPU correlation confirms exactly 48 H2D copies (3072 bytes),
48 D2H copies (24576 bytes), and **zero D2D copies** throughout the decode
interval: one 64-byte upload and one 512-byte result download per cycle.
Previously the steady B8 path submitted three metadata uploads, one result
download, eight assistant state copies, and eight unconditional compaction
copies. The trace predates removal of the final unused projection; the
transfer-count changes are the same in the final build.

Same-process, alternating-order continuous-batch text tests kept all emitted
tokens identical. Final medians were 348.46 → 348.82 tok/s at B1 and
2316.24 → 2328.06 aggregate tok/s at B8. These are small gains (0.1% / 0.5%),
not evidence that PCIe traffic was the main bottleneck. One optimized B8 trial
accepted 569 rather than 568 draft tokens while emitting the same sequence;
the other trials matched acceptance. Raw records are in
`benchmarks/transfer-cull-final-text-ab-20260905.jsonl`.
The final unprofiled image pairs likewise preserved all output tokens and
accepted-draft counts: medians were 381.91 → 382.70 tok/s at B1 and
2260.46 → 2268.14 aggregate tok/s at B8 (0.2% / 0.3%). Raw records are
`benchmarks/transfer-cull-final-image-ab-20260905.jsonl`.

The live OpenAI regression used 24 requests with varying prompt lengths and
output limits of 1, 2, 7, 32, 64, and 128, at concurrency eight. It exposed a
preexisting refill bug: post-prefill admissions did not check whether prefill
had already completed the request, allowing a one-token limit to produce two
tokens. Both synchronous and asynchronous refill admission now apply that
completion check and compact only the newly admitted survivors as needed.
The regression then passed all limits, exercised B1--B8 and 16 successful
refills, and completed SSE with content, a finish reason, and `[DONE]`.
Result: `benchmarks/transfer-cull-oai-refill-20260905.json`.

Attention and projection experiments remain opt-in, not production changes:

- `G4_ALIGNED_DIRECT_ATTENTION=1` separates aligned score/probability row
  strides from the true attention length, without rounding up the context or
  reading extra KV tokens. Fixed 4K B8 improved slightly, but the real image
  medians were effectively unchanged (382.66 → 382.45 B1; 2284.96 → 2282.62 B8).
  All output pairs matched. Artifacts: `aligned-attention-4k-ab-20260905.jsonl`
  and `aligned-attention-image-ab-20260905.jsonl` under `benchmarks/`.
- `G4_TUNE_FP8_PROJECTIONS=1` times up to 32 small-M cuBLASLt heuristic
  candidates during plan creation. It rejects candidates that differ on the
  sampled projection output, but that sample does **not** prove parity for
  other layers or later inputs. Full verifier checks exposed changed token
  hashes and a B1 slowdown, so the tuner is not enabled in serving. B8 alone
  showed roughly 2% improvement; that does not justify promoting it. Raw
  measurements and selected indices are `benchmarks/tuned-fp8-4k-ab-20260905.jsonl`
  and `profiles/tuned-fp8-4k-ab-20260905.log`.
- `G4_TUNED_B8_PROJECTIONS=1` restricts the trial to two consistent winners
  (sliding Q and attention output at padded M=64). Fixed output hashes matched,
  but B8 latency improved only about 0.2%, so the existing defaults remain.
  Artifact: `benchmarks/tuned-b8-qo-4k-ab-20260905.jsonl`.

Use `G4_BATCH_SERVE_SWEEP_CONTINUOUS=1` with the serving A/B harness to exercise
the scheduler. `G4_PROFILE_SWEEP_VARIANT=0` or `1` brackets one warmed B8 wave
for Nsight's CUDA-profiler-API capture. The projection tuner and algorithm
selection switches affect plan creation, so use `target-throughput-sweep`
(which recreates the runners) rather than toggling them on an already planned
resident server. No experimental benchmark artifacts were uploaded.

### Isolated tensor-core attention and target gate/up experiments (2026-09-05)

The standalone `tiled-attention-bench` loads captured, real verifier tensors,
not random Q/K/V. Captures cover text prompts of 505, 3,865, and 32,025 tokens.
Each contains five verifier positions; BS1 replay selects the first request
but still uses five positions, **not** production BS1's four-position geometry.
This is an attention microbenchmark, not emitted tokens/s or TTFT. Sliding
attention retains its 1,024-token window even for the longer prompts.

The baseline reproduces production cuBLAS QK, BF16 softmax, and PV. Every
non-ragged replay matched the captured output exactly. Ragged tests shorten
logical contexts by 17 tokens per request; they check masking on the same
stored K/V, not independently generated ragged requests. They do not compare
against the captured output (`capture_checked=false`).

The prototype uses BF16 WMMA QK/PV, tiled softmax, FP32 split-KV partials,
and a parallel merge. Tested tile widths 64/128/256 and 1/2/4/8 chunks per
partition. Iterations reduced shared memory, processed several tiles per
partition, and parallelized merge statistics. Three repeated sweeps produced
324 measurements, all finite. Relative RMS differences reached 0.277%, with
maximum absolute difference 0.125. Local probability rounding differs from
the baseline, so this is **not** a parity-approved serving implementation.

Median microseconds for global attention (D=512), baseline versus the
best fixed prototype configuration at each shape:

| Prompt tokens | BS1 baseline | BS1 tiled | BS8 baseline | BS8 tiled |
| --- | ---: | ---: | ---: | ---: |
| 505 | 10.70 | 36.16 | 14.37 | 45.02 |
| 3,865 | 33.59 | 41.20 | 89.98 | 250.97 |
| 32,025 | 287.04 | 256.31 | 769.87 | 2,044.54 |

Sliding D=256 at the 3,865-token prompt measured 14.51 -> 13.69 us BS1
and 28.19 -> 30.31 us BS8. The isolated BS1 long-context win is promising,
but does not justify enabling the prototype. The production path is unchanged.

Nsight Compute on a warmed BS8 D512, 128-key tile identified the main
partition kernel at 267.17 us: 16.13% compute throughput, 30.57% DRAM throughput,
22.87% achieved occupancy. Shared memory limits it to three blocks per SM.
There are no local-memory spills in the compiled partition kernels. The next
attention iteration should address tile data reuse, cooperative/asynchronous
staging, and occupancy before expanding fusion. A lower launch count alone
did not improve this implementation. Profiling-instrumented timings are not
used in the table.

Artifacts remain local:

- `benchmarks/tiled-attention-replay-20260905.jsonl`
- `profiles/attention-real-prompt{512,4k,32k}/d{256,512}-b8.bin`
- `profiles/tiled-attention-4k-b8-ncu.ncu-rep`

The earlier `profiles/attention-real-4k/` directory contains startup-warmup
captures, not 4K prompts; it was excluded from these results. The minimum
context capture filter was added to prevent that mistake.

Target gate/up fusion packs each FP8 pair once (no requantization), preserves
their separate scales, and lets GeGLU consume the combined strided output
without a split kernel. The ~340 MiB sidecar is shared across batch runners.
It replaces the BS8 concurrent pair with one GEMM, but remains opt-in:

- Fixed 4K verifier median: BS1 5,911.93 -> 5,923.31 us; BS8
  7,962.18 -> 7,763.21 us (~2.5% lower latency). All token hashes matched.
- Short text continuous serving: BS1 349.13 -> 347.38 emitted decode tok/s;
  BS8 aggregate 2,323.35 -> 2,318.59 tok/s.
- Real 3,865-token text continuous serving: BS1 297.59 -> 296.26 tok/s;
  BS8 aggregate 1,947.49 -> 1,949.03 tok/s. All output pairs matched.
  This ~0.08% BS8 difference does not reproduce the fixed-verifier gain.
- Image continuous serving: BS1 495.42 -> 492.37 tok/s; BS8 aggregate
  3,101.67 -> 3,103.14 tok/s. This test used
  `infiyomi-bench/viewer/public/generated/sample-context-blurred.jpg`, so its
  absolute rates are not comparable to earlier image-workload measurements.

Serving comparisons used 128-token caps, three alternating pairs, and the
continuous scheduler. Text and image output pairs matched. The tiny BS8
image difference is not a meaningful demonstrated gain. Results are in
`benchmarks/fused-target-gate-up-{4k,text,text4k,image}-ab-20260905.jsonl`.
Neither target gate/up fusion nor tiled attention was promoted to defaults.

### Async attention staging and softmax race repair (2026-09-05)

Added coalesced 16-byte `cp.async` K/V staging with zero-filled tails, query
staging, padded shared-memory score/probability rows, compact 64-feature
tiles, and one/two/three query-tile groups per block. Full-width K/V copies
can overlap softmax arithmetic; compact feature tiles currently synchronize
each feature stage (not a double-buffered pipeline). The compact single-query
group uses 24,192 bytes of dynamic shared memory, versus 95,872 bytes for
the full-width D512/S64 staged tile. Larger query groups were generally
slower at BS8, but two groups helped the four-position BS1 global 4K shape.
No new per-step CPU operations were added to serving.

Nsight Compute on the earlier full-width staged D512/S64 kernel measured
179.10 us, 54.02% DRAM throughput, and only 8.30% achieved occupancy. It also
reported substantial shared-memory bank conflicts. These findings motivated
the compact feature tiles and padded intermediate layouts; that profile is
not a measurement of the final compact kernel.
Artifact: `profiles/staged-attention-4k-b8-full.ncu-rep`.

Final sweep: three repetitions, 594 measurements, 505/3,865/32,025-token real
prompt fixtures. BS1 now trims each captured query head to **four** verifier
positions; BS8 keeps five. It retains the original KV allocation, including
the masked unused candidate slot for BS1. This still replays the captured
direct-cache layout, not the entire production BS1 attention dispatch.
All non-ragged baseline outputs matched their captured outputs, including
the trimmed BS1 queries. All 162 checked same-tile staged/unstaged comparisons
were exact, with no nonfinite results. The experimental softmax/PV calculation
still differs from production: maximum relative RMS error 0.288%, maximum
absolute error 0.125. No end-to-end generation validation of tiled attention
has been performed, and it remains disabled in serving.

Median microseconds, fixed best experimental configuration per shape:

| Prompt | Attention | BS1 baseline | BS1 experimental | BS8 baseline | BS8 experimental |
| --- | --- | ---: | ---: | ---: | ---: |
| 505 | sliding | 9.61 | 8.12 | 17.44 | 13.76 |
| 505 | global | 10.55 | 12.13 | 14.37 | 21.96 |
| 3,865 | sliding | 14.47 | 9.18 | 28.16 | 23.52 |
| 3,865 | global | 35.43 | 25.81 | 74.15 | 143.50 |
| 32,025 | sliding | 14.44 | 9.13 | 28.26 | 23.45 |
| 32,025 | global | 295.18 | 187.26 | 767.07 | 1,682.28 |

The BS1 32K global winner is the existing **unstaged** 128-key/two-chunk
variant, now tested with four positions. Do not attribute that gain to async
staging or compare it directly with the older five-position BS1 measurements.
The 4K sliding candidates reduce latency about 37% BS1 and 16% BS8; these
are isolated attention improvements, not emitted-token throughput gains.
The global BS8 prototype is still unsuitable.

Artifacts: `benchmarks/staged-attention-native-query-replay-20260905.jsonl`
(four-position BS1) and `benchmarks/staged-attention-replay-20260905.jsonl`
(five-position BS1 compatibility sweep). All remain local.

CUDA Racecheck exposed a missing barrier in the replay baseline softmax:
warp zero could overwrite the shared maximum with its sum before other warps
consumed it. The same unprotected reuse existed in production
`ragged_attention_softmax_kernel` and `ragged_gemm_softmax_kernel`.
Added the consumption barrier in all three. Other production softmax variants
already had it. This is a correctness repair, not an optimization promotion.
The final measurements above use the repaired replay baseline; older timing
tables used the pre-repair baseline, which matched captures in ordinary runs
but failed under race instrumentation.

Added `attention-softmax-test` / CTest `attention_softmax_correctness`: eight
uniform-logit cases exercise global/sliding, candidate/non-candidate, both
production softmax implementations, and ragged contexts from 1 to 1,100 tokens.
It passes CUDA Racecheck with zero hazards. The replay also passes Memcheck
and, after the repair, Racecheck on the ragged D512 505-token fixture.
The existing single/batched decode-attention CTests pass too.
Logs: `profiles/attention-softmax-production-racecheck.log`,
`profiles/staged-attention-memcheck.log`, and
`profiles/staged-attention-racecheck-fixed.log`.
Native four-position BS1/D256 Memcheck also reports zero errors:
`profiles/staged-attention-b1-d256-memcheck.log`.
The 40-repetition full-verifier regression matched all 12 pre-repair sequence
hashes across BS1/BS8 and fused/unfused target gate/up trials, with repeatable
sessions: `benchmarks/softmax-barrier-verifier40-regression-20260905.jsonl`.

### Experimental staged attention in serving (2026-09-05)

Integrated behind **unset-by-default** `G4_STAGED_ATTENTION`. Selection:
BS1 sliding uses full-width staged S32 at up to 512 keys, S64 above;
BS8 sliding uses compact S64; BS1 global at 1,024–8,192 actual context
positions uses compact S64 with two query tiles and the direct-cache QKV
preparation path. Other batch sizes/global contexts keep the existing path.
Scratch partials are GPU-resident and stream ordered; launch attributes are
initialized once outside graph capture. No per-layer host copies or device
synchronizations were introduced. Graph keys distinguish experimental/global
dispatch. Ring restoration still uses the existing output-unpack kernel.

**This did not pass generation parity.** All six workload/batch combinations
produced different token sequences in every compared pair. The staged math
changes the target forward pass, not merely draft proposals, so these results
must not be described as lossless speculative-decoding improvements. Neither
quality equivalence nor a serving-default promotion is established.

Continuous serving-core benchmark (not HTTP transport), 128-token caps,
three alternating measured pairs after warmup, native three/four drafts for
BS1/BS8 and the existing 32K text / 64K image hot-vocabulary maps. Median
emitted post-prefill tok/s; BS8 is aggregate across eight requests:

| Workload | BS1 baseline | BS1 staged | Change | BS8 baseline | BS8 staged | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text, 25 prompt tokens | 344.13 | 361.48 | +5.0% | 2,325.91 | 2,233.83 | −4.0% |
| Text, 3,865 prompt tokens | 297.19 | 352.09 | +18.5% | 1,946.03 | 2,083.93 | +7.1% |
| Image, 551 prompt tokens | 494.16 | 504.44 | +2.1% | 3,113.26 | 2,914.56 | −6.4% |

Image fixture: `infiyomi-bench/viewer/public/generated/sample-context-blurred.jpg`.
These are complete experimental-path measurements, including BS1 global
layout changes and changed acceptance/outputs, **not isolated kernel gains**.
Output length also changes in the short BS1 case (127 baseline, 128 staged).

Median mean-per-request TTFT in milliseconds, baseline -> staged:

| Workload | BS1 | BS8 |
| --- | ---: | ---: |
| Short text | 8.80 -> 8.80 | 10.54 -> 10.56 |
| 3,865-token text | 141.17 -> 140.46 | 22.97 -> 22.84 |
| Image | 63.16 -> 62.91 | 480.54 -> 480.62 |

TTFT is effectively unchanged; prefill/image work is not modified. These are
warm, repeated-prefix measurements with the benchmark running BS1 before
BS8. In particular, the lower BS8 long-text TTFT includes existing prefix
reuse and must not be read as a cold-request BS8 advantage.

Mean draft acceptance (accepted/proposed), baseline -> staged:

| Workload | BS1 | BS8 |
| --- | ---: | ---: |
| Short text | 42.51% -> 43.64% | 34.80% -> 33.49% |
| 3,865-token text | 42.26% -> 49.02% | 39.80% -> 45.00% |
| Image | 78.76% -> 76.52% | 67.15% -> 60.14% |

The paired sequences share 32/16 initial tokens for short-text BS1/BS8,
14/34 for long text, and 63 for both image batches before divergence.
Consequently, changed speculative acceptance is a material confounder in
the apparent text speedups and image slowdown. All results remain opt-in.

The A/B harness now reports mean TTFT, mean acceptance, mismatched request
counts, and mean matching token-prefix lengths. Its normal mismatch failure
remains; only explicit `G4_SWEEP_ALLOW_TOKEN_MISMATCH=1` permits recording
the rest of an experimental sweep. The first member of each pair compares
to itself; parity must be assessed using the second member, irrespective of
which flag value ran first.

Artifacts: `benchmarks/staged-serving-text-detailed-ab-20260905.jsonl`,
`benchmarks/staged-serving-text4k-detailed-ab-20260905.jsonl`, and
`benchmarks/staged-serving-image-ab-20260905.jsonl`. Earlier text runs without
the added acceptance/prefix fields are also retained as
`staged-serving-{text,text4k}-ab-20260905.jsonl`. Nothing was uploaded.
Opt-in CUDA-graph smoke tests also completed repeatably at BS1 and BS8 with
3,865-token prompts and 64-token caps (`profiles/staged-serving-graph-b{1,8}.log`).
These check execution/repeatability, not parity against baseline or HTTP/SSE.

### Parity-preserving serving attention optimization, promoted (2026-09-05)

`G4_EXACT_ATTENTION_OPT` is now enabled by default. Set it to `0` to restore
the previous path; leave the individual development overrides
`G4_FAST_GEMM_SOFTMAX` and `G4_FUSED_ATTENTION_UNPACK` unset for that comparison.
The earlier `G4_STAGED_ATTENTION` experiment remains off. No attention QK/PV
GEMM algorithm or speculative sampling policy was changed in this promotion.

Implementation:

- For at most 256 keys, one warp emulates the original 256-thread softmax
  reduction tree. Virtual lanes preserve its 128/64/32 summation stages, then
  shuffle reductions preserve the 16/8/4/2/1 stages. Both BF16 rounding steps
  remain. There are no block-wide barriers or intermediate probability stores.
- For 257–1,280 keys, a 256-thread kernel holds 2/4/5 probability values per
  thread in registers and uses the same-tree warp finish for reductions.
  For 1,281–4,096 keys only the reduction finish changes; longer global
  contexts retain the existing kernel. Sliding attention benefits at long
  contexts because its active window remains bounded. These are target
  verifier changes; assistant attention dispatch remains unchanged.
- The packed attention output is transposed and quantized for the output
  projection in one kernel. It still writes the BF16 tensor for fallback use,
  produces identical FP8 bytes/scales in tests, and marks the activation cache
  ready on the existing stream. Ring restoration uses additional blocks in
  that same launch, independently of the per-output-row quantization blocks.
- Options are evaluated once per verifier call, not per layer. Scratch and
  weights do not gain a sidecar; there are no added host/device transfers or
  device synchronizations. Graph keys distinguish the selected implementations.

The initial softmax-only variants gained mostly fractions of a percent. The
first unpack/quantization fusion was slower despite saving launches. Nsight's
warmed, CUDA-profiler-API capture identified why: at BS8 its D256 fused kernel
took 8.57 us per call, while the old unpack/restoration took 2.12 us plus the
separate quantizer. Restoration had been confined to one block per output row.
The corrected kernel restores the wide restoration grid while retaining one
launch. The profile comparison also verified 1,530 fewer quantization launches
over 51 verifier cycles: **30 removed launches per cycle**. The numerical
softmax changes remove barriers/stores but do not reduce its launch count.
Profiles of the diagnostic fusion revision are
`profiles/fused-unpack-v2-b8-{off,on}.nsys-rep`; their kernel times are not
claimed as timings of the final corrected kernel.

Final paired continuous-serving results: three alternating pairs after
warmup, native BS1/BS8 draft lengths, the existing 32K/64K text/image hot maps,
and a **256-token cap**. Median emitted post-prefill tok/s, BS8 aggregate:

| Workload | BS1 previous | BS1 optimized | Change | BS8 previous | BS8 optimized | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text, 25 prompt tokens | 344.65 | 349.44 | +1.39% | 2,325.62 | 2,361.95 | +1.56% |
| Text, 3,865 prompt tokens | 325.67 | 330.48 | +1.48% | 2,241.97 | 2,265.12 | +1.03% |
| Image, 551 prompt tokens | 515.93 | 522.77 | +1.33% | 3,470.96 | 3,513.48 | +1.23% |

Every compared token sequence and accepted-draft count matched. These gains
are not confounded by changed output or acceptance. Image input is the same
`sample-context-blurred.jpg` used above. Actual completion lengths vary: short
text ends below the cap, image BS8 emits 228 tokens/request, and long-text BS8
reaches 256. Consequently do not compare absolute rates directly against the
older 128-token-cap tables. Prefill is unchanged; measured TTFT fluctuated
without a demonstrated improvement (about 8.9/10.6 ms short text, 140/23 ms
warm repeated-prefix long text, and 63–65/488 ms image, BS1/BS8 respectively).

A separate 32,025-token prompt check, with a 32-token cap, also matched all
tokens and improved median decode from 270.54 to 273.04 tok/s BS1 and
1,214.90 to 1,223.88 aggregate tok/s BS8. This checks the long-global-context
fallback and sliding-cache path; it is not an exhaustive native-limit test.

Validation:

- `fast-softmax-test`: 36 exact comparisons, nonuniform scores, ragged
  contexts, candidate/non-candidate and global/sliding masks, key lengths
  spanning 17 through 4,096 and dispatch boundaries.
- `attention-unpack-test`: 18 exact BF16/FP8/scale cases across batch sizes
  1/2/8 and 1/4/5 query positions, plus a full mirrored-cache comparison at
  contexts 2,046 and 4,096. Total 19 cases.
- Both tests passed CUDA Memcheck and Racecheck with zero errors/hazards.
- CUDA-graph smoke with BS1 warmup and repeatable BS8 generation at a
  3,865-token prompt completed successfully.
- Temporary loopback OAI server passed 24 mixed-length requests at concurrency
  eight, token limits 1/2/7/32/64/128, cohort refills, and SSE content/finish.
  It emitted 936 completion tokens and was stopped after testing.

Artifacts: `benchmarks/exact-attention-{text,text4k,image,text32k}-ab-20260905.jsonl`,
`benchmarks/exact-attention-oai-refill-20260905.json`,
`profiles/exact-attention-{softmax,unpack}-{memcheck,racecheck}.log`, and
`profiles/exact-attention-graphs.log`. Earlier softmax/unpack trial artifacts
remain local under `fast-softmax-*` and `fused-unpack-*`. Nothing was uploaded.

### Structural serving experiments: expert plans and fused prefill (2026-09-05)

All new paths in this section are **opt-in**. The parity-preserving serving
default from the previous section remains unchanged.

The warmed fixed-verifier capture
`profiles/structural-baseline-b8-c4096/g4.nsys-rep` attributes about 24% of
summed GPU kernel duration to CUTLASS device kernels (mostly experts), with
small projection and attention GEMMs also prominent. Kernel-duration shares
are not exclusive wall-time shares when streams overlap.

Refactored the dynamic expert plan to support multiple compile-time GEMM
traits while sharing its existing GPU shapes, routing, weight pointers and
scale layouts. `G4_PREPARE_EXPERT_TILES=1` prepares alternatives at load:

| Selection | M x N x K tile | Schedule |
| --- | --- | --- |
| default | 128 x 128 x 256 | ping-pong |
| `G4_EXPERT_NARROW_TILES` | 128 x 64 x 256 | ping-pong |
| `G4_EXPERT_WIDE_TILES` | 128 x 256 x 128 | ping-pong |
| `G4_EXPERT_COOPERATIVE` | 128 x 128 x 128 | cooperative |

The initially attempted 128 x 256 x 256 tile cannot fit the required two
pipeline stages in this SM120's shared memory. The shallower wide variant
compiles but has a 1,176-byte stack frame versus 24 bytes for the default
(`cuobjdump --dump-resource-usage`); it is dramatically slower. Alternative
workspaces are private, no weights are duplicated, and graph keys separate
the selections. Environment checks for selection only execute when the
experimental plans were explicitly prepared.

Layer-0 isolated expert block, microseconds, skewed routes, real weights:

| Input rows | Default | Narrow | Wide | Cooperative |
| --- | ---: | ---: | ---: | ---: |
| 4 (BS1 verification geometry) | 66.91 | 87.23 | 279.90 | 69.57 |
| 40 (BS8 verification geometry) | 83.88 | 109.83 | 529.35 | 89.50 |
| 272 | 116.99 | 114.56 | 564.28 | 119.46 |
| 512 | 136.47 | 133.32 | 580.30 | 135.89 |

These are screening measurements, not serving tok/s. All alternatives
matched every compared BF16 expert-output byte in these cases, using
`G4_EXPERT_COMPARE_BASELINE=1`. The three-pair full-verifier narrow A/B at
4,096 context also matched sequence hashes but regressed median latency:
BS1 5,894.7 -> 6,116.4 us; BS8 7,913.4 -> 8,244.8 us. None was promoted.
Logs: `profiles/expert-{tiles,cooperative}-isolated.log` and
`benchmarks/expert-narrow-verifier-ab-20260905.jsonl`.

The more useful structural experiment is `G4_CUDNN_TEXT_PREFILL=1`:

- Replace materialized QK scores, standalone softmax and PV GEMM for the
  25 sliding layers with fused BF16 cuDNN GQA. Q has 16 D256 heads; KV has
  eight. Existing Q scaling is retained; SDPA's additional scale is one.
- Causal bottom-right alignment plus a 1,024-token left window covers both
  initial and chronological cached-prefix chunks. Sliding queries no longer
  inherit the 64-row global-attention scratch cap at long absolute contexts.
- Single-request, uniform batched and ragged per-session prefill dispatches
  are covered. Global D512 layers retain the existing implementation. Any
  prefill containing image islands retains the old masking implementation;
  no dense image mask is constructed for this experiment.
- Plans/workspaces are cached by batch, query length and key length. There
  is no new Python/Torch serving dependency. Existing global-attention scratch
  is still required, so this does not eliminate all quadratic allocations.

**Not generation-parity preserving.** Fused attention changes intermediate
rounding; neither quality equivalence nor default promotion is established.
The CPU attention oracle passed within the existing 0.00390625 absolute-error
threshold; observed sliding max error was 0.00097656 at 1,057 tokens. That
does not establish model-level quality. Large oracle fixtures sample query
rows 0, 1, 1023, 1024, 1025 and the last row. Chunked fixtures use 257-query
chunks and truncate the chronological prefix after the window wraps.

Final v2 continuous-serving A/B, 3,865-token text prompts, 128-token cap,
three alternating pairs after warmup, prefix reuse disabled, text hot map
32K. Times are medians; BS8 prefill covers the complete eight-request wave
(seven packed prefill chunks), not one request's independent prefill:

| Metric | BS1 baseline | BS1 fused | BS8 baseline | BS8 fused |
| --- | ---: | ---: | ---: | ---: |
| Target prefill ms | 135.13 | 80.48 | 802.97 | 740.86 |
| Mean request TTFT ms | 140.53 | 85.93 | 810.80 | 748.68 |
| Emitted decode tok/s (BS8 aggregate) | 300.54 | 335.99 | 2,027.15 | 2,444.95 |
| Draft acceptance | 42.26% | 52.38% | 42.16% | 56.86% |

The demonstrated prefill reductions are **40.4% BS1 and 7.7% BS8**. Decode
code was not changed by this flag: the decode-rate differences are confounded
by changed tokens/acceptance and must not be presented as decode kernel gains.
All compared generated sequences differed.

Initial no-prefix-only version on 537-token prompts reduced prefill from
17.16 to 16.02 ms BS1 and 80.96 to 72.68 ms BS8. At 3,865 tokens its BS8
prefill only fell from 803.38 to 796.25 ms because later cached-prefix chunks
fell back; this motivated the v2 extension. Initial-version reports remain
separate from v2 results.

Warmed prefill Nsight captures, using
`bash scripts/profile_text_prefill.sh 8 120 baseline` / `fused`, are in
`profiles/text-prefill-b8-r120-{baseline,fused}/g4.nsys-rep`. No model loading
was profiled. Summed kernel time falls from 856.4 to 794.8 ms. The fused trace
contains 175 sliding cuDNN attention calls, totaling about 110 ms; notably
cuDNN selects an **SM80 WMMA flash kernel**, not a Blackwell-specialized one.
Expert GEMMs remain about 170 ms in both captures, while FP8 projections,
global attention, transforms and finalization remain substantial. This is
evidence for investigating an SM120-native fused attention implementation
and prefill-specific projection tuning, not for fusing the whole pipeline
indiscriminately.

Artifacts: `benchmarks/cudnn-prefill-v2-text3865-ab-20260905.jsonl`, initial
`benchmarks/cudnn-prefill-text{537,3865}-ab-20260905.jsonl`,
`profiles/structural-prefill-ctest.log`, and
`profiles/cudnn-text-prefill-memcheck.log`. Reports stay local.

Prior art inspected: [TensorRT-LLM dense MoE experiment](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md)
inspired checking redundant-work/tiling tradeoffs, but its SM100/SM103 TP8
results do not transfer directly to this single SM120. Also reviewed the
[SGLang MoE layer](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py),
[vLLM FlashInfer CUTLASS expert backend](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py),
and [cuDNN attention support](https://docs.nvidia.com/deeplearning/cudnn/v1.28.0/operations/Attention.html).

Prefill FP8 plan tuning was also extended behind `G4_TUNE_FP8_PREFILL=1`.
The existing initialization-time tuner previously only covered at most 64
padded rows. Larger-row trials now consider the same 32 cuBLASLt heuristics,
reject candidates differing from the first successful candidate's sampled
BF16 output, and cache the fastest qualifying plan. This adds startup-time
GPU/CPU comparisons and synchronizations, never per-layer steady-state copies.
It is not enabled in normal serving; a new untuned shape can incur substantial
first-use latency with the flag, so it is a development tuning tool.

`tools/bench_prefill_plans.py` uses separate processes because changing an
environment flag after plan initialization is not a valid A/B. It warms each
shape and disables prefix reuse. One screening pair at 3,865 prompt tokens
and a 32-token cap found no useful prefill gain: BS1 142.95 -> 144.41 ms,
BS8 791.52 -> 793.58 ms. Generated tokens and accepted-draft counts matched.
These screening timings are not directly comparable with the earlier
continuous-scheduler table. No tuned plan was promoted. Local records:
`benchmarks/fp8-prefill-plan-b{1,8}-ab-20260905.jsonl`.

An additional `G4_PRECOMPUTED_ROPE=1` experiment removes repeated per-head,
per-layer rotary transcendental evaluation in prefill. A process-owned
192 MiB GPU table stores BF16 sine/cosine pairs for 128 sliding frequencies
and 64 global frequencies across all 262,144 native positions. The global
non-rotating dimensions retain exact identity pairs. GPU initialization is
once, outside steady-state prefill; no new per-layer kernel or host copy is
needed. The table is shared by single, uniform-batch and ragged prefills.
Decode remains on its prior path.

`rope-table-test` passes 32 byte-exact whole-Q/K/V comparisons covering
D256/D512, 1/4/40/272 rows, FP8 scales, tied global K/V, ragged positions,
window boundaries and the last native position. Memcheck reports zero
errors. All text serving A/B sequences and accepted-draft counts matched,
and the default hashes matched the pre-refactor baseline. Nevertheless,
the three-pair 3,865-token serving test found essentially no benefit:
BS1 prefill 136.100 -> 136.294 ms, BS8 807.088 -> 806.509 ms. This rules out
repeated rotary transcendental evaluation as a major prefill bottleneck in
these kernels; do not pay the VRAM cost in default serving. Artifacts:
`benchmarks/rope-table-text3865-ab-20260905.jsonl`,
`profiles/rope-table-{correctness,memcheck}.log`.

The cuDNN text flag's image-fallback serving A/B passed all sequence and
acceptance comparisons at BS1/BS8 with the 551-token image fixture and
32-token caps: `benchmarks/cudnn-prefill-image-fallback-ab-20260905.jsonl`.
The cached-prefix cuDNN attention oracle also passed Memcheck:
`profiles/cudnn-text-prefill-chunks-memcheck.log`.

Final checks: all 12 selected regression CTests passed
(`profiles/structural-final-ctest.log`); experimental text prefill plus rotary
tables completed a repeatable BS8 serving run with BS1 warmup and decode
CUDA graphs (`profiles/structural-serving-graphs.log`). No server was left
running. The v2 3,865-token prefill times correspond to 28.6k -> 48.0k target
prefill tok/s BS1 and 38.5k -> 41.7k aggregate target prefill tok/s BS8. These
exclude tokenization, image processing and output decode; TTFT is reported
separately above. This remains an experimental speed result, not a validated
quality-equivalent replacement.

### GPU source organization, behavior-preserving (2026-09-05)

The 13,053-line `src/gpu.cu` was split into 40 private CUDA implementation
fragments and three separately compiled host C++ units. The 81-line CUDA
assembly file retains one translation unit; kernels still precede runtime
launchers, and public diagnostics are grouped at the end. All 12,705 lines
of fragment contents were checked against the original source unchanged.
Device selection, stream/graph management and weight-arena uploads moved to
`src/gpu/`; see [the layout guide](code-layout.md).

All 35 CTests passed (`profiles/gpu-refactor-ctest.log`). All 12 pre/post
fixed-verifier trials at BS1/BS8 matched sequence hashes and remained
repeatable. Default-path median verifier times were 5,804.77 -> 5,811.61 us
BS1 and 7,818.14 -> 7,830.81 us BS8, changes below 0.2%; this reorganization
does not demonstrate a performance change. Records:
`profiles/refactor-verifier-{before,after}.jsonl`.
The final BS8 serving run with BS1 warmup and decode CUDA graphs was also
repeatable (`profiles/gpu-refactor-graphs.log`). No server remains running.

### SM120-native fused sliding prefill (2026-09-05)

Added an **opt-in** AOT-compiled native SM120 D256 attention backend, using
NVIDIA's MIT CuTeDSL implementation. It is linked into the C++ executable;
Python/torch are not serving dependencies. Dynamic batch/chunk shapes share
two compiled tile variants. See [build, dispatch and correctness details](sm120-attention.md).
The original attention, global D512 layers, image-query masking and decode
attention remain unchanged by default.

Warmed BS8/3865-token serving Nsight capture, compared with the preceding
cuDNN fused-prefill capture:

| Sliding attention | cuDNN SM80-named kernel | Native SM120 |
| --- | ---: | ---: |
| Launches | 175 | 175 |
| Summed GPU time | 110.288 ms | 70.614 ms |
| Mean launch duration | 630.216 us | 403.508 us |

That is **36.0% less GPU time for this stage**, not a launch-count reduction.
The native capture contains the SM120 kernel in place of all 175 cuDNN
sliding-attention launches. It still uses `mma.sync`; architecture naming
alone is not the explanation for the improvement. Reports:
`profiles/text-prefill-b8-r120-{fused,native}/{g4.nsys-rep,kernels.csv}`.
Summed GPU durations are not exclusive wall time when streams overlap.

Final continuous-serving A/B, three alternating pairs per batch after warmup,
text-only 3865 tokens per request, output cap 128, shared-prefix cache disabled,
32K text assistant hot map. Baseline below is **cuDNN fused prefill**, not the
original unfused attention:

| Median | BS1 cuDNN | BS1 native | BS8 cuDNN | BS8 native |
| --- | ---: | ---: | ---: | ---: |
| Prefill ms | 80.226 | 76.074 | 743.966 | 705.563 |
| TTFT ms | 87.251 | 82.872 | 752.125 | 713.688 |
| Observed aggregate decode tok/s | 326.261 | 377.772 | 2369.119 | 2355.338 |
| Draft acceptance | 52.38% | 64.84% | 56.86% | 55.77% |

Prefill time improves **5.2% at both BS1 and BS8**. BS8 prefill is the complete
eight-request wave (30,920 input tokens, seven packed chunks), not the time
for one independent request. TTFT includes host/request preparation.
Record: `benchmarks/sm120-auto-serving-text3865-ab-20260905.jsonl`.
The forced-64 and forced-128 preliminary serving runs are preserved in
`benchmarks/sm120-q{64,128}-serving-text3865-ab-20260905.jsonl`; both similarly
reduced prefill time by roughly 4–5%. The simple automatic tile heuristic is
not an established optimum across every possible shape.

**All compared generated sequences differ between cuDNN and native** (the
first three output tokens match on this fixture). Therefore the observed
decode/acceptance changes are workload changes, **not decode-kernel speedups**.
No model-quality equivalence or promotion to the default serving path is
claimed. Both native tile variants agree bitwise in the isolated fixtures.

Final isolated attention samples (five alternating trials, each 100 launches;
reported median, without model loading or CPU oracle inside timing):

| B | Sq / Sk | cuDNN us | Native us | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 537 / 537 | 33.42 | 24.55 | 1.36x |
| 1 | 3865 / 3865 | 470.18 | 293.62 | 1.60x |
| 8 | 512 / 1536 | 650.07 | 382.35 | 1.70x |
| 8 | 3865 / 3865 | 4186.89 | 2555.81 | 1.64x |
| 8 | 4608 / 5632 | 5663.17 | 3245.61 | 1.74x |

Tiny B1/Sq=1 cached-prefill work is effectively tied (41.40 versus 41.71 us).
Clocks were not locked; raw timing samples show variation, especially on the
longer fixtures. These isolated results do not replace the serving A/B above.
All batches 1..8 were exercised, including non-tile-aligned tails and cached
prefixes; all passed the independent sampled oracle, finite-output comparison,
exact tile-variant comparison and exact CUDA-graph replay checks. Records:
`benchmarks/sm120-attention-auto-isolated-20260905.jsonl` and preliminary
`benchmarks/sm120-attention-q{64,128}-isolated-20260905.jsonl`.

Seven focused CTests passed (`profiles/sm120-attention-ctest.log`), including
the original mask-aware multimodal test and the native chunked window test.
Compute Sanitizer memcheck exercised both native tiles at B8/Sq129/Sk1153:
**zero errors**, `profiles/sm120-attention-memcheck.log`. BS8 full serving with
BS1 warmup and decode graphs remained repeatable:
`profiles/sm120-serving-graphs.log`.
The 12 image-serving A/B records (BS1/BS8, output cap 32) have zero mismatched
requests and identical output hashes/accepted counts, confirming the fixture
stays on its image-mask fallback:
`benchmarks/sm120-image-fallback-ab-20260905.jsonl`.
The final complete suite passed **39/39** tests in 104.87 seconds:
`profiles/sm120-final-ctest.log`.

### Full Infiyomi quality gate and serving profile (2026-09-05)

The later [full-corpus evaluation](infiyomi-sm120-evaluation.md) tests the fast
bundle through the OAI server, rather than requiring byte-identical outputs.
All runs use concurrency 8, 240 pages, five attempts and retry-incomplete.
Translation scores below use the same fixed vLLM 0.24 judge, including a local
re-score of the historical reference; the reference generation timing is old.

| Serving setup | Output cap | Successful pages/s | Quality / 10 |
| --- | ---: | ---: | ---: |
| Native default | 4096 | 1.1227 | 8.318 |
| Fast eager | 4096 | 1.1859 | 8.320 |
| Native default | 8192 | 0.9400 | 8.283 |
| **Recommended fast eager** | **8192** | **1.1226** | **8.371** |
| Historical int8 + MTP reference | 4096 | 1.1773 | 8.449 |

The recommended 8192 profile improves successful-page throughput 19.4% over
the matched native default and meets the predeclared 0.10 mean-quality-drop
tolerance versus the reference. Individual regressions and statistical
uncertainty remain: this is not a lossless or proven-equivalent quality claim.
The smaller cap is faster but misses that reference-quality gate. Use
`bash scripts/serve_infiyomi_fast.sh 8080` with the SM120 AOT build and client
`max_tokens: 8192`; raw runtime defaults remain available for rollback.
Detailed timings, coverage, paired intervals, the ragged-KV growth fix and
reproduction instructions are in the linked report. Compact local JSON:
`benchmarks/infiyomi-sm120-summary-20260905.json`.
