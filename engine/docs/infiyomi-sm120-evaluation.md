# Infiyomi serving evaluation — 2026-09-05

Local-only evaluation of the SM120-era serving bundle. Results are not uploaded,
and the external benchmark's archived results and viewer are not modified.
The fast **8192-cap Infiyomi serving profile is recommended** on the measured
aggregate-quality gate below; numerical identity is not the acceptance criterion.
The raw runtime's defaults remain available for rollback and other workloads.

## Setup

RTX PRO 6000 Blackwell, concurrency 8, all 240 pages / 3328 labelled regions of
`manga109-s-10x24`. Harness commit
`da43ed47831f4513a31c497e47ea34ff4c98ed88`. Five attempts, retry incomplete pages,
no retry specifically for empty translations; responses saved. This matches
the published reference's retry policy. Prefix prewarm and native shared-prefix
caching remain enabled. Timings exclude loading and the separate quality pass.

Candidate environment (in addition to the pinned GPU):

```sh
G4_CUDNN_TEXT_PREFILL=1
G4_SM120_TEXT_PREFILL=1
G4_FUSED_TARGET_GATE_UP=1
G4_ASSISTANT_HOT_VOCAB_TEXT=assets/gemma4-assistant-hot-vocab-32768.bin
G4_ASSISTANT_HOT_VOCAB_MULTIMODAL=assets/gemma4-assistant-hot-vocab-65536.bin
```

Default MTP lengths remain 3 at BS1 and 4 at BS2–8. Graphs and staged attention
are excluded: adding either made the first-24-page smoke run slower. The first
24 pages are one sparse chapter, not a representative substitute for all 240.
This is a bundle comparison, not an isolated attribution to SM120 attention;
image-mask prefill retains its existing fallback.

## Matched 4096-output-cap runs

| Configuration | Successful pages | Wall seconds | Successful pages/s | Returned / translated regions | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native default | 238 / 240 | 211.989 | 1.1227 | 3312 / 3286 | 276 |
| Fast eager bundle | 238 / 240 | 200.687 | 1.1859 | 3303 / 3280 | 273 |
| Archived vLLM int8 + MTP reference | 239 / 240 | 203.015 | 1.1773 | 3317 / 3299 | 248 |

The candidate gains 5.6% successful pages/s over the native default, but also
generates less work: 305622 versus 315961 completion tokens across all attempts.
Attempted completion throughput is 1522.9 versus 1490.5 tok/s (+2.2%). These are
whole-run throughput figures, including prefill/retries, not isolated decode
rates. The historical reference was not rerun for this timing comparison.

Artifacts: `benchmarks/infiyomi-full240-current-vs-fast-20260905.json` and
`benchmarks/infiyomi-sm120-evaluation-20260905.json` contain the run paths and
provenance. Quality has not yet been established from these timing results.

## Larger-cap failure and fix

Several dense pages repeatedly exhausted 4096 output tokens. The initial
8192-cap candidate run exposed an existing ragged direct-global-attention bug:
batched GEMMs read the longest row's span from every row's KV pointer, but short
rows had only their own logical span physically mapped. Crossing the initial
8192-token mapping poisoned the CUDA context. That failed run is preserved and
**is not a throughput result**.

The fix maps the full direct-GEMM read span in every participating cache and
zeros newly committed pages (masked `0 * NaN` is still NaN). Growth is ordered
on the caller's stream, and resident coverage tracks physical page capacity to
avoid per-token driver bookkeeping. Independently smaller virtual reservations
fall back to the safe packed path.

The isolated ragged BS8 test (one 8225-token row, seven 4096-token rows, four
drafts) went from **2755 Compute Sanitizer invalid reads to zero**. Its output
hash, `7974818885363009435`, matches the packed-attention control. Logs:
`profiles/ragged-kv-growth-{before,after}-memcheck.log` and
`profiles/ragged-kv-growth-packed-control.log`.

Post-fix CTest initially passed 39/40, with one short multimodal golden-output
mismatch. That same unmodified test then passed individually and on five
consecutive fresh-process repeats. The transient mismatch remains recorded;
the golden was not relaxed. Logs:
`profiles/ragged-kv-growth-final-ctest.log` and
`profiles/ragged-kv-growth-repeatability-recheck.log`.

## Matched repaired 8192-output-cap runs

| Configuration | Successful pages | Wall seconds | Successful pages/s | Returned / translated regions | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native default | 238 / 240 | 253.201 | 0.9400 | 3313 / 3282 | 265 |
| Fast eager bundle | 239 / 240 | 212.891 | 1.1226 | 3328 / 3310 | 267 |

The fast bundle gives 19.4% higher successful-page throughput here, with almost
the same attempted completion count (321267 versus 320032). Whole-run attempted
completion throughput is 1509.1 versus 1263.9 tok/s. First-page latency is 1.421
versus 1.552 seconds; this is **TTFP**, not streaming token TTFT.

This larger difference is not an isolated 19% kernel gain. In particular, the
default's tail includes three standalone 8192-cap retry waves, each spending
about 17 seconds in the verifier. Changed outputs change retries, acceptance,
batch occupancy and the amount of serial tail work. These are single complete
runs of a dynamically batched task, not confidence intervals for speed.

Neither repaired run has an inference CUDA/HTTP failure. The candidate's one
incomplete page contains an extra region; all gold regions are returned. Nine
candidate attempts exceed 8192 total tokens, reaching 11576. This is boundary
coverage for the fix, **not validation of the full native context limit**.
Record: `benchmarks/infiyomi-full240-8192-default-vs-fast-20260905.json`.

## Fixed-backend quality assessment

The archived quality score used vLLM 0.23. All comparisons here use one fixed
vLLM 0.24 int8-weight-only + official four-token MTP judge in eager mode, with
FP8 KV and a 32768-token context. See `scripts/infiyomi_reference_judge.sh` and
`benchmarks/infiyomi-vllm-int8-judge.json`. The native and reference backends
never run concurrently. The judge sees gold source text and saved model text,
not images. This is an automated, same-model-family translation assessment,
not independent human validation or a broad language-model benchmark.

The reference was copied locally and re-scored; checksums confirm its original
archived summary, page results and quality files remain unchanged. Its new
score is **8.449/10**, versus archived 8.388. It scores 3326/3328 regions; two
remain missing after three judge attempts. Assigning both unknown scores 0 or
10 bounds its full-dataset mean to **[8.4437, 8.4497]**.

At the matched 8192 output cap, native default scores **8.283**, fast **8.371**,
both over all 3328 regions. Five thousand paired page-cluster bootstrap draws
give these candidate-minus-comparator differences:

| Comparator | Paired mean difference | 95% interval |
| --- | ---: | ---: |
| Native default, 8192 | +0.0886 | [-0.0298, +0.2168] |
| Re-scored reference, common 3326 regions | -0.0756 | [-0.1703, +0.0254] |

The fast bundle meets the **0.10 mean-score-drop tolerance recorded before
judging**, improves coverage and throughput relative to the matched default,
and has no inference CUDA/HTTP failure. The reference interval nevertheless
includes losses greater than 0.10: this is a pragmatic workload-specific gate,
**not a statistical non-inferiority or lossless-quality claim**.

Individual regressions are real. Inspection of the worst paired pages found
judge-flagged misassigned dialogue on `sample-01-kyokugencyclone-p020`, lost text and wrong OCR
on `sample-03-everydayosakanachan-p017`, and a wrong line on
`sample-02-yamatonohane-p020`. Their mean page deltas versus native default are
-2.267, -1.667 and -1.667. The paired JSONs retain the ten worst pages; the full
saved responses and judge reasons remain local for inspection. Aggregate
improvement does not mean every request is better.

The 4096-cap default scores **8.318**, fast **8.320**, both with all 3328 regions
scored. The fast 4096 mean is 0.129 below the re-scored reference and therefore
does not pass the predeclared 0.10 tolerance. Its 1.1859 pages/s is the fastest
of these native full runs, but it is not the recommended quality-qualified cap.

## Recommended serving profile

Build the [optional SM120 AOT backend](sm120-attention.md), then run:

```sh
bash scripts/serve_infiyomi_fast.sh 8080
```

Use `max_tokens: 8192` in the client; the local benchmark config is
`benchmarks/infiyomi-native-8192.json`. The launcher pins the RTX PRO 6000 and
the measured bundle, clearing inherited `G4_*` experiment flags in its child
environment. It loads only the native target and official assistant; vLLM and
Python are **not serving dependencies**. Prefix caching, continuous batching,
SSE streaming and existing context handling remain enabled. Graph and staged
attention experiments remain off. Explicit client caps are always honored.

This promotes a workload profile, not all experimental flags or raw-library
defaults. To roll back, start `./build/g4 oai-serve 8080` from a shell without
experimental `G4_*` variables. Broader text quality, per-request equivalence and
near-native-limit throughput have not been established by this manga benchmark.
The 8192 profile is still about 4.6% below the historical reference's pages/s,
at a different output cap; do not describe it as beating that reference.

Final validation passed **40/40 CTests** in 470.94 seconds, including the
unmodified multimodal golden, ragged KV growth and native attention tests:
`profiles/infiyomi-final-ctest-20260905.log`. Three CPU-only comparison-tool unit
tests also pass. The earlier transient golden mismatch remains documented
above rather than erased by the successful reruns.

The actual launcher was tested from `/tmp` with deliberately inherited graph
and staged-attention flags; its recorded child environment contains only the
measured five `G4_*` flags. Its 24-page OAI smoke completed **24/24** in 9.672s,
174/174 regions, 173 translations, no retries or request errors. This is a
sparse-chapter smoke, not another full-corpus speed claim. Artifacts:
`profiles/infiyomi-promoted-launcher-environment-20260905.txt` and
`profiles/infiyomi-promoted-launcher-smoke24-20260905/20260905-113842-g4-native-8192-manga109-s-10x24`.

Both temporary servers are stopped. All benchmark and quality artifacts remain
local; no result or dataset was uploaded. Compact machine-readable report:
`benchmarks/infiyomi-sm120-summary-20260905.json`.
