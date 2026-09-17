# Jev-inspired decisions with DiffusionGemma

Research date: 2026-09-18. Proposed experiments below are hypotheses, not measured results.

Confirmed workload: many questions at once. Prioritize multiple questions about a shared state; treat unrelated states as a separate request-batching axis. The principal experiment is how latency, accuracy, and calibration scale with question count.

## What we can reproduce

The useful target is an interface that accepts state and questions and returns typed decisions with distributions, quickly. We can test this using an open diffusion model without assuming that its internals match Jev.

TypeSafe launched Jev on September 15. Its launch describes a new architecture, parallel sampler, and Reinforcement Learning for Calibrated Decisions (RLCD). The reviewed primary materials do not provide enough architecture or algorithm detail to reproduce that stack or establish that Jev uses diffusion. Treat the advertised 193.6× speed and 444.6× cost advantages as vendor workflow results, not expected local speedups. [Launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [vendor homepage](https://typesafe.ai/).

The documented primitives are Choice (option and distribution), Score (rubric-based score and distribution), and Noul (probability of a statement being true). TypeSafe says questions are evaluated independently and in parallel against the same state. This distinction matters: packing questions into an ordinary bidirectional canvas does not isolate them. [Introduction](https://docs.typesafe.ai/introduction).

Score is the probability-weighted mean of level indices. Its confidence describes concentration of the distribution; even confidence 1 does not guarantee correctness. Our implementation should expose probabilities, entropy, and a selected answer explicitly, without claiming an undocumented compatibility formula. [Score documentation](https://docs.typesafe.ai/primitives/score).

Three separate targets must be measured: schema validity, task accuracy, and probability calibration. Restricting the answer vocabulary can guarantee valid enums while still producing confidently wrong decisions.

## Why DiffusionGemma is worth trying

Google describes a 26B-total / approximately 4B-active MoE with a context encoder and bidirectional denoising over 256-token canvases. Its reference settings use up to 48 steps, temperature annealing from 0.8 to 0.4, an entropy bound of 0.1, and adaptive stopping. Those settings provide a baseline rather than an optimum for classification. [Google overview](https://ai.google.dev/gemma/docs/diffusiongemma).

Use `google/diffusiongemma-26B-A4B-it` as the initial checkpoint. [Model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it).

The Hugging Face integration exposes `EntropyBoundScheduler`, `BlockRefinementScheduler`, and `DiscreteDDIMScheduler`, including a predictor-corrector option. This supplies useful alternative samplers before implementing novel ones. [Pipeline documentation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/diffusion_gemma.md).

The inspected implementation uses random-token initialization, rounds generation length to a canvas multiple, and invokes its callback after scheduler updates. A post-step canvas edit alone cannot correctly enforce pre-sampling constraints, synchronize scheduler state, or fix stopping logic. Use a small explicit denoising loop or carefully adapted scheduler with instrumentation. Record raw logits before temperature shaping. [Inspected source](https://github.com/huggingface/diffusers/blob/7221eef4573574925b67a69e9fc1482bf093e569/src/diffusers/pipelines/diffusion_gemma/pipeline_diffusion_gemma.py).

The related uniform-diffusion paper derives leave-one-out posterior conversions and inference-time predictor-corrector improvements. It motivates a sampler ablation; it does not establish calibration or correctness for our constrained task. [Paper](https://arxiv.org/abs/2605.22765).

## Recommended representation

Begin with one-token aliases for each categorical answer. For example, a routing question maps A/B/C to billing/technical/account, a boolean maps A/B to false/true, and an ordinal question maps aliases to rubric levels. Describe mappings in the prompt; select only verified token IDs at known answer positions. Serialize the returned Python values to JSON in code.

Verify tokenization in the complete template: a visible character is not necessarily one token, and whitespace can change its token ID. Reject collisions and unsupported cardinalities. Randomize alias assignments and option order during evaluation to detect token and position bias.

For answer slot i with legal token set A_i, compute a provisional distribution:

`p_i(c) = softmax(raw_logits[i, A_i] / T_cal)[c]`

Here `T_cal` is fitted on calibration data; it is distinct from the sampling temperature. Save unconstrained probability mass assigned to A_i before renormalization. A near-certain choice after masking may otherwise hide that the model assigned almost no mass to any legal answer.

Denoiser logits depend on timestep and current noisy canvas. These distributions are decision scores until validated; they are not automatically exact sequence probabilities or calibrated posteriors. Compare several readout times and average distributions across independent noise seeds as a separate, costed experiment.

Use a fixed template containing answer slots and immutable delimiters. Keep delimiters fixed throughout denoising, and project clean-token predictions onto each slot's legal set. Maintain the original uniform-noise transition for noisy states in the first implementation; constraining noisy states themselves changes the process and needs its own ablation. Do not copy an absorbing-mask sampler blindly.

A final serializer guarantees enum types. Joint rules, such as mutually exclusive actions or conditional fields, need a valid-assignment projection or explicit application logic. Independent per-token masks do not enforce general JSON Schema constraints. Start with finite categorical outputs; arbitrary strings and nested variable-length JSON are later work.

## Experiment order

| Stage | Comparison | Question answered |
| --- | --- | --- |
| 0 | Stock reference sampler, normal answer generation | Can the checkpoint run correctly and solve the selected tasks? |
| 1 | One question, one alias slot; then 4/16/64 packed questions | Does constrained parallel prediction preserve accuracy and amortize canvas cost? |
| 2 | Entropy-bound, block refinement, discrete DDIM, predictor-corrector | Which sampler gives the best decision quality per measured forward and millisecond? |
| 3 | Fixed budgets of 1/2/4/8/16/32/48 decoder forwards | How much iterative refinement do these decisions need? |
| 4 | Answer-only adaptive stopping versus fixed budget | Can we stop early without premature confident errors? |
| 5 | Single-seed logits versus 4/8-seed probability averages | Does noise averaging improve calibration enough to justify cost? |
| 6 | Calibration on held-out data; optional supervised adapter later | Is inference alone sufficient, or is task training needed? |

Do not run the full Cartesian product initially. Establish stages 0–1 on a small development set, sweep samplers at 4/8/16/48 forwards, then expand the promising settings.

For adaptive stopping, inspect only answer slots: immutable punctuation and filler would dilute whole-canvas entropy. Require stable answers and bounded distribution change over consecutive passes; choose thresholds on development data. Stability is not correctness. Record actual forwards, including correctors and readout passes, since nominal step counts and runtime can differ.

Compare three arrangements: packed questions sharing a canvas; separately batched question prompts; and later an isolated-attention/shared-state design if the first two justify implementation. The last is an architecture intervention and may change model behavior. Perturb or add irrelevant questions and measure whether existing answers change. Do not claim Jev-like independence from packing alone.

Keep 256-token canvases for the first experiment. Requesting a shorter output does not itself reduce canvas compute. A smaller canvas or position-selective vocabulary projection is a later optimization requiring correctness checks; fewer output symbols alone do not guarantee lower latency.

## Evaluation

Start with synthetic, deterministically labeled rule decisions to validate the machinery. Add labeled intent routing, boolean factual judgments against supplied state, and ordinal rubric tasks. Include missing information, ambiguous examples, conflicting evidence, and shifted wording. A synthetic test validates mechanics, not general intelligence.

Split by source case into development, calibration, and untouched test sets; all questions derived from a case remain in one split. Use human or deterministic labels where available. Keep any model-generated reference judgments separately identified.

Record per-question accuracy/macro-F1, schema validity, joint-constraint violations, negative log likelihood, multiclass Brier score, reliability plots, and accuracy versus abstention coverage. Add MAE for ordinal expected scores. ECE is supplementary and sensitive to binning. Use paired bootstrap intervals grouped by case when comparing settings.

Measure end-to-end p50/p95 latency, prefill and denoising time, decisions/second, actual forwards, and peak allocated/reserved VRAM. Separate cold load/compile from warm runs. Synchronize CUDA around timings. Sweep input lengths such as 512/2,048/8,192 tokens and request batch sizes independently from questions per request. Never substitute generated tokens/second for decision latency.

Include a competent autoregressive baseline with one-token constrained decisions and batched questions, not just verbose JSON generation. This tests whether diffusion adds value beyond eliminating unnecessary text. A smaller classifier is also a useful efficiency baseline. No Jev API comparison has been run.

## Local hardware and implementation plan

Observed with `nvidia-smi` on 2026-09-18:

| Device | Total memory | Free at inspection |
| --- | --- | --- |
| RTX PRO 6000 Blackwell Workstation Edition | 97,887 MiB | 97,273 MiB |
| GeForce RTX 5090 | 32,607 MiB | 29,696 MiB |

Driver: 610.57.04. Free memory is transient. Select the PRO 6000 by GPU UUID via `CUDA_VISIBLE_DEVICES`; inside that process it becomes cuda:0. Avoid automatic placement across both cards.

A 26B-parameter model at two bytes per parameter implies roughly 52 GB / 48.4 GiB for weights, before caches, activations, workspaces, and implementation-specific overhead. This makes BF16 a reasonable starting hypothesis on the 96 GB card, not a measured memory guarantee. Begin with short context and batch size 1. Active parameter count does not determine resident weight memory.

Establish BF16 quality first. Evaluate lower precision only afterward, because small logit changes can affect probabilities and calibration. Verify the actual PyTorch/CUDA and kernel combination supports this Blackwell GPU. Pin model revision, tokenizer revision, Transformers and Diffusers commits, environment versions, seed, prompt, and sampler configuration for each run. The inspected Diffusers main revision was `7221eef4573574925b67a69e9fc1482bf093e569`; compatibility with an installed environment has not been tested.

Suggested first implementation consists of a schema-to-slot compiler, an instrumented single-canvas denoising loop, a raw-logit probability readout, and an evaluation CLI emitting JSONL. Begin uncompiled for debugging; compile only after parity checks. Keep all model-specific details in one backend so an autoregressive comparator can share datasets and metrics.

First milestone: reproduce stock inference, then compare 1/4/16/32/64 categorical questions at 4/8/16/48 forwards on the same labeled cases. Start with 16 and 64 questions; the single-question case is a control. Test 128 only if the verified tokenized answer layout fits the canvas. Encode shared state and question definitions once per packed request, then refine all answer slots together. Report total request latency and decisions/second: amortized time per answer is not individual request latency. Longer question lists increase prefill work even with a fixed output canvas.

Advance if constrained slots preserve useful accuracy and improve latency or throughput relative to the baselines. Fit calibration only once a useful operating point exists. If inference changes fall short, try supervised adapter training with proper probability losses before attempting any speculative RLCD recreation.

No weights have been downloaded, dependencies installed, or GPU experiments run as part of this research pass.
