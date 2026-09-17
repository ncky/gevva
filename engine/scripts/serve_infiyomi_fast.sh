#!/usr/bin/env bash
# Benchmark-qualified Infiyomi serving profile for this workstation.
# See docs/infiyomi-sm120-evaluation.md for measured quality and limitations.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reproduce the measured bundle, without inheriting unrelated experiment flags.
# This only changes the child process environment, never the caller's shell.
for g4_option in "${!G4_@}"; do
  unset "$g4_option"
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
export G4_CUDNN_TEXT_PREFILL=1
export G4_SM120_TEXT_PREFILL=1
export G4_FUSED_TARGET_GATE_UP=1
export G4_ASSISTANT_HOT_VOCAB_TEXT="$repo_root/assets/gemma4-assistant-hot-vocab-32768.bin"
export G4_ASSISTANT_HOT_VOCAB_MULTIMODAL="$repo_root/assets/gemma4-assistant-hot-vocab-65536.bin"

# Requires the optional SM120 AOT backend in this build. The server defaults to
# 8192 output tokens; clients should request that explicitly for this benchmark.
# An explicitly smaller client cap is honored, not silently overridden.
exec "$repo_root/build/g4" oai-serve "${1:-8080}"
