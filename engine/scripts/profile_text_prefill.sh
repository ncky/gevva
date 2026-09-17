#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch="${1:-8}"
repeats="${2:-120}"
variant="${3:-baseline}"
profile_dir="${G4_PROFILE_DIR:-$repo_root/profiles/text-prefill-b${batch}-r${repeats}-${variant}}"
case "$variant" in
  baseline) unset G4_CUDNN_TEXT_PREFILL G4_SM120_TEXT_PREFILL ;;
  fused) export G4_CUDNN_TEXT_PREFILL=1; unset G4_SM120_TEXT_PREFILL ;;
  native) export G4_CUDNN_TEXT_PREFILL=1 G4_SM120_TEXT_PREFILL=1 ;;
  *) echo 'variant must be baseline, fused, or native' >&2; exit 2 ;;
esac
mkdir -p "$profile_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
export G4_BATCH_SERVE_TEXT_ONLY=1
export G4_BATCH_SERVE_PROMPT_REPEATS="$repeats"
export G4_BATCH_SERVE_TEST_TOKENS=1
export G4_BATCH_SERVE_WARMUP=1
export G4_BATCH_SERVE_WARMUP_SAME_PREFIX=1
export G4_DISABLE_SHARED_PREFIX_CACHE=1
export G4_PROFILE_BATCH_SERVE=1
unset G4_BATCH_SERVE_SWEEP_ENV

# Capture only a warmed full serving prefill, never checkpoint loading.
exec nsys profile --force-overwrite=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none --trace=cuda,nvtx,cublas \
  --output="$profile_dir/g4" \
  "$repo_root/build/g4" batch-serve-test unused "$batch"
