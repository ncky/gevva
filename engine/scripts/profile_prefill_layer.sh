#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
layer="${1:-0}"
tokens="${2:-272}"
profile_dir="${G4_PROFILE_DIR:-$repo_root/profiles/prefill-l${layer}-t${tokens}-scoped}"

mkdir -p "$profile_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
export G4_PROFILE_PREFILL=1

exec nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cublas \
  --output="$profile_dir/g4" \
  "$repo_root/build/g4" prefill-layer-bench "$layer" "$tokens"
