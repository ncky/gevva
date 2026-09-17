#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch="${1:-8}"
context="${2:-4096}"
drafts="${3:-4}"
profile_dir="${G4_PROFILE_DIR:-$repo_root/profiles/fixed-b${batch}-c${context}-scoped}"

mkdir -p "$profile_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d
export G4_PROFILE_FIXED=1

# Capture only the warmed timed loop. Profiling process startup/weight upload
# has previously exercised a driver/IOMMU failure on this workstation.
exec nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx,cublas \
  --output="$profile_dir/g4" \
  "$repo_root/build/g4" target-batch-bench "$batch" "$context" "$drafts"
