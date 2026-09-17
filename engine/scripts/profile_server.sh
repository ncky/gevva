#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${1:-/mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/infiyomi-rendered-browser-page1.jpg}"
profile_dir="${G4_PROFILE_DIR:-$repo_root/profiles/multimodal-serve-warm}"
mkdir -p "$profile_dir"

export CUDA_VISIBLE_DEVICES="GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

warmup="{\"id\":\"warmup\",\"image\":\"$image\",\"prompt\":\"<|image|>Describe this image.\",\"max_tokens\":16}"
profiled="{\"id\":\"profiled\",\"image\":\"$image\",\"prompt\":\"<|image|>Describe this image.\",\"max_tokens\":16,\"profile\":true}"

printf '%s\n' "$warmup" "$profiled" | nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$profile_dir/g4" \
  "$repo_root/build/g4" multimodal-serve
