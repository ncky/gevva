#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile_dir="${G4_PROFILE_DIR:-$repo_root/profiles}"
mkdir -p "$profile_dir"

# The binary itself checks physical PCI identity; this UUID also prevents CUDA
# libraries and profilers from creating a context on the 5090.
export CUDA_VISIBLE_DEVICES="GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --force-overwrite=true \
  --output="$profile_dir/g4" \
  "$repo_root/build/g4" "$@"
