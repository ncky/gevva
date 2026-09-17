#!/usr/bin/env bash
# Temporary independent quality judge, not the native serving deployment.
# Keep this process separate from all native/GPU benchmarks on this machine.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d

# vLLM 0.24 replaces the unavailable archived 0.23 executable. Eager mode
# avoids compiling another serving deployment just for this untimed judge.
# Rejudge the saved reference with this same process before comparing scores.
exec /mnt/SSD/gemma4/.venvs/vllm-latest/bin/vllm serve \
  /mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model \
  --served-model-name page-vlm --host 127.0.0.1 --port "${1:-8080}" \
  --dtype auto --gpu-memory-utilization 0.75 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":1,"audio":0}' \
  --mm-processor-kwargs '{"max_soft_tokens":560}' \
  --quantization int8_per_channel_weight_only --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 10G --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --speculative-config '{"method":"mtp","model":"/mnt/SSD/g4-models/gemma4-26b-a4b-assistant","num_speculative_tokens":4}' \
  --max-num-seqs 8 --max-num-batched-tokens 16384 --enforce-eager
