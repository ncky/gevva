#!/usr/bin/env python3
"""Pack a tied BF16 vocabulary matrix as rowwise symmetric INT8."""

import argparse
import json
import os
from pathlib import Path


TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"


def resolve_shard(model: Path, tensor: str) -> Path:
    index = model / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return model / weight_map[tensor]
    return model / "model.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name or str(props.pci_bus_id).lower() not in {
        "15",
        "0000:0f:00.0",
    }:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")

    shard = resolve_shard(args.model, args.tensor)
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor(args.tensor)
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError(f"expected a BF16 matrix, got {weight.dtype} {tuple(weight.shape)}")

    rows, columns = weight.shape
    packed = torch.empty((rows, columns), dtype=torch.int8)
    scales = torch.empty(rows, dtype=torch.float32)
    for begin in range(0, rows, args.chunk_rows):
        end = min(rows, begin + args.chunk_rows)
        chunk = weight[begin:end].float().cuda()
        scale = chunk.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        quantized = torch.round(chunk / scale[:, None]).clamp(-127, 127).to(torch.int8)
        packed[begin:end].copy_(quantized.cpu())
        scales[begin:end].copy_(scale.cpu())
        print(f"packed.rows={end}/{rows}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {"weight": packed, "scale": scales},
        args.output / "model.safetensors",
        metadata={
            "format": "g4.rowwise_int8_vocab.v1",
            "source_model": str(args.model),
            "source_tensor": args.tensor,
        },
    )
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "format": "g4.rowwise_int8_vocab.v1",
                "source_model": str(args.model),
                "source_tensor": args.tensor,
                "rows": rows,
                "columns": columns,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"packed.output={args.output}")


if __name__ == "__main__":
    main()
