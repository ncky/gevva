#!/usr/bin/env python3
"""Pack a tied BF16 vocabulary matrix as rowwise symmetric INT8."""

import argparse
import json
from pathlib import Path


from _preparation import add_launch_arguments, settings, select_gpu


def resolve_shard(model: Path, tensor: str) -> Path:
    index = model / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return model / weight_map[tensor]
    return model / "model.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tensor", default="model.language_model.embed_tokens.weight")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    add_launch_arguments(parser)
    args = parser.parse_args()
    if args.chunk_rows < 1: parser.error("chunk-rows must be positive")
    config = settings(args)
    args.model = args.model or config.model.target
    args.output = args.output or config.model.vocab
    if (args.output / "model.safetensors").is_file() and not args.force:
        print(f"skip existing={args.output}", flush=True)
        return
    select_gpu(config)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

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
            "format": "gevva.rowwise_int8_vocab.v1",
            "source_model": str(args.model),
            "source_tensor": args.tensor,
        },
    )
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "format": "gevva.rowwise_int8_vocab.v1",
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
