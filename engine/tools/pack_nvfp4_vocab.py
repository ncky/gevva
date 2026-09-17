#!/usr/bin/env python3
"""Offline-pack a tied Gemma 4 vocabulary matrix for SM120 W4A4."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
DEFAULT_OUTPUT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-target-vocab-nvfp4")
TENSOR = "model.language_model.embed_tokens.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tensor", default=TENSOR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.output / "model.safetensors"
    if destination.exists() and not args.force:
        print(f"skip existing={destination}")
        return

    import torch
    from flashinfer import SfLayout, nvfp4_block_scale_interleave, nvfp4_quantize

    if torch.cuda.get_device_properties(0).major != 12:
        raise RuntimeError("vocabulary packing must run on the SM120 target GPU")
    index_path = args.source / "model.safetensors.index.json"
    source_file = (args.source / json.loads(index_path.read_text())["weight_map"][args.tensor]
                   if index_path.exists() else args.source / "model.safetensors")
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.tensor).cuda().contiguous()
    if weight.ndim != 2 or weight.shape[0] != 262144 or weight.shape[1] % 128:
        raise RuntimeError(f"unexpected vocabulary shape {tuple(weight.shape)}")
    rows, columns = weight.shape

    maximum = weight.float().abs().nan_to_num().amax()
    output_scale = torch.where(
        maximum > 0, maximum / (448.0 * 6.0), torch.ones_like(maximum)
    ).float()
    packed, scales = nvfp4_quantize(
        weight, 1.0 / output_scale,
        sfLayout=SfLayout.layout_linear, backend="cuda"
    )
    scales = scales.view(torch.uint8).reshape(rows, columns // 16)
    scales = nvfp4_block_scale_interleave(scales).contiguous()
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            # Safetensors writes keys lexicographically. Alpha is four bytes;
            # this padding keeps the following TMA-loaded weight 16-byte
            # aligned inside DeviceModel's contiguous device arena.
            "weight": packed.view(torch.uint8).cpu().contiguous(),
            "weight_scale": scales.cpu().contiguous(),
            "alpha": output_scale.cpu().reshape(1),
            "padding": torch.zeros(12, dtype=torch.uint8),
        },
        destination,
        metadata={
            "format": "g4.cutlass_nvfp4_vocab.v1",
            "architecture": "sm120f",
            "source_tensor": args.tensor,
        },
    )
    config = {
        "format": "g4.cutlass_nvfp4_vocab.v1",
        "source": str(args.source),
        "source_tensor": args.tensor,
        "shape": [rows, columns],
        "activation_quantization": "dynamic_fp4_block16",
        "architecture": "sm120f",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"packed.output={args.output}")
    print(f"packed.gib={destination.stat().st_size / (1 << 30):.3f}")


if __name__ == "__main__":
    main()
