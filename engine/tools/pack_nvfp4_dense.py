#!/usr/bin/env python3
"""Offline-pack Gemma 4 target dense projections for native SM120 W4A4."""

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
DEFAULT_OUTPUT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-dense-nvfp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from flashinfer import SfLayout, nvfp4_block_scale_interleave, nvfp4_quantize
    from safetensors import safe_open
    from safetensors.torch import save_file

    if torch.cuda.get_device_properties(0).major != 12:
        raise RuntimeError("dense packing must run on the SM120 target GPU")
    index = json.loads(
        (args.source / "model.safetensors.index.json").read_text()
    )["weight_map"]
    handles = {}

    def get(name: str) -> torch.Tensor:
        filename = index[name]
        if filename not in handles:
            handles[filename] = safe_open(
                args.source / filename, framework="pt", device="cpu"
            )
        return handles[filename].get_tensor(name)

    args.output.mkdir(parents=True, exist_ok=True)
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "router.proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for layer in range(30):
        destination = args.output / f"dense-{layer:02d}.safetensors"
        if destination.exists() and not args.force:
            print(f"skip layer={layer} existing={destination}", flush=True)
            continue
        tensors = {}
        for suffix in suffixes:
            name = f"model.language_model.layers.{layer}.{suffix}"
            if name not in index:
                continue
            weight = get(name).cuda().contiguous()
            amax = weight.float().abs().nan_to_num().amax()
            decode_scale = torch.where(
                amax > 0, amax / (448.0 * 6.0), torch.ones_like(amax)
            ).float()
            packed, scales = nvfp4_quantize(
                weight,
                1.0 / decode_scale,
                sfLayout=SfLayout.layout_linear,
                backend="cuda",
            )
            rows, columns = weight.shape
            scales = scales.view(torch.uint8).reshape(rows, columns // 16)
            scales = nvfp4_block_scale_interleave(scales).contiguous()
            base = suffix.removesuffix(".weight")
            tensors[base + ".weight"] = packed.view(torch.uint8).cpu().contiguous()
            tensors[base + ".weight_scale"] = scales.cpu().contiguous()
            tensors[base + ".alpha"] = decode_scale.cpu().reshape(1)
            del weight, packed, scales
        save_file(
            tensors,
            destination,
            metadata={
                "format": "g4.cutlass_nvfp4_dense.v1",
                "layer": str(layer),
                "architecture": "sm120f",
            },
        )
        gib = destination.stat().st_size / (1 << 30)
        print(f"packed layer={layer} gib={gib:.3f}", flush=True)

    manifest = {
        "format": "g4.cutlass_nvfp4_dense.v1",
        "source": str(args.source),
        "layers": 30,
        "activation_quantization": "dynamic_fp4_block16_global_one",
        "architecture": "sm120f",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed.output={args.output}")


if __name__ == "__main__":
    main()
