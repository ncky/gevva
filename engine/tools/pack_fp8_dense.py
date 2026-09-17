#!/usr/bin/env python3
"""Offline-pack Gemma 4 target dense projections as scaled FP8 E4M3."""

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
DEFAULT_OUTPUT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-dense-fp8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    props = torch.cuda.get_device_properties(0)
    if props.major != 12 or props.minor != 0:
        raise RuntimeError(f"FP8 packing requires the SM120 target, got {props.name}")
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
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    for layer in range(30):
        destination = args.output / f"dense-{layer:02d}.safetensors"
        if destination.exists() and not args.force:
            print(f"skip layer={layer} existing={destination}", flush=True)
            continue
        tensors = {}
        for suffix in suffixes:
            source_name = f"model.language_model.layers.{layer}.{suffix}"
            if source_name not in index:
                continue
            weight = get(source_name).cuda().contiguous()
            scale = (weight.float().abs().nan_to_num().amax() / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            base = f"layers.{layer}." + suffix.removesuffix(".weight")
            tensors[base + ".weight"] = (
                (weight / scale).to(torch.float8_e4m3fn).cpu().contiguous()
            )
            tensors[base + ".scale"] = scale.float().cpu().reshape(1)
            del weight
        # safetensors groups F32 scales before F8 payloads. Pad that prefix to
        # 16 bytes so every matrix satisfies cuBLASLt's alignment contract.
        projection_count = sum(name.endswith(".weight") for name in tensors)
        padding_floats = (-projection_count) % 4
        if padding_floats:
            tensors[f"__g4_fp8_alignment_padding_{layer}__"] = torch.zeros(
                padding_floats, dtype=torch.float32
            )
        save_file(
            tensors,
            destination,
            metadata={
                "format": "g4.fp8_dense.v1",
                "layer": str(layer),
                "architecture": "sm120f",
            },
        )
        print(
            f"packed layer={layer} gib={destination.stat().st_size / (1 << 30):.3f}",
            flush=True,
        )

    manifest = {
        "format": "g4.fp8_dense.v1",
        "source": str(args.source),
        "layers": 30,
        "weight_quantization": "per_tensor_e4m3",
        "activation_quantization": "dynamic_per_token_e4m3",
        "architecture": "sm120f",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed.output={args.output}")


if __name__ == "__main__":
    main()
