#!/usr/bin/env python3
"""Offline-pack Gemma 4 MTP assistant projections as FP8 E4M3."""

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-assistant")
DEFAULT_OUTPUT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-assistant-fp8")


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
    source_file = args.source / "model.safetensors"
    output_file = args.output / "model.safetensors"
    if output_file.exists() and not args.force:
        print(f"skip existing={output_file}")
        return

    names = ["pre_projection.weight", "post_projection.weight"]
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    names.extend(
        f"model.layers.{layer}.{suffix}"
        for layer in range(4)
        for suffix in suffixes
    )
    tensors = {}
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    with safe_open(source_file, framework="pt", device="cpu") as source:
        for index, name in enumerate(names, 1):
            weight = source.get_tensor(name).cuda().contiguous()
            scale = (weight.float().abs().nan_to_num().amax() / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            tensors[name + ".fp8"] = (
                (weight / scale).to(torch.float8_e4m3fn).cpu().contiguous()
            )
            tensors[name + ".scale"] = scale.float().cpu().reshape(1)
            print(f"packed {index}/{len(names)} {name}", flush=True)
            del weight

    padding_floats = (-len(names)) % 4
    if padding_floats:
        tensors["__g4_fp8_alignment_padding__"] = torch.zeros(
            padding_floats, dtype=torch.float32
        )
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output_file,
        metadata={"format": "g4.fp8_assistant.v1", "architecture": "sm120f"},
    )
    manifest = {
        "format": "g4.fp8_assistant.v1",
        "source": str(args.source),
        "projections": len(names),
        "weight_quantization": "per_tensor_e4m3",
        "activation_quantization": "dynamic_per_token_e4m3",
        "architecture": "sm120f",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed.output={args.output}")


if __name__ == "__main__":
    main()
