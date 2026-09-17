#!/usr/bin/env python3
"""Offline-pack Gemma 4 vision projections as scaled FP8 E4M3."""

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
DEFAULT_OUTPUT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-vision-fp8")


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
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(f"FP8 packing requires SM120, got {props.name}")
    destination = args.output / "vision.safetensors"
    if destination.exists() and not args.force:
        print(f"skip existing={destination}")
        return

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

    names = [
        "model.vision_tower.patch_embedder.input_proj.weight",
        "model.embed_vision.embedding_projection.weight",
    ]
    suffixes = [
        "self_attn.q_proj.linear.weight",
        "self_attn.k_proj.linear.weight",
        "self_attn.v_proj.linear.weight",
        "self_attn.o_proj.linear.weight",
        "mlp.gate_proj.linear.weight",
        "mlp.up_proj.linear.weight",
        "mlp.down_proj.linear.weight",
    ]
    for layer in range(27):
        prefix = f"model.vision_tower.encoder.layers.{layer}."
        names.extend(prefix + suffix for suffix in suffixes)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    tensors = {}
    for index_in_model, name in enumerate(names):
        weight = get(name).cuda().contiguous()
        scale = (weight.float().abs().nan_to_num().amax(dim=1) / fp8_max).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        tensors[name + ".fp8"] = (
            (weight / scale[:, None]).to(torch.float8_e4m3fn).cpu().contiguous()
        )
        tensors[name + ".scale"] = scale.float().cpu().contiguous()
        del weight
        print(f"packed {index_in_model + 1}/{len(names)} {name}", flush=True)

    # F32 scale tensors sort before most FP8 payloads. Pad their aggregate
    # prefix so each following matrix retains cuBLASLt's 16-byte alignment.
    scale_floats = sum(tensor.numel() for name, tensor in tensors.items()
                       if name.endswith(".scale"))
    padding_floats = (-scale_floats) % 4
    if padding_floats:
        tensors["__g4_fp8_alignment_padding__"] = torch.zeros(
            padding_floats, dtype=torch.float32
        )
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        destination,
        metadata={"format": "g4.fp8_vision.v1", "architecture": "sm120f"},
    )
    manifest = {
        "format": "g4.fp8_vision.v1",
        "source": str(args.source),
        "projections": len(names),
        "weight_quantization": "per_output_channel_e4m3",
        "activation_quantization": "dynamic_per_token_e4m3",
        "architecture": "sm120f",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packed.output={args.output}")


if __name__ == "__main__":
    main()
