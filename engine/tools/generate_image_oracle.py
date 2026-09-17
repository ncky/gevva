#!/usr/bin/env python3
"""Export Gemma 4 image preprocessing tensors for the native differential."""

import argparse
import json
import os
from pathlib import Path

TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionEncoderLayer,
    Gemma4VisionModel,
    Gemma4VisionRotaryEmbedding,
)


DEFAULT_MODEL = Path("/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model")
DEFAULT_IMAGE = Path(
    "/mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/"
    "infiyomi-rendered-browser-page1.jpg"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--soft-tokens", type=int, default=280)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--output", type=Path, default=Path("goldens/generated/image"))
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
        result = processor.image_processor(
            images=[image], max_soft_tokens=args.soft_tokens, return_tensors="pt"
        )
    pixels = np.ascontiguousarray(result["pixel_values"].numpy(), dtype=np.float32)
    positions = np.ascontiguousarray(result["image_position_ids"].numpy(), dtype=np.int32)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(args.model / index[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name or str(props.pci_bus_id).lower() not in {
        "15",
        "0000:0f:00.0",
    }:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")
    with torch.inference_mode():
        values = result["pixel_values"].cuda()
        patch_weight = tensor("model.vision_tower.patch_embedder.input_proj.weight").cuda()
        position_table = tensor(
            "model.vision_tower.patch_embedder.position_embedding_table"
        ).cuda()
        padding = (result["image_position_ids"] == -1).all(dim=-1).cuda()
        clamped = result["image_position_ids"].clamp(min=0).cuda()
        patch_hidden = torch.nn.functional.linear(
            (2 * (values - 0.5)).to(patch_weight.dtype), patch_weight
        )
        position_hidden = (
            position_table[0, clamped[..., 0]] + position_table[1, clamped[..., 1]]
        )
        position_hidden = torch.where(padding.unsqueeze(-1), 0.0, position_hidden)
        patch_hidden = np.ascontiguousarray(
            (patch_hidden + position_hidden).float().cpu().numpy(), dtype=np.float32
        )
        valid_patches = int(result["num_soft_tokens_per_image"][0]) * 9
        config = AutoConfig.from_pretrained(args.model, local_files_only=True).vision_config
        config._attn_implementation = args.attention
        layer = Gemma4VisionEncoderLayer(config, 0)
        prefix = "model.vision_tower.encoder.layers.0."
        state = {
            name.removeprefix(prefix): tensor(name)
            for name in index
            if name.startswith(prefix)
        }
        layer.load_state_dict(state, strict=True)
        layer = layer.to(device="cuda", dtype=torch.bfloat16).eval()
        rotary = Gemma4VisionRotaryEmbedding(config).cuda()
        valid_hidden = torch.from_numpy(patch_hidden[:, :valid_patches]).cuda().to(torch.bfloat16)
        valid_positions = result["image_position_ids"][:, :valid_patches].cuda()
        rope = rotary(valid_hidden, valid_positions)
        layer0_hidden = layer(
            valid_hidden,
            position_embeddings=rope,
            position_ids=valid_positions,
            attention_mask=None,
        )
        layer0_hidden = np.ascontiguousarray(
            layer0_hidden.float().cpu().numpy(), dtype=np.float32
        )
        del layer, rotary, rope, valid_hidden
        vision = Gemma4VisionModel(config)
        vision_state = {}
        vision_prefix = "model.vision_tower."
        for shard_name in sorted({
            shard for name, shard in index.items() if name.startswith(vision_prefix)
        }):
            with safe_open(args.model / shard_name, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name.startswith(vision_prefix):
                        vision_state[name.removeprefix(vision_prefix)] = source.get_tensor(name)
        vision.load_state_dict(vision_state, strict=True)
        vision = vision.to(device="cuda", dtype=torch.bfloat16).eval()
        full_layer_outputs = {}
        hooks = []
        for layer_index in (0, 1, 4, 9, 26):
            def capture(_module, _inputs, output, index=layer_index):
                full_layer_outputs[index] = output.detach().clone()

            hooks.append(vision.encoder.layers[layer_index].register_forward_hook(capture))
        vision_output = vision(
            pixel_values=result["pixel_values"].cuda(),
            pixel_position_ids=result["image_position_ids"].cuda(),
        ).last_hidden_state
        for hook in hooks:
            hook.remove()
        projection_weight = tensor("model.embed_vision.embedding_projection.weight").cuda()
        mean_squared = vision_output.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps
        projected_input = (vision_output.float() * torch.pow(mean_squared, -0.5)).to(
            torch.bfloat16
        )
        projected = torch.nn.functional.linear(projected_input, projection_weight)
        vision_output = np.ascontiguousarray(
            vision_output.float().cpu().numpy(), dtype=np.float32
        )
        projected = np.ascontiguousarray(projected.float().cpu().numpy(), dtype=np.float32)
        trimmed_positions = result["image_position_ids"][:, :valid_patches].cuda()
        trimmed_padding = torch.zeros(
            (1, valid_patches), dtype=torch.bool, device="cuda"
        )
        trimmed = vision.patch_embedder(
            result["pixel_values"].cuda(),
            result["image_position_ids"].cuda(),
            (result["image_position_ids"] == -1).all(dim=-1).cuda(),
        )[:, :valid_patches]
        trimmed_rope = vision.encoder.rotary_emb(trimmed, trimmed_positions)
        for encoder_layer in vision.encoder.layers:
            trimmed = encoder_layer(
                trimmed,
                attention_mask=None,
                position_embeddings=trimmed_rope,
                position_ids=trimmed_positions,
            )
        trimmed, _ = vision.pooler(
            trimmed, trimmed_positions, trimmed_padding, output_length=valid_patches // 9
        )
        trimmed = (trimmed - vision.std_bias.float()) * vision.std_scale.float()
        trimmed = trimmed.to(torch.bfloat16)
        trimmed_mean_squared = trimmed.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps
        trimmed_norm = (trimmed.float() * torch.pow(trimmed_mean_squared, -0.5)).to(
            torch.bfloat16
        )
        projected_trimmed = torch.nn.functional.linear(trimmed_norm, projection_weight)
        projected_trimmed = np.ascontiguousarray(
            projected_trimmed.float().cpu().numpy(), dtype=np.float32
        )
    args.output.mkdir(parents=True, exist_ok=True)
    pixels.tofile(args.output / "pixel_values.f32")
    positions.tofile(args.output / "position_ids.i32")
    patch_hidden.tofile(args.output / "patch_hidden.f32")
    layer0_hidden.tofile(args.output / "vision_layer0_hidden.f32")
    for layer_index, layer_output in full_layer_outputs.items():
        np.ascontiguousarray(
            layer_output.float().cpu().numpy(), dtype=np.float32
        ).tofile(args.output / f"vision_full_layer{layer_index}_hidden.f32")
    vision_output.tofile(args.output / "vision_soft_tokens.f32")
    projected.tofile(args.output / "vision_projected.f32")
    projected_trimmed.tofile(args.output / "vision_projected_trimmed.f32")
    metadata = {
        "source": str(args.image),
        "pixel_shape": list(pixels.shape),
        "position_shape": list(positions.shape),
        "soft_token_count": int(result["num_soft_tokens_per_image"][0]),
        "pixel_checksum": float(pixels.sum(dtype=np.float64)),
        "patch_hidden_shape": list(patch_hidden.shape),
        "patch_hidden_checksum": float(patch_hidden.sum(dtype=np.float64)),
        "valid_patches": valid_patches,
        "layer0_hidden_shape": list(layer0_hidden.shape),
        "layer0_hidden_checksum": float(layer0_hidden.sum(dtype=np.float64)),
        "vision_soft_tokens_shape": list(vision_output.shape),
        "vision_soft_tokens_checksum": float(vision_output.sum(dtype=np.float64)),
        "vision_projected_shape": list(projected.shape),
        "vision_projected_checksum": float(projected.sum(dtype=np.float64)),
        "vision_projected_trimmed_checksum": float(
            projected_trimmed.sum(dtype=np.float64)
        ),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
