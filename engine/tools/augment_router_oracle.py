#!/usr/bin/env python3
"""Add router intermediates without loading the 48 GiB Transformers model."""

import json
import math
import os
from pathlib import Path

import numpy as np


MODEL = Path("/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model")
ORACLE = Path("goldens/generated")
GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"


def main() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    import torch
    from safetensors import safe_open

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name: str) -> torch.Tensor:
        with safe_open(MODEL / index[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).cuda()

    hidden = torch.from_numpy(np.fromfile(ORACLE / "hidden_0.bin", np.float32).reshape(3, 2816)).cuda().bfloat16()
    attention = torch.from_numpy(
        np.fromfile(ORACLE / "layer0_self_attn_0.bin", np.float32).reshape(3, 2816)
    ).cuda().bfloat16()
    post_weight = weight("model.language_model.layers.0.post_attention_layernorm.weight")
    mean_squared = attention.float().pow(2).mean(-1, keepdim=True) + 1e-6
    post = (attention.float() * torch.pow(mean_squared, -0.5) * post_weight.float()).bfloat16()
    residual = hidden + post

    mean_squared = residual.float().pow(2).mean(-1, keepdim=True) + 1e-6
    router_norm = (residual.float() * torch.pow(mean_squared, -0.5)).bfloat16()
    router_scale = weight("model.language_model.layers.0.router.scale")
    router_scaled = router_norm * router_scale
    router_input = router_scaled * (2816**-0.5)
    router_weight = weight("model.language_model.layers.0.router.proj.weight")
    logits = torch.nn.functional.linear(router_input, router_weight)

    manifest_path = ORACLE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, tensor in {
        "layer0.router_norm": router_norm,
        "layer0.router_scaled": router_scaled,
        "layer0.router_input": router_input,
        "layer0.router_logits": logits,
    }.items():
        array = tensor.float().cpu().numpy()
        filename = name.replace(".", "_") + ".bin"
        array.tofile(ORACLE / filename)
        manifest["arrays"][name] = {
            "file": filename,
            "dtype": "float32",
            "shape": list(array.shape),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("added router_norm, router_scaled, router_input, and router_logits")


if __name__ == "__main__":
    main()
