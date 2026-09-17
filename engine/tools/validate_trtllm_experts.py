#!/usr/bin/env python3
"""Compare a packed layer's fused SM120 output with direct ModelOpt dequantization."""

import argparse
import json
import os
import statistics
from pathlib import Path


TARGET_UUID = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
SOURCE = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
PACKED = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-trtllm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_UUID
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    import torch
    import torch.nn.functional as functional
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
    from safetensors import safe_open

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name:
        raise SystemExit(f"refusing GPU: {props.name}")
    device = torch.device("cuda")
    ids = torch.tensor([[3, 17, 31, 52, 73, 91, 105, 127]], dtype=torch.int32, device=device)
    route_weights = torch.tensor(
        [[0.24, 0.19, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]],
        dtype=torch.float32,
        device=device,
    )
    activation = (
        torch.sin(torch.arange(2816, device=device, dtype=torch.float32) * 0.019)
        + 0.31 * torch.cos(torch.arange(2816, device=device, dtype=torch.float32) * 0.007)
    ).to(torch.bfloat16).reshape(1, -1)

    prefix = f"layers.{args.layer}."
    packed_path = PACKED / f"experts-{args.layer:02d}.safetensors"
    with safe_open(packed_path, framework="pt", device=0) as packed:
        w13 = packed.get_tensor(prefix + "w13.weight")
        s13 = packed.get_tensor(prefix + "w13.weight_scale")
        w2 = packed.get_tensor(prefix + "w2.weight")
        s2 = packed.get_tensor(prefix + "w2.weight_scale")
        a1_quant = packed.get_tensor(prefix + "g1.input_scale_quant")
        a2_quant = packed.get_tensor(prefix + "g2.input_scale_quant")
        g1_alpha = packed.get_tensor(prefix + "g1.alpha")
        g2_alpha = packed.get_tensor(prefix + "g2.alpha")
        def run_fused():
            return cutlass_fused_moe(
                activation,
                ids,
                route_weights,
                w13.view(torch.long),
                w2.view(torch.long),
                torch.bfloat16,
                quant_scales=[
                    a1_quant,
                    s13.view(torch.int32),
                    g1_alpha,
                    a2_quant,
                    s2.view(torch.int32),
                    g2_alpha,
                ],
                activation_type=ActivationType.Geglu,
            )[0]

        for _ in range(20):
            fused = run_fused()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused = run_fused()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(300):
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) * 1000.0)

    index = json.loads((SOURCE / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name: str):
        filename = index[name]
        if filename not in handles:
            handles[filename] = safe_open(SOURCE / filename, framework="pt", device=0)
        return handles[filename].get_tensor(name)

    def dequantize(base: str) -> torch.Tensor:
        weight = get(base + ".weight")
        scales = get(base + ".weight_scale").float()
        global_scale = get(base + ".weight_scale_2").float()
        low = weight & 0x0F
        high = weight >> 4
        codes = torch.stack((low, high), dim=-1).reshape(weight.shape[0], -1)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device
        )
        decoded = magnitudes[(codes & 7).long()] * torch.where((codes & 8).bool(), -1.0, 1.0)
        return (decoded * scales.repeat_interleave(16, dim=1) * global_scale).to(torch.bfloat16)

    routed = []
    for expert in ids[0].tolist():
        base = f"model.language_model.layers.{args.layer}.experts.{expert}."
        gate = activation @ dequantize(base + "gate_proj").T
        up = activation @ dequantize(base + "up_proj").T
        product = functional.gelu(gate, approximate="tanh") * up
        routed.append(product @ dequantize(base + "down_proj").T)
    reference = torch.stack(routed, dim=1)
    reference = (reference * route_weights.to(torch.bfloat16).unsqueeze(-1)).sum(dim=1)
    difference = (fused.float() - reference.float()).abs()
    print(f"gpu={props.name}")
    print(f"layer={args.layer}")
    print(f"max_abs_error={difference.max().item():.6f}")
    print(f"mean_abs_error={difference.mean().item():.6f}")
    print(f"fused_norm={fused.float().norm().item():.6f}")
    print(f"fused_abs_checksum={fused.float().abs().sum().item():.6f}")
    print(f"reference_norm={reference.float().norm().item():.6f}")
    cosine = functional.cosine_similarity(fused.float(), reference.float()).item()
    print(f"cosine_similarity={cosine:.6f}")
    print(f"graph_median_us={statistics.median(timings):.3f}")
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        fused.float().cpu().numpy().tofile(args.dump)
        print(f"dump={args.dump}")
    if cosine < 0.8:
        raise SystemExit("packed expert validation failed: cosine similarity below 0.8")


if __name__ == "__main__":
    main()
