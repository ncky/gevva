#!/usr/bin/env python3
"""Offline-pack Gemma 4 ModelOpt experts for the SM120 fused MoE ABI.

The resulting per-layer safetensors are runtime artifacts.  Python, Torch,
SGLang, and FlashInfer are only converter dependencies and are not linked or
imported by gevva-engine itself.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path


from _preparation import add_launch_arguments, settings, select_gpu


def parse_layers(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            first, last = (int(x) for x in part.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"layers must be within [0, {count - 1}]")
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", default="all", help="all, comma list, or inclusive ranges")
    parser.add_argument("--force", action="store_true")
    add_launch_arguments(parser)
    args = parser.parse_args()
    config = settings(args)
    args.source = args.source or config.model.target
    args.output = args.output or config.model.experts
    props = select_gpu(config)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from flashinfer import nvfp4_block_scale_interleave

    def prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, s13, s2, hidden_size, intermediate_size, num_experts
    ):
        """Pack the measured cutlass_fused_moe ABI: raw rows, swizzled scales."""
        w13 = w13.reshape(num_experts, 2 * intermediate_size, hidden_size // 2)
        w2 = w2.reshape(num_experts, hidden_size, intermediate_size // 2)
        s13 = s13.reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
        s2 = s2.reshape(num_experts, hidden_size, intermediate_size // 16)
        packed_s13 = torch.stack(
            [nvfp4_block_scale_interleave(s13[i].view(torch.uint8)) for i in range(num_experts)]
        ).view_as(s13)
        packed_s2 = torch.stack(
            [nvfp4_block_scale_interleave(s2[i].view(torch.uint8)) for i in range(num_experts)]
        ).view_as(s2)
        return w13.contiguous(), packed_s13, w2.contiguous(), packed_s2

    config = json.loads((args.source / "config.json").read_text())
    text_config = config["text_config"]
    layers = parse_layers(args.layers, text_config["num_hidden_layers"])
    experts = text_config["num_experts"]
    hidden = text_config["hidden_size"]
    intermediate = text_config["moe_intermediate_size"]
    index_path = args.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    args.output.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        shards = {
            filename: stack.enter_context(
                safe_open(args.source / filename, framework="pt", device="cpu")
            )
            for filename in sorted(set(index.values()))
        }

        def get(name: str) -> torch.Tensor:
            return shards[index[name]].get_tensor(name)

        def expert_stack(layer: int, projection: str, suffix: str) -> torch.Tensor:
            return torch.stack(
                [
                    get(
                        f"model.language_model.layers.{layer}.experts.{expert}."
                        f"{projection}.{suffix}"
                    )
                    for expert in range(experts)
                ]
            )

        for layer in layers:
            destination = args.output / f"experts-{layer:02d}.safetensors"
            if destination.exists() and not args.force:
                print(f"skip layer={layer} existing={destination}", flush=True)
                continue
            gate_w = expert_stack(layer, "gate_proj", "weight").cuda()
            up_w = expert_stack(layer, "up_proj", "weight").cuda()
            down_w = expert_stack(layer, "down_proj", "weight").cuda()
            gate_s = expert_stack(layer, "gate_proj", "weight_scale").cuda()
            up_s = expert_stack(layer, "up_proj", "weight_scale").cuda()
            down_s = expert_stack(layer, "down_proj", "weight_scale").cuda()
            # TensorRT-LLM's fused gated GEMM ABI is blockwise [up; gate]
            # before its row-interleave/shuffle transform.
            w13 = torch.cat((up_w, gate_w), dim=1).contiguous()
            s13 = torch.cat((up_s, gate_s), dim=1).contiguous()
            packed_w13, packed_s13, packed_w2, packed_s2 = (
                prepare_static_weights_for_trtllm_fp4_moe(
                    w13,
                    down_w,
                    s13,
                    down_s,
                    hidden,
                    intermediate,
                    experts,
                )
            )
            gate_input = expert_stack(layer, "gate_proj", "input_scale").float().reshape(experts)
            up_input = expert_stack(layer, "up_proj", "input_scale").float().reshape(experts)
            if not torch.equal(gate_input, up_input):
                raise RuntimeError(f"layer {layer}: gate/up activation scales differ")
            gate_global = expert_stack(layer, "gate_proj", "weight_scale_2").float().reshape(experts)
            up_global = expert_stack(layer, "up_proj", "weight_scale_2").float().reshape(experts)
            down_input = expert_stack(layer, "down_proj", "input_scale").float().reshape(experts)
            down_global = expert_stack(layer, "down_proj", "weight_scale_2").float().reshape(experts)
            prefix = f"layers.{layer}."
            tensors = {
                prefix + "w13.weight": packed_w13.cpu().contiguous(),
                prefix + "w13.weight_scale": packed_s13.cpu().contiguous(),
                prefix + "w2.weight": packed_w2.cpu().contiguous(),
                prefix + "w2.weight_scale": packed_s2.cpu().contiguous(),
                prefix + "w13.input_scale": gate_input.contiguous(),
                prefix + "w2.input_scale": down_input.contiguous(),
                prefix + "w13.weight_scale_2": torch.stack((gate_global, up_global), dim=1).contiguous(),
                prefix + "w2.weight_scale_2": down_global.contiguous(),
                prefix + "g1.alpha": (gate_input * gate_global).contiguous(),
                prefix + "g1.alpha_up": (up_input * up_global).contiguous(),
                prefix + "g2.alpha": (down_input * down_global).contiguous(),
                prefix + "g1.input_scale_quant": (1.0 / gate_input).contiguous(),
                prefix + "g2.input_scale_quant": (1.0 / down_input).contiguous(),
            }
            metadata = {
                "format": "gevva.cutlass_nvfp4_experts.v1",
                "layer": str(layer),
                "architecture": "sm120f",
                "activation": "geglu_tanh",
            }
            temporary = destination.with_suffix(".tmp")
            save_file(tensors, temporary, metadata=metadata)
            temporary.replace(destination)
            gib = destination.stat().st_size / (1 << 30)
            print(f"packed layer={layer} gib={gib:.3f} path={destination}", flush=True)
            del tensors, packed_w13, packed_s13, packed_w2, packed_s2
            del w13, s13, gate_w, up_w, down_w, gate_s, up_s, down_s
            torch.cuda.empty_cache()

    source_digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest = {
        "format": "gevva.cutlass_nvfp4_experts.v1",
        "source": str(args.source.resolve()),
        "source_index_sha256": source_digest,
        "target_gpu": props.name,
        "target_gpu_uuid": str(getattr(props, "uuid", "")),
        "architecture": "sm120f",
        "layers_requested": layers,
        "experts": experts,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
