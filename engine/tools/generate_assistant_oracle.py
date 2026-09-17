#!/usr/bin/env python3
"""Generate a framework-only oracle for one native Gemma 4 MTP draft step."""

import argparse
import json
import os
from pathlib import Path


TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/SSD/g4-models/gemma4-26b-a4b-assistant"),
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--drafts", type=int, default=1)
    parser.add_argument(
        "--target-model",
        type=Path,
        default=Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4"),
    )
    parser.add_argument("--output", type=Path, default=Path("goldens/generated"))
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    import torch
    from transformers import Gemma4AssistantForCausalLM

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name or str(props.pci_bus_id).lower() not in {
        "15",
        "0000:0f:00.0",
    }:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")
    model = Gemma4AssistantForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    columns = torch.arange(5632, device="cuda", dtype=torch.float32)
    inputs = (torch.sin(columns * 0.013) * 0.4 + torch.cos(columns * 0.007) * 0.1)
    inputs = inputs.to(torch.bfloat16).view(1, 1, 5632)
    shared = {
        "sliding_attention": (
            torch.zeros((1, 8, min(args.context, 1024), 256), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((1, 8, min(args.context, 1024), 256), dtype=torch.bfloat16, device="cuda"),
        ),
        "full_attention": (
            torch.zeros((1, 2, args.context, 512), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((1, 2, args.context, 512), dtype=torch.bfloat16, device="cuda"),
        ),
    }
    position_ids = torch.tensor([[args.context - 1]], dtype=torch.long, device="cuda")
    target_embedding = None
    if args.drafts > 1:
        from safetensors import safe_open

        tensor_name = "model.language_model.embed_tokens.weight"
        index = json.loads((args.target_model / "model.safetensors.index.json").read_text())
        shard = args.target_model / index["weight_map"][tensor_name]
        with safe_open(shard, framework="pt", device="cpu") as source:
            target_embedding = source.get_tensor(tensor_name).cuda()
    tokens = []
    with torch.inference_mode():
        for draft in range(args.drafts):
            result = model(
                inputs_embeds=inputs,
                position_ids=position_ids,
                shared_kv_states=shared,
                use_cache=False,
            )
            token = int(result.logits[0, 0].argmax())
            tokens.append(token)
            if draft + 1 < args.drafts:
                embedding = target_embedding[token].to(torch.bfloat16)
                embedding = embedding * torch.tensor(2816**0.5, dtype=torch.bfloat16, device="cuda")
                inputs = torch.cat((embedding, result.last_hidden_state[0, 0])).view(1, 1, 5632)
    state = result.last_hidden_state[0, 0].float().cpu()
    token = tokens[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.drafts == 1 else f"_drafts{args.drafts}"
    state.numpy().tofile(args.output / f"assistant_state_ctx{args.context}{suffix}.f32")
    (args.output / f"assistant_step_ctx{args.context}{suffix}.json").write_text(
        json.dumps(
            {
                "context": args.context,
                "selected_token": token,
                "drafted_tokens": tokens,
                "state_checksum": float(state.sum()),
                "position_mode": "single_position",
                "transformers_class": type(model).__name__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"assistant.context={args.context}")
    print(f"assistant.selected_token={token}")
    print("assistant.drafted_tokens=" + ",".join(map(str, tokens)))
    print(f"assistant.state_checksum={float(state.sum()):.8f}")


if __name__ == "__main__":
    main()
