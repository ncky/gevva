#!/usr/bin/env python3
"""Generate small Transformers tensors for native differential tests.

This is a development tool only. The g4 executable never imports Python/torch.
"""

import argparse
import json
import os
from pathlib import Path


TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
DEFAULT_MODEL = Path("/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("goldens/transformers_bf16.npz"))
    parser.add_argument("--raw-dir", type=Path, default=Path("goldens/generated"))
    parser.add_argument("--prompt", default="Hello world!")
    args = parser.parse_args()

    # Must happen before importing torch.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU

    import numpy as np
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()

    captured: dict[str, np.ndarray] = {}

    def store(name: str, value) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                if torch.is_tensor(item):
                    store(f"{name}.{index}", item)
            return
        if torch.is_tensor(value):
            captured[name] = value.detach().float().cpu().numpy()

    handles = []
    for layer_index in (0, 5):
        layer = model.model.language_model.layers[layer_index]
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp",
            "router",
            "pre_feedforward_layernorm_2",
            "experts",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
            "post_feedforward_layernorm",
        ):
            module = getattr(layer, name)
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, key=f"layer{layer_index}.{name}": store(key, output)
            ))

    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    for index, hidden in enumerate(outputs.hidden_states):
        store(f"hidden.{index}", hidden)
    store("logits", outputs.logits)
    captured["input_ids"] = ids.cpu().numpy().astype(np.int32)
    captured["logits_top_ids"] = outputs.logits[0, -1].float().topk(32).indices.cpu().numpy().astype(np.int32)
    captured["logits_top_values"] = outputs.logits[0, -1].float().topk(32).values.cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **captured)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"prompt": args.prompt, "arrays": {}}
    for name, array in captured.items():
        safe_name = name.replace(".", "_")
        # Float tensors are deliberately normalized to f32 by store(); ids are i32.
        file_name = f"{safe_name}.bin"
        contiguous = np.ascontiguousarray(array)
        contiguous.tofile(args.raw_dir / file_name)
        manifest["arrays"][name] = {
            "file": file_name,
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
        }
    (args.raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} and {args.raw_dir} ({len(captured)} arrays)")


if __name__ == "__main__":
    main()
