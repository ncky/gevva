#!/usr/bin/env python3
"""Record exact BF16 Transformers greedy multimodal+MTP output offline."""

import argparse
import json
import os
from pathlib import Path


TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
DEFAULT_MODEL = Path("/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model")
DEFAULT_ASSISTANT = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-assistant")
DEFAULT_IMAGE = Path(
    "/mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/"
    "infiyomi-rendered-browser-page1.jpg"
)


def format_user_turn(content: str) -> str:
    return (
        "<bos><|turn>user\n"
        + content
        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--assistant", type=Path, default=DEFAULT_ASSISTANT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="<|image|>Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("goldens/generated/multimodal_generation.json"),
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Gemma4AssistantForCausalLM,
        Gemma4ForConditionalGeneration,
    )

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name or str(props.pci_bus_id).lower() not in {
        "15",
        "0000:0f:00.0",
    }:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    with Image.open(args.image) as source:
        inputs = processor(
            text=format_user_turn(args.prompt),
            images=source.convert("RGB"),
            max_soft_tokens=280,
            return_tensors="pt",
        )
    inputs = {name: value.cuda() for name, value in inputs.items()}
    target = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
    ).cuda().eval()
    assistant = Gemma4AssistantForCausalLM.from_pretrained(
        args.assistant,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    assistant.generation_config.num_assistant_tokens = 4
    assistant.generation_config.num_assistant_tokens_schedule = "constant"

    with torch.inference_mode():
        sequence = target.generate(
            **inputs,
            assistant_model=assistant,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )[0]
    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated = sequence[prompt_tokens:].tolist()
    result = {
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "token_ids": generated,
        "text": processor.tokenizer.decode(generated),
        "target_dtype": "BF16",
        "assistant_drafts": 4,
        "assistant_position_mode": "single_position",
        "transformers_target_class": type(target).__name__,
        "transformers_assistant_class": type(assistant).__name__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
