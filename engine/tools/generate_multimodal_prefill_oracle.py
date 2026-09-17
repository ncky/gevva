#!/usr/bin/env python3
"""Run the exact BF16 Gemma 4 image prompt and record its first prediction."""

import argparse
import json
import os
from pathlib import Path

TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration


DEFAULT_MODEL = Path("/mnt/Drive_2_lin/Programming/infiyomi/models/page-vlm/model")
DEFAULT_IMAGE = Path(
    "/mnt/Drive_2_lin/Programming/infiyomi/data/component-tests/"
    "infiyomi-rendered-browser-page1.jpg"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="<|image|>Describe this image.")
    parser.add_argument(
        "--output", type=Path, default=Path("goldens/generated/multimodal_prefill.json")
    )
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    if "PRO 6000" not in props.name or str(props.pci_bus_id).lower() not in {
        "15",
        "0000:0f:00.0",
    }:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    with Image.open(args.image) as source:
        inputs = processor(
            text=args.prompt,
            images=source.convert("RGB"),
            max_soft_tokens=280,
            return_tensors="pt",
        )
    inputs = {name: value.cuda() for name, value in inputs.items()}
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).cuda().eval()
    with torch.inference_mode():
        output = model(**inputs, logits_to_keep=1, output_hidden_states=True)
    last_hidden = output.hidden_states[-1][0, -1].float()
    token = int(output.logits[0, -1].argmax())
    result = {
        "prompt": args.prompt,
        "tokens": int(inputs["input_ids"].shape[1]),
        "image_tokens": int((inputs["mm_token_type_ids"] == 1).sum()),
        "selected_token": token,
        "decoded": processor.tokenizer.decode([token]),
        "last_hidden_checksum": float(last_hidden.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
