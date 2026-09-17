#!/usr/bin/env python3
"""Exercise two requests against one resident native serving process."""

import argparse
import json
import subprocess


def read_record(process: subprocess.Popen[str]) -> dict:
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read()
        raise RuntimeError(f"server exited before a response: {stderr}")
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("image")
    args = parser.parse_args()
    process = subprocess.Popen(
        [args.executable, "multimodal-serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = read_record(process)
        if ready.get("event") != "ready" or ready.get("protocol") != "g4-jsonl-v1":
            raise RuntimeError(f"invalid ready record: {ready}")
        request = {
            "image": args.image,
            "prompt": "<|image|>Describe this image.",
            "max_tokens": 8,
        }
        responses = []
        for request_id in ("first", "second"):
            request["id"] = request_id
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            responses.append(read_record(process))
        for response in responses:
            if not response.get("ok"):
                raise RuntimeError(f"request failed: {response}")
        first = responses[0]["result"]
        second = responses[1]["result"]
        # Golden for the deployed target path: fused cuDNN vision SDPA, FP8
        # language projections, native NVFP4 experts, and BF16 MTP assistant.
        expected_tokens = [236776, 9351, 236764, 2764, 236772, 624, 236772, 7986]
        if first["prompt_tokens"] != 285:
            raise RuntimeError(f"unexpected expanded prompt length: {first['prompt_tokens']}")
        if first["token_ids"] != expected_tokens:
            raise RuntimeError(f"optimized generation regression: {first['token_ids']}")
        if first["token_ids"] != second["token_ids"] or first["text"] != second["text"]:
            raise RuntimeError("persistent requests produced different greedy output")
        if first["accepted_drafts"] != second["accepted_drafts"]:
            raise RuntimeError("persistent requests produced different MTP acceptance")
        print("multimodal_serve.requests=2")
        print("multimodal_serve.outputs_repeatable=true")
        print(f"multimodal_serve.warm_wall_ms={second['wall_ms']:.3f}")
        print(
            "multimodal_serve.warm_decode_postprefill_tps="
            f"{second['decode_postprefill_tps']:.3f}"
        )
    finally:
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
