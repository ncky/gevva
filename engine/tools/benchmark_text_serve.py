#!/usr/bin/env python3
"""Benchmark the resident native server on the localmaxxing text workload."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path


PROMPT = (
    "Explain why local inference benchmarks should report prompt prefill "
    "throughput, decode throughput, and time to first token."
)


def read_record(process: subprocess.Popen[str]) -> dict:
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(
            "server exited before a response: " + process.stderr.read()
        )
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
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
        if ready.get("event") != "ready":
            raise RuntimeError(f"invalid ready record: {ready}")
        records = []
        total = args.warmup + args.iterations
        for index in range(total):
            request = {
                "id": f"text-{index}",
                "prompt": PROMPT,
                "max_tokens": args.max_tokens,
            }
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = read_record(process)
            if not response.get("ok"):
                raise RuntimeError(f"request failed: {response}")
            result = response["result"]
            if result["prompt_tokens"] != 34:
                raise RuntimeError(
                    f"expected 34 prompt tokens, got {result['prompt_tokens']}"
                )
            if len(result["token_ids"]) != args.max_tokens:
                raise RuntimeError(
                    f"expected {args.max_tokens} output tokens, "
                    f"got {len(result['token_ids'])}"
                )
            if index >= args.warmup:
                records.append(result)
            print(
                f"iteration={index + 1} warmup={index < args.warmup} "
                f"tps={result['decode_postprefill_tps']:.3f} "
                f"acceptance={result['draft_acceptance']:.4f} "
                f"assistant_ms={result['assistant_ms']:.3f} "
                f"verifier_ms={result['verifier_ms']:.3f}",
                flush=True,
            )

        rates = [record["decode_postprefill_tps"] for record in records]
        summary = {
            "workload": {
                "prompt": PROMPT,
                "prompt_tokens": 34,
                "output_tokens": args.max_tokens,
                "batch_size": 1,
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            "ready": ready,
            "samples": records,
            "decode_postprefill_tps": {
                "mean": statistics.mean(rates),
                "median": statistics.median(rates),
                "minimum": min(rates),
                "maximum": max(rates),
            },
        }
        encoded = json.dumps(summary, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
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
