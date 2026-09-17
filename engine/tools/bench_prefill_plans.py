#!/usr/bin/env python3
"""Process-isolated prefill plan A/B; warm each shape, disable prefix reuse.

Print JSONL locally. Plan tuning changes initialization, so toggling flags
inside an already initialized runtime is not a valid A/B for this experiment.
"""
import argparse
import json
import os
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="./build/g4")
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 1 or args.rounds < 1:
        parser.error("repeats, tokens and rounds must be positive")
    environment = os.environ.copy()
    for key in ("G4_CUDNN_TEXT_PREFILL", "G4_PREPARE_EXPERT_TILES",
                "G4_TUNE_FP8_PREFILL", "G4_FP8_ALGO_START",
                "G4_BATCH_SERVE_SWEEP_ENV", "G4_PROFILE_BATCH_SERVE"):
        environment.pop(key, None)
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d",
        "G4_BATCH_SERVE_TEXT_ONLY": "1",
        "G4_BATCH_SERVE_PROMPT_REPEATS": str(args.repeats),
        "G4_BATCH_SERVE_TEST_TOKENS": str(args.tokens),
        "G4_BATCH_SERVE_WARMUP": "1",
        "G4_BATCH_SERVE_WARMUP_SAME_PREFIX": "1",
        "G4_DISABLE_SHARED_PREFIX_CACHE": "1",
        "G4_ASSISTANT_HOT_VOCAB_TEXT": "assets/gemma4-assistant-hot-vocab-32768.bin",
    })
    for round_index in range(args.rounds):
        reference = None
        reference_acceptance = None
        for enabled in ((False, True) if round_index % 2 == 0 else (True, False)):
            env = environment.copy()
            if enabled:
                env["G4_TUNE_FP8_PREFILL"] = "1"
            process = subprocess.run(
                [args.executable, "batch-serve-test", "unused", str(args.batch)],
                env=env, text=True, capture_output=True, timeout=600)
            if process.returncode:
                raise RuntimeError(process.stdout + "\n" + process.stderr)
            sessions = [{} for _ in range(args.batch)]
            for line in process.stdout.splitlines():
                match = re.match(r"batch_serve.session\[(\d+)\]\.([^=]+)=(.*)", line)
                if match:
                    sessions[int(match[1])][match[2]] = match[3]
            sequences = [session["token_ids"] for session in sessions]
            acceptance = [session["accepted_drafts"] for session in sessions]
            if reference is None:
                reference = sequences
                reference_acceptance = acceptance
            matches = sequences == reference
            record = {"round": round_index, "batch": args.batch,
                      "prompt_repeats": args.repeats, "token_cap": args.tokens,
                      "tuned": enabled, "matches_pair": matches,
                      "acceptance_matches_pair": acceptance == reference_acceptance,
                      "sessions": sessions,
                      "plan_log": [line for line in process.stderr.splitlines()
                                   if line.startswith("fp8_tune ")]}
            print(json.dumps(record), flush=True)
            if not matches or acceptance != reference_acceptance:
                raise RuntimeError("prefill plan A/B changed tokens or draft acceptance")


if __name__ == "__main__":
    main()
