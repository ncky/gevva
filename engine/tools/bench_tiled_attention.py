#!/usr/bin/env python3
"""Sequential, fixture-based attention replay; never loads model weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary", default="./build/g4")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--bs1-tokens", type=int, default=4,
                        help="BS1 verifier positions; 0 keeps the full capture")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES="GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d")
    env.pop("G4_PROFILE_ATTENTION", None)
    env.pop("G4_ATTENTION_CHECK_ONLY", None)
    with args.output.open("x") as output:
        for round_id in range(args.rounds):
            for fixture in args.fixtures:
                for batch in (1, 8):
                    for ragged in (0, 1) if batch == 8 else (0,):
                        result = subprocess.run(
                            [args.binary, "tiled-attention-bench", str(fixture),
                             str(batch), str(ragged), str(args.bs1_tokens if batch == 1 else 0)], env=env, check=True,
                            text=True, capture_output=True)
                        rows = [json.loads(line) for line in result.stdout.splitlines()
                                if line.startswith("{")]
                        if not rows:
                            raise RuntimeError(f"no benchmark rows: {fixture}")
                        for row in rows:
                            if row["nonfinite"] or row.get("staging_mismatches", 0) or (row["capture_checked"] and
                                                     row["capture_mismatches"]):
                                raise RuntimeError(f"invalid replay: {row}")
                            row["round"] = round_id
                            output.write(json.dumps(row) + "\n")
                        output.flush()
                        best = min(rows, key=lambda row: row["tiled_us"])
                        print(f"round={round_id} {fixture} B{batch} ragged={ragged}: "
                              f"baseline={best['baseline_us']:.2f}us "
                              f"best_tiled={best['tiled_us']:.2f}us", flush=True)


if __name__ == "__main__":
    main()
