#!/usr/bin/env python3
"""Export a generated oracle NPZ as dependency-free raw arrays plus JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="Hello world!")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"prompt": args.prompt, "arrays": {}}
    with np.load(args.npz) as archive:
        for name in archive.files:
            array = np.ascontiguousarray(archive[name])
            file_name = f"{name.replace('.', '_')}.bin"
            array.tofile(args.output / file_name)
            manifest["arrays"][name] = {
                "file": file_name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['arrays'])} raw arrays to {args.output}")


if __name__ == "__main__":
    main()

