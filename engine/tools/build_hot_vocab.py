#!/usr/bin/env python3
"""Build a reproducible frequency-ranked token map for draft-only decoding."""

import argparse
import collections
import json
import pathlib
import struct

from tokenizers import Tokenizer


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".md", ".py",
    ".rst", ".txt", ".json", ".jsonl",
}


def strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def documents(path):
    candidates = path.rglob("*") if path.is_dir() else [path]
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = candidate.read_text(errors="ignore")
            if candidate.suffix.lower() == ".json":
                yield from strings(json.loads(text))
            elif candidate.suffix.lower() == ".jsonl":
                for line in text.splitlines():
                    try:
                        yield from strings(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            elif text.strip():
                yield text
        except (OSError, json.JSONDecodeError):
            continue


def fallback_ids(tokenizer_json, vocabulary_size):
    model = tokenizer_json.get("model", {})
    vocab = model.get("vocab", [])
    if isinstance(vocab, list):
        scored = sorted(enumerate(vocab), key=lambda pair: pair[1][1], reverse=True)
        return [token_id for token_id, _ in scored]
    if isinstance(vocab, dict):
        return [token_id for _, token_id in sorted(vocab.items(), key=lambda pair: pair[1])]
    return list(range(vocabulary_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("corpus", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_json = json.loads(args.tokenizer.read_text())
    # The tied LM head excludes tokenizer-only added IDs.
    vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=False)
    if args.size < 2048 or args.size > vocabulary_size or args.size % 2048:
        raise SystemExit("--size must be a multiple of 2048 within the vocabulary")

    counts = collections.Counter()
    documents_seen = 0
    tokens_seen = 0
    for root in args.corpus:
        for document in documents(root):
            ids = tokenizer.encode(document, add_special_tokens=False).ids
            counts.update(ids)
            documents_seen += 1
            tokens_seen += len(ids)

    # Added/special tokens and corpus-observed IDs take priority. Fill a sparse
    # corpus deterministically with the tokenizer model's own unigram scores.
    required = set()
    for added in tokenizer_json.get("added_tokens", []):
        if added.get("special"):
            required.add(int(added["id"]))
    ranked = sorted(counts, key=lambda token_id: (-counts[token_id], token_id))
    ordered = []
    seen = set()
    for token_id in sorted(required) + ranked + fallback_ids(tokenizer_json, vocabulary_size):
        if 0 <= token_id < vocabulary_size and token_id not in seen:
            ordered.append(token_id)
            seen.add(token_id)
            if len(ordered) == args.size:
                break
    if len(ordered) != args.size:
        raise SystemExit("could not fill requested hot vocabulary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(struct.pack(f"<{len(ordered)}I", *ordered))
    metadata = {
        "format": "little-endian uint32 token IDs",
        "size": len(ordered),
        "vocabulary_size": vocabulary_size,
        "documents": documents_seen,
        "tokens": tokens_seen,
        "observed_unique_tokens": len(counts),
        "corpora": [str(path.resolve()) for path in args.corpus],
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
