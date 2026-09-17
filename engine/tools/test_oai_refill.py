#!/usr/bin/env python3
"""Check ragged decode metadata, cohort refills, tiny limits, and SSE."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"

    def request(index):
        limit = (1, 2, 7, 32, 64, 128)[index % 6]
        prefix = "Explain the practical engineering tradeoffs. " * (index % 5 * 9)
        payload = {
            "model": "page-vlm",
            "messages": [{"role": "user", "content": prefix +
                          "Describe how batching and caching affect model inference."}],
            "max_tokens": limit,
            "temperature": 0,
        }
        req = Request(endpoint, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            result = json.load(response)
        assert len(result["choices"]) == 1, result
        assert result["choices"][0]["finish_reason"] in ("stop", "length"), result
        tokens = result["usage"]["completion_tokens"]
        assert 0 < tokens <= limit, result
        assert isinstance(result["choices"][0]["message"]["content"], str), result
        return tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(request, range(24)))

    payload = {"model": "page-vlm", "messages": [{"role": "user", "content":
               "Explain why GPU memory bandwidth matters."}], "max_tokens": 32,
               "temperature": 0, "stream": True}
    req = Request(endpoint, data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    chunks, done, finish = 0, False, False
    with urlopen(req, timeout=120) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            result = json.loads(line[6:])
            for choice in result["choices"]:
                chunks += bool(choice.get("delta", {}).get("content"))
                finish |= choice.get("finish_reason") in ("stop", "length")
    assert chunks > 0 and done and finish, (chunks, done, finish)
    print(json.dumps({"requests": 24, "concurrency": 8,
                      "completion_tokens": sum(counts), "limits_valid": True,
                      "stream_content_chunks": chunks, "stream_finished": True}))


if __name__ == "__main__":
    main()
