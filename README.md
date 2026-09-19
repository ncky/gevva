# Gevva

Inference setup for running Gemma 4 26B-A4B in a similar shape to
[Jev](https://typesafe.ai/), for local testing.

**Blackwell SM120 only.** Tested on the RTX 5090 and RTX PRO 6000 Blackwell.
The tested workloads use about **24 GiB VRAM** for inference.

Supported model: [NVIDIA Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4),
with the prepared sidecars described in [setup](SETUP.md).
See [recommended settings](SETUP.md#recommended-settings).

From a fresh checkout (with the [system prerequisites](SETUP.md#build-and-install) installed):

```sh
./setup.sh
# Edit gevva.toml if needed: defaults are RTX 5090, ./models, 127.0.0.1:8081.
.venv/bin/python tools/prepare_model.py --config gevva.toml --download
.venv/bin/gevva --config gevva.toml
```

Setup builds the engine, installs Python dependencies, and creates `gevva.toml`
without overwriting existing settings. Preparation downloads the model from
Hugging Face and produces the three required runtime sidecars. For an existing
checkpoint, set `model.target` in TOML and omit `--download`.

Override binding at launch with `--host 0.0.0.0 --port 8090`.

Endpoint: `POST /v1/systemone`. [Example request](examples/email-flags.json) ·
[Full example TOML](gevva.example.toml) · [Configuration](SETUP.md#configuration).

Doom demo using structured game state (visible objects and positions) to choose turning and firing:

https://github.com/user-attachments/assets/a4ed26f2-7ddc-4f33-8487-30b72eee1e68

Measured on an **RTX PRO 6000** (median request latency / answers per second):

| Input | 1 question | 5 questions | 10 questions |
| --- | ---: | ---: | ---: |
| Text | 20 ms / 49/s | 23 ms / 213/s | 27 ms / 372/s |
| Image | 31 ms / 32/s | 33 ms / 148/s | 36 ms / 272/s |

Measured with fresh inputs, 140-token images, and 100 requests per cell.
Startup is excluded; results are not cached.
