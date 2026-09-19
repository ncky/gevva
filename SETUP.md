# Setup

Gevva runs the NVIDIA NVFP4 version of Gemma 4 26B-A4B with a local
`POST /v1/systemone` API. The supported gpu selectors
are `5090` and `pro6000` The default is `pro6000`.

## Project layout

The root `CMakeLists.txt` builds `gevva-engine`;
`pyproject.toml` installs the `gevva` API and launcher.

- `src/` and `include/gevva/`: native inference implementation.
- `gevva/`: Python API, configuration, and worker lifecycle.
- `tools/`: model download and preparation.
- `tests/` and `examples/`: checks and runnable request examples.
- `third_party/`, `build/`, and `runs/`: ignored dependencies, build output, and local results.

## Build and install

Requires Linux, Python 3.11+, a CUDA-capable NVIDIA driver, CUDA 13 including
nvJPEG, cuDNN, CMake 3.28+, Ninja, a C++20 compiler, nlohmann-json, libjpeg,
libpng, and OpenMP. The tested toolchain is CUDA 13.2 / driver 610.57.04.
Native GPU kernels target SM120 specifically.

From a fresh checkout:

```sh
./setup.sh
# Review gevva.toml: the supplied example selects 5090, ./models, port 8081.
# Change engine.gpu to "pro6000" for the RTX PRO 6000.
.venv/bin/python tools/prepare_model.py --config gevva.toml --download
.venv/bin/gevva --config gevva.toml --check-config
.venv/bin/gevva --config gevva.toml
```

`setup.sh` fetches pinned CUTLASS and cuDNN frontend headers, builds the engine,
creates `.venv`, and installs the API plus offline preparation dependencies.
It creates `gevva.toml` only when absent. Existing dependency directories with
the required headers are reused. It does not download weights or start a server.
No virtualenv activation is needed for these commands. Use `PYTHON=python3.13`
to select an interpreter or `JOBS=4` to limit build parallelism.

The dependency revisions are CUTLASS `59e3a3338d516ca6ce0e073af8da65289678a35c`
and cuDNN frontend `f77fbc3d21be3f24cd0286b9b368105f7c518b8a`.
For manual builds, fetch those sources into `third_party/cutlass` and
`third_party/cudnn-frontend`, then run `cmake -S . -B build -G Ninja` and
`cmake --build build --parallel 6`. Install Python with
`python3 -m venv .venv` and `.venv/bin/python -m pip install '.[prepare]'`.

If CUDA is outside your compiler search path, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to `./setup.sh` (or when configuring CMake).
To install the native executable elsewhere:
`cmake --install build --prefix /your/install/prefix`.
Set `engine.binary` to that installed executable, or use a bare `gevva-engine` found on PATH.
The Python wheel contains the API, launcher, and warmup image; it does not bundle
the native executable, CUDA libraries, or model weights. The source distribution
includes the native sources and preparation tools for the same root build.

## Model preparation

Supported checkpoint:
[NVIDIA Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4).
The engine also needs three prepared sidecars: packed NVFP4 experts, FP8 dense
projections, and an INT8 vocabulary matrix. They are converted runtime files,
not additional model downloads. The same prepared files work on both cards.

Edit `model.root` and `engine.gpu` in `gevva.toml`, then run:

```sh
# Offline preparation dependencies; not needed by the serving process.
.venv/bin/python -m pip install '.[prepare]'
.venv/bin/python tools/prepare_model.py --config gevva.toml --download
```

Preparation requires CUDA-enabled PyTorch with SM120 support. If your PyTorch
installation is CPU-only, install a CUDA build using the
[PyTorch installer](https://pytorch.org/get-started/locally/) before preparing.
For an already downloaded checkpoint, set `model.target` to its directory and
omit `--download`. `--dry-run` prints the resolved commands without touching
weights. Completed sidecar files are reused; `--force` rebuilds them.

Preparation was validated with PyTorch 2.13.0 (CUDA 13.0), safetensors 0.8.0,
FlashInfer 0.6.17, and huggingface_hub 1.27.0.

The default directory layout is:

```text
models/
  gemma4-26b-a4b-nvfp4/             # NVIDIA checkpoint, tokenizer, config
  gemma4-26b-a4b-trtllm/            # 30 packed expert files
  gemma4-26b-a4b-dense-fp8/         # 30 dense projection files
  gemma4-26b-a4b-target-vocab-int8/ # vocabulary matrix
```

Allow roughly 35 GB of storage for this bundle. Paths can contain spaces and
can be changed without rebuilding the engine. `model.target`, `model.experts`,
`model.dense`, and `model.vocab` override individual directories if needed.
All four components must belong to the same supported checkpoint; this does
not make the engine a general loader for other Gemma sizes or architectures.

## Configuration

A full example is in [gevva.example.toml](gevva.example.toml).
Launch configuration is selected in this order:

1. `--config PATH` or `LocalEvaluator(config=PATH)`.
2. `GEVVA_CONFIG`.
3. `gevva.toml` in the current working directory.
4. Defaults: `pro6000`, model directories under `./models`, loopback port 8081.

An explicitly selected missing or malformed file is an error. Paths inside
TOML are relative to that file, not the launch directory; `~` is supported.
A bare executable name is searched on PATH. In a source checkout the default
executable is `build/gevva-engine`; an installed Python package searches for `gevva-engine`.

Values use **command-line/Python overrides > environment > TOML > defaults**.
`GEVVA_GPU` and `CUDA_VISIBLE_DEVICES` select the GPU.
`GEVVA_ENGINE`, `GEVVA_MODEL_ROOT`, and `GEVVA_MODEL_TARGET`, `GEVVA_MODEL_EXPERTS`,
`GEVVA_MODEL_DENSE`, `GEVVA_MODEL_VOCAB` are also accepted. Paths supplied through the
environment or CLI are relative to the launch directory. A root override moves
only derived component paths; individually configured component paths stay fixed.
Each worker receives its own environment, so selecting one GPU does not mutate
another evaluator's selection in the same Python process.

| Table | Settings |
| --- | --- |
| `engine` | `binary`, `gpu`, `worker_timeout` (seconds, default 120), `startup_timeout` (first response, default 300), optional `cuda_visible_devices` |
| `model` | `root`, optional `target`, `experts`, `dense`, `vocab` |
| `server` | `host`, `port`, `queue_capacity` (8), `max_connections` (32), `api_key_env` (`GEVVA_API_KEY`) |
| `warmup` | `images` (false), `common_shapes` (true) |

Unknown keys and invalid types are rejected. `schema_version = 1` is optional.
When the environment variable named by `api_key_env` is set, requests need
`Authorization: Bearer <value>`. The key itself is not stored or printed in TOML.
`--print-config` prints resolved settings; `--check-config` checks configuration
and required files without loading a model or touching a GPU. It does not check
weight contents or guarantee free GPU memory.

## Launch the API

```sh
gevva --config gevva.toml --check-config
gevva --config gevva.toml
# Equivalent without the console script:
python -m gevva --config gevva.toml
# Bind address, port, or GPU overrides:
python -m gevva --config gevva.toml --host 0.0.0.0 --port 8090 --gpu 5090
```

`--bind` aliases `--host`. `python -m gevva.server` also remains supported.
The server prints a ready record and opens its HTTP port after warmup. Stop it
with Ctrl-C. Cold startup gets up to 300 seconds to load weights; later requests
use the separate 120-second worker timeout. `GET /health` reports readiness; `GET /v1/models` lists aliases.

```sh
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/email-flags.json
```

For in-process use:

```python
from gevva import LocalEvaluator

with LocalEvaluator(config="gevva.toml", gpu="5090") as evaluator:
    response = evaluator.evaluate({
        "state": "Please refund the duplicate charge.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is a refund requested?"}
        },
    })
```

The API supports `choice`, `score`, and `noul` questions. It returns model scores;
probabilities are not calibrated. It follows the Jev request shape for local
experiments and is not the Jev model.

## Recommended settings

Use the example's image warmup and common-shape warmup. Keep a process resident
and use a persistent `gevva.Client`; repeatedly launching the model adds seconds.
One, five, and ten questions per shared state are the measured operating points.
The API defaults to read-only shared prefixes and restricted-option readout.

For fast screenshot decisions, set `parameters.image_soft_tokens` to **140**
on the request. The API default is **280**, which preserves more image detail.
Supported budgets are 70, 140, 280, 560, and 1120. Larger images/contexts/batches
need more time and memory. Tested workloads used about 23–24 GiB for the worker;
the 5090 reached about 26 GiB device-wide with the desktop included. Leave
headroom for other applications.

The README speed table uses an RTX PRO 6000 Blackwell, 100 fresh requests per cell,
1/5/10 questions,
140-token images, and persistent HTTP after warmup. Text cases are email tasks;
image cases are synthetic scenes. Answers/second is
measured across the entire group, not inferred from median latency. Both state
and image caches are reset on each measured request. Timings exclude input
fixture generation and model startup. They are workload measurements, not
latency guarantees.

## Checks

```sh
PYTHONPATH=tests python -m unittest test_worker test_config test_api test_scheduler
python tests/test_gpu_selection.py # requires both installed GPU models
GEVVA_GPU=5090 python tests/test_decision_gpu.py
GEVVA_GPU=5090 python tests/test_multimodal_api_gpu.py
GEVVA_GPU=5090 ctest --test-dir build --output-on-failure
```

Use `GEVVA_CONFIG=/path/to/gevva.toml` when running tools outside the config's
directory. Transient results stay in ignored `runs/`. Local `gevva.toml`, model
weights, dependencies, and build artifacts are not included in the repository.
