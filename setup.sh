#!/usr/bin/env bash
# Build Gevva and install its serving/model-preparation environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage: ./setup.sh [CMake options...]
Builds the engine, installs .venv (including model preparation dependencies),
and creates gevva.toml from the example if it does not already exist.
Requires the system packages listed in SETUP.md; does not install drivers.
Environment: PYTHON=python3, JOBS=6. Extra arguments go to CMake.
Next: edit gevva.toml, then:
  .venv/bin/python tools/prepare_model.py --config gevva.toml --download
  .venv/bin/gevva --config gevva.toml
Already have the checkpoint? Set model.target in TOML and omit --download.
EOF
    exit 0
fi
for command in git cmake ninja "${PYTHON:-python3}"; do
    command -v "$command" >/dev/null || { echo "Missing $command; see SETUP.md." >&2; exit 1; }
done
"${PYTHON:-python3}" -c 'import sys; sys.exit("Python 3.11+ is required") if sys.version_info < (3, 11) else None'

fetch_dependency() {
    local name="$1" revision="$2" header="$3" destination="third_party/$1"
    if [[ -f "$destination/$header" ]]; then
        echo "Using existing $destination"
        return
    fi
    mkdir -p "$destination"
    if [[ ! -d "$destination/.git" ]]; then
        git init -q "$destination"
        git -C "$destination" remote add origin "https://github.com/NVIDIA/$name.git"
    fi
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach FETCH_HEAD
    [[ -f "$destination/$header" ]] || { echo "Missing $header in $destination" >&2; exit 1; }
}
fetch_dependency cutlass 59e3a3338d516ca6ce0e073af8da65289678a35c include/cutlass/cutlass.h
fetch_dependency cudnn-frontend f77fbc3d21be3f24cd0286b9b368105f7c518b8a include/cudnn_frontend.h

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --parallel "${JOBS:-6}"
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install '.[prepare]'
if [[ ! -e gevva.toml ]]; then
    cp gevva.example.toml gevva.toml
    echo "Created gevva.toml (RTX 5090, ./models, 127.0.0.1:8081)."
else
    echo "Keeping existing gevva.toml."
fi
cat <<'EOF'

Build complete. Check GPU and model paths in gevva.toml, then run:
  .venv/bin/python tools/prepare_model.py --config gevva.toml --download
  .venv/bin/gevva --config gevva.toml
Already have the checkpoint? Set model.target in TOML and omit --download.
EOF
