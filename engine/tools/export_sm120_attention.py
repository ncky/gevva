#!/usr/bin/env python3
"""AOT-build NVIDIA's experimental SM120 D256 attention (no serving Python).

Only the data-only config and kernel are loaded from the vendored frontend;
its optional Python cuDNN bindings and torch are intentionally not imported.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import types


def load_kernel(root):
    for name in ("cudnn", "cudnn.frost", "cudnn.frost.tile_dsl", "cudnn.sdpa",
                 "cudnn.sdpa.fwd", "cudnn.sdpa.fwd.kernels"):
        module = types.ModuleType(name)
        module.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = module
    name = "cudnn.sdpa.fwd.kernels.prefill_f16_sm120"
    spec = importlib.util.spec_from_file_location(
        name, root / "cudnn/sdpa/fwd/kernels/prefill_f16_sm120.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/sm120-attention"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query", type=int, default=1057)
    parser.add_argument("--keys", type=int, default=0)
    parser.add_argument("--q-tile", type=int, choices=(64, 128), default=64)
    # D256 uses two BF16 KV tiles: 128 rows exceeds SM120's per-CTA SMEM.
    parser.add_argument("--kv-tile", type=int, choices=(64,), default=64)
    parser.add_argument("--name", default="attention")
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    import cutlass
    import cutlass.cute as cute
    root = Path(__file__).resolve().parents[1]
    module = load_kernel(root / "external/cudnn-frontend/python")
    kernel = module.SM120FusedMultiHeadAttentionForward(
        in_dtype=cutlass.BFloat16, out_dtype=cutlass.BFloat16,
        is_causal=True, causal_bottom_right=True,
        window_size_left=1023, head_tile=256,
        q_tile=args.q_tile, kv_tile=args.kv_tile)
    b, sq, sk = args.batch, args.query, args.keys or args.query
    if args.dynamic:
        b, sq, sk = cute.SymInt(), cute.SymInt(), cute.SymInt()
    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16)
    q = tensor(cutlass.BFloat16, (b, sq, 16, 256))
    k = tensor(cutlass.BFloat16, (b, sk, 8, 256))
    from sm120_attention_launch import launch
    compiled = cute.compile(
        launch, kernel, q, k, k, q, tensor(cutlass.Float32, (b, 16, sq)),
        tensor(cutlass.Float32, (16,)), tensor(cutlass.Int32, (b,)),
        tensor(cutlass.Int32, (b,)), cutlass.Float32(1.4426950408889634),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False))
    args.output.mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(file_path=str(args.output), file_name=args.name,
                         function_prefix=f"g4_sm120_{args.name}")
    print(f"Exported SM120 attention to {args.output}")


if __name__ == "__main__":
    main()
