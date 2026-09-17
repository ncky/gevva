#!/usr/bin/env python3
"""Development-only Marlin reference benchmark for one ModelOpt NVFP4 matrix.

This intentionally imports only SGLang's low-level JIT wrappers; the native g4
runtime has no Python, Torch, or SGLang dependency.
"""

import json
import os
from pathlib import Path


GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"
MODEL = Path("/mnt/SSD/g4-models/gemma4-26b-a4b-nvfp4")
BASE = "model.language_model.layers.0.experts.0.gate_proj"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch
    from safetensors import safe_open
    from sgl_kernel.scalar_type import scalar_types
    from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(suffix: str) -> torch.Tensor:
        name = BASE + suffix
        with safe_open(MODEL / index[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).cuda()

    weight = tensor(".weight")
    scale = tensor(".weight_scale")
    global_scale = tensor(".weight_scale_2")
    n, packed_k = weight.shape
    k = packed_k * 2
    qweight = gptq_marlin_repack(
        weight.view(torch.int32).T.contiguous(),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        k,
        n,
        4,
    )

    scale_perm = []
    for i in range(8):
        scale_perm.extend(i + 8 * j for j in range(8))
    marlin_scale = scale.T.contiguous().to(torch.bfloat16)
    marlin_scale = marlin_scale.reshape(-1, 64)[:, scale_perm].reshape(-1, n).contiguous()
    marlin_scale = marlin_scale.to(torch.half)
    marlin_scale = marlin_scale.view(-1, 4)[:, [0, 2, 1, 3]].reshape(marlin_scale.size(0), -1)
    marlin_scale = (marlin_scale * (2**7)).view(torch.int16) << 1
    marlin_scale = marlin_scale.view(torch.float8_e4m3fn)[:, 1::2].contiguous()

    # SGLang adjusts the tensor-wide scale to account for FP4 versus BF16 exponent bias.
    marlin_global = global_scale.to(torch.bfloat16).reshape(1) * (2.0**119)
    workspace = torch.zeros(
        torch.cuda.get_device_properties(0).multi_processor_count,
        dtype=torch.int32,
        device="cuda",
    )
    activation = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")

    def run() -> torch.Tensor:
        return gptq_marlin_gemm(
            a=activation,
            c=None,
            b_q_weight=qweight,
            b_scales=marlin_scale,
            global_scale=marlin_global,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=1,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=False,
        )

    for _ in range(20):
        run()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(1000):
        output = run()
    end.record()
    end.synchronize()
    print(f"shape=1x{n}x{k} marlin_us={begin.elapsed_time(end):.3f} output_norm={output.float().norm():.5f}")


if __name__ == "__main__":
    main()
