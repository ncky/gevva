#!/usr/bin/env python3
"""Measure SGLang's CUTLASS NVFP4 MoE at Gemma 4's exact decode shape.

This is a reference-backend probe, not part of the standalone runtime.  It uses
synthetic already-quantized weights so setup does not dominate the measurement.
"""

import argparse
import os
import statistics


TARGET_GPU = "GPU-8b5d604a-476b-a8f4-02a2-6e28acf2796d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU
    import torch
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
    from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4
    from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    # PyTorch 2.13 reports this build's PCI field as decimal bus "15";
    # older releases return the full CUDA bus identifier.
    if str(props.pci_bus_id).lower() not in {"15", "0000:0f:00.0"} or "PRO 6000" not in props.name:
        raise SystemExit(f"refusing GPU at {props.pci_bus_id}: {props.name}")

    m, n, k, experts, topk = 1, 704, 2816, 128, 8
    device = torch.device("cuda")
    # CUTLASS FP4 block-scale tensors use its 128-row / four-scale-column
    # padded layout. Constant valid bit patterns are sufficient for timing.
    w1 = torch.full((experts, 2 * n, k // 2), 0x22, dtype=torch.uint8, device=device)
    w2 = torch.full((experts, k, n // 2), 0x22, dtype=torch.uint8, device=device)
    s1 = torch.ones((experts, 2 * n, k // 16), dtype=torch.float8_e4m3fn, device=device)
    s2 = torch.ones((experts, k, n // 16), dtype=torch.float8_e4m3fn, device=device)
    activations = torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.05
    ids = torch.tensor([[3, 17, 31, 52, 73, 91, 105, 127]], dtype=torch.int32, device=device)
    weights = torch.full((m, topk), 1.0 / topk, dtype=torch.float32, device=device)
    global_scale = torch.ones((experts,), dtype=torch.float32, device=device)
    alpha = torch.ones((experts,), dtype=torch.float32, device=device)
    params = CutlassMoEParams(CutlassMoEType.BlockscaledFP4, device, experts, n, k)

    def run():
        return cutlass_moe_fp4(
            a=activations,
            a1_gscale=global_scale,
            w1_fp4=w1,
            w1_blockscale=s1,
            w1_alphas=alpha,
            a2_gscale=global_scale,
            w2_fp4=w2,
            w2_blockscale=s2,
            w2_alphas=alpha,
            topk_weights=weights,
            topk_ids=ids,
            params=params,
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    samples = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        begin.record()
        output = run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    graph_samples = []
    for _ in range(args.iterations):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(begin.elapsed_time(end) * 1000.0)
    print(f"gpu={props.name}")
    print(f"shape=m{m},n{n},k{k},experts{experts},topk{topk}")
    print(f"median_us={statistics.median(samples):.3f}")
    print(f"mean_us={statistics.mean(samples):.3f}")
    print(f"p10_us={sorted(samples)[len(samples) // 10]:.3f}")
    print(f"graph_median_us={statistics.median(graph_samples):.3f}")
    print(f"graph_mean_us={statistics.mean(graph_samples):.3f}")
    print(f"output_checksum={output.float().sum().item():.6f}")
    print(f"graph_output_checksum={graph_output.float().sum().item():.6f}")

    def run_flashinfer():
        return flashinfer_cutlass_fused_moe(
            activations,
            ids,
            weights,
            w1.view(torch.long),
            w2.view(torch.long),
            activations.dtype,
            quant_scales=[
                global_scale,
                s1.view(torch.int32),
                alpha,
                global_scale,
                s2.view(torch.int32),
                alpha,
            ],
            activation_type=ActivationType.Geglu,
        )[0]

    for _ in range(20):
        flash_output = run_flashinfer()
    flash_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flash_graph):
        flash_graph_output = run_flashinfer()
    flash_samples = []
    for _ in range(args.iterations):
        begin.record()
        flash_graph.replay()
        end.record()
        end.synchronize()
        flash_samples.append(begin.elapsed_time(end) * 1000.0)
    print(f"flashinfer_graph_median_us={statistics.median(flash_samples):.3f}")
    print(f"flashinfer_graph_mean_us={statistics.mean(flash_samples):.3f}")
    print(f"flashinfer_output_checksum={flash_graph_output.float().sum().item():.6f}")


if __name__ == "__main__":
    main()
