"""Decode-step bench for fla.ops.kda.fused_recurrent_kda on H100.

Shape:
  B  = 1     (single sequence, decode workload)
  T  = 1     (one new token per call)
  H  = 16    (KV groups)
  HV = 16    (Q heads)
  K  = 128   (qk head dim)
  V  = 128   (v head dim)

Reports per-call wall time via torch.cuda.Event over 500 reps, plus a
torch.profiler trace over 10 reps to surface per-kernel CUDA time and
shape/launch info. Hardware snapshot, grid estimate, and FLA module
sources accompany this script in the same directory.
"""
from __future__ import annotations

import json
import os
import subprocess

import torch
from fla.ops.kda import fused_recurrent_kda


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "hardware.txt"), "w") as f:
        for cmd in (
            ["nvidia-smi"],
            ["uname", "-a"],
            ["nproc"],
        ):
            try:
                f.write(f"\n$ {' '.join(cmd)}\n")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                f.write(r.stdout)
                if r.stderr:
                    f.write("[stderr]\n" + r.stderr)
            except Exception as exc:
                f.write(f"[error] {exc}\n")
        f.write("\n$ torch.cuda.get_device_properties\n")
        prop = torch.cuda.get_device_properties(0)
        for kk in ("name", "multi_processor_count", "shared_memory_per_block",
                   "shared_memory_per_block_optin", "warp_size",
                   "regs_per_multiprocessor", "max_threads_per_multi_processor",
                   "total_memory", "major", "minor", "L2_cache_size"):
            f.write(f"{kk} = {getattr(prop, kk, '?')}\n")

    B, T, H, HV, K, V = 1, 1, 16, 16, 128, 128
    dtype = torch.bfloat16
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(B, T, H,  K, dtype=dtype, device=device)
    k_t = torch.randn(B, T, H,  K, dtype=dtype, device=device)
    v_t = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype=dtype, device=device))
    A_log = torch.zeros(HV, dtype=torch.float32, device=device)

    for _ in range(20):
        fused_recurrent_kda(q=q, k=k_t, v=v_t, g=g, beta=beta, A_log=A_log,
                            use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()

    reps = 500
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fused_recurrent_kda(q=q, k=k_t, v=v_t, g=g, beta=beta, A_log=A_log,
                            use_qk_l2norm_in_kernel=True)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    per_call_us = (total_ms * 1000.0) / reps

    bench = {
        "B": B, "T": T, "H": H, "HV": HV, "K": K, "V": V,
        "dtype": str(dtype), "reps": reps,
        "total_ms": total_ms, "per_call_us": per_call_us,
        "sm_count": prop.multi_processor_count,
        "device": prop.name,
    }
    with open(os.path.join(out_dir, "bench.json"), "w") as f:
        json.dump(bench, f, indent=2)

    from torch.profiler import profile, ProfilerActivity, record_function
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            with record_function("kda_step"):
                fused_recurrent_kda(q=q, k=k_t, v=v_t, g=g, beta=beta, A_log=A_log,
                                    use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
    prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
    with open(os.path.join(out_dir, "profile_table.txt"), "w") as f:
        f.write(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=40
        ))

    # default heuristic for fused_recurrent_kda: BK = min(64, K), BV = min(64, V)
    BK = min(64, K)
    BV = min(64, V)
    NK = (K + BK - 1) // BK
    NV = (V + BV - 1) // BV
    grid_blocks = NK * NV * B * HV
    with open(os.path.join(out_dir, "grid_estimate.txt"), "w") as f:
        f.write(f"# Estimated grid for fused_recurrent_kda fwd\n")
        f.write(f"# (default heuristic: BK = min(64, K), BV = min(64, V))\n")
        f.write(f"BK = {BK}\n")
        f.write(f"BV = {BV}\n")
        f.write(f"NK = ceil(K / BK) = {NK}\n")
        f.write(f"NV = ceil(V / BV) = {NV}\n")
        f.write(f"N  = B = {B}\n")
        f.write(f"HV = {HV}\n")
        f.write(f"grid_blocks = NK * NV * N * HV = {grid_blocks}\n")
        f.write(f"sm_count    = {prop.multi_processor_count}\n")
        f.write(f"blocks_per_sm = {grid_blocks / prop.multi_processor_count:.3f}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    main(args.out_dir)
