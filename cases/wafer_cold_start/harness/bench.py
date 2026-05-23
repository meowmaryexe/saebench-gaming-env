"""Diagnostic harness used to characterize a launch-time spike observed in our
custom-kernel benchmarking pipeline.

The original symptom: a fresh-process benchmark of a custom CUDA kernel
reported it 44% slower than its reference, while a separate benchmark of the
same kernel on the same hardware put it 12% faster. The two differed in
whether the bench process had just compiled new extensions before timing.

To isolate cause, the bench was rewritten to run five labeled arms, each in
a freshly spawned process on the same node. Each arm follows the same
template:

    pin to whatever cpu we land on
    burn-in N launches (drain warmup transients)
    WINDOW A1     -> record M launches
    <arm-specific action>
    WINDOW A2     -> record M launches immediately
    sleep dwell_ms
    WINDOW A3     -> record M launches after dwell

Each record carries (phase, iter, cpu_id, wall_us, gpu_us). The matching
sched.jsonl logs every step with the cpu the process was on at that moment.

We are looking for: which action(s) produce a spike in A2 that A1 and A3 do
not have.

Arms:
    idle_pin    - no action between A1 and A2
    migrate     - sched_setaffinity to a different cpu
    fork_only   - spawn six "/bin/true" subprocesses, wait for each
    load_open   - cpp_extension.load a freshly-written cuda extension, with
                  cpu affinity open across all cpus during the load
    load_pin    - same load, but keep affinity pinned to the current cpu
                  across the load
"""
from __future__ import annotations

import csv
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.cpp_extension import load as load_extension


def sched_getcpu() -> int:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.sched_getcpu.restype = ctypes.c_int
    return int(libc.sched_getcpu())


with open(os.path.join(os.path.dirname(__file__), "kernel.cu")) as f:
    KERNEL_TEMPLATE = f.read()


def compile_probe(fn: str, salt: float, cache_root: str):
    src_root = tempfile.mkdtemp(prefix=f"src_{fn}_", dir="/tmp")
    src_path = os.path.join(src_root, f"{fn}.cu")
    with open(src_path, "w") as f:
        f.write(KERNEL_TEMPLATE.format(fn=fn, salt=f"{salt:.4f}"))
    build_dir = os.path.join(cache_root, fn)
    os.makedirs(build_dir, exist_ok=True)
    return load_extension(
        name=fn,
        sources=[src_path],
        verbose=False,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3"],
    )


def time_window(func, x, n: int, phase: str, rows: list, sched: list):
    sched.append({"event": "window_start", "phase": phase, "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})
    for it in range(n):
        cpu_id = sched_getcpu()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t_a = time.perf_counter_ns()
        s.record()
        _ = func(x)
        e.record()
        torch.cuda.synchronize()
        t_b = time.perf_counter_ns()
        rows.append((phase, it, cpu_id, (t_b - t_a) / 1000.0, s.elapsed_time(e) * 1000.0))
    sched.append({"event": "window_end", "phase": phase, "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})


def pick_other_cpu(current: int, pool: set) -> int:
    others = [c for c in pool if c != current]
    if not others:
        raise RuntimeError(f"only one cpu available; cannot migrate (pool={pool})")
    return max(others, key=lambda c: abs(c - current))


def run_arm(arm: str, label: str, out_dir: str, window: int, burn_in: int, dwell_ms: int):
    os.makedirs(out_dir, exist_ok=True)

    cache_root = "/tmp/torch_ext_cache"
    shutil.rmtree(cache_root, ignore_errors=True)
    os.makedirs(cache_root, exist_ok=True)

    full_affinity = set(os.sched_getaffinity(0))
    os.sched_setaffinity(0, full_affinity)
    cpu_a = sched_getcpu()
    os.sched_setaffinity(0, {cpu_a})

    sched: list = [{
        "event": "arm_start",
        "arm": arm, "label": label, "cpu_a": cpu_a,
        "allowed_cpus": sorted(full_affinity), "n_allowed": len(full_affinity),
        "t_ns": time.perf_counter_ns(),
    }]

    mod = compile_probe("probe_main", 0.0, cache_root)
    sched.append({"event": "compile_done", "fn": "probe_main", "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})

    x = torch.randn(1024, device="cuda")
    func = mod.probe_main
    for _ in range(burn_in):
        _ = func(x)
    torch.cuda.synchronize()
    sched.append({"event": "burn_in_done", "n": burn_in, "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})

    rows: list = []
    time_window(func, x, window, "A1_warm_cpu_a", rows, sched)

    if arm == "idle_pin":
        sched.append({"event": "no_action", "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})
    elif arm == "migrate":
        cpu_b = pick_other_cpu(cpu_a, full_affinity)
        os.sched_setaffinity(0, {cpu_b})
        sched.append({"event": "migrate", "from_cpu": cpu_a, "to_cpu": cpu_b,
                      "observed_cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})
    elif arm == "fork_only":
        cpu_pre = sched_getcpu()
        for _ in range(6):
            subprocess.run(["/bin/true"], check=True)
        cpu_post = sched_getcpu()
        sched.append({"event": "fork_only", "cpu_pre": cpu_pre, "cpu_post": cpu_post,
                      "t_ns": time.perf_counter_ns()})
    elif arm == "load_open":
        os.sched_setaffinity(0, full_affinity)
        cpu_pre = sched_getcpu()
        _ = compile_probe("probe_aux", 0.5, cache_root)
        cpu_post = sched_getcpu()
        sched.append({"event": "load", "affinity": "open",
                      "cpu_pre": cpu_pre, "cpu_post": cpu_post, "t_ns": time.perf_counter_ns()})
    elif arm == "load_pin":
        cpu_pre = sched_getcpu()
        _ = compile_probe("probe_aux", 0.5, cache_root)
        cpu_post = sched_getcpu()
        sched.append({"event": "load", "affinity": "pinned",
                      "cpu_pre": cpu_pre, "cpu_post": cpu_post, "t_ns": time.perf_counter_ns()})
    else:
        raise SystemExit(f"unknown arm: {arm}")

    time_window(func, x, window, "A2_post_action", rows, sched)

    if dwell_ms > 0:
        time.sleep(dwell_ms / 1000.0)
        sched.append({"event": "dwell_done", "ms": dwell_ms, "cpu": sched_getcpu(), "t_ns": time.perf_counter_ns()})

    time_window(func, x, window, "A3_dwell", rows, sched)

    with open(os.path.join(out_dir, "per_iter.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "iter", "cpu_id", "wall_us", "gpu_us"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}"])

    with open(os.path.join(out_dir, "sched.jsonl"), "w") as f:
        for row in sched:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "arm": arm, "label": label, "cpu_a": cpu_a,
            "window": window, "burn_in": burn_in, "dwell_ms": dwell_ms,
            "n_allowed_cpus": len(full_affinity),
        }, f, indent=2)

    sys_path = os.path.join(out_dir, "system.txt")
    with open(sys_path, "w") as f:
        for cmd in (
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.gr,clocks.mem,pstate", "--format=csv"],
            ["uname", "-a"], ["nproc"], ["lscpu"],
            ["taskset", "-pc", str(os.getpid())],
        ):
            try:
                f.write(f"\n$ {' '.join(cmd)}\n")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                f.write(r.stdout)
                if r.stderr:
                    f.write("[stderr]\n" + r.stderr)
            except Exception as exc:
                f.write(f"[error] {exc}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--burn-in", type=int, default=2500)
    ap.add_argument("--dwell-ms", type=int, default=300)
    args = ap.parse_args()
    run_arm(args.arm, args.label, args.out_dir, args.window, args.burn_in, args.dwell_ms)
