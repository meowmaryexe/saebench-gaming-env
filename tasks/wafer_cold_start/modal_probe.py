"""Modal probe: reproduce the cpu-migration cold-cache kernel-launch effect.

Blog under reconstruction: "Where did my microseconds go" (wafer.ai/blog).
The mechanism: every CUDA kernel launch reads CPU-side dispatch structures
held in L1/L2 of whatever core the python process is running on. If the
process migrates to a new core, those structures are cold on the new core
and the next ~50 launches pay the miss penalty. In the blog, this happened
organically because `cpp_extension.load()` forks nvcc which causes CFS to
shuffle the parent; result was a 44% apparent regression that recovered.

On Modal the cgroup-pinned container does not migrate organically (we
verified: 17-core affinity but the scheduler keeps the process on a single
core under low contention). So we drive migration explicitly via
`sched_setaffinity` and measure the launch-time response.

Probe shape:
  * Three arms, each in a fresh container so substrate matches.
  * Each arm: 2000-launch burn-in, then three measurement windows of N iters.
      - `pinned`     : windows are A1/A2/A3 — stay on cpu_a throughout
      - `migrate`    : A1 on cpu_a → migrate to cpu_b → A2 immediate → wait → A3
      - `loadwave`   : A1 on cpu_a → cpp_extension.load() of a NEW kernel →
                       A2 immediate after load → A3 after dwell. Captures the
                       organic version on hardware where ninja forks DO trigger
                       migration; on hardware where they don't, the trace shows
                       no spike and the agent must explain why.

Output per arm at /data/wafer_cold_start/<run_id>/<arm>/
  - per_iter.csv  (phase,iter,cpu_id,wall_us,gpu_us)
  - sched.jsonl   (event markers: arm_start, window_start, migrate, load, ...)
  - summary.json
  - system.txt    (nvidia-smi, taskset, nproc, uname)

Usage:
  modal run modal_probe.py --action probe --window 400 --run-id <id>
"""
from __future__ import annotations

import modal

APP_NAME = "wafer-cold-start-probe"
VOLUME_NAME = "wafer-cold-start"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python=None,
    )
    .apt_install("linux-tools-generic", "linux-tools-common", "util-linux", "procps")
    .pip_install("numpy", "pandas", "psutil")
    .env({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "TORCH_EXTENSIONS_DIR": "/tmp/torch_ext_cache",
        "CUDA_LAUNCH_BLOCKING": "0",
    })
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _sched_getcpu() -> int:
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.sched_getcpu.restype = ctypes.c_int
    return int(libc.sched_getcpu())


KERNEL_TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void {fn}_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 1.0001f + {salt}f;
}}

torch::Tensor {fn}(torch::Tensor x) {{
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    {fn}_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{fn}", &{fn}, "tiny launch-overhead probe ({fn})");
}}
"""


def _compile(fn: str, salt: float):
    import os, tempfile
    from torch.utils.cpp_extension import load as load_extension
    src_root = tempfile.mkdtemp(prefix=f"src_{fn}_", dir="/tmp")
    src_path = os.path.join(src_root, f"{fn}.cu")
    with open(src_path, "w") as f:
        f.write(KERNEL_TEMPLATE.format(fn=fn, salt=f"{salt:.4f}"))
    build_dir = os.path.join("/tmp/torch_ext_cache", fn)
    os.makedirs(build_dir, exist_ok=True)
    return load_extension(
        name=fn,
        sources=[src_path],
        verbose=False,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3"],
    )


def _time_block(mod, fn: str, x, n: int, phase: str, rows: list, sched_log: list):
    import time
    import torch
    sched_log.append({"event": "window_start", "phase": phase, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})
    func = getattr(mod, fn)
    for it in range(n):
        cpu_id = _sched_getcpu()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t_a = time.perf_counter_ns()
        s.record()
        _ = func(x)
        e.record()
        torch.cuda.synchronize()
        t_b = time.perf_counter_ns()
        rows.append((phase, it, cpu_id, (t_b - t_a) / 1000.0, s.elapsed_time(e) * 1000.0))
    sched_log.append({"event": "window_end", "phase": phase, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})


def _write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "iter", "cpu_id", "wall_us", "gpu_us"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.3f}"])


def _write_sched(sched_log, path):
    import json
    with open(path, "w") as f:
        for row in sched_log:
            f.write(json.dumps(row) + "\n")


def _write_system(out_dir):
    import os
    import subprocess
    sys_info_path = os.path.join(out_dir, "system.txt")
    with open(sys_info_path, "w") as f:
        for cmd in (
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.gr,clocks.mem,pstate", "--format=csv"],
            ["uname", "-a"],
            ["nproc"],
            ["lscpu"],
            ["taskset", "-pc", str(os.getpid())],
        ):
            try:
                f.write(f"\n$ {' '.join(cmd)}\n")
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                f.write(r.stdout)
                if r.stderr:
                    f.write("[stderr]\n" + r.stderr)
            except Exception as e:
                f.write(f"[error] {e}\n")


def _allowed_cpus() -> list:
    import os
    # full affinity set: every cpu the cgroup permits
    return sorted(os.sched_getaffinity(0))


def _pick_other_cpu(current: int, pool: set) -> int:
    others = [c for c in pool if c != current]
    if not others:
        raise RuntimeError(f"only one cpu available ({current}); cannot migrate (pool={pool})")
    return max(others, key=lambda c: abs(c - current))


def _run_arm(arm: str, window: int, burn_in: int, dwell_ms: int, out_root: str) -> dict:
    import json
    import os
    import shutil
    import time

    import torch

    out_dir = os.path.join(out_root, arm)
    os.makedirs(out_dir, exist_ok=True)
    _write_system(out_dir)

    # full cache wipe so cpp_extension actually compiles
    shutil.rmtree("/tmp/torch_ext_cache", ignore_errors=True)
    os.makedirs("/tmp/torch_ext_cache", exist_ok=True)

    # ensure affinity is open so we can pin deliberately
    full = set(_allowed_cpus())
    os.sched_setaffinity(0, full)

    # pin to current cpu for stability before burn-in
    cpu_a = _sched_getcpu()
    os.sched_setaffinity(0, {cpu_a})

    sched_log: list = [{
        "event": "arm_start",
        "arm": arm,
        "cpu_a": cpu_a,
        "allowed_cpus": list(sorted(full)),
        "n_allowed": len(full),
        "t_ns": time.perf_counter_ns(),
    }]

    # compile the main probe kernel
    mod = _compile("probe_main", 0.0)
    sched_log.append({"event": "compile_done", "fn": "probe_main", "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    x = torch.randn(1024, device="cuda")
    func = mod.probe_main

    # burn-in (GPU warmup, cublas init, dispatch path caches)
    for _ in range(burn_in):
        _ = func(x)
    torch.cuda.synchronize()
    sched_log.append({"event": "burn_in_done", "n": burn_in, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    rows: list = []

    # WINDOW A1: steady on cpu_a
    _time_block(mod, "probe_main", x, window, "A1_warm_cpu_a", rows, sched_log)

    # ARM-SPECIFIC ACTION
    if arm == "pinned":
        # no-op
        sched_log.append({"event": "no_action", "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})
    elif arm == "migrate":
        cpu_b = _pick_other_cpu(cpu_a, full)
        os.sched_setaffinity(0, {cpu_b})
        # force the scheduler to move us right now
        try:
            time.sleep(0)
        except Exception:
            pass
        sched_log.append({
            "event": "migrate",
            "from_cpu": cpu_a,
            "to_cpu": cpu_b,
            "observed_cpu": _sched_getcpu(),
            "t_ns": time.perf_counter_ns(),
        })
    elif arm == "loadwave":
        # mirror the blog: compile a second kernel mid-measurement
        # leave affinity open during the compile so ninja can migrate us
        os.sched_setaffinity(0, full)
        cpu_pre = _sched_getcpu()
        mod2 = _compile("probe_aux", 0.5)
        cpu_post = _sched_getcpu()
        sched_log.append({
            "event": "loadwave",
            "cpu_pre": cpu_pre,
            "cpu_post": cpu_post,
            "t_ns": time.perf_counter_ns(),
        })
        # don't re-pin; let the scheduler keep us wherever it dropped us
    else:
        raise RuntimeError(f"unknown arm: {arm}")

    # WINDOW A2: immediate
    _time_block(mod, "probe_main", x, window, "A2_post_action", rows, sched_log)

    # DWELL
    if dwell_ms > 0:
        time.sleep(dwell_ms / 1000.0)
        sched_log.append({"event": "dwell_done", "ms": dwell_ms, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    # WINDOW A3: after dwell
    _time_block(mod, "probe_main", x, window, "A3_dwell", rows, sched_log)

    _write_csv(rows, os.path.join(out_dir, "per_iter.csv"))
    _write_sched(sched_log, os.path.join(out_dir, "sched.jsonl"))

    summary = {
        "arm": arm,
        "cpu_a": cpu_a,
        "window": window,
        "burn_in": burn_in,
        "dwell_ms": dwell_ms,
        "n_allowed_cpus": len(full),
        "events": [ev for ev in sched_log if ev["event"] in ("migrate", "loadwave", "compile_done", "no_action")],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {"arm": arm, "out_dir": out_dir, "n_rows": len(rows)}


@app.function(gpu="H100:1", cpu=8.0, timeout=60 * 30, volumes={"/data": volume})
def run_one(arm: str, rep: int, window: int, burn_in: int, dwell_ms: int, run_id: str) -> dict:
    import os
    out_root = f"/data/wafer_cold_start/{run_id}"
    arm_label = f"{arm}_rep{rep:02d}"
    os.makedirs(out_root, exist_ok=True)
    # _run_arm uses its arm string for the subdir and for the action branch,
    # so we pass the bare arm name but write into a rep-suffixed dir
    r = _run_arm_with_label(arm, arm_label, window, burn_in, dwell_ms, out_root)
    volume.commit()
    return r


def _run_arm_with_label(arm: str, label: str, window: int, burn_in: int, dwell_ms: int, out_root: str) -> dict:
    # wrap _run_arm so the on-disk directory uses label (with rep suffix) while
    # the arm-action branch keeps using the bare arm name
    import json
    import os
    import shutil
    import time

    import torch

    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)
    _write_system(out_dir)

    shutil.rmtree("/tmp/torch_ext_cache", ignore_errors=True)
    os.makedirs("/tmp/torch_ext_cache", exist_ok=True)

    full = set(_allowed_cpus())
    os.sched_setaffinity(0, full)

    cpu_a = _sched_getcpu()
    os.sched_setaffinity(0, {cpu_a})

    sched_log: list = [{
        "event": "arm_start",
        "arm": arm,
        "label": label,
        "cpu_a": cpu_a,
        "allowed_cpus": list(sorted(full)),
        "n_allowed": len(full),
        "t_ns": time.perf_counter_ns(),
    }]

    mod = _compile("probe_main", 0.0)
    sched_log.append({"event": "compile_done", "fn": "probe_main", "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    x = torch.randn(1024, device="cuda")
    func = mod.probe_main
    for _ in range(burn_in):
        _ = func(x)
    torch.cuda.synchronize()
    sched_log.append({"event": "burn_in_done", "n": burn_in, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    rows: list = []
    _time_block(mod, "probe_main", x, window, "A1_warm_cpu_a", rows, sched_log)

    if arm == "idle_pin":
        # control: pinned, nothing happens
        sched_log.append({"event": "no_action", "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})
    elif arm == "migrate":
        # explicit migration only, no other workload — rules out / in pure-migration as cause
        cpu_b = _pick_other_cpu(cpu_a, full)
        os.sched_setaffinity(0, {cpu_b})
        sched_log.append({
            "event": "migrate",
            "from_cpu": cpu_a,
            "to_cpu": cpu_b,
            "observed_cpu": _sched_getcpu(),
            "t_ns": time.perf_counter_ns(),
        })
    elif arm == "load_open":
        # blog's bug case: open affinity, then compile a fresh extension.
        # the fork/exec invalidates process caches; if scheduler also migrates,
        # cost compounds.
        os.sched_setaffinity(0, full)
        cpu_pre = _sched_getcpu()
        _ = _compile("probe_aux", 0.5)
        cpu_post = _sched_getcpu()
        sched_log.append({
            "event": "load",
            "affinity": "open",
            "cpu_pre": cpu_pre,
            "cpu_post": cpu_post,
            "t_ns": time.perf_counter_ns(),
        })
    elif arm == "load_pin":
        # blog's claimed fix: keep affinity pinned across the compile
        cpu_pre = _sched_getcpu()
        _ = _compile("probe_aux", 0.5)
        cpu_post = _sched_getcpu()
        sched_log.append({
            "event": "load",
            "affinity": "pinned",
            "cpu_pre": cpu_pre,
            "cpu_post": cpu_post,
            "t_ns": time.perf_counter_ns(),
        })
    elif arm == "fork_only":
        # factor out: fork+exec a no-op subprocess. no .so loading, no CUDA.
        # if this spikes -> any fork is the trigger. if not -> the compile/.so is.
        import subprocess
        cpu_pre = _sched_getcpu()
        # match ~the wall-clock duration of the compile so we see comparable
        # post-action quiescence, but only the fork machinery is exercised.
        # use multiple forks to mirror how ninja+nvcc spawn several children.
        for _ in range(6):
            subprocess.run(["/bin/true"], check=True)
        cpu_post = _sched_getcpu()
        sched_log.append({
            "event": "fork_only",
            "cpu_pre": cpu_pre,
            "cpu_post": cpu_post,
            "t_ns": time.perf_counter_ns(),
        })
    else:
        raise RuntimeError(f"unknown arm: {arm}")

    _time_block(mod, "probe_main", x, window, "A2_post_action", rows, sched_log)

    if dwell_ms > 0:
        time.sleep(dwell_ms / 1000.0)
        sched_log.append({"event": "dwell_done", "ms": dwell_ms, "cpu": _sched_getcpu(), "t_ns": time.perf_counter_ns()})

    _time_block(mod, "probe_main", x, window, "A3_dwell", rows, sched_log)

    _write_csv(rows, os.path.join(out_dir, "per_iter.csv"))
    _write_sched(sched_log, os.path.join(out_dir, "sched.jsonl"))

    summary = {
        "arm": arm,
        "label": label,
        "cpu_a": cpu_a,
        "window": window,
        "burn_in": burn_in,
        "dwell_ms": dwell_ms,
        "n_allowed_cpus": len(full),
        "events": [ev for ev in sched_log if ev["event"] in ("migrate", "loadwave", "compile_done", "no_action")],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {"arm": arm, "label": label, "out_dir": out_dir, "n_rows": len(rows)}


@app.local_entrypoint()
def main(
    action: str = "probe",
    window: int = 500,
    burn_in: int = 2500,
    dwell_ms: int = 300,
    reps: int = 3,
    run_id: str = "",
    arms: str = "idle_pin,migrate,fork_only,load_open,load_pin",
) -> None:
    import time
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    if action == "probe":
        jobs = [(arm, rep, window, burn_in, dwell_ms, run_id)
                for arm in arm_list for rep in range(reps)]
        results = list(run_one.starmap(jobs))
        print(f"[probe] run_id={run_id} jobs={len(jobs)}")
        for r in results:
            print(r)
    else:
        raise SystemExit(f"unknown action: {action}")
