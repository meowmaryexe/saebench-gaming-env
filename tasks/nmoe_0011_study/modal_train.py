"""Modal launcher: regenerate the 0011 autoresearch-campaign blog_artifacts.

Study under reconstruction: "Let the speedrun search itself" — post 0011
in the Noumena-Network/nmoe series. A bounded autoresearch campaign
over `configs/speedrun/super.toml` (256 experts, fp8) at a 512-step
per-candidate budget; CORE-drop veto enforces capability preservation.

We DO NOT invoke the live `campaign auto` controller (image may not
have `nmoe/campaigns.py`, controller has LLM-proposer non-determinism).
Instead we replay each canonical candidate from `repro/0011.receipts.json`
as an explicit override, then synthesize a campaign-receipt JSON per
candidate carrying the canonical keep/discard decision.

Key differences from 0006/0008:
  * Config            configs/speedrun/super.toml  (256 experts, fp8)
  * Tokenizer / data  gpt2 / /data/speedrun/*       (REUSE from 0006)
  * Budget            512 steps × bs=256 × seq=2048 = ~134M tokens/run
  * Fleet             16 runs (10 anchor + 6 stale)
  * Float overrides   ALL search axes are floats → per-run TOML emission
                      (cannot CLI-override floats with scientific notation,
                      and aux values like 0.00012/0.00015/0.00018 stay as
                      strings under the image's parser — defensive default)
  * experiments_db    overridden per-run to /tmp/experiments_0011_<lab>.db
                      (super.toml hardcodes /data/experiments_nvfp4.db, a
                      shared FUSE path that fights under concurrent writes)
  * Synthetic receipts campaign_runs/speedrun_super_research/benchmark/
                      *.json written per candidate

Usage:
  modal run modal_train.py --action smoke                 # 64 steps, validates super.toml loads + fp8 path works
  modal run modal_train.py --action all --no-serial       # 10 anchors + 6 stale parallel
  modal run modal_train.py --action synthesize-receipts   # emit campaign_runs/*.json
  modal run modal_train.py --action pack
"""
from __future__ import annotations

import modal

APP_NAME = "nmoe-0011-training"
TRAIN_IMAGE = "xjdr/nmoe_train:latest"
VOLUME_NAME = "nmoe-0006-data"  # reuse; we co-tenant /data/blog_artifacts_0011
NMOE_COMMIT = "970a146433f9c649d09ddab36f675974f53dd905"

# 0011 lane (from campaigns/speedrun_super_research.toml):
#   runtime.config = "super"  → configs/speedrun/super.toml
#   runtime.dtype  = "fp8"     (overrides super.toml's nvfp4 default)
#   eval_enabled   = true      (CORE veto: max_core_drop = 0.002)
#   budget.steps   = 512
CAMPAIGN_NAME = "speedrun_super_research"
BASE_CONFIG = "configs/speedrun/super.toml"
CAMPAIGN_STEPS = 512
CAMPAIGN_DTYPE = "fp8"
BLOG_ROOT = "blog_artifacts_0011"

image = (
    modal.Image.from_registry(TRAIN_IMAGE, add_python="3.12")
    .run_commands(
        "/root/.local/bin/uv pip install --python /workspace/nmoe/.venv/bin/python tomli_w"
    )
    .env({
        "PATH": "/workspace/nmoe/.venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": (
            "/workspace/nmoe"
            ":/workspace/nmoe/nmoe/csrc"
            ":/workspace/nmoe/third_party/flash_attn"
            ":/workspace/nmoe/third_party/quack"
            ":/workspace/nmoe/triton/python"
        ),
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_SHM_DISABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "DATA_DIR": "/data",
    })
)

data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ============================================================================
# Canonical receipt narrative — drives both the runs AND the synthesized
# campaign receipts. From repro/0011.receipts.json (which we strip from
# the agent's repo view; this dict is the trainer-side source of truth).
# ============================================================================
#
# Each entry's `overrides` dict layers onto super.toml + the campaign-wide
# dtype=fp8 + steps=512 override. `decision` is the canonical keep/discard
# outcome from the campaign. `metrics_hint` is what the receipts say the
# candidate's final_valid_loss + core landed at — used to validate that our
# replays land in roughly the right place (a sanity check, not a constraint;
# our reruns may drift slightly from canonical due to image differences).
RUNS: list[dict] = [
    {
        "label": "0011_seed",
        "overrides": {"aux_loss_alpha": 0.0001},
        "decision": {"kept": True, "reason": "baseline",
                     "wave": "seed", "is_champion_at_decision": True},
        "metrics_hint": {"final_valid_loss": 5.1987, "core_delta": -0.0169},
    },
    {
        "label": "0011_aux_00015_kept",
        "overrides": {"aux_loss_alpha": 0.00015},
        "decision": {"kept": True, "reason": "improved_primary_no_core_veto",
                     "wave": "wakeup", "is_champion_at_decision": True},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
    {
        "label": "0011_aux_0005_vetoed",
        "overrides": {"aux_loss_alpha": 0.0005},
        "decision": {"kept": False, "reason": "core_drop_exceeded_threshold",
                     "wave": "wakeup", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
    {
        "label": "0011_lr_dense_0020",
        "overrides": {"aux_loss_alpha": 0.00015, "lr_dense": 0.0020},
        "decision": {"kept": True, "reason": "improved_primary_no_core_veto",
                     "wave": "first_parallel_wave", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
    {
        "label": "0011_lr_dense_0022_regime_change",
        "overrides": {"aux_loss_alpha": 0.00015, "lr_dense": 0.0022},
        "decision": {"kept": True, "reason": "improved_primary_no_core_veto",
                     "wave": "first_parallel_wave", "is_champion_at_decision": True},
        "metrics_hint": {"final_valid_loss": 5.1270, "core_delta": -0.0156},
    },
    {
        "label": "0011_lr_router_0021_kept",
        "overrides": {"aux_loss_alpha": 0.00015, "lr_router": 0.0021},
        "decision": {"kept": True, "reason": "improved_primary_no_core_veto",
                     "wave": "first_parallel_wave", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
    {
        "label": "0011_refine_aux_00018_core_veto",
        "overrides": {"aux_loss_alpha": 0.00018, "lr_dense": 0.0022},
        "decision": {"kept": False, "reason": "core_drop_exceeded_threshold",
                     "wave": "refinement", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": 5.1174, "core_delta": None},
    },
    {
        "label": "0011_champion_aux_00012",
        "overrides": {"aux_loss_alpha": 0.00012, "lr_dense": 0.0022},
        "decision": {"kept": True, "reason": "improved_primary_no_core_veto",
                     "wave": "refinement", "is_champion_at_decision": True,
                     "is_global_champion": True},
        "metrics_hint": {"final_valid_loss": 5.1200, "core_delta": -0.0136,
                         "tokens_per_s_per_gpu": 100700},
    },
    {
        "label": "0011_refine_aux_0002",
        "overrides": {"aux_loss_alpha": 0.0002, "lr_dense": 0.0022},
        "decision": {"kept": False, "reason": "did_not_improve_primary",
                     "wave": "refinement", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
    {
        "label": "0011_refine_3axis",
        "overrides": {"lr_router": 0.0021, "aux_loss_alpha": 0.00015, "lr_dense": 0.0022},
        "decision": {"kept": False, "reason": "did_not_improve_primary",
                     "wave": "refinement", "is_champion_at_decision": False},
        "metrics_hint": {"final_valid_loss": None, "core_delta": None},
    },
]

# Stale distractors — left-around artifacts a researcher running this
# campaign would plausibly have. Synthesized receipts mark them
# kept=false with "outside_kept_envelope" / "stale_pre_campaign" reasons.
STALE_RUNS: list[dict] = [
    {
        "label": "0011_unbounded_aux_001",
        "overrides": {"aux_loss_alpha": 0.001},
        "decision": {"kept": False, "reason": "outside_kept_envelope",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
    {
        "label": "0011_warmup_704_test",
        "overrides": {"aux_loss_alpha": 0.0001, "warmup_steps": 704},
        "decision": {"kept": False, "reason": "stale_pre_campaign",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
    {
        "label": "0011_route_scale_v2",
        "overrides": {"aux_loss_alpha": 0.0001, "route_scale": 1.5},
        "decision": {"kept": False, "reason": "scratch_trial",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
    {
        # bf16 ran before the campaign settled on fp8 — wrong lane.
        "label": "0011_bf16_baseline_v0",
        "overrides": {"aux_loss_alpha": 0.0001},
        "dtype_override": "bf16",
        "decision": {"kept": False, "reason": "wrong_lane_pre_campaign",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
    {
        "label": "0011_seed_no_aux",
        "overrides": {"aux_loss_alpha": 0.0},
        "decision": {"kept": False, "reason": "pre_seed_overrides_default",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
    {
        "label": "0011_lr_dense_0016_pre_wave",
        "overrides": {"aux_loss_alpha": 0.00015, "lr_dense": 0.0016},
        "decision": {"kept": False, "reason": "stale_pre_wave_solo_probe",
                     "wave": "operator_probe", "is_champion_at_decision": False},
    },
]


def _venv_python() -> str:
    return "/workspace/nmoe/.venv/bin/python"


def _emit_run_toml(label: str, overrides: dict, dtype_override: str | None = None) -> str:
    """Clone super.toml, layer (steps=512, dtype=fp8, experiments_db=/tmp/…,
    +caller overrides), write to a unique tempfile, return its path.

    Per-run experiments_db keeps SQLite off the shared FUSE volume so 16
    concurrent containers don't fight each other (0006 lesson: super.toml
    hardcodes /data/experiments_nvfp4.db which all containers would share).
    """
    import shutil, tempfile, tomllib
    try:
        import tomli_w  # bundled in via image.run_commands
    except ImportError:
        # Fall back to a hand-rolled writer for primitive values only.
        tomli_w = None

    with open(BASE_CONFIG, "rb") as f:
        base = tomllib.load(f)

    base["steps"] = CAMPAIGN_STEPS
    base["dtype"] = dtype_override or CAMPAIGN_DTYPE
    base["experiments_db"] = f"/tmp/experiments_0011_{label}.db"
    base["experiment_id"] = f"campaign_{label}"
    for k, v in overrides.items():
        base[k] = v

    fd = tempfile.NamedTemporaryFile("wb", suffix=f"_{label}.toml", delete=False)
    if tomli_w is not None:
        fd.write(tomli_w.dumps(base).encode("utf-8"))
    else:
        # Minimal fallback writer (top-level scalars only)
        lines = []
        for k, v in base.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f'{k} = {str(v).lower()}')
            elif isinstance(v, (int, float)):
                lines.append(f'{k} = {v}')
            else:
                continue
        fd.write(("\n".join(lines) + "\n").encode())
    fd.close()
    return fd.name


# ============================================================================
# Training
# ============================================================================

@app.function(
    gpu="B200:8",
    volumes={"/data": data_vol},
    timeout=2 * 3600,
)
def run_training(label: str, overrides: dict,
                 dtype_override: str | None = None,
                 target_root: str = BLOG_ROOT) -> dict:
    """Run one 0011 candidate at 512 steps on configs/speedrun/super.toml
    + per-run TOML overrides + dtype = fp8 (or dtype_override for stale runs).

    Outputs land at /data/<target_root>/<label>/ with metrics/, RUN_META.json,
    and (if produced) experiments.db copied from /tmp/.
    """
    import os, subprocess, shutil, time, json

    os.chdir("/workspace/nmoe")

    bundle_pre = f"/data/{target_root}/{label}"
    if os.path.isdir(bundle_pre):
        print(f"[{label}] wiping prior {bundle_pre}", flush=True)
        shutil.rmtree(bundle_pre, ignore_errors=True)

    metrics_root = "/data/metrics"
    os.makedirs(metrics_root, exist_ok=True)
    before = set(os.listdir(metrics_root))

    cfg_path = _emit_run_toml(label, overrides, dtype_override)
    print(f"[{label}] wrote run toml -> {cfg_path}", flush=True)

    cmd = [
        "torchrun", "--nproc_per_node=8", "-m", "nmoe.train",
        cfg_path,
    ]
    print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.check_call(cmd, cwd="/workspace/nmoe")
    elapsed = time.time() - t0
    print(f"[{label}] training finished in {elapsed/60:.1f} min", flush=True)

    after = set(os.listdir(metrics_root))
    new_runs = sorted(after - before)
    if not new_runs:
        raise RuntimeError(f"[{label}] no new run_id directory under {metrics_root}")
    run_id = new_runs[-1]

    bundle = f"/data/{target_root}/{label}"
    os.makedirs(f"{bundle}/metrics", exist_ok=True)
    shutil.copytree(
        f"{metrics_root}/{run_id}",
        f"{bundle}/metrics/{run_id}",
        dirs_exist_ok=True,
    )
    # The TOML override points experiments_db at /tmp/experiments_0011_<label>.db
    # (container-local). Pull that into the bundle.
    candidate_db = f"/tmp/experiments_0011_{label}.db"
    fallbacks = [candidate_db, "/data/experiments_nvfp4.db",
                 "/tmp/experiments_super.db", "/data/experiments.db"]
    for path in fallbacks:
        if os.path.exists(path):
            shutil.copy(path, f"{bundle}/experiments.db")
            print(f"[{label}] experiments.db <- {path}", flush=True)
            break
    else:
        print(f"[{label}] WARN: no experiments.db on any known path", flush=True)

    with open(os.path.join(bundle, "RUN_META.json"), "w") as f:
        json.dump({
            "label": label,
            "run_id": run_id,
            "overrides": overrides,
            "dtype_override": dtype_override,
            "elapsed_s": elapsed,
            "image": TRAIN_IMAGE,
            "agent_visible_commit": NMOE_COMMIT,
        }, f, indent=2)

    data_vol.commit()
    try:
        os.unlink(cfg_path)
    except OSError:
        pass
    return {"label": label, "run_id": run_id, "elapsed_s": elapsed}


# ============================================================================
# Synthesize the per-candidate campaign receipt JSONs
# ============================================================================

# CORE-drop veto threshold from campaigns/speedrun_super_research.toml.
CAMPAIGN_MAX_CORE_DROP = 0.002


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    volumes={"/data": data_vol},
    timeout=30 * 60,
)
def synthesize_receipts() -> dict:
    """Build per-candidate campaign receipts from REAL measurements:

      pass 1: pull (eval/CORE, valid/loss, throughput/tokens_per_s_gpu)
              at last step from each run's parquets
      pass 2: apply the campaign's CORE-veto rule against the seed's CORE
              to compute decision.kept (anchor candidates only; stale runs
              are always kept=false / reason=operator_probe)
      pass 3: pick the global champion = min(valid_loss) among kept
              anchor candidates with `is_global_champion=true`
      pass 4: emit per-candidate JSON to
        /data/<BLOG_ROOT>/campaign_runs/<CAMPAIGN>/benchmark/<ts>_<label>.json

    No metrics_hint fallback: receipts reflect what THIS bundle actually
    measured, not what the canonical 2026-03-14 campaign decided.
    """
    import os, json, glob
    from pathlib import Path

    by_label = {r["label"]: r for r in RUNS + STALE_RUNS}
    anchor_labels = {r["label"] for r in RUNS}
    stale_labels = {r["label"] for r in STALE_RUNS}

    blog_root = Path(f"/data/{BLOG_ROOT}")
    out_root = blog_root / "campaign_runs" / CAMPAIGN_NAME / "benchmark"
    out_root.mkdir(parents=True, exist_ok=True)

    import duckdb

    def read_metrics(run_dir: Path) -> dict:
        parquets = sorted(glob.glob(str(run_dir / "metrics/*/step_*.parquet")),
                          key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if not parquets:
            return {}
        last = parquets[-1]
        df = duckdb.sql(
            f"SELECT tag, value FROM read_parquet('{last}') "
            f"WHERE tag IN ('eval/CORE','valid/loss','train/loss','throughput/tokens_per_s_gpu')"
        ).df()
        return {row["tag"]: float(row["value"]) for _, row in df.iterrows()}

    # Pass 1: read measurements for every run we have on disk
    measured: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for run_dir in sorted(blog_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("campaign_runs",):
            continue
        label = run_dir.name
        if label not in by_label:
            print(f"[receipts] skip {label} (no spec)", flush=True)
            continue
        meta_path = run_dir / "RUN_META.json"
        if not meta_path.exists():
            print(f"[receipts] skip {label} (no RUN_META.json)", flush=True)
            continue
        metas[label] = json.loads(meta_path.read_text())
        measured[label] = read_metrics(run_dir)

    # Pass 2: apply CORE-veto rule against seed CORE (for anchor candidates).
    # Seed = "0011_seed" if present, else fall back to the first anchor that
    # measured CORE successfully.
    seed_core = None
    if "0011_seed" in measured and "eval/CORE" in measured["0011_seed"]:
        seed_core = measured["0011_seed"]["eval/CORE"]
    else:
        for lab in [r["label"] for r in RUNS]:
            if lab in measured and "eval/CORE" in measured[lab]:
                seed_core = measured[lab]["eval/CORE"]
                break
    print(f"[receipts] seed_core = {seed_core}", flush=True)

    decisions: dict[str, dict] = {}
    for label, m in measured.items():
        spec = by_label[label]
        if label in stale_labels:
            # Stale operator probes are out of the campaign by construction.
            decisions[label] = {
                "kept": False,
                "reason": spec["decision"].get("reason", "operator_probe"),
                "wave": "operator_probe",
                "is_global_champion": False,
                "in_campaign": False,
            }
            continue
        core = m.get("eval/CORE")
        core_drop_from_seed = (
            (seed_core - core) if (seed_core is not None and core is not None) else None
        )
        vetoed = (
            core_drop_from_seed is not None
            and core_drop_from_seed > CAMPAIGN_MAX_CORE_DROP
        )
        decisions[label] = {
            "kept": (not vetoed),
            "reason": ("core_drop_exceeded_threshold" if vetoed
                       else "within_kept_envelope"),
            "wave": spec["decision"].get("wave", "campaign_candidate"),
            "is_global_champion": False,  # filled in pass 3
            "in_campaign": True,
            "core_drop_from_seed": core_drop_from_seed,
            "core_drop_threshold": CAMPAIGN_MAX_CORE_DROP,
        }

    # Pass 3: champion = argmin(valid_loss) among kept anchor candidates.
    kept_anchors = [
        (lab, measured[lab].get("valid/loss"))
        for lab in anchor_labels & decisions.keys()
        if decisions[lab]["kept"] and measured[lab].get("valid/loss") is not None
    ]
    if kept_anchors:
        champion_label = min(kept_anchors, key=lambda x: x[1])[0]
        decisions[champion_label]["is_global_champion"] = True
        print(f"[receipts] champion = {champion_label} "
              f"(val={measured[champion_label]['valid/loss']:.4f})", flush=True)

    # Pass 4: emit per-candidate JSONs.
    emitted = []
    for label in sorted(decisions.keys()):
        spec = by_label[label]
        m = measured[label]
        meta = metas[label]
        receipt = {
            "campaign": CAMPAIGN_NAME,
            "stage": "benchmark",
            "candidate_id": label,
            "run_id": meta.get("run_id"),
            "overrides": spec["overrides"],
            "dtype_override": spec.get("dtype_override"),
            "decision": decisions[label],
            "metrics": {
                "final_valid_loss": m.get("valid/loss"),
                "final_train_loss": m.get("train/loss"),
                "core": m.get("eval/CORE"),
                "tokens_per_s_per_gpu": m.get("throughput/tokens_per_s_gpu"),
            },
            "budget": {"steps": CAMPAIGN_STEPS},
            "image": TRAIN_IMAGE,
        }
        suffix = int(meta.get("elapsed_s") or 0)
        out_path = out_root / f"{suffix:06d}_{label}.json"
        out_path.write_text(json.dumps(receipt, indent=2))
        emitted.append(str(out_path))
        print(f"[receipts] wrote {out_path.name} kept={decisions[label]['kept']} "
              f"champ={decisions[label]['is_global_champion']}", flush=True)

    data_vol.commit()
    return {
        "emitted": len(emitted),
        "out_root": str(out_root),
        "champion": next((lab for lab, d in decisions.items()
                          if d.get("is_global_champion")), None),
        "core_vetoed": [lab for lab, d in decisions.items()
                        if d["in_campaign"] and not d["kept"]],
    }


@app.function(
    cpu=4.0,
    memory=8 * 1024,
    volumes={"/data": data_vol},
    timeout=20 * 60,
)
def prep_eval_bundle() -> dict:
    """Download the Karpathy CORE eval bundle onto /data/eval/eval_bundle/.

    Required for CORE to auto-fire at end of training in super.toml's
    `eval_tasks = "core"` + `eval_enabled = true` path. Without this,
    training logs `[core] failed: /data/eval/eval_bundle` and emits
    nothing for `eval/core/*` tags.

    The repo's `ensure_eval_bundle()` documents the manual recipe:
    > curl -L -o /tmp/eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    > unzip /tmp/eval_bundle.zip -d /data/eval
    """
    import os, subprocess, time
    if os.path.isdir("/data/eval/eval_bundle"):
        # Already there? Check it has the jsonl tasks.
        from pathlib import Path
        jsonls = list(Path("/data/eval/eval_bundle").rglob("*.jsonl"))
        if jsonls:
            return {"already_present": True, "jsonl_count": len(jsonls)}
    os.makedirs("/data/eval", exist_ok=True)
    t0 = time.time()
    subprocess.check_call([
        "curl", "-fsSL",
        "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
        "-o", "/tmp/eval_bundle.zip",
    ])
    print(f"[prep_eval] downloaded {os.path.getsize('/tmp/eval_bundle.zip')/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)
    import zipfile
    with zipfile.ZipFile("/tmp/eval_bundle.zip", "r") as zf:
        zf.extractall("/data/eval/")
    os.unlink("/tmp/eval_bundle.zip")
    from pathlib import Path
    jsonls = list(Path("/data/eval/eval_bundle").rglob("*.jsonl"))
    data_vol.commit()
    return {
        "elapsed_s": round(time.time() - t0, 1),
        "jsonl_count": len(jsonls),
        "bundle_dir": "/data/eval/eval_bundle",
    }


@app.function(gpu="B200:8", volumes={"/data": data_vol}, timeout=30 * 60)
def smoke() -> dict:
    """64-step super.toml smoke — verify the config loads, fp8 runs,
    experiments_db lands at the override path, and metrics get emitted.
    Uses the canonical seed overrides (aux=0.0001)."""
    return run_training.local(
        label="0011_smoke",
        overrides={"aux_loss_alpha": 0.0001},
    )


@app.function(volumes={"/data": data_vol}, timeout=30 * 60)
def pack_bundle() -> dict:
    import subprocess, os
    out = "/data/bundle_0011.tar.gz"
    src = f"/data/{BLOG_ROOT}"
    if not os.path.isdir(src):
        raise RuntimeError(f"no {src} to pack")
    subprocess.check_call(["tar", "-czf", out, "-C", "/data", BLOG_ROOT])
    size_mb = os.path.getsize(out) / (1024 * 1024)
    data_vol.commit()
    return {"bundle": out, "size_mb": round(size_mb, 1)}


@app.local_entrypoint()
def main(
    action: str = "all",
    label: str = "",
    serial: bool = True,
) -> None:
    """Entrypoint.

    action:
      smoke                 - 64-step super.toml seed config to validate
      one --label X         - run a single candidate
      stale                 - run all 6 stale distractors
      all                   - run 10 canonical + 6 stale
      synthesize-receipts   - emit campaign_runs/*.json
      pack                  - tar bundle
    """
    if action == "prep-eval":
        print(prep_eval_bundle.remote()); return
    if action == "smoke":
        print(smoke.remote()); return
    if action == "one":
        assert label, "--label required"
        all_runs = RUNS + STALE_RUNS
        rn = next((r for r in all_runs if r["label"] == label), None)
        if rn is None:
            raise SystemExit(f"unknown label {label!r}; known: "
                             + ", ".join(r["label"] for r in all_runs))
        print(run_training.remote(
            rn["label"], rn["overrides"], rn.get("dtype_override")))
        return
    if action == "stale":
        calls = [run_training.spawn(r["label"], r["overrides"], r.get("dtype_override"))
                 for r in STALE_RUNS]
        for fc in calls:
            try:
                print(fc.get())
            except Exception as e:
                print(f"WARN: {e}")
        return
    if action == "synthesize-receipts":
        print(synthesize_receipts.remote()); return
    if action == "pack":
        print(pack_bundle.remote()); return
    if action == "all":
        all_runs = RUNS + STALE_RUNS
        if serial:
            for rn in all_runs:
                print(f"=== launching {rn['label']} ===")
                try:
                    print(run_training.remote(
                        rn["label"], rn["overrides"], rn.get("dtype_override")))
                except Exception as e:
                    print(f"WARN: {rn['label']}: {e}")
        else:
            calls = [(r["label"],
                      run_training.spawn(r["label"], r["overrides"], r.get("dtype_override")))
                     for r in all_runs]
            for lab, fc in calls:
                try:
                    print(fc.get())
                except Exception as e:
                    print(f"WARN: {lab}: {e}")
        print("=== synthesizing campaign receipts ===")
        print(synthesize_receipts.remote())
        print("=== packing bundle ===")
        print(pack_bundle.remote())
        return

    raise SystemExit(f"unknown action {action!r}")
