"""Modal launcher: generate the 0008 expert-learning-rate blog_artifacts.

Study under reconstruction: "Do MoE Experts Need Different Learning Rates?"
— post 0008 in the Noumena-Network/nmoe series. The study kills Moonlet's
historical `lr_expert = 15 × lr_dense` rule for the bf16/AdamW lane.

Key differences from `nmoe_0006_study/modal_train.py`:
  * Config         configs/moonlet.toml       (Moonlet 7B2A, single-node)
  * Tokenizer      o200k_harmony              (NOT gpt2; vocab=201088, eos=199999)
  * Dataset        HuggingFaceFW/fineweb-edu  (NOT karpathy/fineweb-edu-100b-shuffle)
  * Data path      /data/fineweb_edu          (NOT /data/speedrun/*)
  * Token budget   ~120M total                (NOT 10B — runs are 200 steps × bs=8 × seq=4096)
  * Fleet          24 runs (18 anchor + 6 stale)
  * Wall budget    ~30-45 min fully parallel, ~$110-130

Same image as 0006 (`xjdr/nmoe_train:latest`). Same Modal Volume
(`nmoe-0006-data`) — Moonlet data lives at `/data/fineweb_edu` which
doesn't collide with `/data/speedrun/{train,val}`.

Usage:
  modal run modal_train.py --action prefetch       # ~30-60s, pull HF parquets
  modal run modal_train.py --action tokenize       # ~5-10 min, write /data/fineweb_edu/manifest.json
  modal run modal_train.py --action smoke          # ~3 min on B200:8, 16-step Moonlet bf16 m=1 sanity
  modal run modal_train.py --action smoke-nvfp4    # ~3 min on B200:8, 16-step Moonlet nvfp4 m=1 — verifies FP4 kernels work end-to-end
  modal run modal_train.py --action all --no-serial   # fire all 24 in parallel
  modal run modal_train.py --action pack           # tar /data/blog_artifacts_0008 -> /data/bundle_0008.tar.gz
  modal volume get nmoe-0006-data /bundle_0008.tar.gz ../../cases/nmoe_0008_study/bundle.tar.gz
"""
from __future__ import annotations

import modal

APP_NAME = "nmoe-0008-training"
TRAIN_IMAGE = "xjdr/nmoe_train:latest"
VOLUME_NAME = "nmoe-0006-data"   # reuse — different namespace under /data
NMOE_COMMIT = "970a146433f9c649d09ddab36f675974f53dd905"

# Moonlet's tokenizer + dataset (per configs/moonlet.toml header comment)
HF_DATASET = "HuggingFaceFW/fineweb-edu"
TOKENIZER = "o200k_harmony"
VOCAB_SIZE = 201088
EOS_TOKEN_ID = 199999

# Token budget: Moonlet runs are bs=8, seq=4096, max 200 steps = 6.55M tokens/run.
# 24-run fleet ≈ 130M tokens. Tokenize 500M for comfortable margin.
TRAIN_TOKEN_BUDGET = 500_000_000

# Where Moonlet expects its tokenized data on the Volume
DATA_DIR_VOL = "/data/fineweb_edu"

# HF cache for prefetched parquets (FUSE; we mirror to local NVMe for tokenize)
CACHE_REPO_DIR = "hub/datasets--HuggingFaceFW--fineweb-edu"

image = (
    modal.Image.from_registry(TRAIN_IMAGE, add_python="3.12")
    .run_commands(
        "/root/.local/bin/uv pip install --python /workspace/nmoe/.venv/bin/python hf_transfer"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/hf_cache_0008",
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
# 18 anchor runs from repro/0008.receipts.json's commands block.
# All use configs/moonlet.toml; all floats are plain-decimal so CLI override
# works through the image's parser (no scientific notation, which would stay
# a string and crash AdamW per the playbook).
# ============================================================================
RUNS: list[dict] = [
    # --- bf16 main sweep: 2 seeds × {0.5, 1, 2, 4, 15}x ---
    {"label": "0008_bf16_m0p5_s42",  "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.0015","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m1_s42",    "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m2_s42",    "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.006","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m4_s42",    "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.012","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m15_s42",   "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m0p5_s43",  "args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.0015","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m1_s43",    "args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m2_s43",    "args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.006","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m4_s43",    "args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.012","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_m15_s43",   "args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
    # --- beta2_expert=0.95 controls (m=1, both seeds) — the b95 ablation
    #     that flipped across seeds and was NOT promoted to a rule ---
    {"label": "0008_bf16_m1_b95_s42","args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.95","--log_every=10"]},
    {"label": "0008_bf16_m1_b95_s43","args": ["--dtype=bf16","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.95","--log_every=10"]},
    # --- bf16 updateproof: m=1 + m=15, seed 42 (separate study root in receipts) ---
    {"label": "0008_bf16_updateproof_m1_s42",  "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_bf16_updateproof_m15_s42", "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
    # --- nvfp4 updateproof: m=1 + m=15 (diagnostic, single-seed) ---
    {"label": "0008_nvfp4_updateproof_m1_s42", "args": ["--dtype=nvfp4","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_nvfp4_updateproof_m15_s42","args": ["--dtype=nvfp4","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
    # --- nvfp4 grad-health canary: 50 steps ---
    {"label": "0008_nvfp4_gradhealth_m1_s42",  "args": ["--dtype=nvfp4","--steps=50","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    {"label": "0008_nvfp4_gradhealth_m15_s42", "args": ["--dtype=nvfp4","--steps=50","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
]

# ============================================================================
# 6 stale distractors — abandoned earlier exploration that the researcher
# left in the folder. Force scope-selection to do real work.
# ============================================================================
STALE_RUNS: list[dict] = [
    # Pre-correction fp8 attempt at the bad multiplier — abandoned because
    # the fp8 optimizer-update noise was different. Agent sees dtype=fp8
    # in config_json and excludes.
    {"label": "0008_bf16_m15_s44_predtype_switch",
     "args": ["--dtype=fp8","--steps=200","--seed=44","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--log_every=10"]},
    # Out-of-grid 8x point — explored before the team settled on the
    # canonical {0.5, 1, 2, 4, 15} log grid. Not in the receipts' sweep.
    {"label": "0008_bf16_m8_s42_intermediate",
     "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.024","--adam_beta2_expert=0.99","--log_every=10"]},
    # Longer-horizon m=1 — breaks the 200-step matched comparison contract.
    {"label": "0008_bf16_m1_s42_long_500step",
     "args": ["--dtype=bf16","--steps=500","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    # Long-warmup rescue attempt at m=15 — confounds the schedule. Abandoned.
    {"label": "0008_bf16_m15_s42_warmup512",
     "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.045","--adam_beta2_expert=0.99","--warmup_steps=512","--log_every=10"]},
    # Frozen-router probe — old curiosity, doesn't match 0008's
    # router-trains contract.
    {"label": "0008_bf16_m1_s42_routeonly",
     "args": ["--dtype=bf16","--steps=200","--seed=42","--lr_dense=0.003","--lr_router=0.0","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
    # Attempted second-seed for nvfp4 lane that the receipts did NOT
    # promote — nvfp4 stayed single-seed-diagnostic.
    {"label": "0008_nvfp4_m1_s43_solo",
     "args": ["--dtype=nvfp4","--steps=200","--seed=43","--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003","--adam_beta2_expert=0.99","--log_every=10"]},
]


def _venv_python() -> str:
    return "/workspace/nmoe/.venv/bin/python"


# ============================================================================
# Prefetch + tokenize
# ============================================================================

@app.function(
    cpu=16.0,
    memory=32 * 1024,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=2 * 3600,
)
def prefetch_shards(num_shards: int = 15) -> dict:
    """Parallel-download enough HuggingFaceFW/fineweb-edu parquet shards
    to cover ~500M tokens of Moonlet training. Fineweb-edu's HF repo
    layout is `data/CC-MAIN-*/*.parquet` (different from karpathy's
    flat `shard_NNNNN.parquet` namespace), so we list-then-pick.
    """
    import os, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = HF_DATASET
    all_parquets = sorted(
        f for f in list_repo_files(repo_id=repo, repo_type="dataset")
        if f.endswith(".parquet")
    )
    print(f"[prefetch] repo has {len(all_parquets)} parquets", flush=True)

    # Take the first N — fineweb-edu shards are largeish (~700M tokens each
    # per HF's CC-MAIN sharding); 15 is way more than the 500M-token budget
    # but cheap to overshoot.
    to_fetch = all_parquets[:num_shards]
    print(f"[prefetch] fetching {len(to_fetch)} parquets", flush=True)

    t0 = time.time()
    bytes_fetched = 0

    def _one(fname: str) -> int:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fname)
        return os.path.getsize(path)

    with ThreadPoolExecutor(max_workers=64) as ex:
        futures = {ex.submit(_one, f): f for f in to_fetch}
        for i, fut in enumerate(as_completed(futures)):
            try:
                bytes_fetched += fut.result()
            except Exception as e:
                print(f"[prefetch] ERROR on {futures[fut]}: {e}", flush=True)
                raise
            if (i + 1) % 5 == 0 or i == len(to_fetch) - 1:
                dt = time.time() - t0
                gb = bytes_fetched / (1024**3)
                print(f"[prefetch] {i+1}/{len(to_fetch)} done "
                      f"({gb:.1f} GiB in {dt:.0f}s = {gb*1024/dt:.0f} MiB/s)",
                      flush=True)

    data_vol.commit()
    return {
        "elapsed_s": round(time.time() - t0, 1),
        "files": len(to_fetch),
        "gib": round(bytes_fetched / (1024**3), 2),
    }


def _snapshot_root(hf_home: str) -> str:
    import os
    snaps_dir = os.path.join(hf_home, CACHE_REPO_DIR, "snapshots")
    ids = [d for d in os.listdir(snaps_dir) if os.path.isdir(os.path.join(snaps_dir, d))]
    if not ids:
        raise FileNotFoundError(f"no snapshot under {snaps_dir}")
    return os.path.join(snaps_dir, ids[0])


def _list_cached_parquets(hf_home: str = "/data/hf_cache_0008") -> list[str]:
    """Walk the cached HF snapshot dir recursively (fineweb-edu uses
    nested `data/CC-MAIN-*/*.parquet`, not flat shard names)."""
    import os
    root = _snapshot_root(hf_home)
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".parquet"):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


@app.function(
    cpu=32.0,
    memory=64 * 1024,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=2 * 3600,
)
def tokenize_data() -> dict:
    """Tokenize prefetched fineweb-edu parquets via the o200k_harmony
    tokenizer into /data/fineweb_edu/. Moonlet has no validation set
    (`validation_enabled = False`), so just train shards needed.

    Same lessons as 0006: mirror cached files to local NVMe first
    (FUSE+pyarrow errno 22), use --source parquet so the CLI's
    _shard_paths can stripe across workers if we ever want to scale.
    Single-container path is fine here because the budget is tiny.
    """
    import os, subprocess, time, shutil, json
    from concurrent.futures import ThreadPoolExecutor

    os.chdir("/workspace/nmoe")
    t0 = time.time()

    train_dir = DATA_DIR_VOL
    if os.path.isdir(train_dir) and not os.path.exists(os.path.join(train_dir, "manifest.json")):
        print(f"[tokenize] wiping incomplete {train_dir}", flush=True)
        shutil.rmtree(train_dir, ignore_errors=True)
    if os.path.exists(os.path.join(train_dir, "manifest.json")):
        return {"skipped": "manifest already present", "output": train_dir}
    os.makedirs(train_dir, exist_ok=True)

    vol_parquets = _list_cached_parquets()
    print(f"[tokenize] cached parquets: {len(vol_parquets)}", flush=True)

    # Mirror to local NVMe
    local_dir = "/root/local_fineweb"
    os.makedirs(local_dir, exist_ok=True)

    def _copy(src: str) -> str:
        dst = os.path.join(local_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        return dst
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        local_paths = list(ex.map(_copy, vol_parquets))
    print(f"[tokenize] copied {len(local_paths)} files in {time.time()-t_copy:.1f}s",
          flush=True)

    env = os.environ.copy()
    env["HF_HOME"] = "/root/.hf_cache_0008_unused"

    cmd = [
        _venv_python(), "-m", "nmoe.data.cli", "prep",
        "--source", "parquet",
        "--paths", *local_paths,
        "--output", train_dir,
        "--name", "fineweb_edu",
        "--tokenizer", TOKENIZER,
        "--vocab-size", str(VOCAB_SIZE),
        "--eos-token-id", str(EOS_TOKEN_ID),
        "--max-tokens-total", str(TRAIN_TOKEN_BUDGET),
        "--num-shards", "16",
        "--workers", "24",
        "--batch-size", "5000",
        "--parallel",
    ]
    t_tok = time.time()
    subprocess.check_call(cmd, cwd="/workspace/nmoe", env=env)
    print(f"[tokenize] done in {time.time()-t_tok:.1f}s", flush=True)

    # Patch the manifest's source_info.source so Moonlet's `_manifest_ok`
    # (if it has one) accepts the canonical HF dataset name. ArrowSource
    # produces "arrow:N_files" which would otherwise fail any prefix check.
    mpath = os.path.join(train_dir, "manifest.json")
    if os.path.exists(mpath):
        m = json.loads(open(mpath).read())
        m["source_info"] = {"source": HF_DATASET}
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
    shutil.rmtree(local_dir, ignore_errors=True)

    data_vol.commit()
    return {
        "elapsed_s": round(time.time() - t0, 1),
        "manifest": mpath,
        "total_tokens": int(m.get("total_tokens", 0)) if os.path.exists(mpath) else 0,
    }


# ============================================================================
# Training
# ============================================================================

@app.function(
    gpu="B200:8",
    volumes={"/data": data_vol},
    timeout=2 * 3600,
)
def run_training(label: str, extra_args: list[str], target_root: str = "blog_artifacts_0008") -> dict:
    """Run one Moonlet training command, then repackage outputs under
    /data/<target_root>/<label>/."""
    import os, subprocess, shutil, time, json

    os.chdir("/workspace/nmoe")

    bundle_pre = f"/data/{target_root}/{label}"
    if os.path.isdir(bundle_pre):
        print(f"[{label}] wiping prior {bundle_pre}", flush=True)
        shutil.rmtree(bundle_pre, ignore_errors=True)

    metrics_root = "/data/metrics"
    os.makedirs(metrics_root, exist_ok=True)
    before = set(os.listdir(metrics_root))

    # Force experiments_db onto container-local /tmp so 24 concurrent
    # containers don't fight over the shared FUSE /data/experiments.db
    # (moonlet.toml leaves it at the Config default of /data/).
    per_run_db = f"/tmp/experiments_0008_{label}.db"
    cmd = [
        "torchrun", "--nproc_per_node=8", "-m", "nmoe.train",
        "configs/moonlet.toml",
        f"--experiments_db={per_run_db}",
    ] + list(extra_args)
    print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.check_call(cmd, cwd="/workspace/nmoe")
    elapsed = time.time() - t0
    print(f"[{label}] training finished in {elapsed/3600:.2f} hr", flush=True)

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
    # Moonlet uses the same /tmp/experiments_*.db pattern. Probe both
    # for safety. Files in /tmp are container-local so each pod's db
    # is its own.
    for candidate in (per_run_db,
                      "/tmp/experiments_moonlet.db",
                      "/tmp/experiments_moonlet_dev.db",
                      "/tmp/experiments_super.db",
                      "/data/experiments.db"):
        if os.path.exists(candidate):
            shutil.copy(candidate, f"{bundle}/experiments.db")
            print(f"[{label}] experiments.db <- {candidate}", flush=True)
            break
    else:
        print(f"[{label}] WARN: no experiments.db on any known path", flush=True)

    with open(os.path.join(bundle, "RUN_META.json"), "w") as f:
        json.dump({
            "label": label,
            "run_id": run_id,
            "cmd": cmd,
            "elapsed_s": elapsed,
            "image": TRAIN_IMAGE,
            "agent_visible_commit": NMOE_COMMIT,
        }, f, indent=2)

    data_vol.commit()
    return {"label": label, "run_id": run_id, "elapsed_s": elapsed}


@app.function(gpu="B200:8", volumes={"/data": data_vol}, timeout=30 * 60)
def smoke(label_suffix: str = "_smoke", extra_args: tuple = ()) -> dict:
    """16-step Moonlet bf16 m=1 sanity. Override args via extra_args."""
    args = list(extra_args) or [
        "--dtype=bf16","--steps=16","--seed=42",
        "--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003",
        "--adam_beta2_expert=0.99","--log_every=4",
    ]
    return run_training.local(label=f"0008{label_suffix}", extra_args=args)


@app.function(gpu="B200:8", volumes={"/data": data_vol}, timeout=30 * 60)
def smoke_nvfp4() -> dict:
    """Verify nvfp4 kernels actually run end-to-end on B200 before we
    commit the full nvfp4 lane to compute. 16-step Moonlet at m=1."""
    args = [
        "--dtype=nvfp4","--steps=16","--seed=42",
        "--lr_dense=0.003","--lr_router=0.003","--lr_expert=0.003",
        "--adam_beta2_expert=0.99","--log_every=4",
    ]
    return run_training.local(label="0008_smoke_nvfp4", extra_args=args)


@app.function(volumes={"/data": data_vol}, timeout=30 * 60)
def pack_bundle() -> dict:
    """Tar /data/blog_artifacts_0008 into /data/bundle_0008.tar.gz."""
    import subprocess, os
    out = "/data/bundle_0008.tar.gz"
    src = "/data/blog_artifacts_0008"
    if not os.path.isdir(src):
        raise RuntimeError(f"no {src} to pack")
    subprocess.check_call(["tar", "-czf", out, "-C", "/data", "blog_artifacts_0008"])
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
      prefetch     - parallel HF prefetch of fineweb-edu parquets
      tokenize     - run prep CLI to write /data/fineweb_edu/manifest.json
      prep         - prefetch + tokenize
      smoke        - 16-step bf16 m=1 Moonlet sanity
      smoke-nvfp4  - 16-step nvfp4 m=1 Moonlet sanity (verify FP4 kernels)
      one          - run a single entry from RUNS or STALE_RUNS by --label
      stale        - run all 6 stale distractors
      all          - 18 anchors + 6 stale (24 runs)
      pack         - tar the bundle
    """
    if action == "prefetch":
        print(prefetch_shards.remote()); return
    if action == "tokenize":
        print(tokenize_data.remote()); return
    if action == "prep":
        print("=== prefetch ===")
        print(prefetch_shards.remote())
        print("=== tokenize ===")
        print(tokenize_data.remote())
        return
    if action == "smoke":
        print(smoke.remote()); return
    if action == "smoke-nvfp4":
        print(smoke_nvfp4.remote()); return
    if action == "one":
        assert label, "--label required for action=one"
        all_runs = RUNS + STALE_RUNS
        rn = next((r for r in all_runs if r["label"] == label), None)
        if rn is None:
            raise SystemExit(f"unknown label {label!r}; known: "
                             + ", ".join(r["label"] for r in all_runs))
        print(run_training.remote(rn["label"], rn["args"]))
        return
    if action == "stale":
        calls = [run_training.spawn(r["label"], r["args"]) for r in STALE_RUNS]
        # NB: do not raise on .get() — one crash should not kill the others
        for fc in calls:
            try:
                print(fc.get())
            except Exception as e:
                print(f"WARN: stale run failed: {e}")
        return
    if action == "pack":
        print(pack_bundle.remote()); return
    if action == "all":
        all_runs = RUNS + STALE_RUNS
        if serial:
            for rn in all_runs:
                print(f"=== launching {rn['label']} ===")
                try:
                    print(run_training.remote(rn["label"], rn["args"]))
                except Exception as e:
                    print(f"WARN: {rn['label']} failed: {e}")
        else:
            calls = [(r["label"], run_training.spawn(r["label"], r["args"]))
                     for r in all_runs]
            for lab, fc in calls:
                try:
                    print(fc.get())
                except Exception as e:
                    print(f"WARN: {lab} failed: {e}")
        print("=== packing bundle ===")
        print(pack_bundle.remote())
        return

    raise SystemExit(f"unknown action {action!r}")
