# NMoE study reconstruction — playbook

This is the operational doc for building HUD reconstruction tasks from
Noumena-Network/nmoe blog-post studies. We did this for 0006 (Super-4096
sparsity collapse). Same pattern lifts to 0008 (expert learning rate),
0011, or anything else the repo has receipts for.

Audience: future curator (probably you). Not seen by agents — keep it
outside the case bundle.

---

## Inputs you need per study

| input | where |
|---|---|
| `repro/<NNNN>.receipts.json` in the repo | drives the `runs:` list + `commands:` list per study |
| Training image | `xjdr/nmoe_train:latest` on Docker Hub. Built ~2026-03-08 from a commit ≈ a week before `970a146`. Pin this — building a fresh image from the Dockerfile chain is 30-50min and almost never worth it. |
| Modal workspace with B200:8 access | We use `hud-evals`. fp8 default on B200; bf16 also works. |
| `hf-secret` Modal secret | Anonymous HF downloads are throttled to ~30s/shard; with the secret + `HF_HUB_ENABLE_HF_TRANSFER=1` you get ~270 MiB/s. |

Public artifact data is **not** on GitHub. `blog_artifacts/` is `.gitignore`'d. You generate it yourself.

---

## Pipeline (per study)

End-to-end this takes ~3-4 hours wall and $200-300 of B200 compute for a
5–10 run fleet.

### 1. Prefetch HF shards into a Modal Volume (~70s)

`HfHubParquetSource` calls `hf_hub_download` serially on each parquet,
which is ~30s/shard even with auth. Workaround: pre-pull the shards
you'll actually need in parallel.

```python
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=64) as ex:
    list(ex.map(lambda f: hf_hub_download(repo_id=..., filename=f), files))
```

For a 10B-token train budget on `karpathy/fineweb-edu-100b-shuffle`,
prefetch the first ~210 shards (~18 GB). Each shard is ~55M tokens.

### 2. Tokenize via parallel-striped CLI invocations (~16 min)

`HfHubParquetSource.__iter__` is single-threaded by design — the main
process iterates pyarrow row-groups and feeds the worker pool. With
`--workers 24`, the pool sits at ~1.5 cores of 32 used: workers idle
waiting for the producer.

Fix: don't use `hub_parquet`. Use `--source parquet --paths <local files>`
which has CLI-level path sharding via `--num-workers / --worker-index`.
Stripe the prefetched parquets across K=8 Modal containers, each running
`nmoe.data.cli prep` on its slice. Each writes its own subdir under
`/data/speedrun/train_parts/worker_<I>/`. Then a merge step:

- Symlinks each `worker_<I>/` subtree into `/data/speedrun/train/`.
- Sums `total_tokens` and concatenates `shards` across the 8 manifests.
- Sets `source_info.source = "karpathy/fineweb-edu-100b-shuffle"`
  (ArrowSource produces `"arrow:N_files"` which fails `_manifest_ok`'s
  `startswith` check in `cli/main.py::ensure_speedrun_data`).

A pyarrow FUSE pitfall lurks here: reads from `/data/hf_cache` via the
Modal Volume FUSE mount eventually throw `OSError: [Errno 22] Invalid
argument` under concurrent pressure. **Copy each worker's stripe to
local NVMe (`/root/local_parquets_*`) before passing to the CLI.**
`shutil.copy2` follows symlinks, which writes the blob content under
the parquet's logical name.

### 3. Smoke (~5 min, ~$10)

Run the main study config with `--steps=64` on B200:8 to validate
end-to-end: NCCL init, kernel autotune, manifest readback, parquet
load, checkpoint save. Catches:

- Image / CUDA arch mismatch
- Tokenizer manifest validation failure
- Volume FUSE pyarrow blow-up
- Any `Config(**cfg_dict)` `unexpected kwarg` errors (image is older than
  some receipts — see Pitfalls below)

### 4. Fire all canonical + stale runs in parallel (~2-3 h, ~$200-260)

Modal `function.spawn(...)` for each run, then `fc.get()` to collect.
B200:8 capacity is the gate — Modal will queue past ~3-4 concurrent on
`hud-evals`. Don't worry about it.

**Do NOT use `for fc in calls: print(fc.get())` if any single run can
crash.** One CalledProcessError bubbles up, takes down the whole app,
and any container that hadn't fully launched gets killed mid-startup.
Wrap each `.get()` in a try/except, or run flaky/distractor runs as
separate `modal run --action one` invocations.

### 5. Pack + download

`tar -czf /data/bundle.tar.gz -C /data blog_artifacts`. Modal Volume
state lags briefly after the final `.commit()` from each run; if you
pack immediately after the spawn loop returns, the tar comes out 100KB
of empty file entries. Re-run `pack_bundle` 30s later, or just
`modal volume get` the individual files.

---

## Critical pitfalls (and the fixes that work)

These are the load-bearing gotchas. Skip these and you waste compute.

### `--collect_update_stats=false` is in the receipts but not in the image

`xjdr/nmoe_train:latest` predates that CLI flag (the image is
2026-03-08, our pinned receipts commit is 970a146 = 2026-03-15).

Symptom: `TypeError: Config.__init__() got an unexpected keyword
argument 'collect_update_stats'`. Fix: drop the flag from your training
commands. Behaviorally a no-op on the older image.

### `experiments.db` doesn't land at `/data/experiments.db`

Receipts say it should. AGENTS.md says it should. The
`configs/speedrun/small_moe_super.toml` file overrides it:

```toml
experiments_db = "/tmp/experiments_super.db"
```

The authors do this deliberately so concurrent pods don't fight over one
SQLite file on a shared mount. Your `run_training` function needs to
copy from `/tmp/experiments_super.db` (container-local) into the bundle
dir before the container exits — not `/data/experiments.db`.

### `--lr_dense=6e-4` crashes in AdamW init

The image's CLI parser stores CLI overrides as raw strings. `--steps=2048`
gets coerced to int because Config has `steps: int`, but `--lr_dense=6e-4`
stays the string `"6e-4"` and AdamW does `if lr <= 0` → comparison fails:

> `TypeError: '<=' not supported between instances of 'float' and 'str'`

This affects any `float`-typed config field. Workaround: only override
**integer or string** fields via CLI (`--steps`, `--n_routed_experts`,
`--dtype`, etc). For float fields, set in a TOML override or skip that
run idea.

### `--collect_update_stats` is gone, so is the "clean stack" toggle

The receipts notes mention "output probes, MoE grad-health scans, and
update-stat collection disabled in-pod" — those are the toggles that
distinguish "old contaminated baseline" from the "corrected r2_clean"
rerun. The image doesn't expose them.

To still tell the stale-run story, stand in with a config delta the
image **does** support — we use `--dtype=fp8` for older baselines vs
`--dtype=bf16` for canonical. An agent reading `config_json` from
`experiments.db` sees the dtype mismatch and treats fp8 runs as
pre-correction.

### HfHubParquetSource is single-threaded; ArrowSource shards by paths

For tokenizing, use `--source parquet` not `--source hub_parquet`. Only
the former plumbs `--num-workers` / `--worker-index` through to file
sharding. See pipeline step 2.

### Modal Volume FUSE + pyarrow = errno 22

After ~100 sequential parquet reads from a FUSE-mounted Volume, pyarrow
throws `OSError: [Errno 22] Invalid argument` on `pread`. Copy to local
NVMe first. `df -h` confirms ~500GB ephemeral disk per Modal container,
so you have plenty of room.

### `shutil.copytree` from FUSE is glacial

Single-threaded reads from FUSE ~25 MB/s. For 18 GB of cache that's 8-10
minutes. Use `ThreadPoolExecutor(max_workers=16)` parallel copy → ~30-60s.

### Worker manifest merge needs canonical `source_info.source`

`ensure_speedrun_data` validates a manifest via
`_manifest_ok(dir, min_tokens=...)`:

```python
src = str(obj.get("source_info", {}).get("source", ""))
if not src.startswith(hf_dataset):   # "karpathy/fineweb-edu-100b-shuffle"
    return False
```

ArrowSource sets `source_info.source = "arrow:N_files"`. That fails the
prefix check and training refuses to start. Your merge step must
explicitly set `source_info.source` to the canonical HF repo id.

---

## Spoiler stripping for the case bundle

For study NNNN, the agent must NOT see:

| path glob | why |
|---|---|
| `content/NNNN-*.md` | the writeup |
| `content/000{N-1..N-2}-*.md`, `content/000{N+1..}-*.md` | adjacent posts almost always synopsize the study |
| `repro/NNNN.receipts.json` | the answer key in JSON form |
| `repro/falsification_ledger/NNNN.md` | the falsified-vs-supported claim table |
| `static/figures/<study-keyword>*.svg` | derived plots tell the collapse story visually |
| `scripts/repro/build_post_receipts.py` | contains `_build_NNNN()` with the run-id list, summary text, and notes |
| `papers/*-foundations.pdf` etc. | for any companion theory paper (e.g., 0007's Atlas Foundations) |

For 0006 we also stripped `content/0002-make-it-measurable.md` and
`repro/0002.receipts.json` because both directly named "Super-4096
collapse" as the comparison case. Always grep the surviving content/*
and repro/*.json for the study slug + study keywords before shipping.

---

## Per-study notes

### 0008 — expert learning rate

Receipts: `repro/0008.receipts.json`. Anchor finding (from grep):
"lr_expert = lr_dense is the best tested multiplier in the 200-step
window; 0.5x is not better, and larger multipliers worsen loss or
collapse routing earlier."

Anchor runs likely vary `lr_expert` relative to `lr_dense`. **Watch
out**: this study touches the exact CLI-string-as-float bug above. You
will need to set `lr_expert` via TOML override per-run, not CLI.

Generate a per-run TOML in `run_training`:

```python
import toml, tempfile
base = toml.load("configs/speedrun/small_moe_super.toml")
base["lr_expert"] = 0.0027   # whatever the variant
with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
    toml.dump(base, f)
    cfg_path = f.name
# pass cfg_path as the first positional to nmoe.train, not the canonical
```

Or accept that the image cannot run multipliers other than `1.0x` and
skip studies that rely on tuning floats. Probably the former.

### 0011 — check `repro/0011.receipts.json` for the runs list

You'll need to look up what 0011 actually studied. Suspect it
references metrics not in step_*.parquet (e.g., per-token diagnostics),
in which case go back to the playbook in the 0007 conversation thread:
either accept the existing parquet surface or patch nmoe.metrics to add
the missing tags.

### Cost ballpark per study

| component | wall | cost |
|---|---|---|
| prefetch | 70s | $1 |
| tokenize | 16 min | $5 |
| smoke | 5 min | $10 |
| 4 canonical runs (parallel) | 1.5 h | $80 |
| 5-6 stale distractors (parallel + queued) | 1-1.5 h | $80 |
| repack + download | 5 min | $0 |
| **total** | **~3-4 h** | **~$180-220** |

---

## What lives where

```
environments/ml-triage-tasks/
  Dockerfile.hud         # agent env: now has pyarrow + duckdb + sqlite3
  env.py                 # diagnose_research_study scenario (0-N axis judge + caps + bonus)
  cases/
    nmoe_0006_study/
      repo/              # nmoe @ 970a146 with 0006-spoiler files stripped
      blog_artifacts/    # the 10 run outputs (4 canonical + 6 stale)
  tasks/
    NMOE_STUDY_PLAYBOOK.md   # this file
    nmoe_0006_study/
      task.py            # prompt + rubric + caps + weights
      modal_train.py     # the pipeline; copy + adapt for 0008/0011
```

`modal_train.py` is the load-bearing piece. For 0008/0011: copy it,
swap `RUNS` and `STALE_RUNS` for that study's runs, copy + adapt the
`task.py` rubric. Everything else (prefetch, tokenize, merge, pack)
should work unchanged.
