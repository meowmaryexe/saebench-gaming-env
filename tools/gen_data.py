"""Generate cases/sae_heist/ data bundle from the REAL SynthSAEBench-16k model.

Downloads decoderesearch/synth-sae-bench-16k-v1 from HuggingFace (small synthetic
model, not a language model), samples activations, picks a real SCR task with
hierarchy-aware T/S feature pairs, and saves all arrays the scoring pipeline needs.

Run with:
    .venv/bin/python tools/gen_data.py

Produces in cases/sae_heist/:
  gt_directions.npy  (num_gt_features, d_in)  — unit-norm GT directions
  synth_acts.npy     (N, d_in)                — hidden activations
  t_label.npy        (N,)  bool               — T-feature label per sample
  s_label.npy        (N,)  bool               — S-feature label per sample
  sae_W_enc.npy      (n_latents, d_model)     — random-init SAE encoder
  sae_W_dec.npy      (d_model, n_latents)     — random-init SAE decoder
  sae_meta.json

sae_lens shape convention: W_enc (d_in, d_sae), W_dec (d_sae, d_in).
Our numpy convention:       W_enc (n_latents, d_model) = TRANSPOSED.
The conversion is applied when loading into score_scr() / toy_check.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root on sys.path so vendored saebench_audit is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saebench_audit.synthetic.data_gen import D_SAE, V1_REPO, generate
from sae_lens import StandardSAE, StandardSAEConfig
from sae_lens.synthetic import SyntheticModel

DEVICE = "cpu"
N_SAMPLES = 60_000     # paper standard (NUM_SAMPLES in data_gen.py)
SEED = 42
SAMPLES_PER_CLASS_TARGET = 100    # audit-repo min_per_cell_default; error path needs <20

OUT = Path(__file__).resolve().parent.parent / "cases" / "sae_heist"
TMP_DIR = Path("/tmp/sae_heist_data_gen")

# ---------------------------------------------------------------------------
# Step 1: generate the audit-repo data bundle (scr_tasks.json, scr_labels.pt, etc.)
# ---------------------------------------------------------------------------
print(f"[gen_data] Generating audit data from {V1_REPO} (N={N_SAMPLES}, seed={SEED})")
print(f"[gen_data] Output dir: {TMP_DIR}  (cached; delete to regenerate)")
generate(V1_REPO, None, SEED, TMP_DIR, device=DEVICE, num_samples=N_SAMPLES)

# ---------------------------------------------------------------------------
# Step 2: load generated tensors
# ---------------------------------------------------------------------------
print("[gen_data] Loading generated tensors …")
hidden = torch.load(TMP_DIR / "hidden.pt")          # (N, d_in)
features = torch.load(TMP_DIR / "features.pt")      # (N, num_gt)

with open(TMP_DIR / "scr_tasks.json") as f:
    scr_tasks = json.load(f)

scr_labels = torch.load(TMP_DIR / "scr_labels.pt")  # {task_name: {"t": Tensor, "s": Tensor}}

print(f"  hidden:   {tuple(hidden.shape)}")
print(f"  features: {tuple(features.shape)}")
print(f"  scr_tasks: {len(scr_tasks)}")

# ---------------------------------------------------------------------------
# Step 3: get GT feature directions from the live model
# ---------------------------------------------------------------------------
print("[gen_data] Loading SyntheticModel for GT feature directions …")
model = SyntheticModel.from_pretrained(V1_REPO, device=DEVICE)
gt_vectors = model.feature_dict.feature_vectors.detach().cpu()   # (num_gt_features, d_in)
print(f"  gt_vectors: {tuple(gt_vectors.shape)}  norms: {gt_vectors.norm(dim=1).min():.4f}–{gt_vectors.norm(dim=1).max():.4f}")

# ---------------------------------------------------------------------------
# Step 4: pick the best SCR task (most balanced class counts)
# ---------------------------------------------------------------------------
def _six_class_min(task_name: str) -> int:
    labels = scr_labels[task_name]
    t = labels["t"].bool()
    s = labels["s"].bool()
    counts = [
        t.sum().item(),            # CLASS_T
        (~t).sum().item(),         # NEG_T
        s.sum().item(),            # CLASS_S
        (~s).sum().item(),         # NEG_S
        (t & s).sum().item(),      # CLASS_BIAS
        (~t & ~s).sum().item(),    # NEG_BIAS
    ]
    return min(counts)

best_task = max(scr_tasks, key=lambda t: _six_class_min(t["name"]))
task_name = best_task["name"]
task_labels = scr_labels[task_name]
t_label = task_labels["t"].bool().numpy()    # (N,)
s_label = task_labels["s"].bool().numpy()    # (N,)

t = t_label
s = s_label
class_counts = {
    "T=1":         int(t.sum()),
    "T=0":         int((~t).sum()),
    "S=1":         int(s.sum()),
    "S=0":         int((~s).sum()),
    "T=1 & S=1":   int((t & s).sum()),
    "T=0 & S=0":   int((~t & ~s).sum()),
}
min_count = min(class_counts.values())

print(f"[gen_data] Selected SCR task: {task_name}")
print(f"  t_feats={best_task['t_feats']}  s_feats={best_task['s_feats']}")
print(f"  6-class counts: {class_counts}")
print(f"  min class count: {min_count}")

# Hard gate: must have enough samples for meaningful probing
assert min_count >= SAMPLES_PER_CLASS_TARGET, (
    f"Insufficient samples in smallest SCR class: {min_count} < {SAMPLES_PER_CLASS_TARGET}. "
    f"Increase N_SAMPLES (currently {N_SAMPLES}) and re-run.  "
    f"(<20 causes run_scr_for_task to return error→0.0)"
)
print(f"  PASS: min_count={min_count} >= {SAMPLES_PER_CLASS_TARGET}")
effective_spc = min(1500, min_count)
print(f"  effective samples_per_class = {effective_spc}")

# ---------------------------------------------------------------------------
# Step 5: random-init StandardSAE (our "weaker" baseline, in sae_lens shapes)
# ---------------------------------------------------------------------------
d_in = int(hidden.shape[1])
cfg = StandardSAEConfig(d_in=d_in, d_sae=D_SAE, device=DEVICE, dtype="float32")
torch.manual_seed(SEED)
sae_weak = StandardSAE(cfg)

# sae_lens: W_enc (d_in, d_sae), W_dec (d_sae, d_in)
# our numpy: W_enc (n_latents, d_model) = W_dec_T, W_dec (d_model, n_latents) = W_dec_T
W_enc_sael = sae_weak.W_enc.data.numpy()   # (d_in, d_sae)
W_dec_sael = sae_weak.W_dec.data.numpy()   # (d_sae, d_in)

sae_W_enc_np = W_enc_sael.T.copy().astype(np.float32)   # (d_sae=n_latents, d_in=d_model)
sae_W_dec_np = W_dec_sael.T.copy().astype(np.float32)   # (d_in=d_model, d_sae=n_latents)

# ---------------------------------------------------------------------------
# Step 6: save to cases/sae_heist/
# ---------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)

np.save(OUT / "gt_directions.npy",  gt_vectors.numpy().astype(np.float32))
np.save(OUT / "synth_acts.npy",     hidden.numpy().astype(np.float32))
np.save(OUT / "t_label.npy",        t_label.astype(np.int8))
np.save(OUT / "s_label.npy",        s_label.astype(np.int8))
np.save(OUT / "sae_W_enc.npy",      sae_W_enc_np)
np.save(OUT / "sae_W_dec.npy",      sae_W_dec_np)

meta = {
    "d_model":            d_in,
    "n_latents":          D_SAE,
    "threshold":          0.0,
    "samples_per_class":  effective_spc,
    "scr_task":           task_name,
    "t_feats":            best_task["t_feats"],
    "s_feats":            best_task["s_feats"],
    "t_op":               best_task["t_op"],
    "s_op":               best_task["s_op"],
    "n_samples":          N_SAMPLES,
    "seed":               SEED,
    "source":             V1_REPO,
}
with open(OUT / "sae_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------------
# Step 7: report
# ---------------------------------------------------------------------------
print("\n=== cases/sae_heist/ shapes ===")
for fname in ["gt_directions.npy", "synth_acts.npy", "t_label.npy", "s_label.npy",
              "sae_W_enc.npy", "sae_W_dec.npy"]:
    arr = np.load(OUT / fname)
    size_mb = arr.nbytes / 1e6
    print(f"  {fname:<22} {str(arr.shape):<20} dtype={arr.dtype}  {size_mb:.1f} MB")
print(f"  meta: {meta}")
