"""Train a real SAE on SynthSAEBench-16k and score it vs the oracle.

Uses sae_lens.synthetic.training.train_toy_sae — the REAL training API, no hand-rolling.

ACTUAL RESULT (500K samples, lr=3e-4, l1=4e-4):
  GT-MCC trajectory: random(0.1479) → 5-step(0.1481) → 50-step(0.1503) → 500K(0.1488)
  Early apparent gains were noise; the trained SAE did NOT learn GT-aligned features.
  GT-MCC ~0.15 ≈ random baseline. MSE fell (202→9.5) but features aren't GT-aligned.
  STATUS: superseded. The experiment uses decoderesearch/synth-sae-bench-16k-v1-saes
  (GT-MCC=0.676) as the starting SAE instead (see tools/search_loop.py).

Why not decoderesearch/sae-snapshot-panels:
  Those SAEs target Gemma-2-2b (d_in=2560, real LLM).  Our gt_directions.npy is (16384,768)
  from synth-sae-bench-16k-v1.  Shape mismatch → GT-MCC comparison is invalid.

Run with: .venv/bin/python tools/train_sae.py  (takes ~10–20 min CPU)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from saebench_audit.synthetic.eval_scr import run_scr_for_task
from saebench_audit.synthetic.data_gen import V1_REPO
from sae_lens import StandardSAE, StandardSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic import SyntheticModel
from sae_lens.synthetic.training import train_toy_sae
from scoring import score_gt_mcc

CASE   = REPO / "cases" / "sae_heist"
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
print("[train_sae] Loading case bundle ...")
gt_arr   = np.load(CASE / "gt_directions.npy")    # (16384, 768)
meta     = json.load(open(CASE / "sae_meta.json"))
d_model  = int(meta["d_model"])      # 768
n_latents = int(meta["n_latents"])   # 4096
spc       = int(meta.get("samples_per_class", 422))

acts_np  = np.load(CASE / "synth_acts.npy")       # (60K, 768)
acts     = torch.tensor(acts_np, dtype=torch.float32)
t_label  = torch.tensor(np.load(CASE / "t_label.npy").astype(np.int64))
s_label  = torch.tensor(np.load(CASE / "s_label.npy").astype(np.int64))

print("[train_sae] Loading SyntheticModel ...")
model = SyntheticModel.from_pretrained(V1_REPO, device=DEVICE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_training_sae(seed: int = 42) -> StandardTrainingSAE:
    torch.manual_seed(seed)
    cfg = StandardTrainingSAEConfig(
        d_in=d_model,
        d_sae=n_latents,
        dtype="float32",
        device=DEVICE,
        l1_coefficient=4e-4,
    )
    return StandardTrainingSAE(cfg)


def _extract_inference_sae(sae_tr: StandardTrainingSAE) -> StandardSAE:
    """Copy trained weights into a StandardSAE (inference-only, works with run_scr_for_task)."""
    cfg_inf = StandardSAEConfig(d_in=d_model, d_sae=n_latents, device=DEVICE, dtype="float32")
    sae_inf = StandardSAE(cfg_inf)
    with torch.no_grad():
        sae_inf.W_enc.data.copy_(sae_tr.W_enc.data)
        sae_inf.W_dec.data.copy_(sae_tr.W_dec.data)
        sae_inf.b_enc.data.copy_(sae_tr.b_enc.data)
        sae_inf.b_dec.data.copy_(sae_tr.b_dec.data)
    return sae_inf


def _score_sae(sae_inf: StandardSAE, label: str) -> tuple[float, dict]:
    # Shape assertion before every GT-MCC call — the transpose is the likeliest silent bug
    assert sae_inf.W_dec.data.shape == (n_latents, d_model), (
        f"{label} W_dec shape {sae_inf.W_dec.data.shape} != ({n_latents}, {d_model}); "
        "sae_lens convention is (d_sae, d_in) = (n_latents, d_model)"
    )
    W_dec_np = sae_inf.W_dec.data.T.numpy()   # (d_in, d_sae) = (768, 4096) — our convention
    assert W_dec_np.shape == (d_model, n_latents), \
        f"After .T: {W_dec_np.shape} != ({d_model}, {n_latents})"

    gt_mcc = score_gt_mcc(W_dec_np, gt_arr)

    result = run_scr_for_task(
        sae_inf, acts, t_label, s_label,
        n_values=[2, 5, 10, 50], device=DEVICE,
        samples_per_class=spc,
    )
    if "error" in result:
        return gt_mcc, {"error": result["error"]}
    return gt_mcc, result["metrics"]


def _best_scr(metrics: dict) -> float:
    vals = [v for k, v in metrics.items() if k.startswith("scr_avg_")]
    return float(max(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Self-test 1: 5-step dry run (training_samples = 5 × 1024 = 5120)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Self-test 1: 5-step dry run (5120 samples)")
print("PREDICTION: GT-MCC must move from random (≈0.01); training broken if it doesn't")
print("=" * 60)

sae_tr_5 = _make_training_sae()
sae_inf_init = _extract_inference_sae(sae_tr_5)
gt_mcc_init = score_gt_mcc(sae_inf_init.W_dec.data.T.numpy(), gt_arr)
print(f"  GT-MCC at step 0 (random init): {gt_mcc_init:.4f}")

t0 = time.time()
train_toy_sae(sae_tr_5, model.feature_dict, model.activation_generator,
              training_samples=5 * 1024, batch_size=1024, lr=3e-4, device=DEVICE)
print(f"  5-step run: {time.time()-t0:.1f}s")

sae_inf_5 = _extract_inference_sae(sae_tr_5)
gt_mcc_5 = score_gt_mcc(sae_inf_5.W_dec.data.T.numpy(), gt_arr)
print(f"  GT-MCC after 5 steps: {gt_mcc_5:.4f}")

if abs(gt_mcc_5 - gt_mcc_init) < 1e-5:
    print("FAIL: GT-MCC did not change after 5 steps — training may be broken")
    sys.exit(1)
else:
    print(f"  PASS: GT-MCC moved by {gt_mcc_5 - gt_mcc_init:+.4f}")

# ---------------------------------------------------------------------------
# Self-test 2: 50-step run (51200 samples), must be detectably higher than 5-step
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Self-test 2: 50-step run (51200 samples)")
print("PREDICTION: GT-MCC must be higher than after 5 steps")
print("=" * 60)

sae_tr_50 = _make_training_sae()
t0 = time.time()
train_toy_sae(sae_tr_50, model.feature_dict, model.activation_generator,
              training_samples=50 * 1024, batch_size=1024, lr=3e-4, device=DEVICE)
print(f"  50-step run: {time.time()-t0:.1f}s")

sae_inf_50 = _extract_inference_sae(sae_tr_50)
gt_mcc_50 = score_gt_mcc(sae_inf_50.W_dec.data.T.numpy(), gt_arr)
print(f"  GT-MCC after 50 steps: {gt_mcc_50:.4f}")

if gt_mcc_50 <= gt_mcc_5:
    print(f"FAIL: GT-MCC after 50 steps ({gt_mcc_50:.4f}) not higher than after 5 ({gt_mcc_5:.4f})")
    sys.exit(1)
else:
    print(f"  PASS: GT-MCC 5→50 steps: {gt_mcc_5:.4f} → {gt_mcc_50:.4f} (+{gt_mcc_50-gt_mcc_5:.4f})")

# ---------------------------------------------------------------------------
# Full training run — time-boxed ~20 min CPU
# TRAINING_SAMPLES chosen to reach GT-MCC ~0.3–0.6 without full convergence
# ---------------------------------------------------------------------------
TRAINING_SAMPLES = 500_000   # 488 steps at batch=1024; ~10-20 min CPU

print()
print("=" * 60)
print(f"Full training run: {TRAINING_SAMPLES:,} samples ({TRAINING_SAMPLES//1024} steps)")
print("PREDICTION (before run): GT-MCC ~0.3–0.6 (partial convergence)")
print("SCR: unknown — this is what we want to measure")
print("Paper predicts: trained SCR ≥ oracle SCR (0.5990) while GT-MCC ≪ oracle (1.0)")
print("=" * 60)

sae_tr = _make_training_sae()
t0 = time.time()
train_toy_sae(sae_tr, model.feature_dict, model.activation_generator,
              training_samples=TRAINING_SAMPLES, batch_size=1024, lr=3e-4, device=DEVICE)
elapsed = time.time() - t0
print(f"  Full run: {elapsed:.1f}s ({elapsed/60:.1f} min)")

sae_trained = _extract_inference_sae(sae_tr)

print()
print("Scoring trained SAE ...")
t0 = time.time()
trained_gt_mcc, trained_metrics = _score_sae(sae_trained, "trained SAE")
print(f"  Scoring: {time.time()-t0:.1f}s")

# Full scr_avg sweep (not cherry-picked)
scr_avgs = {k: v for k, v in trained_metrics.items() if k.startswith("scr_avg_")}
trained_scr = _best_scr(trained_metrics)

# Sanity gate
assert not (trained_scr < -0.1 or trained_scr > 1.1 or trained_scr != trained_scr), \
    f"DEGENERATE trained SCR={trained_scr:.4f} — check probe pipeline"

# Oracle comparison (from committed run)
ORACLE_GT_MCC = 1.0000
ORACLE_SCR    = 0.5990

print()
print("=" * 60)
print("RESULTS — full 2×2 table")
print("=" * 60)
print(f"{'SAE':<20} {'GT-MCC':>8} {'SCR (max avg)':>14}")
print("-" * 45)
print(f"{'oracle':<20} {ORACLE_GT_MCC:>8.4f} {ORACLE_SCR:>14.4f}")
print(f"{'trained':<20} {trained_gt_mcc:>8.4f} {trained_scr:>14.4f}")
print()
print("Full scr_avg sweep for trained SAE (all n_values, not cherry-picked):")
for k in sorted(scr_avgs):
    print(f"  {k} = {scr_avgs[k]:.4f}")
print()
if trained_scr >= ORACLE_SCR and trained_gt_mcc < 0.95:
    print("THESIS FIRES: trained SCR >= oracle SCR while trained GT-MCC << oracle GT-MCC")
    print("An SAE with lower feature quality achieves equal or higher benchmark score.")
elif trained_gt_mcc < 0.15:
    print("NOTE: Trained SAE is still near random (~0.15 GT-MCC threshold).")
    print("This may mean training was too short. Report as limitation.")
    print("The thesis still needs a polysemantic SAE (GT-MCC 0.3-0.6) to fire.")
else:
    print("Thesis did not fire with these numbers.")
    print("Possible reasons: (a) training too short, (b) this architecture/task combo")
    print("doesn't exhibit the polysemantic-inflation effect, (c) SCR metric variant differs.")
    print("Report these numbers honestly — do not adjust n_values to force it.")

# ---------------------------------------------------------------------------
# Save trained weights + results
# ---------------------------------------------------------------------------
W_dec_our  = sae_trained.W_dec.data.T.numpy()    # (768, 4096) our convention
W_enc_our  = sae_trained.W_enc.data.T.numpy()    # (4096, 768) our convention
np.save(CASE / "trained_W_enc.npy", W_enc_our.astype(np.float32))
np.save(CASE / "trained_W_dec.npy", W_dec_our.astype(np.float32))

results = {
    "trained_gt_mcc": trained_gt_mcc,
    "trained_scr":    trained_scr,
    "trained_scr_all_n": {k: float(v) for k, v in scr_avgs.items()},
    "oracle_gt_mcc":  ORACLE_GT_MCC,
    "oracle_scr":     ORACLE_SCR,
    "training_samples": TRAINING_SAMPLES,
    "batch_size": 1024,
    "lr": 3e-4,
    "l1_coefficient": 4e-4,
    "d_model": d_model,
    "n_latents": n_latents,
}
with open(CASE / "trained_meta.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved trained_W_enc.npy, trained_W_dec.npy, trained_meta.json to {CASE}")
