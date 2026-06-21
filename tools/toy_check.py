"""Real sanity check: prove oracle_SCR < weaker_SCR using the REAL pipeline.

Uses:
- PerfectSAE oracle  (decoder = first n_latents GT directions; encoder = lookup table)
- Random-init StandardSAE as the "weaker" baseline
- REAL sae_bench probe pipeline via saebench_audit.synthetic.eval_scr.run_scr_for_task
- REAL eval_sae_on_synthetic_data to VALIDATE the scipy GT-MCC implementation

Expected results per Chanin et al.:
  oracle GT-MCC  ≈ 1.0  (perfect feature recovery)
  oracle SCR     < weaker SCR  (benchmark under-rates the perfect SAE)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from saebench_audit.diagnostic.perfect_oracle import PerfectSAE, perfect_oracle
from saebench_audit.synthetic.data_gen import D_SAE, V1_REPO
from saebench_audit.synthetic.eval_scr import run_scr_for_task
from scoring import score_gt_mcc
from sae_lens import StandardSAE, StandardSAEConfig
from sae_lens.synthetic import SyntheticModel
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data

CASE   = REPO / "cases" / "sae_heist"
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Load case bundle
# ---------------------------------------------------------------------------
print("[toy_check] Loading case bundle ...")
gt_arr   = np.load(CASE / "gt_directions.npy")   # (16384, 768)  unit-norm
W_enc_np = np.load(CASE / "sae_W_enc.npy")        # (4096, 768)  our convention
W_dec_np = np.load(CASE / "sae_W_dec.npy")        # (768, 4096)  our convention
t_arr    = np.load(CASE / "t_label.npy").astype(bool)
s_arr    = np.load(CASE / "s_label.npy").astype(bool)
meta     = json.load(open(CASE / "sae_meta.json"))

d_model   = meta["d_model"]    # 768
n_latents = meta["n_latents"]  # 4096
spc       = int(meta.get("samples_per_class", 422))
N_VALUES  = [2, 5, 10, 50]

t_label   = torch.tensor(t_arr, dtype=torch.long)
s_label   = torch.tensor(s_arr, dtype=torch.long)

# synth_acts.npy was saved as float32 from the exact same hidden.pt tensor used
# to build the oracle's lookup table (byte-identical float32 values).
acts_np    = np.load(CASE / "synth_acts.npy")                           # (60K, 768) float32
acts_torch = torch.tensor(acts_np, dtype=torch.float32)                 # (60K, 768)

# ---------------------------------------------------------------------------
# Build oracle
# ---------------------------------------------------------------------------
print("[toy_check] Loading SyntheticModel and building PerfectSAE oracle ...")
model    = SyntheticModel.from_pretrained(V1_REPO, device=DEVICE)

# features.pt from gen_data.py output; contains continuous GT feature activations
TMP_DATA = Path("/tmp/sae_heist_data_gen")
features_t = torch.load(TMP_DATA / "features.pt")                       # (60K, 16384)

# oracle_pool_hidden MUST be the tensor we also pass to run_scr_for_task.
# PerfectSAE's encode() resolves inputs via a lookup table keyed on the exact
# float32 values of oracle_pool_hidden. Passing a different tensor (even a copy
# of a different dataset) would cause KeyError in _lookup.
oracle_pool_hidden = acts_torch   # (60K, 768) — same object as scr_hidden below

oracle = perfect_oracle(
    model, oracle_pool_hidden, features_t,
    d_sae=n_latents, device=DEVICE,
)

# Hard assertion: scr_hidden MUST be the identical Python object as oracle_pool_hidden
# so that future refactors cannot accidentally break the lookup-table contract.
scr_hidden = oracle_pool_hidden
assert scr_hidden is oracle_pool_hidden, (
    "scr_hidden must be the exact same tensor object as oracle_pool_hidden. "
    "PerfectSAE._lookup() uses float32 byte patterns from oracle_pool_hidden as keys; "
    "passing a different tensor (even a copy) will cause KeyError. "
    "If you need to slice/filter, do it AFTER this assertion and pass the same rows."
)

# ---------------------------------------------------------------------------
# Build weaker StandardSAE from saved weights
# ---------------------------------------------------------------------------
cfg_weak = StandardSAEConfig(d_in=d_model, d_sae=n_latents, device=DEVICE, dtype="float32")
sae_weak = StandardSAE(cfg_weak)
with torch.no_grad():
    # our numpy: W_enc (n_latents, d_model) = (d_sae, d_in)
    # sae_lens:  W_enc (d_in, d_sae)  → need .T
    sae_weak.W_enc.data = torch.tensor(W_enc_np.T.copy(), dtype=torch.float32)
    sae_weak.W_dec.data = torch.tensor(W_dec_np.T.copy(), dtype=torch.float32)
    sae_weak.b_enc.data = torch.zeros(n_latents)
    sae_weak.b_dec.data = torch.zeros(d_model)

# ---------------------------------------------------------------------------
# GT-MCC — scipy implementation
# ---------------------------------------------------------------------------
# oracle W_dec in sae_lens = (d_sae, d_in); score_gt_mcc expects (d_model, n_latents) = (d_in, d_sae)
oracle_W_dec_np = oracle.W_dec.data.T.numpy()   # (768, 4096) — our numpy convention
oracle_gt_mcc_scipy = score_gt_mcc(oracle_W_dec_np, gt_arr)
weak_gt_mcc_scipy   = score_gt_mcc(W_dec_np, gt_arr)

# ---------------------------------------------------------------------------
# VALIDATION (condition 1): compare scipy GT-MCC against eval_sae_on_synthetic_data
# ---------------------------------------------------------------------------
# Oracle: eval_sae_on_synthetic_data would fail (fresh activations not in lookup table).
#         Analytic truth: oracle W_dec = first n_latents GT directions → MCC = 1.0 exactly.
#         We assert the scipy implementation gives ≥ 0.999 (floating-point rounding).
print("[toy_check] Validating oracle GT-MCC (analytic truth = 1.0) ...")
assert oracle_gt_mcc_scipy > 0.999, (
    f"scipy oracle GT-MCC = {oracle_gt_mcc_scipy:.4f}, expected ≥ 0.999. "
    "If W_dec = first n_latents GT directions, Hungarian matching must yield 1.0."
)
print(f"  oracle scipy GT-MCC = {oracle_gt_mcc_scipy:.4f}  ✓ (analytic truth = 1.0)")

# Weak SAE: run real eval_sae_on_synthetic_data and compare
print("[toy_check] Validating weak SAE GT-MCC via eval_sae_on_synthetic_data ...")
VALIDATION_N = 5000
weak_eval_result = eval_sae_on_synthetic_data(
    sae_weak,
    model.feature_dict,
    model.activation_generator,
    num_samples=VALIDATION_N,
    batch_size=1000,
)
weak_gt_mcc_real = weak_eval_result.mcc
diff = abs(weak_gt_mcc_scipy - weak_gt_mcc_real)
print(f"  weak scipy={weak_gt_mcc_scipy:.4f}  real={weak_gt_mcc_real:.4f}  |diff|={diff:.4f}")

VALIDATION_TOL = 0.05   # 5% tolerance — different sample sets; both are noisy estimates
if diff > VALIDATION_TOL:
    print(f"WARNING: scipy and eval_sae_on_synthetic_data diverge by {diff:.4f} > {VALIDATION_TOL}")
    print("Using eval_sae_on_synthetic_data value for reporting (real implementation).")
    weak_gt_mcc = weak_gt_mcc_real
else:
    print(f"VALIDATED: |scipy − real| = {diff:.4f} ≤ {VALIDATION_TOL} — scipy implementation is correct")
    weak_gt_mcc = weak_gt_mcc_scipy

# Use validated implementations
oracle_gt_mcc = oracle_gt_mcc_scipy   # analytic truth; scipy confirmed correct

# ---------------------------------------------------------------------------
# SCR — REAL sae_bench probe pipeline
# ---------------------------------------------------------------------------
print(f"\n[toy_check] Scoring oracle SCR  (samples_per_class={spc}) ...")
oracle_scr_result = run_scr_for_task(
    oracle,
    scr_hidden,      # ← same object as oracle_pool_hidden (assertion above)
    t_label,
    s_label,
    n_values=N_VALUES,
    device=DEVICE,
    samples_per_class=spc,
)

print(f"[toy_check] Scoring weaker SAE SCR ...")
weak_scr_result = run_scr_for_task(
    sae_weak,
    acts_torch,
    t_label,
    s_label,
    n_values=N_VALUES,
    device=DEVICE,
    samples_per_class=spc,
)


def _best_scr_avg(result: dict) -> float:
    if "error" in result:
        return float("nan")
    vals = [v for k, v in result["metrics"].items() if k.startswith("scr_avg_")]
    return float(max(vals)) if vals else float("nan")


oracle_scr = _best_scr_avg(oracle_scr_result)
weak_scr   = _best_scr_avg(weak_scr_result)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print()
print("=" * 50)
print("RESULTS (REAL SynthSAEBench-16k pipeline)")
print("=" * 50)
print(f"oracle GT-MCC : {oracle_gt_mcc:.4f}  (assert > 0.95)")
print(f"oracle SCR    : {oracle_scr:.4f}")
print(f"weaker GT-MCC : {weak_gt_mcc:.4f}")
print(f"weaker SCR    : {weak_scr:.4f}")
print()

# ---------------------------------------------------------------------------
# Sanity gates
# ---------------------------------------------------------------------------
assert oracle_gt_mcc > 0.95, f"FAIL oracle GT-MCC={oracle_gt_mcc:.4f} should be > 0.95"
print("PASS: oracle GT-MCC > 0.95")


def _check_sane_scr(name: str, val: float) -> None:
    if val != val:  # NaN
        raise AssertionError(f"DEGENERATE {name} SCR = NaN (run_scr_for_task returned error)")
    if not (-0.1 <= val <= 1.1):
        raise AssertionError(
            f"DEGENERATE {name} SCR = {val:.4f}  expected [-0.1, 1.1].\n"
            "This is a RED FLAG — do NOT claim success. Check the probe pipeline."
        )
    print(f"PASS: {name} SCR = {val:.4f} in sane range [-0.1, 1.1]")


_check_sane_scr("oracle", oracle_scr)
_check_sane_scr("weaker", weak_scr)

print()
if oracle_scr < weak_scr:
    print("THESIS CONFIRMED: oracle_SCR < weaker_SCR")
    print("SCR under-rates the perfect SAE — the benchmark is gameable.")
else:
    gap = oracle_scr - weak_scr
    print(f"UNEXPECTED: oracle_SCR >= weaker_SCR   gap = {gap:.4f}")
    print("Report this finding honestly. Do not massage the data.")
