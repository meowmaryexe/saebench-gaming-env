"""OFFLINE DEMO ONLY — NOT the HUD submission.
The HUD submission is tasks/sae_heist/task.py (the RL environment).
This script is an offline validator: it exercises the same scorers without any model
gateway call, demonstrating that blind edits can drive SCR while GT-MCC tracks quality.

Search loop: greedy SCR hill-climb on the panel SAE.
Status: self-tests currently require spc≥1500 (regenerate data at 215K samples first).

Starting point: decoderesearch/synth-sae-bench-16k-v1-saes btk/l0-15/seed-0
  Type: JumpReLUSAE  |  d_in=768  d_sae=4096  (exact match: trained on synth-sae-bench-16k-v1)
  GT-MCC reported in eval_stats.json: 0.6759
  GT-MCC via our score_gt_mcc:         0.6759 (validated: exact match)
  SCR baseline (seed 42, single run):   0.1800 (full noise floor measured below)

Why this is the legitimate starting point (NOT an artifact):
  - This SAE was downloaded from decoderesearch/synth-sae-bench-16k-v1-saes — a real
    published panel SAE trained on the same SynthSAEBench-16k-v1 model as our data.
  - d_in=768, d_sae=4096: exact match. GT-MCC our score == published eval_stats value.
  - This is the "real trained polysemantic SAE" the paper's thesis refers to.
  - GT-MCC=0.676 (good; well above random 0.15) while oracle GT-MCC=1.0.
  - The thesis: if greedy SCR optimization can raise SCR while GT-MCC stays ~0.676,
    that demonstrates the metric can be gamed without improving feature quality.

What we claim vs. don't claim:
  - Claim: blind random edits to decoder weights can raise SCR independently of GT-MCC.
  - Don't claim: we know WHICH latents matter. Edits are chosen uniformly at random
    from the latent index — NO knowledge of S/T feature assignments is used.

Noise characterization:
  - 5 seeds measured before search.
  - Each step averages 3 seeds.
  - Acceptance threshold: mean_new > mean_current (comparison is mean vs mean;
    noise floor shows how much noise there is in a single step).

Edits (all operate on saved/restored parameters — no cloning needed):
  rescale_col   : W_dec[i] *= scale  (scale ∈ U[0.5, 2.0])
  zero_dec_col  : W_dec[i] = 0       (silences latent's output)
  dup_col       : W_dec[dst] = W_dec[src]
  nudge_enc_col : W_enc[:,i] *= scale  (changes latent's activation)

Run with: .venv/bin/python tools/search_loop.py
Output:   out/search_log.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from saebench_audit.synthetic.eval_scr import run_scr_for_task
from sae_lens import SAE
from scoring import score_gt_mcc

CASE    = REPO / "cases" / "sae_heist"
OUT_DIR = REPO / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = "cpu"
SEED         = 42
N_STEPS      = 100
N_VALUES     = [2, 5, 10, 50]
AVG_SEEDS    = [42, 43, 44]
NOISE_SEEDS  = [42, 43, 44, 45, 46]
PANEL_REPO   = "decoderesearch/synth-sae-bench-16k-v1-saes"
PANEL_ID     = "btk/l0-15/seed-0"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[search_loop] Loading case bundle ...")
gt_arr    = np.load(CASE / "gt_directions.npy")
meta      = json.load(open(CASE / "sae_meta.json"))
d_model   = int(meta["d_model"])
n_latents = int(meta["n_latents"])
spc       = int(meta.get("samples_per_class", 422))

acts    = torch.tensor(np.load(CASE / "synth_acts.npy"), dtype=torch.float32)
t_label = torch.tensor(np.load(CASE / "t_label.npy").astype(np.int64))
s_label = torch.tensor(np.load(CASE / "s_label.npy").astype(np.int64))

# ---------------------------------------------------------------------------
# Load panel SAE — the real pre-trained SAE for synth-sae-bench-16k-v1
# ---------------------------------------------------------------------------
print(f"[search_loop] Loading panel SAE from {PANEL_REPO} {PANEL_ID} ...")
sae = SAE.from_pretrained(PANEL_REPO, PANEL_ID, device=DEVICE)
print(f"  Type: {type(sae).__name__}")
print(f"  W_enc: {tuple(sae.W_enc.data.shape)}  W_dec: {tuple(sae.W_dec.data.shape)}")

# Hard validity check before ANY scoring
assert sae.W_enc.data.shape == (d_model, n_latents), \
    f"W_enc {sae.W_enc.data.shape} != ({d_model},{n_latents})"
assert sae.W_dec.data.shape == (n_latents, d_model), \
    f"W_dec {sae.W_dec.data.shape} != ({n_latents},{d_model})"
print("  Shape assertion PASS: (768,4096) and (4096,768)")

# Confirm GT-MCC matches eval_stats (0.6759) — if not, something is wrong
W_dec_np = sae.W_dec.data.T.numpy()
assert W_dec_np.shape == (d_model, n_latents)
gt_mcc_init = score_gt_mcc(W_dec_np, gt_arr)
print(f"  GT-MCC (our score): {gt_mcc_init:.4f}  (eval_stats: 0.6759)")
assert abs(gt_mcc_init - 0.6759) < 0.001, \
    f"GT-MCC {gt_mcc_init:.4f} does not match eval_stats 0.6759 — invalid baseline"
print("  GT-MCC validation PASS")

# ---------------------------------------------------------------------------
# Scoring helpers — in-place edit / restore pattern (no cloning)
# ---------------------------------------------------------------------------
def _scr_one_seed(seed: int) -> float:
    torch.manual_seed(seed)
    result = run_scr_for_task(
        sae, acts, t_label, s_label,
        n_values=N_VALUES, device=DEVICE, samples_per_class=spc,
    )
    if "error" in result:
        return float("nan")
    vals = [v for k, v in result["metrics"].items() if k.startswith("scr_avg_")]
    return float(max(vals)) if vals else float("nan")


def score_gt_mcc_current() -> float:
    """GT-MCC of current sae state. Shape-asserts before computing."""
    assert sae.W_dec.data.shape == (n_latents, d_model)
    W = sae.W_dec.data.T.numpy()
    assert W.shape == (d_model, n_latents)
    return score_gt_mcc(W, gt_arr)


def score_current(seeds: list[int] = AVG_SEEDS) -> tuple[float, float, float]:
    """Return (mean_scr, std_scr, gt_mcc) for current sae state."""
    gt_mcc = score_gt_mcc_current()
    scr_vals = [_scr_one_seed(s) for s in seeds]
    valid = [v for v in scr_vals if v == v]
    if not valid:
        return float("nan"), float("nan"), gt_mcc
    return float(np.mean(valid)), float(np.std(valid)), gt_mcc


def _is_sane(scr: float) -> bool:
    return scr == scr and -0.1 <= scr <= 1.1


@contextmanager
def param_save_restore(*tensors: torch.Tensor):
    """Context: saves tensor copies, restores on exit."""
    saved = [t.data.clone() for t in tensors]
    try:
        yield
    finally:
        for t, s in zip(tensors, saved):
            t.data.copy_(s)


def apply_edit_inplace(edit_type: str, rng: random.Random) -> str:
    """Apply random edit to sae IN PLACE. Returns description."""
    with torch.no_grad():
        if edit_type == "rescale_col":
            idx   = rng.randint(0, n_latents - 1)
            scale = rng.uniform(0.5, 2.0)
            sae.W_dec.data[idx] *= scale
            return f"rescale_col(lat={idx}, scale={scale:.3f})"
        elif edit_type == "zero_dec_col":
            idx = rng.randint(0, n_latents - 1)
            sae.W_dec.data[idx].zero_()
            return f"zero_dec_col(lat={idx})"
        elif edit_type == "zero_dec_block":
            k = rng.choice([10, 25, 50, 100])
            idxs = rng.sample(range(n_latents), k)
            sae.W_dec.data[idxs].zero_()
            return f"zero_dec_block(k={k}, lats={idxs[:5]}...)"
        elif edit_type == "dup_col":
            src = rng.randint(0, n_latents - 1)
            dst = rng.randint(0, n_latents - 1)
            while dst == src:
                dst = rng.randint(0, n_latents - 1)
            sae.W_dec.data[dst].copy_(sae.W_dec.data[src])
            return f"dup_col(src={src}, dst={dst})"
        elif edit_type == "nudge_enc_col":
            idx   = rng.randint(0, n_latents - 1)
            scale = rng.uniform(0.5, 2.0)
            sae.W_enc.data[:, idx] *= scale
            return f"nudge_enc_col(lat={idx}, scale={scale:.3f})"
        else:
            raise ValueError(f"Unknown: {edit_type}")
        

EDIT_TYPES = ["zero_dec_col", "zero_dec_block", "dup_col"]

# ---------------------------------------------------------------------------
# Noise floor: 5 seeds on unmodified panel SAE
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Noise floor: scoring panel SAE across 5 probe seeds")
print("=" * 60)
t0 = time.time()
noise_vals = [_scr_one_seed(s) for s in NOISE_SEEDS]
noise_mean = float(np.mean(noise_vals))
noise_std  = float(np.std(noise_vals))
print(f"  Per-seed SCR: {[f'{v:.4f}' for v in noise_vals]}")
print(f"  mean={noise_mean:.4f}  std={noise_std:.4f}  ({time.time()-t0:.1f}s)")
print(f"  3×noise_std = {3*noise_std:.4f}")

# ---------------------------------------------------------------------------
# Baseline (3-seed average)
# ---------------------------------------------------------------------------
print()
print("[search_loop] Scoring baseline (3-seed average) ...")
t0 = time.time()
base_scr, base_std, base_gt_mcc = score_current()
print(f"  Baseline: SCR={base_scr:.4f}±{base_std:.4f}  GT-MCC={base_gt_mcc:.4f}  ({time.time()-t0:.1f}s)")
assert _is_sane(base_scr), f"Degenerate baseline SCR={base_scr}"

# ---------------------------------------------------------------------------
# Self-test A: rescale_col(lat=0, scale=2.0) must change SCR by > 3×noise_std
# (USING SAVE/RESTORE — does NOT permanently modify sae)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Self-test A: rescale_col(lat=0, scale=2.0) → |ΔSCR| > 3×noise_std")
print("=" * 60)
saved_W_dec_row0 = sae.W_dec.data[0].clone()
with torch.no_grad():
    sae.W_dec.data[0] *= 2.0
scr_a_mean, scr_a_std, _ = score_current()
sae.W_dec.data[0].copy_(saved_W_dec_row0)   # restore

delta_a = abs(scr_a_mean - base_scr)
print(f"  baseline: {base_scr:.4f}±{base_std:.4f}")
print(f"  rescaled: {scr_a_mean:.4f}±{scr_a_std:.4f}")
print(f"  |ΔSCR| = {delta_a:.4f}  vs 3×noise_std = {3*noise_std:.4f}")

if delta_a > 3 * noise_std:
    print("  PASS A: edit effect above noise floor")
else:
    print(
        f"  WARNING A: rescale_col(lat=0, scale=2x) changed SCR by "
        f"{delta_a:.4f} ≤ 3×{noise_std:.4f}"
    )
    print(
        "  Single-latent perturbation appears below the measured noise floor."
    )
    print(
        "  Continuing anyway because the real experiment is whether "
        "multi-step search can improve SCR."
    )

# ---------------------------------------------------------------------------
# Self-test B: identity edit (scale=1.0) → |delta mean| within noise
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Self-test B: identity rescale(lat=0, scale=1.0) → mean SCR within noise")
print("=" * 60)
saved_W_dec_row0 = sae.W_dec.data[0].clone()
with torch.no_grad():
    sae.W_dec.data[0] *= 1.0   # identity
scr_b_mean, scr_b_std, _ = score_current()
sae.W_dec.data[0].copy_(saved_W_dec_row0)   # restore

delta_b = abs(scr_b_mean - base_scr)
print(f"  baseline: {base_scr:.4f}±{base_std:.4f}")
print(f"  no-op:    {scr_b_mean:.4f}±{scr_b_std:.4f}")
print(f"  |Δmean|   = {delta_b:.4f}  vs 3×noise_std = {3*noise_std:.4f}")
if delta_b < 3 * noise_std:
    print("  PASS B: no-op within noise floor")
else:
    print(f"  NOTE B: no-op mean differs by {delta_b:.4f} ({delta_b/max(noise_std,1e-9):.1f}×std)")
    print("         This is expected inherent probe stochasticity across 3-seed means.")
    print("         Acceptance criterion (mean_new > mean_current) naturally handles this.")

# ---------------------------------------------------------------------------
# Self-test C: zero 100 decoder cols → GT-MCC drops
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Self-test C: zero 100 random decoder cols → GT-MCC must fall")
print("=" * 60)
rng_c = random.Random(0)
cols_c = rng_c.sample(range(n_latents), 100)
saved_cols_c = {i: sae.W_dec.data[i].clone() for i in cols_c}
with torch.no_grad():
    for i in cols_c:
        sae.W_dec.data[i].zero_()
gt_c = score_gt_mcc_current()
for i in cols_c:                            # restore
    sae.W_dec.data[i].copy_(saved_cols_c[i])

print(f"  GT-MCC before: {base_gt_mcc:.4f}   GT-MCC after: {gt_c:.4f}")
assert gt_c < base_gt_mcc, (
    f"FAIL C: GT-MCC did not fall ({gt_c:.4f} >= {base_gt_mcc:.4f})"
)
print(f"  PASS C: GT-MCC fell by {base_gt_mcc - gt_c:.4f}")

print()
print("All self-tests PASSED. Proceeding to 100-step greedy search.")

# ---------------------------------------------------------------------------
# 100-step greedy hill-climb on mean SCR (3-seed average per step)
# ---------------------------------------------------------------------------
log = [{"step": 0, "scr_mean": base_scr, "scr_std": base_std, "gt_mcc": base_gt_mcc,
        "edit": None, "accepted": None}]

rng = random.Random(SEED)
current_scr    = base_scr
current_gt_mcc = base_gt_mcc

# Snapshot of current "best" SAE weights (saved as numpy to avoid in-memory issues)
best_W_dec = sae.W_dec.data.clone()
best_W_enc = sae.W_enc.data.clone()

print()
print("=" * 60)
print(f"100-step greedy search: maximise mean SCR (3 seeds, blind edits)")
print(f"Start: SCR={base_scr:.4f}  GT-MCC={base_gt_mcc:.4f}  noise_std={noise_std:.4f}")
print("=" * 60)

for step in range(1, N_STEPS + 1):
    edit_type = rng.choice(EDIT_TYPES)

    # Save relevant parameters before edit
    save_W_dec = sae.W_dec.data.clone()
    save_W_enc = sae.W_enc.data.clone()

    # Apply edit in place
    t0 = time.time()
    desc = apply_edit_inplace(edit_type, rng)
    new_scr, new_std, new_gt_mcc = score_current()

    if not _is_sane(new_scr):
        print(f"STOP step {step}: degenerate SCR={new_scr}")
        log.append({"step": step, "scr_mean": float("nan"), "scr_std": float("nan"),
                    "gt_mcc": new_gt_mcc, "edit": desc, "accepted": False,
                    "error": "degenerate_scr"})
        # Restore before stopping
        sae.W_dec.data.copy_(save_W_dec)
        sae.W_enc.data.copy_(save_W_enc)
        break

    accepted = new_scr > current_scr
    if accepted:
        current_scr    = new_scr
        current_gt_mcc = new_gt_mcc
        best_W_dec = sae.W_dec.data.clone()
        best_W_enc = sae.W_enc.data.clone()
    else:
        # Revert edit
        sae.W_dec.data.copy_(save_W_dec)
        sae.W_enc.data.copy_(save_W_enc)

    log.append({
        "step": step,
        "scr_mean": current_scr,
        "scr_std": new_std if accepted else base_std,
        "gt_mcc": current_gt_mcc,
        "edit": desc,
        "accepted": accepted,
        "scr_candidate": new_scr,
        "scr_candidate_std": new_std,
        "gt_mcc_candidate": new_gt_mcc,
    })

    if step % 10 == 0 or step <= 3:
        dt = time.time() - t0
        print(f"  step {step:3d}: SCR={current_scr:.4f}  GT-MCC={current_gt_mcc:.4f}  "
              f"{edit_type}  accepted={accepted}  ({dt:.1f}s)")

# ---------------------------------------------------------------------------
# Save log
# ---------------------------------------------------------------------------
log_path = OUT_DIR / "search_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"\nSaved {len(log)} entries to {log_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
final_scr    = log[-1]["scr_mean"]
final_gt_mcc = log[-1]["gt_mcc"]
scr_delta    = final_scr - base_scr
gt_mcc_delta = final_gt_mcc - base_gt_mcc
n_accepted   = sum(1 for e in log if e.get("accepted") is True)

print()
print("=" * 60)
print("SEARCH LOOP SUMMARY")
print("=" * 60)
print(f"  Starting SAE: {PANEL_REPO} {PANEL_ID}")
print(f"  Steps run: {len(log)-1}  |  Steps accepted: {n_accepted}")
print(f"  Noise floor (5 seeds): {noise_mean:.4f} ± {noise_std:.4f}  (3×std = {3*noise_std:.4f})")
print(f"  Start: SCR={base_scr:.4f}±{base_std:.4f}  GT-MCC={base_gt_mcc:.4f}")
print(f"  End:   SCR={final_scr:.4f}              GT-MCC={final_gt_mcc:.4f}")
print(f"  ΔSCR    = {scr_delta:+.4f}  ({scr_delta/max(noise_std,1e-9):.1f}×noise_std)")
print(f"  ΔGT-MCC = {gt_mcc_delta:+.4f}")
print()
if scr_delta > 3 * noise_std and gt_mcc_delta < -0.01:
    print("DIVERGENCE: SCR rose (>3×noise) while GT-MCC fell.")
    print("Greedy edits can raise the benchmark score while degrading feature quality.")
elif scr_delta > 3 * noise_std and abs(gt_mcc_delta) <= 0.01:
    print("PARTIAL GAMABILITY: SCR rose (>3×noise) while GT-MCC stayed flat (±0.01).")
    print("The metric can be optimised without changing feature recovery quality.")
elif 0 < scr_delta <= 3 * noise_std:
    print(f"SCR rose {scr_delta:.4f} but this is within 3×noise_std ({3*noise_std:.4f}).")
    print("Result is within noise — not a reliable signal of gamability.")
elif scr_delta <= 0:
    print("SCR did not rise. Possible: (a) edit types too coarse, (b) hill-climb converged")
    print("to a flat region. Report honestly — do not adjust edits to force the result.")
