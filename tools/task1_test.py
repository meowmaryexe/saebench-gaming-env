"""Task 1 trivial known-answer test for the SCR ablation mechanism.

PREDICTION (from source reading):
  Zero SAE: W_enc=0, W_dec=0, b_enc=0, b_dec=0
    encode(x) = ReLU(x @ 0 + 0) = 0
    decode(0) = 0 @ 0 + 0 = 0
    error     = x - 0 = x
    ablation: f[..., to_ablate] = 0.0  (already zero, no change)
    modified_acts = decode(0) + error = 0 + x = x  (identical to input)
  -> changed_acc = original_acc (same data seen by probe)
  -> SCR = (changed_acc - original_acc) / (T_clean - original_acc) = 0 / ... = 0.0 EXACTLY

If any scr_avg_threshold_N != 0.0 for the zero SAE, the mechanism is not as read.

Run with: .venv/bin/python tools/task1_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from saebench_audit.synthetic.eval_scr import run_scr_for_task
from sae_lens import StandardSAE, StandardSAEConfig

CASE = REPO / "cases" / "sae_heist"
meta = json.load(open(CASE / "sae_meta.json"))
d_model   = int(meta["d_model"])
n_latents = int(meta["n_latents"])
spc       = int(meta.get("samples_per_class", 422))

acts_np = np.load(CASE / "synth_acts.npy")
acts    = torch.tensor(acts_np, dtype=torch.float32)
t_label = torch.tensor(np.load(CASE / "t_label.npy").astype(np.int64))
s_label = torch.tensor(np.load(CASE / "s_label.npy").astype(np.int64))

print("=" * 60)
print("Task 1 trivial test: zero SAE → SCR must equal 0.0 exactly")
print("=" * 60)
print(f"d_model={d_model}  n_latents={n_latents}  samples_per_class={spc}")
print()

# Build zero SAE: all parameters zero
cfg = StandardSAEConfig(d_in=d_model, d_sae=n_latents, device="cpu", dtype="float32")
zero_sae = StandardSAE(cfg)
with torch.no_grad():
    zero_sae.W_enc.data.zero_()
    zero_sae.W_dec.data.zero_()
    zero_sae.b_enc.data.zero_()
    zero_sae.b_dec.data.zero_()

# Verify shapes before scoring (guard against silent transpose bug)
assert zero_sae.W_enc.data.shape == (d_model, n_latents), \
    f"W_enc shape {zero_sae.W_enc.data.shape} != ({d_model}, {n_latents})"
assert zero_sae.W_dec.data.shape == (n_latents, d_model), \
    f"W_dec shape {zero_sae.W_dec.data.shape} != ({n_latents}, {d_model})"
print(f"W_enc.shape = {tuple(zero_sae.W_enc.data.shape)}  (expected ({d_model}, {n_latents}))")
print(f"W_dec.shape = {tuple(zero_sae.W_dec.data.shape)}  (expected ({n_latents}, {d_model}))")
print()

# Verify the zero-SAE prediction manually
x_sample = acts[:3]
f = zero_sae.encode(x_sample)
xhat = zero_sae.decode(f)
error = x_sample - xhat
print(f"encode(x_sample).max() = {f.abs().max().item():.6f}  (expected 0.0)")
print(f"decode(0).max()        = {xhat.abs().max().item():.6f}  (expected 0.0)")
print(f"error = x - xhat:  max|error - x_sample| = {(error - x_sample).abs().max().item():.6f}  (expected 0.0)")
print()

print("PREDICTED: scr_avg_threshold_N = 0.0 for all N")
print("Running run_scr_for_task with zero SAE ...")
result = run_scr_for_task(
    zero_sae, acts, t_label, s_label,
    n_values=[2, 5, 10, 50], device="cpu",
    samples_per_class=spc,
)

if "error" in result:
    print(f"FAIL: run_scr_for_task returned error: {result['error']}")
    sys.exit(1)

metrics = result["metrics"]
scr_avg_vals = {k: v for k, v in metrics.items() if k.startswith("scr_avg_")}

print()
print("Observed scr_avg values:")
all_pass = True
for k, v in sorted(scr_avg_vals.items()):
    status = "PASS" if abs(v) < 1e-6 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  {k} = {v:.8f}   [{status}]")

print()
if all_pass:
    print("VERIFIED: zero SAE → SCR = 0.0 exactly for all n_values")
    print("Mechanism confirmed: error_BLD dominates when SAE reconstruction is zero,")
    print("modified_acts = original_acts, probe sees identical data, changed_acc = original_acc.")
else:
    print("MECHANISM MISMATCH: zero SAE did NOT give SCR=0.")
    print("Either the source reading is wrong or there is a sign/formula discrepancy.")
    print("Full metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k} = {v}")
    sys.exit(1)
