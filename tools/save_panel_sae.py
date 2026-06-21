"""One-shot: save panel SAE weights to cases/sae_heist/ and verify GT-MCC.

Replaces the random-init sae_W_enc.npy / sae_W_dec.npy with the HuggingFace
panel SAE (decoderesearch/synth-sae-bench-16k-v1-saes btk/l0-15/seed-0)
that achieves GT-MCC≈0.676.

Run with: .venv/bin/python tools/save_panel_sae.py

Gates:
  - W_enc shape == (4096, 768)  our convention: (n_latents, d_model)
  - W_dec shape == (768, 4096)  our convention: (d_model, n_latents)
  - GT-MCC in [0.674, 0.678]   ±0.002 tolerance around known value
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sae_lens import SAE
from scoring import score_gt_mcc

CASE = REPO / "cases" / "sae_heist"

print("[save_panel_sae] Loading SAE from HuggingFace cache ...")
result = SAE.from_pretrained(
    "decoderesearch/synth-sae-bench-16k-v1-saes",
    "btk/l0-15/seed-0",
)
# from_pretrained returns (sae, cfg_dict, log_sparsities) in some versions,
# or just the SAE object. Handle both.
if isinstance(result, tuple):
    sae = result[0]
else:
    sae = result

print(f"  SAE type: {type(sae).__name__}")
print(f"  W_enc sae_lens shape: {tuple(sae.W_enc.shape)}  (d_in, d_sae)")
print(f"  W_dec sae_lens shape: {tuple(sae.W_dec.shape)}  (d_sae, d_in)")

# sae_lens convention: W_enc (d_in=768, d_sae=4096), W_dec (d_sae=4096, d_in=768)
# our convention:      W_enc (n_latents=4096, d_model=768) = .T
#                      W_dec (d_model=768, n_latents=4096) = .T
W_enc_np = sae.W_enc.data.T.numpy().astype("float32")   # (4096, 768)
W_dec_np = sae.W_dec.data.T.numpy().astype("float32")   # (768, 4096)

assert W_enc_np.shape == (4096, 768), f"W_enc shape {W_enc_np.shape} != (4096, 768)"
assert W_dec_np.shape == (768, 4096), f"W_dec shape {W_dec_np.shape} != (768, 4096)"
print(f"  Our conv W_enc: {W_enc_np.shape}  W_dec: {W_dec_np.shape}  ✓")

np.save(CASE / "sae_W_enc.npy", W_enc_np)
np.save(CASE / "sae_W_dec.npy", W_dec_np)
print(f"  Saved to {CASE}/")

print("\n[save_panel_sae] Verifying GT-MCC ...")
gt = np.load(CASE / "gt_directions.npy")
gt_mcc = score_gt_mcc(W_dec_np, gt)
print(f"  GT-MCC: {gt_mcc:.6f}  (expected ~0.6759)")
assert abs(gt_mcc - 0.6759) < 0.002, (
    f"GT-MCC {gt_mcc:.6f} not near 0.6759 — possible transpose error or wrong SAE"
)
print("  PASS ✓")
