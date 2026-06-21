"""Degraded SAE variants used as controls in the paper.

These constructors take a base SAE and produce a degraded copy for use as a
diagnostic alongside the trained SAEs in the synthetic panel
(\\S 4.2 of the paper).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import torch
from sae_lens import SAE


def permute_decoder(base: SAE[Any], seed: int = 0) -> SAE[Any]:
    """Shuffle rows of ``W_dec``; encoder output is unchanged.

    Encoder-only metrics (sparse probing) score identically to the base SAE
    because ``encode(x)`` does not touch the decoder. Decoder-sensitive
    metrics (TPP, SCR, RAVEL) crash to noise.
    """
    new = deepcopy(base)
    W_dec = cast(torch.Tensor, new.W_dec)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(int(W_dec.shape[0]), generator=g).to(W_dec.device)
    with torch.no_grad():
        W_dec.copy_(W_dec[perm])
    return new


def random_init_variant(base: SAE[Any], seed: int = 0) -> SAE[Any]:
    """Replace SAE weights with random init; bias-free, threshold = 1.0."""
    new = deepcopy(base)
    W_enc = cast(torch.Tensor, new.W_enc)
    W_dec = cast(torch.Tensor, new.W_dec)
    b_enc = cast(torch.Tensor, new.b_enc)
    b_dec = cast(torch.Tensor, new.b_dec)
    d_in = int(W_enc.shape[0])
    d_sae = int(W_enc.shape[1])
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        W_enc.copy_(torch.randn(d_in, d_sae, generator=g) / d_in**0.5)
        W_dec.copy_(torch.randn(d_sae, d_in, generator=g) / d_sae**0.5)
        b_enc.zero_()
        b_dec.zero_()
        threshold = getattr(new, "threshold", None)
        if isinstance(threshold, torch.Tensor):
            threshold.fill_(1.0)
    return new


def random_l0_matched(
    base: SAE[Any],
    calibration_hidden: torch.Tensor,
    target_l0: float = 25.0,
    seed: int = 0,
    chunk: int = 4096,
) -> SAE[Any]:
    """Random init, threshold tuned so calibration L0 matches ``target_l0``.

    The threshold is set to the ``(1 - target_l0 / d_sae)`` quantile of the
    pre-activations on the calibration hidden batch.
    """
    new = random_init_variant(base, seed=seed)
    W_enc = cast(torch.Tensor, new.W_enc)
    W_dec = cast(torch.Tensor, new.W_dec)
    b_enc = cast(torch.Tensor, new.b_enc)
    b_dec = cast(torch.Tensor, new.b_dec)
    device = W_enc.device
    preact_chunks: list[torch.Tensor] = []
    apply_b_dec = bool(getattr(new.cfg, "apply_b_dec_to_input", False))
    with torch.no_grad():
        for i in range(0, calibration_hidden.shape[0], chunk):
            x = calibration_hidden[i : i + chunk].to(device).float()
            x_in = x - b_dec if apply_b_dec else x
            preact = x_in @ W_enc + b_enc
            preact_chunks.append(preact.cpu())
    preacts = torch.cat(preact_chunks, dim=0)
    d_sae = int(W_dec.shape[0])
    target_q = max(0.0, 1.0 - target_l0 / d_sae)
    flat = preacts.flatten().float()
    if flat.numel() > 1_000_000:
        idx = torch.randperm(
            flat.numel(),
            generator=torch.Generator().manual_seed(seed),
        )[:1_000_000]
        flat = flat[idx]
    tau = torch.quantile(flat, target_q).item()
    with torch.no_grad():
        threshold = getattr(new, "threshold", None)
        if isinstance(threshold, torch.Tensor):
            threshold.fill_(tau)
    return new


def best_k_variant(
    base: SAE[Any],
    calibration_hidden: torch.Tensor,
    k: int = 512,
    chunk: int = 4096,
) -> SAE[Any]:
    """Keep the K latents with highest ``mean(|z|) * ||W_dec[k]||`` on calibration data."""
    new = deepcopy(base)
    W_enc = cast(torch.Tensor, new.W_enc)
    W_dec = cast(torch.Tensor, new.W_dec)
    b_enc = cast(torch.Tensor, new.b_enc)
    d_sae = int(W_dec.shape[0])
    if k >= d_sae:
        return new
    device = W_enc.device
    abs_sum = torch.zeros(d_sae, device=device)
    n_total = 0
    with torch.no_grad():
        for i in range(0, calibration_hidden.shape[0], chunk):
            x = calibration_hidden[i : i + chunk].to(device).float()
            z = base.encode(x)
            abs_sum += z.abs().sum(dim=0)
            n_total += x.shape[0]
        mean_abs = abs_sum / max(n_total, 1)
        dec_norm = W_dec.norm(dim=-1)
        score = mean_abs * dec_norm
        keep_idx = torch.topk(score, k).indices
        keep_mask = torch.zeros(d_sae, dtype=torch.bool, device=device)
        keep_mask[keep_idx] = True
        not_keep = ~keep_mask
        W_dec[not_keep] = 0.0
        W_enc[:, not_keep] = 0.0
        b_enc[not_keep] = 0.0
        threshold = getattr(new, "threshold", None)
        if isinstance(threshold, torch.Tensor):
            threshold[not_keep] = 1e9
    return new
