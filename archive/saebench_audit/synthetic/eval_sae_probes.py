"""Sparse-probing evaluation on synthetic SAE activations.

Mirrors the SAEBench sparse-probing eval (fixed C=1 L2 sklearn LR) and the
sae-probes variant (L1-regularised LR with 10-fold CV over the regularisation
strength) — both are run on the same SAE for direct comparison in the paper's
synthetic-validity panel.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from sae_lens import SAE
from sae_probes.run_sae_evals import get_sorted_indices, mean_act_normalization
from sae_probes.utils_training import find_best_reg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ProbeMode = Literal["sae_bench", "sae_probes"]
ProbeReg = Literal["l1", "l2"]


@torch.no_grad()
def encode_in_chunks(
    sae: SAE[Any], hidden: torch.Tensor, batch_size: int = 1024, device: str = "cuda"
) -> torch.Tensor:
    """Encode ``hidden`` through ``sae`` in chunks; returns ``(N, d_sae)``."""
    chunks: list[torch.Tensor] = []
    for i in range(0, hidden.shape[0], batch_size):
        x = hidden[i : i + batch_size].to(device)
        z = sae.encode(x).detach().cpu()
        chunks.append(z)
    return torch.cat(chunks, dim=0)


def balanced_split(
    labels: torch.Tensor,
    max_train: int,
    target_train_pos: int,
    target_test_pos: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced train/test split for a binary label tensor."""
    rng = np.random.RandomState(seed)
    pos = np.where(labels.numpy() == 1)[0]
    neg = np.where(labels.numpy() == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_train = min(target_train_pos, len(pos) // 2)
    pos_test = min(target_test_pos, len(pos) - pos_train)
    if pos_train < 5 or pos_test < 5:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    neg_train = min(pos_train, len(neg) // 2)
    neg_test = min(pos_test, len(neg) - neg_train)
    if pos_train + neg_train > max_train:
        pos_train = max_train // 2
        neg_train = max_train // 2
    train_idx = np.concatenate([pos[:pos_train], neg[:neg_train]])
    test_idx = np.concatenate(
        [pos[pos_train : pos_train + pos_test], neg[neg_train : neg_train + neg_test]]
    )
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return torch.from_numpy(train_idx).long(), torch.from_numpy(test_idx).long()


def _empty_metrics(ks: list[int]) -> dict[int, dict[str, float]]:
    return {
        k: {"test_acc": 0.5, "test_auc": 0.5, "test_f1": 0.0, "val_auc": 0.5}
        for k in ks
    }


def _fit_sae_bench_probe(
    Xtr: npt.NDArray[np.floating[Any]],
    Xte: npt.NDArray[np.floating[Any]],
    y_train: npt.NDArray[np.int_],
    y_test: npt.NDArray[np.int_],
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xte)
        try:
            proba = clf.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = 0.5
    return {
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_auc": auc,
        "test_f1": float(f1_score(y_test, pred)),
        "val_auc": 0.5,
    }


def _fit_sae_probes_probe(
    Xtr: npt.NDArray[np.floating[Any]],
    Xte: npt.NDArray[np.floating[Any]],
    y_train: npt.NDArray[np.int_],
    y_test: npt.NDArray[np.int_],
    reg_type: ProbeReg,
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = find_best_reg(
            Xtr, y_train, Xte, y_test, n_jobs=1, parallel=False, penalty=reg_type
        )
    return {
        "test_acc": float(results.metrics.test_acc),
        "test_auc": float(results.metrics.test_auc),
        "test_f1": float(results.metrics.test_f1),
        "val_auc": float(results.metrics.val_auc),
    }


def run_sae_probes_for_task(
    sae_acts: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    ks: list[int],
    mode: ProbeMode = "sae_bench",
    reg_type: ProbeReg = "l2",
) -> dict[int, dict[str, float]]:
    """Run a top-k sparse probe on SAE latents.

    ``mode`` chooses between the SAEBench fixed-C L2 probe and the sae-probes
    L1-regularised CV probe used by the SAEBench sparse_probing_sae_probes
    eval. Top-k feature selection follows the sae-probes mean-act-normalised
    sorting in both modes.
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return _empty_metrics(ks)

    X_train = sae_acts[train_idx].float()
    X_test = sae_acts[test_idx].float()
    y_train = labels[train_idx].long()
    y_test = labels[test_idx].long()
    sorted_indices = get_sorted_indices(X_train, y_train, mean_act_normalization)

    out: dict[int, dict[str, float]] = {}
    y_train_np = y_train.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    for k in ks:
        top_k = sorted_indices[:k]
        Xtr = X_train[:, top_k].cpu().numpy()
        Xte = X_test[:, top_k].cpu().numpy()
        if len(set(y_train_np.tolist())) < 2 or len(set(y_test_np.tolist())) < 2:
            out[k] = {
                "test_acc": 0.5,
                "test_auc": 0.5,
                "test_f1": 0.0,
                "val_auc": 0.5,
            }
            continue
        if mode == "sae_bench":
            out[k] = _fit_sae_bench_probe(Xtr, Xte, y_train_np, y_test_np)
        else:
            out[k] = _fit_sae_probes_probe(
                Xtr, Xte, y_train_np, y_test_np, reg_type=reg_type
            )
    return out
