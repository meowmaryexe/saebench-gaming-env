"""TPP eval on synthetic data.

Each TPP task is a sibling group under a common parent (4 mutually-exclusive
depth-3 children) — the synthetic analogue of SAEBench's bias-in-bios 4-class
TPP setup. We reuse SAEBench's primitives verbatim for probe training, node
effects, and feature ablation, only swapping the data source for our
hierarchy-aware synthetic activations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from sae_bench.evals.scr_and_tpp.main import (
    create_tpp_plotting_dict,
    get_all_node_effects_for_one_sae,
    get_probe_test_accuracy,
    perform_feature_ablations,
)
from sae_bench.evals.sparse_probing.probe_training import (
    train_probe_on_activations,
)
from sae_lens import SAE


def _build_class_acts(
    hidden: torch.Tensor,
    per_class_indices: dict[str, torch.Tensor],
    samples_per_class: int,
    device: str,
    min_pos: int = 25,
) -> dict[str, torch.Tensor]:
    """Return ``{class_name -> [B, 1, D]}`` for a sibling group, balanced."""
    eligible = {k: v for k, v in per_class_indices.items() if len(v) >= min_pos}
    if len(eligible) < 2:
        return {}
    target = min(samples_per_class, min(len(v) for v in eligible.values()))
    out: dict[str, torch.Tensor] = {}
    for k, idx in eligible.items():
        out[k] = hidden[idx[:target]].to(device).unsqueeze(1)
    return out


def _split_train_test(
    class_acts: dict[str, torch.Tensor], train_frac: float = 0.8
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train: dict[str, torch.Tensor] = {}
    test: dict[str, torch.Tensor] = {}
    for k, v in class_acts.items():
        n = v.shape[0]
        ntr = int(n * train_frac)
        train[k] = v[:ntr]
        test[k] = v[ntr:]
    return train, test


def _meaned(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.squeeze(1).float() for k, v in d.items()}


def run_tpp_for_sibling_group(
    sae: SAE[Any],
    hidden: torch.Tensor,
    per_class_indices: dict[str, torch.Tensor],
    n_values: Sequence[int],
    device: str,
    samples_per_class: int = 2000,
    sae_batch_size: int = 1024,
    probe_test_batch_size: int = 1024,
    probe_train_batch_size: int = 16,
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    probe_l1_penalty: float | None = 1e-3,
    early_stopping_patience: int = 20,
) -> dict[str, Any]:
    """Run TPP on one sibling group; returns per-sibling and overall metrics.

    Default probe hyperparameters match SAEBench's canonical
    ``ScrAndTppEvalConfig`` settings.
    """
    class_acts_all = _build_class_acts(
        hidden, per_class_indices, samples_per_class, device
    )
    if len(class_acts_all) < 2:
        return {"per_task": {}, "overall": {}}

    train_acts, test_acts = _split_train_test(class_acts_all)
    train_meaned = _meaned(train_acts)
    test_meaned = _meaned(test_acts)

    probes_raw, _ = train_probe_on_activations(
        train_meaned,
        test_meaned,
        select_top_k=None,
        use_sklearn=False,
        perform_scr=False,
        batch_size=probe_train_batch_size,
        epochs=probe_epochs,
        lr=probe_lr,
        verbose=False,
        l1_penalty=probe_l1_penalty,
        early_stopping_patience=early_stopping_patience,
    )
    probes = cast(dict[str, Any], probes_raw)

    clean_accs = get_probe_test_accuracy(
        probes,
        list(test_meaned.keys()),
        test_meaned,
        probe_test_batch_size,
        perform_scr=False,
    )

    node_effects = get_all_node_effects_for_one_sae(
        sae,
        probes,
        list(train_acts.keys()),
        perform_scr=False,
        indirect_effect_acts=train_acts,
        sae_batch_size=sae_batch_size,
    )

    ablated_class_accuracies = perform_feature_ablations(
        probes,
        sae,
        sae_batch_size,
        test_acts,
        node_effects,
        list(n_values),
        list(test_acts.keys()),
        probe_test_batch_size,
        perform_scr=False,
    )

    overall, per_class = create_tpp_plotting_dict(ablated_class_accuracies, clean_accs)
    return {"per_task": per_class, "overall": overall, "clean_accs": clean_accs}
