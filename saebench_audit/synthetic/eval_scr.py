"""Synthetic SCR eval.

SAEBench's SCR is hardwired to the bias-in-bios paired-class structure
(``male/female``, ``professor/nurse``, ``male_professor/female_nurse``). We
monkey-patch ``dataset_info.PAIRED_CLASS_KEYS`` for the duration of an SCR run
to mirror our synthetic ``(T, S)`` setup, then call the SAEBench primitives
directly. SCR scores are computed via a parameterised copy of
``get_scr_plotting_dict``.

Class keys (``"positive"`` is the named class; ``"_neg"`` is where negatives
come from for ``prepare_probe_data`` when ``perform_scr=True``):

* ``"S"`` — samples where ``S=1``;     ``"S_neg"`` — samples where ``S=0``
* ``"T"`` — samples where ``T=1``;     ``"T_neg"`` — samples where ``T=0``
* ``"bias"`` — samples where ``T=S=1``; ``"bias_neg"`` — samples where ``T=S=0``

The SCR metric is
``(changed_acc - original_acc) / (clean_acc - original_acc)``:

* ``original_acc``: biased probe on un-confounded T-only data (low — leaks ``S``).
* ``clean_acc``: T probe on un-confounded T-only data (high).
* ``changed_acc``: biased probe on un-confounded T-only data after ablating
  the top-N SAE latents most associated with ``S``.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, cast

import sae_bench.sae_bench_utils.dataset_info as dataset_info
import torch
from sae_bench.evals.scr_and_tpp.main import (
    get_all_node_effects_for_one_sae,
    get_probe_test_accuracy,
    perform_feature_ablations,
)
from sae_bench.evals.sparse_probing.probe_training import (
    train_probe_on_activations,
)
from sae_lens import SAE

CLASS_S = "S"
CLASS_T = "T"
CLASS_BIAS = "bias"
NEG_S = "S_neg"
NEG_T = "T_neg"
NEG_BIAS = "bias_neg"

SCR_PAIRED_CLASS_KEYS: dict[str, str] = {
    CLASS_S: NEG_S,
    CLASS_T: NEG_T,
    CLASS_BIAS: NEG_BIAS,
}


@contextmanager
def scr_paired_keys():  # type: ignore[no-untyped-def]
    """Monkey-patch ``dataset_info.PAIRED_CLASS_KEYS`` for the SCR run."""
    original = dataset_info.PAIRED_CLASS_KEYS
    dataset_info.PAIRED_CLASS_KEYS = SCR_PAIRED_CLASS_KEYS
    try:
        yield
    finally:
        dataset_info.PAIRED_CLASS_KEYS = original


def build_scr_class_acts(
    hidden: torch.Tensor,
    t_label: torch.Tensor,
    s_label: torch.Tensor,
    samples_per_class: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Build the six class activation tensors for SCR.

    Each returned value has shape ``[B, 1, D]``; ``B`` is balanced across
    the six classes to the smallest eligible class (capped at
    ``samples_per_class``).
    """
    t = t_label.bool()
    s = s_label.bool()
    index = {
        CLASS_S: torch.nonzero(s, as_tuple=True)[0],
        NEG_S: torch.nonzero(~s, as_tuple=True)[0],
        CLASS_T: torch.nonzero(t, as_tuple=True)[0],
        NEG_T: torch.nonzero(~t, as_tuple=True)[0],
        CLASS_BIAS: torch.nonzero(t & s, as_tuple=True)[0],
        NEG_BIAS: torch.nonzero(~t & ~s, as_tuple=True)[0],
    }
    min_count = min(len(v) for v in index.values())
    if min_count < 20:
        return {}
    target = min(samples_per_class, min_count)
    out: dict[str, torch.Tensor] = {}
    for k, idx in index.items():
        out[k] = hidden[idx[:target]].to(device).unsqueeze(1)
    return out


def _split_train_test(
    d: dict[str, torch.Tensor], train_frac: float = 0.8
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train: dict[str, torch.Tensor] = {}
    test: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        n = v.shape[0]
        ntr = int(n * train_frac)
        train[k] = v[:ntr]
        test[k] = v[ntr:]
    return train, test


def _meaned(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.squeeze(1).float() for k, v in d.items()}


def compute_scr_metrics(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    clean_accs: dict[str, float],
) -> dict[str, float]:
    """Parameterised copy of SAEBench's ``get_scr_plotting_dict``.

    ``class_accuracies[ablated_class][threshold][evaluated_class]`` follows
    the SAEBench nested-dict format. We emit both directions of intervention
    plus a canonical SAEBench-style metric per threshold.
    """
    results: dict[str, float] = {}

    bias_on_T_key = f"{CLASS_BIAS} probe on {CLASS_T} data"
    bias_on_S_key = f"{CLASS_BIAS} probe on {CLASS_S} data"
    T_clean = clean_accs[CLASS_T]
    S_clean = clean_accs[CLASS_S]
    bias_on_T_orig = clean_accs[bias_on_T_key]
    bias_on_S_orig = clean_accs[bias_on_S_key]

    for threshold in class_accuracies[CLASS_S]:
        changed = class_accuracies[CLASS_S][threshold][bias_on_T_key]
        if (T_clean - bias_on_T_orig) < 0.001:
            scr_score = 0.0
        else:
            scr_score = (changed - bias_on_T_orig) / (T_clean - bias_on_T_orig)
        results[f"scr_dir1_threshold_{threshold}"] = scr_score

    for threshold in class_accuracies[CLASS_T]:
        changed = class_accuracies[CLASS_T][threshold][bias_on_S_key]
        if (S_clean - bias_on_S_orig) < 0.001:
            scr_score = 0.0
        else:
            scr_score = (changed - bias_on_S_orig) / (S_clean - bias_on_S_orig)
        results[f"scr_dir2_threshold_{threshold}"] = scr_score

    dir1_acc_bias = bias_on_T_orig
    dir2_acc_bias = bias_on_S_orig
    for threshold in class_accuracies[CLASS_S]:
        d1 = results[f"scr_dir1_threshold_{threshold}"]
        d2 = results[f"scr_dir2_threshold_{threshold}"]
        results[f"scr_avg_threshold_{threshold}"] = 0.5 * (d1 + d2)
        if dir1_acc_bias < dir2_acc_bias:
            results[f"scr_metric_threshold_{threshold}"] = d1
        elif dir1_acc_bias > dir2_acc_bias:
            results[f"scr_metric_threshold_{threshold}"] = d2
        else:
            results[f"scr_metric_threshold_{threshold}"] = 0.5 * (d1 + d2)

    results["_diag_T_clean"] = T_clean
    results["_diag_S_clean"] = S_clean
    results["_diag_bias_on_T_orig"] = bias_on_T_orig
    results["_diag_bias_on_S_orig"] = bias_on_S_orig
    return results


def run_scr_for_task(
    sae: SAE[Any],
    hidden: torch.Tensor,
    t_label: torch.Tensor,
    s_label: torch.Tensor,
    n_values: Sequence[int],
    device: str,
    samples_per_class: int = 1500,
    sae_batch_size: int = 1024,
    probe_test_batch_size: int = 1024,
    probe_train_batch_size: int = 16,
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    probe_l1_penalty: float | None = 1e-3,
    early_stopping_patience: int = 20,
) -> dict[str, Any]:
    """End-to-end SCR run for one ``(T, S)`` task on one SAE."""
    class_acts = build_scr_class_acts(
        hidden, t_label, s_label, samples_per_class, device
    )
    if not class_acts:
        return {"error": "insufficient_samples"}

    train_acts, test_acts = _split_train_test(class_acts)
    train_meaned = _meaned(train_acts)
    test_meaned = _meaned(test_acts)

    with scr_paired_keys():
        probes_raw, _ = train_probe_on_activations(
            train_meaned,
            test_meaned,
            select_top_k=None,
            use_sklearn=False,
            perform_scr=True,
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
            perform_scr=True,
        )

        chosen_classes = list(SCR_PAIRED_CLASS_KEYS.keys())
        node_effects = get_all_node_effects_for_one_sae(
            sae,
            probes,
            chosen_classes,
            perform_scr=True,
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
            chosen_classes,
            probe_test_batch_size,
            perform_scr=True,
        )

    metrics = compute_scr_metrics(ablated_class_accuracies, clean_accs)
    return {"metrics": metrics, "clean_accs": clean_accs}
