"""Synthetic-validity panel driver.

For one ``(base_model, seed)`` and one or more SAEs, runs:

* ``ground-truth``: GT-MCC, GT-F1, L0, etc. via SAELens's
  ``eval_sae_on_synthetic_data``.
* ``sparse-probing``: SAEBench-style fixed-C probe and (optionally) the
  sae-probes CV variant on every SP task.
* ``TPP``: each sibling group, at every requested ablation top-N.
* ``SCR``: each ``(T, S)`` pair, at every requested ablation top-N.

Output is one JSON record per SAE, written under
``<results-root>/<variation>/seed_<seed>/per_sae/<sae-name>.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from sae_lens.synthetic import SyntheticModel
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data

from saebench_audit.diagnostic.perfect_oracle import PerfectSAE
from saebench_audit.synthetic.data_gen import V1_REPO, generate, model_tag_for_repo
from saebench_audit.synthetic.eval_sae_probes import (
    ProbeMode,
    ProbeReg,
    balanced_split,
    encode_in_chunks,
    run_sae_probes_for_task,
)
from saebench_audit.synthetic.eval_scr import run_scr_for_task
from saebench_audit.synthetic.eval_tpp import run_tpp_for_sibling_group

PROBE_MAX_TRAIN = 1024
PROBE_TARGET_TRAIN_POS = 512
PROBE_TARGET_TEST_POS = 1500

GT_NUM_SAMPLES = 100_000
GT_BATCH = 4096
SAE_ENCODE_BATCH = 2048


@dataclass
class PanelEntry:
    """One SAE in the panel along with its source-of-truth metadata."""

    name: str
    family: str
    arch: str
    base_model_repo: str
    base_model_revision: str | None


def shared_data_dir(
    root: Path,
    base_model_repo: str,
    base_model_revision: str | None,
    seed: int,
) -> Path:
    """Return the shared-data directory for a ``(model, seed)`` pair."""
    return (
        root / model_tag_for_repo(base_model_repo, base_model_revision) / f"seed_{seed}"
    )


def compute_ground_truth(
    sae: SAE[Any],
    base_model_repo: str,
    base_model_revision: str | None,
    *,
    device: str = "cuda",
    num_samples: int = GT_NUM_SAMPLES,
    batch_size: int = GT_BATCH,
) -> dict[str, float | int]:
    """Compute GT-MCC, GT-F1, etc. for ``sae`` on the base model.

    The ``PerfectSAE`` oracle is short-circuited: its lookup-table contract
    only works for activations from the shared pool, but ``eval_sae_on_synthetic_data``
    samples *fresh* activations, so we report the analytic ``MCC = F1 = 1.0``
    instead.
    """
    if isinstance(sae, PerfectSAE):
        return {
            "mcc": 1.0,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "accuracy": 1.0,
            "explained_variance": 1.0,
            "uniqueness": 1.0,
            "dead_latents": 0,
        }
    if base_model_revision is None:
        model = SyntheticModel.from_pretrained(base_model_repo, device=device)
    else:
        model = SyntheticModel.from_pretrained(
            base_model_repo, model_path=base_model_revision, device=device
        )
    torch.manual_seed(0)
    with torch.no_grad():
        res = eval_sae_on_synthetic_data(
            sae,
            model.feature_dict,
            model.activation_generator,
            num_samples=num_samples,
            batch_size=batch_size,
        )
    return {
        "mcc": float(res.mcc),
        "f1": float(res.classification.f1_score),
        "precision": float(res.classification.precision),
        "recall": float(res.classification.recall),
        "accuracy": float(res.classification.accuracy),
        "sae_l0": float(res.sae_l0),
        "true_l0": float(res.true_l0),
        "uniqueness": float(res.uniqueness),
        "explained_variance": float(res.explained_variance),
        "dead_latents": int(res.dead_latents),
    }


def run_one_sae(  # noqa: PLR0913
    *,
    entry: PanelEntry,
    sae: SAE[Any],
    shared_dir: Path,
    out_path: Path,
    probe_ks: list[int],
    tpp_ns: list[int],
    scr_ns: list[int],
    probe_mode: ProbeMode = "sae_bench",
    probe_reg: ProbeReg = "l2",
    probe_train_batch_size: int = 16,
    skip_sp: bool = False,
    device: str = "cuda",
    force: bool = False,
) -> dict[str, Any] | None:
    """Run all three benchmarks for one SAE; returns the JSON record written."""
    if out_path.exists() and not force:
        return None

    t_start = time.time()
    hidden = torch.load(shared_dir / "hidden.pt")
    sp_labels: dict[int, torch.Tensor] = torch.load(shared_dir / "sp_labels.pt")
    tpp_indices: dict[str, dict[str, torch.Tensor]] = torch.load(
        shared_dir / "tpp_sibling_indices.pt"
    )
    scr_labels: dict[str, dict[str, torch.Tensor]] = torch.load(
        shared_dir / "scr_labels.pt"
    )
    with open(shared_dir / "sp_tasks.json") as f:
        sp_tasks: list[dict[str, Any]] = json.load(f)
    with open(shared_dir / "tpp_tasks.json") as f:
        tpp_tasks: list[dict[str, Any]] = json.load(f)
    with open(shared_dir / "scr_tasks.json") as f:
        scr_tasks: list[dict[str, Any]] = json.load(f)

    gt = compute_ground_truth(
        sae, entry.base_model_repo, entry.base_model_revision, device=device
    )

    sae_acts = encode_in_chunks(sae, hidden, batch_size=SAE_ENCODE_BATCH, device=device)

    sp_results: dict[int, dict[int, dict[str, float]]] = {}
    if not skip_sp:
        for task in sp_tasks:
            labels = sp_labels[task["id"]]
            train_idx, test_idx = balanced_split(
                labels,
                PROBE_MAX_TRAIN,
                PROBE_TARGET_TRAIN_POS,
                PROBE_TARGET_TEST_POS,
                seed=int(task["id"]),
            )
            if len(train_idx) == 0:
                continue
            per_k = run_sae_probes_for_task(
                sae_acts,
                labels,
                train_idx,
                test_idx,
                probe_ks,
                mode=probe_mode,
                reg_type=probe_reg,
            )
            sp_results[int(task["id"])] = per_k

    tpp_results: dict[str, dict[str, Any]] = {}
    for task in tpp_tasks:
        per_class = tpp_indices[task["name"]]
        try:
            res = run_tpp_for_sibling_group(
                sae,
                hidden,
                per_class,
                n_values=tpp_ns,
                device=device,
                samples_per_class=2000,
                sae_batch_size=512,
                probe_train_batch_size=probe_train_batch_size,
            )
            tpp_results[task["name"]] = {
                "category": task["category"],
                "result": res,
            }
        except Exception as exc:  # noqa: BLE001
            tpp_results[task["name"]] = {
                "category": task["category"],
                "error": repr(exc),
            }

    scr_results: dict[str, dict[str, Any]] = {}
    for task in scr_tasks:
        labels = scr_labels[task["name"]]
        t_lab = labels["t"].to(torch.int64)
        s_lab = labels["s"].to(torch.int64)
        try:
            res = run_scr_for_task(
                sae,
                hidden,
                t_lab,
                s_lab,
                n_values=scr_ns,
                device=device,
                samples_per_class=1500,
                sae_batch_size=512,
                probe_train_batch_size=probe_train_batch_size,
            )
            scr_results[task["name"]] = {
                "t_cat": task["t_cat"],
                "s_cat": task["s_cat"],
                "t_op": task["t_op"],
                "s_op": task["s_op"],
                "cell_counts": task["cell_counts"],
                "result": res,
            }
        except Exception as exc:  # noqa: BLE001
            scr_results[task["name"]] = {
                "t_cat": task["t_cat"],
                "s_cat": task["s_cat"],
                "error": repr(exc),
            }

    rec: dict[str, Any] = {
        "sae_name": entry.name,
        "family": entry.family,
        "arch": entry.arch,
        "base_model_repo": entry.base_model_repo,
        "base_model_revision": entry.base_model_revision,
        "ground_truth": gt,
        "sparse_probing": sp_results,
        "tpp": tpp_results,
        "scr": scr_results,
        "probe_ks": probe_ks,
        "tpp_n_values": tpp_ns,
        "scr_n_values": scr_ns,
        "probe_train_batch_size": probe_train_batch_size,
        "probe_mode": probe_mode,
        "probe_reg": probe_reg,
        "elapsed_s": time.time() - t_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    tmp.rename(out_path)

    del sae_acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def run_panel(
    *,
    panel: Iterable[tuple[PanelEntry, SAE[Any]]],
    shared_data_root: Path,
    results_root: Path,
    seed: int,
    probe_ks: list[int],
    tpp_ns: list[int],
    scr_ns: list[int],
    probe_mode: ProbeMode = "sae_bench",
    probe_reg: ProbeReg = "l2",
    probe_train_batch_size: int = 16,
    skip_sp: bool = False,
    device: str = "cuda",
    force: bool = False,
    paper_fixture: str | Path | None = None,
) -> None:
    """Generate (if needed) the shared data and evaluate every SAE in the panel.

    ``paper_fixture`` is forwarded to :func:`generate`; pass ``"v1_seed_1234"``
    (with ``seed=1234`` and the v1 model) to load the same SP/TPP/SCR task
    feature picks as the paper's Section 4 figures.
    """
    panel_list = list(panel)
    if not panel_list:
        return
    first = panel_list[0][0]
    sd = shared_data_dir(
        shared_data_root, first.base_model_repo, first.base_model_revision, seed
    )
    generate(
        first.base_model_repo,
        first.base_model_revision,
        seed,
        sd,
        device=device,
        paper_fixture=paper_fixture,
    )

    out_dir = results_root / f"seed_{seed}" / "per_sae"
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, sae in panel_list:
        run_one_sae(
            entry=entry,
            sae=sae,
            shared_dir=sd,
            out_path=out_dir / f"{entry.name}.json",
            probe_ks=probe_ks,
            tpp_ns=tpp_ns,
            scr_ns=scr_ns,
            probe_mode=probe_mode,
            probe_reg=probe_reg,
            probe_train_batch_size=probe_train_batch_size,
            skip_sp=skip_sp,
            device=device,
            force=force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=V1_REPO)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shared-data-root", default="shared_data", help="Where shared data lives."
    )
    parser.add_argument(
        "--results-root", default="results", help="Where to write JSONs."
    )
    parser.add_argument(
        "--probe-ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--tpp-ns", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 500]
    )
    parser.add_argument(
        "--scr-ns", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 500]
    )
    parser.add_argument(
        "--probe-mode", choices=["sae_bench", "sae_probes"], default="sae_bench"
    )
    parser.add_argument("--probe-reg", choices=["l1", "l2"], default="l2")
    parser.add_argument("--probe-train-batch-size", type=int, default=16)
    parser.add_argument("--skip-sp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--paper-fixture",
        default=None,
        help=(
            "Optional name of a fixture under "
            "saebench_audit/synthetic/paper_fixtures/ (e.g. 'v1_seed_1234') "
            "that pins task feature picks to the paper's Section 4 inputs."
        ),
    )
    args = parser.parse_args()

    sd = shared_data_dir(
        Path(args.shared_data_root),
        args.base_model,
        args.revision,
        args.seed,
    )
    generate(
        args.base_model,
        args.revision,
        args.seed,
        sd,
        device=args.device,
        paper_fixture=args.paper_fixture,
    )
    print(f"Shared data generated at {sd}")
    print(
        "Pass an iterable of (PanelEntry, SAE) pairs to run_panel() to evaluate "
        "your panel; this CLI just prepares the synthetic-task data."
    )


if __name__ == "__main__":
    main()
