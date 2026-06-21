"""Snapshot-evaluation driver (Section 5 of the paper).

Walks a directory tree of SAE training snapshots, loads each SAE from disk,
and runs every SAEBench evaluation on it. Skip-if-exists logic in each eval
makes reruns idempotent and partial progress cheap to resume.

Usage:
    python -m saebench_audit.runners.snapshots \\
        --snapshots-root path/to/snapshots \\
        --output-root results/snapshots

Each snapshot directory is identified by the presence of a ``cfg.json`` at
its root (the convention used by SAELens's ``SAE.save`` method).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sae_lens import SAE

from saebench_audit.runners.eval import Eval
from saebench_audit.runners.reseed import default_evals
from saebench_audit.runners.run_all import run_all_evals
from saebench_audit.runners.sae_compat import patch_sae


def list_snapshot_paths(snapshots_root: Path) -> list[Path]:
    """Return every directory under ``snapshots_root`` that holds a SAELens SAE."""
    return sorted(p.parent for p in snapshots_root.rglob("cfg.json"))


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_snapshot(
    sae_path: Path,
    *,
    output_root: Path,
    shared_dir: Path,
    evals: list[Eval] | None = None,
    random_seed: int = 42,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Load one SAE snapshot from disk and run every eval on it.

    Results are written under ``output_root / <relative-snapshot-path>``.
    """
    eval_list = evals if evals is not None else default_evals()
    sae = SAE.load_from_disk(str(sae_path), device=_resolve_device(device))
    sae = patch_sae(sae)
    run_all_evals(
        sae,
        results_dir=output_root / sae_path.name,
        shared_dir=shared_dir,
        evals=eval_list,
        random_seed=random_seed,
        force=force,
    )


def evaluate_snapshots(
    snapshots_root: Path,
    *,
    output_root: Path,
    shared_dir: Path,
    evals: list[Eval] | None = None,
    random_seed: int = 42,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Evaluate every SAE snapshot under ``snapshots_root``."""
    eval_list = evals if evals is not None else default_evals()
    for sae_path in list_snapshot_paths(snapshots_root):
        rel = sae_path.relative_to(snapshots_root)
        evaluate_snapshot(
            sae_path,
            output_root=output_root / rel.parent,
            shared_dir=shared_dir,
            evals=eval_list,
            random_seed=random_seed,
            device=device,
            force=force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-root", required=True)
    parser.add_argument("--output-root", default="results/snapshots")
    parser.add_argument("--shared-dir", default="results/_shared")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ravel", action="store_true")
    parser.add_argument("--no-autointerp", action="store_true")
    args = parser.parse_args()

    evals = default_evals(
        include_ravel=not args.no_ravel,
        include_autointerp=not args.no_autointerp,
    )
    evaluate_snapshots(
        Path(args.snapshots_root),
        output_root=Path(args.output_root),
        shared_dir=Path(args.shared_dir),
        evals=evals,
        random_seed=args.seed,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
