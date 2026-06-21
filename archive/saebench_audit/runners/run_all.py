"""Run a list of SAEBench evaluations on one SAE.

The reseed-noise (\\S 3) and snapshot (\\S 5) runners both share this loop:
take an SAE, run a list of ``Eval`` wrappers, write each eval's result to
``results_dir``, and skip evaluations whose expected output JSON already
exists. Errors in one eval don't abort the others by default.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE

from saebench_audit.runners.eval import Eval


def run_all_evals(
    sae: SAE[Any],
    *,
    results_dir: Path,
    shared_dir: Path,
    evals: list[Eval],
    random_seed: int = 42,
    force: bool = False,
    crash_on_error: bool = False,
) -> None:
    """Run every eval in ``evals``, skipping those whose results already exist.

    ``shared_dir`` is the directory under which evals are allowed to cache
    cross-SAE artefacts (model activations, RAVEL templates, etc.) — typically
    the same path is reused across all SAEs in a panel.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    for eval_obj in evals:
        torch.set_grad_enabled(True)
        if eval_obj.has_eval_run(results_dir) and not force:
            continue
        try:
            eval_obj.run(sae, results_dir, shared_dir, random_seed=random_seed)
        except Exception:
            if crash_on_error:
                raise
            print(
                f"[run_all_evals] {eval_obj.__class__.__name__} failed:\n"
                f"{traceback.format_exc()}"
            )
