"""Reseed-noise driver (Section 3 of the paper).

For one SAE, run each SAEBench evaluation ``len(seeds)`` times, varying the
``random_seed`` field of the SAEBench eval config each time. Each run writes
its results to a seed-specific subdirectory so SAEBench's skip-if-exists
logic doesn't short-circuit later seeds.

Usage:
    python -m saebench_audit.runners.reseed \\
        --sae-release gemma-scope-2b-pt-res-canonical \\
        --sae-id layer_12/width_65k/canonical \\
        --output-dir results/reseed/65k

The default seed list ``[42, 123, 456, 789, 2024]`` matches Appendix A.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sae_lens import SAE

from saebench_audit.runners.eval import (
    AutointerpEval,
    AutointerpOptions,
    CoreEval,
    CoreOptions,
    Eval,
    RAVELEval,
    RAVELOptions,
    SAEProbesEval,
    SAEProbesOptions,
    ScrAndTppOptions,
    SCREval,
    SparseProbingEval,
    SparseProbingOptions,
    TPPEval,
)
from saebench_audit.runners.run_all import run_all_evals
from saebench_audit.runners.sae_compat import patch_sae

DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 2024]


def default_evals(
    *,
    include_ravel: bool = True,
    include_autointerp: bool = True,
) -> list[Eval]:
    """Default list of SAEBench evals used for the Section-3 reseed sweep.

    ``include_ravel`` should be False for non-Gemma SAEs; the SAEBench RAVEL
    wrapper hard-codes Gemma-2-2b datasets. ``include_autointerp`` should be
    False if no ``OPENAI_API_KEY`` is available.
    """
    evals: list[Eval] = [
        CoreEval(CoreOptions()),
        SparseProbingEval(SparseProbingOptions()),
        SAEProbesEval(SAEProbesOptions(ks=[1, 2, 5, 10, 16])),
        SCREval(ScrAndTppOptions()),
        TPPEval(ScrAndTppOptions()),
    ]
    if include_autointerp:
        evals.append(AutointerpEval(AutointerpOptions()))
    if include_ravel:
        evals.append(RAVELEval(RAVELOptions()))
    return evals


def reseed_for_sae(
    sae: SAE[Any],
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    output_dir: Path,
    shared_dir: Path,
    evals: list[Eval] | None = None,
    force: bool = False,
) -> None:
    """Run each eval once per seed in ``seeds``.

    Output layout: ``output_dir / "seed_<seed>" / <SAEBench eval JSON>``.
    """
    sae = patch_sae(sae)
    eval_list = evals if evals is not None else default_evals()
    for seed in seeds:
        run_all_evals(
            sae,
            results_dir=output_dir / f"seed_{seed}",
            shared_dir=shared_dir,
            evals=eval_list,
            random_seed=seed,
            force=force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--output-dir", default="results/reseed")
    parser.add_argument("--shared-dir", default="results/_shared")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ravel", action="store_true")
    parser.add_argument("--no-autointerp", action="store_true")
    args = parser.parse_args()

    sae, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    evals = default_evals(
        include_ravel=not args.no_ravel,
        include_autointerp=not args.no_autointerp,
    )
    reseed_for_sae(
        sae,
        seeds=args.seeds,
        output_dir=Path(args.output_dir),
        shared_dir=Path(args.shared_dir),
        evals=evals,
        force=args.force,
    )


if __name__ == "__main__":
    main()
