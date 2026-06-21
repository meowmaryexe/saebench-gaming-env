"""Train a cross-architecture panel SAE on Gemma-2-2b layer 12.

The cross-architecture panel from Section 5 / Appendix C has four SAEs:
``BatchTopK k in {50, 100}`` and ``Matryoshka BatchTopK k in {50, 100}``,
each trained for 1.5B tokens on the residual stream after layer 12 of
Gemma-2-2b. The Matryoshka variant uses three nested inner widths
``(d_sae/16, d_sae/4, d_sae) = (2048, 8192, 32768)``.

Usage:
    python -m saebench_audit.training.train_cross_arch --variant btk --k 50
    python -m saebench_audit.training.train_cross_arch --variant matryoshka --k 100

This script delegates to SAELens's training runner. Intermediate SAEs are saved
as snapshots on the paper's exact (irregular) schedule -- 28 snapshots, dense
early in training and sparser later.
"""

from __future__ import annotations

import argparse
import time
from typing import Literal

from sae_lens import BatchTopKTrainingSAEConfig

from saebench_audit.saes.matryoshka_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from saebench_audit.training.common import (
    D_IN,
    D_SAE,
    TrainingResult,
    build_snapshots,
    make_runner_config,
    run_training,
)

TRAINING_TOKENS_DEFAULT = 1_500_000_000

Variant = Literal["btk", "matryoshka"]


def _snapshot_token_amounts(training_tokens: int) -> list[int]:
    """Token counts at which the cross-architecture panel saves a snapshot.

    The schedule is dense early in training and sparser later: 10M and 25M
    tokens, then every 50M up to 1B, then every 100M up to 1.5B. Combined with
    the implicit step-1 snapshot this is the 28-snapshot schedule the paper
    reports for this panel.

    Amounts past ``training_tokens`` are dropped so shorter runs (e.g. tests)
    still produce a valid schedule.
    """
    amounts = [10_000_000, 25_000_000]
    amounts += list(range(50_000_000, 1_000_000_001, 50_000_000))
    amounts += list(range(1_100_000_000, 1_500_000_001, 100_000_000))
    return [tokens for tokens in amounts if tokens <= training_tokens]


def train_cross_arch(
    variant: Variant,
    k: int,
    *,
    output_path: str,
    seed: int = 0,
    training_tokens: int = TRAINING_TOKENS_DEFAULT,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> TrainingResult:
    """Train one cross-architecture panel SAE.

    ``variant`` selects the sparsifying activation; ``k`` is the BatchTopK
    activations-per-step count (50 or 100 in the paper).
    """
    if variant == "btk":
        sae_cfg = BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=k)
        override_sae = None
    elif variant == "matryoshka":
        matryoshka_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            k=k,
            matryoshka_widths=[D_SAE // 16, D_SAE // 4, D_SAE],
            level_selection_mode="fixed",
            use_matryoshka_aux_loss=True,
        )
        sae_cfg = matryoshka_cfg
        override_sae = MatryoshkaBatchTopKTrainingSAE(matryoshka_cfg)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    run_name = (
        f"{variant}-cross-arch-k-{k}-seed-{seed}-{time.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    cfg = make_runner_config(
        sae_cfg=sae_cfg,
        training_tokens=training_tokens,
        seed=seed,
        output_path=output_path,
        run_name=run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    snapshots = build_snapshots(output_path, _snapshot_token_amounts(training_tokens))
    return run_training(cfg, snapshots=snapshots, override_sae=override_sae)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["btk", "matryoshka"], required=True)
    parser.add_argument("--k", type=int, required=True, choices=[50, 100])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-tokens", type=int, default=TRAINING_TOKENS_DEFAULT)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    train_cross_arch(
        variant=args.variant,
        k=args.k,
        output_path=args.output_path,
        seed=args.seed,
        training_tokens=args.training_tokens,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
