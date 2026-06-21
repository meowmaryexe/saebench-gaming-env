"""Train a sampled-Matryoshka SAE on Gemma-2-2b layer 12.

The sampled-Matryoshka panel from Section 5 / Appendix C trains four
Matryoshka BatchTopK SAEs ($k=150$) for 300M tokens each, varying only the
number of inner-width prefixes sampled per training step from a
``LogUniform(64, d_sae)`` distribution: ``n in {1, 2, 3, 4}``.

Usage:
    python -m saebench_audit.training.train_sampled_matryoshka --n-levels 1
    python -m saebench_audit.training.train_sampled_matryoshka --n-levels 4

Intermediate SAEs are saved as snapshots every 30M tokens; combined with the
implicit step-1 snapshot this is the 11-snapshot schedule the paper reports for
this panel.
"""

from __future__ import annotations

import argparse
import time

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

TRAINING_TOKENS_DEFAULT = 300_000_000
K_DEFAULT = 150
MIN_MATRYOSHKA_WIDTH_DEFAULT = 64
SNAPSHOT_TOKEN_INTERVAL = 30_000_000


def _snapshot_token_amounts(training_tokens: int) -> list[int]:
    """Token counts at which the sampled-Matryoshka panel saves a snapshot.

    A snapshot every 30M tokens. Combined with the implicit step-1 snapshot
    this is the 11-snapshot schedule the paper reports for the default
    300M-token run.
    """
    return list(
        range(SNAPSHOT_TOKEN_INTERVAL, training_tokens + 1, SNAPSHOT_TOKEN_INTERVAL)
    )


def train_sampled_matryoshka(
    n_levels: int,
    *,
    output_path: str,
    k: int = K_DEFAULT,
    seed: int = 0,
    training_tokens: int = TRAINING_TOKENS_DEFAULT,
    min_matryoshka_width: int = MIN_MATRYOSHKA_WIDTH_DEFAULT,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> TrainingResult:
    """Train one sampled-Matryoshka SAE."""
    matryoshka_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=k,
        level_selection_mode="log_uniform",
        num_sampled_levels=n_levels,
        min_matryoshka_width=min_matryoshka_width,
        use_matryoshka_aux_loss=True,
    )
    override_sae = MatryoshkaBatchTopKTrainingSAE(matryoshka_cfg)

    run_name = (
        f"sampled-matryoshka-n{n_levels}-k{k}-seed{seed}-"
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    cfg = make_runner_config(
        sae_cfg=matryoshka_cfg,
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
    parser.add_argument("--n-levels", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-tokens", type=int, default=TRAINING_TOKENS_DEFAULT)
    parser.add_argument(
        "--min-matryoshka-width", type=int, default=MIN_MATRYOSHKA_WIDTH_DEFAULT
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    train_sampled_matryoshka(
        n_levels=args.n_levels,
        output_path=args.output_path,
        k=args.k,
        seed=args.seed,
        training_tokens=args.training_tokens,
        min_matryoshka_width=args.min_matryoshka_width,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
