"""Shared training defaults for the discriminability panels.

The two panels in the paper share most of their hyperparameters; only the
sparsifying activation, training-token count, and matryoshka configuration
differ. This module factors the common values and a small helper that wraps
SAELens's ``LanguageModelSAETrainingRunner`` with snapshot scheduling.

Snapshots are inference-mode copies of the SAE saved at fixed points during
training (see :class:`SnapshotSAETrainer`). They are how the paper studies how
discriminability evolves over a training run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import wandb
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    TrainingSAE,
    TrainingSAEConfig,
    logger,
)
from sae_lens.constants import SPARSITY_FILENAME
from sae_lens.llm_sae_training_runner import LLMSaeEvaluator
from sae_lens.saes.sae import TrainStepOutput
from sae_lens.training.prefetch import PrefetchingIterator
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider
from safetensors.torch import save_file
from tqdm.auto import tqdm

# Paper's canonical training settings (Appendix C "SAE training setup").
GEMMA_2_2B = "google/gemma-2-2b"
LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
D_IN = 2304
D_SAE = 16 * D_IN  # 32_768
LR = 3e-4
BATCH_SIZE = 4096
DATASET_PATH = "monology/pile-uncopyrighted"
CONTEXT_SIZE = 1024


@dataclass
class TrainingResult:
    """Where a training run ended up on disk."""

    final_dir: str
    checkpoints_dir: str | None
    snapshot_dirs: list[str] = field(default_factory=list)


def build_snapshots(
    output_path: str,
    snapshot_token_amounts: list[int],
    *,
    batch_size: int = BATCH_SIZE,
) -> dict[int, str]:
    """Map training steps to the directories their snapshots are written to.

    A snapshot is always taken at step 1 (right after the very first optimiser
    step), then at the step that lands on each requested token count. Directory
    names encode both the step and the token count so snapshots sort
    meaningfully on disk.

    ``snapshot_token_amounts`` must be expressed in tokens; each is converted to
    a step via ``tokens // batch_size``.
    """
    snapshots: dict[int, str] = {1: f"{output_path}/snapshots/step-1-tokens-0"}
    for tokens in snapshot_token_amounts:
        step = tokens // batch_size
        snapshots[step] = f"{output_path}/snapshots/step-{step}-tokens-{tokens}"
    return snapshots


class SnapshotSAETrainer(SAETrainer):  # type: ignore[type-arg]
    """``SAETrainer`` that also writes inference-mode SAE snapshots mid-training.

    A snapshot differs from a SAELens checkpoint: a checkpoint stores resumable
    *training* state (optimizer, LR scheduler, ...), whereas a snapshot stores
    the SAE in the same inference format ``SAE.load_from_disk`` expects, so it
    can be evaluated exactly like a finished SAE.

    ``snapshots`` maps a 1-indexed training step to the directory the snapshot
    for that step is written to.
    """

    def __init__(
        self,
        *args: Any,
        snapshots: Mapping[int, str | Path] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # JSON/CLI plumbing can hand us str keys; coerce so the `in` check
        # against the int ``n_training_steps`` still matches.
        self.snapshots: dict[int, str | Path] = {
            int(step): path for step, path in (snapshots or {}).items()
        }

    @torch.no_grad()
    def _update_pbar(
        self,
        step_output: TrainStepOutput,
        pbar: tqdm,  # type: ignore[type-arg]
        update_interval: int = 100,
    ) -> None:
        # ``SAETrainer.fit`` increments ``n_training_steps`` immediately before
        # calling ``_update_pbar``, so here it holds the just-completed step
        # count -- the point at which a snapshot for that step should be taken.
        # Hooking in here (rather than re-implementing the long ``fit`` method)
        # keeps us robust to upstream changes in SAELens's training loop.
        if self.n_training_steps in self.snapshots:
            self._save_snapshot(self.snapshots[self.n_training_steps])
        super()._update_pbar(step_output, pbar, update_interval)

    def _save_snapshot(self, path: str | Path) -> None:
        """Write the current SAE to ``path`` as an inference-mode snapshot."""
        snapshot_path = Path(path)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        # save_inference_model writes the SAE in the format SAE.load_from_disk
        # expects; we save sparsity.safetensors alongside it so a snapshot
        # directory has the same layout as a finished, evaluatable SAE.
        self.sae.save_inference_model(str(snapshot_path))
        save_file(
            {"sparsity": self.log_feature_sparsity},
            snapshot_path / SPARSITY_FILENAME,
        )


class SnapshotTrainingRunner(LanguageModelSAETrainingRunner):
    """``LanguageModelSAETrainingRunner`` that saves inference-mode snapshots.

    Identical to the stock runner except that it drives training with a
    :class:`SnapshotSAETrainer`, forwarding the ``snapshots`` step→directory map.
    """

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig[TrainingSAEConfig],
        *,
        snapshots: Mapping[int, str | Path] | None = None,
        override_sae: TrainingSAE | None = None,
    ) -> None:
        super().__init__(cfg, override_sae=override_sae)
        self.snapshots: dict[int, str | Path] = {
            int(step): path for step, path in (snapshots or {}).items()
        }

    def run(self) -> TrainingSAE:
        # Adapted from LanguageModelSAETrainingRunner.run; the only change is
        # swapping SAETrainer for SnapshotSAETrainer so mid-training snapshots
        # get written. SAELens offers no hook to inject a custom trainer, so the
        # surrounding setup has to be repeated.
        self._set_sae_metadata()
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = LLMSaeEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
        )

        data_provider: DataProvider = self.activations_store
        if self.cfg.prefetch_llm_batches:
            # Order matters: bool is a subclass of int, so check bool first.
            prefetch_size = (
                1
                if isinstance(self.cfg.prefetch_llm_batches, bool)
                else self.cfg.prefetch_llm_batches
            )
            data_provider = PrefetchingIterator(
                iter(self.activations_store), prefetch=prefetch_size
            )

        trainer = SnapshotSAETrainer(
            sae=self.sae,
            data_provider=data_provider,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            snapshots=self.snapshots,
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            self.sae.load_weights_from_checkpoint(self.cfg.resume_from_checkpoint)
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.output_path is not None:
            self.save_final_sae(
                sae=sae,
                output_path=self.cfg.output_path,
                log_feature_sparsity=trainer.log_feature_sparsity,
            )

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return sae


def _make_logger(
    *,
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> LoggingConfig:
    if wandb_project is None:
        return LoggingConfig(log_to_wandb=False)
    return LoggingConfig(
        log_to_wandb=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity or "",
        wandb_log_frequency=100,
        run_name=run_name,
    )


def make_runner_config(
    sae_cfg: TrainingSAEConfig,
    *,
    training_tokens: int,
    seed: int,
    output_path: str,
    run_name: str,
    n_checkpoints: int = 0,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    """Build the SAELens runner config used by both training panels.

    Mirrors the hyperparameters listed in Appendix C of the paper:

    * Adam optimizer with default $\\beta_1, \\beta_2$ from SAELens, peak
      learning rate $3 \\times 10^{-4}$, no warmup, linear LR decay over the
      final fifth of training.
    * Train batch size 4{,}096 tokens, ``n_batches_in_buffer=256``.
    * The Pile dataset, context length 1024.
    * Mixed-precision (bf16 autocast) for both the LM and the SAE.

    Intermediate SAEs are saved as *snapshots* (see :class:`SnapshotSAETrainer`),
    not SAELens checkpoints, so ``n_checkpoints`` defaults to 0. It is kept as a
    knob only for crash-recovery checkpointing, which is orthogonal to the
    snapshot schedule the panels rely on.
    """
    total_steps = training_tokens // BATCH_SIZE
    lr_decay_steps = total_steps // 5
    return LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=GEMMA_2_2B,
        hook_name=HOOK_NAME,
        dataset_path=DATASET_PATH,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=CONTEXT_SIZE,
        training_tokens=training_tokens,
        device="cuda",
        lr=LR,
        lr_warm_up_steps=0,
        lr_decay_steps=lr_decay_steps,
        n_batches_in_buffer=256,
        train_batch_size_tokens=BATCH_SIZE,
        store_batch_size_prompts=12,
        eval_batch_size_prompts=6,
        autocast=True,
        autocast_lm=True,
        adam_beta1=0.9,
        n_checkpoints=n_checkpoints,
        save_final_checkpoint=True,
        checkpoint_path=f"{output_path}/checkpoints",
        output_path=output_path,
        seed=seed,
        exclude_special_tokens=True,
        logger=_make_logger(
            run_name=run_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        ),
    )


def run_training(
    cfg: LanguageModelSAERunnerConfig[TrainingSAEConfig],
    *,
    snapshots: Mapping[int, str | Path] | None = None,
    override_sae: TrainingSAE | None = None,
) -> TrainingResult:
    """Run a single SAE training and return the on-disk paths.

    ``snapshots`` maps a 1-indexed training step to the directory the
    inference-mode SAE snapshot for that step is written to (see
    :func:`build_snapshots`). When omitted, no snapshots are saved and training
    behaves like the stock SAELens runner.

    ``override_sae`` is forwarded to the runner and is only needed for SAE
    classes that the SAELens config registry does not know how to instantiate
    from a config alone (e.g. our matryoshka SAE before registration).
    """
    runner = SnapshotTrainingRunner(cfg, snapshots=snapshots, override_sae=override_sae)
    runner.run()
    return TrainingResult(
        final_dir=str(cfg.output_path),
        checkpoints_dir=cfg.checkpoint_path,
        snapshot_dirs=sorted(str(path) for path in (snapshots or {}).values()),
    )
