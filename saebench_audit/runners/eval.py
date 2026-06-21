"""SAEBench eval wrappers used by the reseed and snapshot drivers.

Each wrapper hides SAEBench's per-eval boilerplate behind a uniform interface:

* ``has_eval_run(results_dir)`` — True if the wrapper's expected output JSON
  is already present (used to make reruns idempotent).
* ``run(sae, results_dir, shared_dir, random_seed)`` — actually run the eval,
  writing results under ``results_dir`` and using ``shared_dir / "artifacts"``
  for any cross-SAE cached artefacts. The SAEBench eval configs all accept a
  ``random_seed`` field; reseed-noise runs vary it per call.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval as run_autointerp_eval
from sae_bench.evals.core.main import multiple_evals
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.main import run_eval as run_ravel_eval
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import run_eval as run_scr_tpp_eval
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.sparse_probing.main import run_eval as run_sparse_probing_eval
from sae_lens import SAE
from sae_probes.constants import Setting
from sae_probes.generate_model_activations import generate_dataset_activations
from sae_probes.run_sae_evals import DATASETS as _SAE_PROBES_DATASETS
from sae_probes.run_sae_evals import run_sae_evals as run_sae_probes_evals


def _glob_matches(base_dir: Path, pattern: str) -> bool:
    return len(list(base_dir.glob(pattern))) > 0


def _try_fold_W_dec_norm(sae: SAE[Any]) -> None:
    """Best-effort fold of the decoder norms into the encoder.

    Some SAE classes don't implement folding; we silently skip in that case.
    """
    try:
        sae.fold_W_dec_norm()
    except NotImplementedError:
        pass


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Eval(ABC):
    """Common interface for the SAEBench eval wrappers."""

    @abstractmethod
    def has_eval_run(self, results_dir: Path) -> bool: ...

    @abstractmethod
    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None: ...


# ---------- sparse probing -----------------------------------------------


@dataclass
class SparseProbingOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128
    k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10])


class SparseProbingEval(Eval):
    def __init__(self, options: SparseProbingOptions | None = None) -> None:
        self.options = options or SparseProbingOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir,
            "**/saebench_sparse_probing_custom_sae_eval_results.json",
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        _try_fold_W_dec_norm(sae)
        cfg = SparseProbingEvalConfig(
            random_seed=random_seed,
            model_name=sae.cfg.metadata.model_name,
            llm_dtype=self.options.llm_dtype,
            llm_batch_size=self.options.llm_batch_size,
            k_values=self.options.k_values,
        )
        run_sparse_probing_eval(
            config=cfg,
            device=_resolve_device(self.options.device),
            output_path=str(results_dir),
            selected_saes=[("saebench_sparse_probing", sae)],
            artifacts_path=str(shared_dir / "artifacts"),
        )


# ---------- TPP / SCR ----------------------------------------------------


@dataclass
class ScrAndTppOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128
    sae_batch_size: int = 32
    n_values: list[int] = field(default_factory=lambda: [2, 5, 10, 20, 50, 100, 500])


class TPPEval(Eval):
    def __init__(self, options: ScrAndTppOptions | None = None) -> None:
        self.options = options or ScrAndTppOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir, "**/saebench_tpp_custom_sae_eval_results.json"
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        _try_fold_W_dec_norm(sae)
        cfg = ScrAndTppEvalConfig(
            random_seed=random_seed,
            model_name=sae.cfg.metadata.model_name,
            llm_dtype=self.options.llm_dtype,
            llm_batch_size=self.options.llm_batch_size,
            perform_scr=False,
            n_values=self.options.n_values,
            sae_batch_size=self.options.sae_batch_size,
        )
        run_scr_tpp_eval(
            config=cfg,
            device=_resolve_device(self.options.device),
            output_path=str(results_dir),
            selected_saes=[("saebench_tpp", sae)],
            artifacts_path=str(shared_dir / "artifacts"),
        )


class SCREval(Eval):
    def __init__(self, options: ScrAndTppOptions | None = None) -> None:
        self.options = options or ScrAndTppOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir, "**/saebench_scr_custom_sae_eval_results.json"
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        _try_fold_W_dec_norm(sae)
        cfg = ScrAndTppEvalConfig(
            random_seed=random_seed,
            model_name=sae.cfg.metadata.model_name,
            llm_dtype=self.options.llm_dtype,
            llm_batch_size=self.options.llm_batch_size,
            perform_scr=True,
            n_values=self.options.n_values,
            sae_batch_size=self.options.sae_batch_size,
        )
        run_scr_tpp_eval(
            config=cfg,
            device=_resolve_device(self.options.device),
            output_path=str(results_dir),
            selected_saes=[("saebench_scr", sae)],
            artifacts_path=str(shared_dir / "artifacts"),
        )


# ---------- autointerp ---------------------------------------------------


@dataclass
class AutointerpOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int = 128
    api_key: str | None = None


class AutointerpEval(Eval):
    def __init__(self, options: AutointerpOptions | None = None) -> None:
        self.options = options or AutointerpOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir, "**/saebench_autointerp_custom_sae_eval_results.json"
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        api_key = self.options.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY is not set; pass options.api_key or set the env var."
            )
        _try_fold_W_dec_norm(sae)
        cfg = AutoInterpEvalConfig(
            random_seed=random_seed,
            model_name=sae.cfg.metadata.model_name,
            llm_dtype=self.options.llm_dtype,
            llm_batch_size=self.options.llm_batch_size,
        )
        run_autointerp_eval(
            config=cfg,
            device=_resolve_device(self.options.device),
            output_path=str(results_dir),
            selected_saes=[("saebench_autointerp", sae)],
            api_key=api_key,
            artifacts_path=str(shared_dir / "artifacts"),
        )


# ---------- RAVEL --------------------------------------------------------


@dataclass
class RAVELOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    llm_batch_size: int | None = None


class RAVELEval(Eval):
    def __init__(self, options: RAVELOptions | None = None) -> None:
        self.options = options or RAVELOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir, "**/saebench_ravel_custom_sae_eval_results.json"
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        _try_fold_W_dec_norm(sae)
        cfg = RAVELEvalConfig(
            random_seed=random_seed,
            model_name=sae.cfg.metadata.model_name,
            llm_dtype=self.options.llm_dtype,
            artifact_dir=str(shared_dir / "artifacts" / "ravel_shared"),
        )
        if self.options.llm_batch_size is not None:
            cfg.llm_batch_size = self.options.llm_batch_size
        run_ravel_eval(
            config=cfg,
            selected_saes=[("saebench_ravel", sae)],
            device=_resolve_device(self.options.device),
            output_path=str(results_dir),
            artifacts_path=str(shared_dir / "artifacts"),
        )


# ---------- sae-probes ---------------------------------------------------


@dataclass
class SAEProbesOptions:
    device: str | None = None
    llm_batch_size: int = 64
    settings: list[Setting] = field(default_factory=lambda: ["normal"])
    ks: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 16])
    generate_model_activations: bool = False


class SAEProbesEval(Eval):
    """sae-probes-style sparse probing (113 datasets, L1+CV, AUC/F1/acc)."""

    def __init__(self, options: SAEProbesOptions | None = None) -> None:
        self.options = options or SAEProbesOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        for setting in self.options.settings:
            existing = list(results_dir.glob(f"sae_probes/*/{setting}_setting/*.json"))
            if len(existing) < len(_SAE_PROBES_DATASETS):
                return False
        return True

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,
        random_seed: int = 42,
    ) -> None:
        _try_fold_W_dec_norm(sae)
        device = _resolve_device(self.options.device)
        model_name = sae.cfg.metadata.model_name
        hook_name = sae.cfg.metadata.hook_name
        model_cache_path = shared_dir / "artifacts" / "sae_probes" / "model_cache"
        if self.options.generate_model_activations:
            generate_dataset_activations(
                model_name=model_name,
                hook_names=[hook_name],
                batch_size=self.options.llm_batch_size,
                device=device,
                model_cache_path=model_cache_path,
            )
        for setting in self.options.settings:
            run_sae_probes_evals(
                sae=sae,
                model_name=model_name,
                hook_name=hook_name,
                reg_type="l1",
                setting=setting,
                results_path=results_dir / "sae_probes",
                model_cache_path=model_cache_path,
                ks=self.options.ks,
                seed=random_seed,
            )


# ---------- core ---------------------------------------------------------


@dataclass
class CoreOptions:
    device: str | None = None
    llm_dtype: str = "float32"
    batch_size_prompts: int = 16
    n_eval_reconstruction_batches: int = 10
    n_eval_sparsity_variance_batches: int = 1
    dataset: str = "Skylion007/openwebtext"
    context_size: int = 128


class CoreEval(Eval):
    """SAEBench's "core" reconstruction / sparsity / KL / CE evaluation."""

    def __init__(self, options: CoreOptions | None = None) -> None:
        self.options = options or CoreOptions()

    def has_eval_run(self, results_dir: Path) -> bool:
        return _glob_matches(
            results_dir, "**/saebench_core_custom_sae_eval_results.json"
        )

    def run(
        self,
        sae: SAE[Any],
        results_dir: Path,
        shared_dir: Path,  # noqa: ARG002
        random_seed: int = 42,  # noqa: ARG002 — core eval has no seed knob
    ) -> None:
        _try_fold_W_dec_norm(sae)
        multiple_evals(
            selected_saes=[("saebench_core", sae)],
            n_eval_reconstruction_batches=self.options.n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=self.options.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=self.options.batch_size_prompts,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            exclude_special_tokens_from_reconstruction=True,
            dataset=self.options.dataset,
            context_size=self.options.context_size,
            output_folder=str(results_dir),
            verbose=False,
            dtype=self.options.llm_dtype,
            device=_resolve_device(self.options.device),
        )
