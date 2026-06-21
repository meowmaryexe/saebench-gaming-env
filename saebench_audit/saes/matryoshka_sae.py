"""Matryoshka BatchTopK SAE with optional log-uniform prefix sampling.

This is the SAE architecture used in the paper's discriminability panels
(Section 5 / Appendix C). Two variants are exercised:

  - ``level_selection_mode="fixed"``: a fixed list of matryoshka prefix widths,
    summing one BatchTopK reconstruction loss per prefix at every step. This is
    the cross-architecture panel's Matryoshka variant.
  - ``level_selection_mode="log_uniform"``: at each training step, draws
    ``num_sampled_levels`` widths from a log-uniform distribution between
    ``min_matryoshka_width`` and ``d_sae`` and uses those as that step's
    matryoshka prefixes. This is the sampled-Matryoshka panel.

The reconstruction loss is the sum of BatchTopK reconstruction MSEs over the
selected prefixes (and the full SAE if ``include_outer_loss=True``). The
top-K dead-feature aux loss is split across the same prefixes.

This implementation is deliberately minimal: the paper does not use the
floating-decoder, frequency-sorted, encoder-pinning, or probability-based
prefix-skipping Matryoshka variants, so they are not included here.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import torch
from sae_lens import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.batchtopk_sae import BatchTopK
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput
from typing_extensions import override

from saebench_audit.saes.util import calculate_topk_aux_acts


@dataclass
class MatryoshkaBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """Config for a BatchTopK matryoshka SAE.

    Attributes:
      matryoshka_widths: prefix widths for ``level_selection_mode="fixed"``.
        Must be strictly increasing and end with ``d_sae``; an outer width
        equal to ``d_sae`` is appended automatically if missing.
      include_outer_loss: also include a reconstruction loss for the full
        ``d_sae`` SAE in addition to each prefix.
      skip_final_matryoshka_width: skip the explicit final prefix loss when
        ``include_outer_loss`` already provides the full reconstruction. The
        full reconstruction is then the only loss on the final prefix.
      level_selection_mode: "fixed" or "log_uniform".
      num_sampled_levels: number of widths to draw per step in sampling modes.
      min_matryoshka_width: lower bound on sampled widths.
      use_matryoshka_aux_loss: split the BatchTopK dead-feature aux loss
        across the matryoshka prefixes (rather than one global aux loss).
    """

    matryoshka_widths: list[int] = field(default_factory=list)
    include_outer_loss: bool = True
    skip_final_matryoshka_width: bool = True
    level_selection_mode: str = "fixed"
    num_sampled_levels: int = 1
    min_matryoshka_width: int = 1
    use_matryoshka_aux_loss: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka_batchtopk"


class MatryoshkaBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    cfg: MatryoshkaBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(
        self,
        cfg: MatryoshkaBatchTopKTrainingSAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__(cfg, use_error_term)
        _validate_matryoshka_config(cfg)
        self._current_step_widths: list[int] | None = None

    def _sample_step_widths(self) -> list[int]:
        d_sae = self.cfg.d_sae
        min_w = self.cfg.min_matryoshka_width
        n = self.cfg.num_sampled_levels
        if self.cfg.level_selection_mode == "log_uniform":
            inner_widths = sample_log_uniform_widths(min_w, d_sae, n)
        else:
            raise ValueError(
                f"Unknown level_selection_mode: {self.cfg.level_selection_mode}"
            )
        return inner_widths + [d_sae]

    def _get_step_widths(self) -> list[int]:
        if self.cfg.level_selection_mode == "fixed":
            return self.cfg.matryoshka_widths
        if self._current_step_widths is None:
            self._current_step_widths = self._sample_step_widths()
        return self._current_step_widths

    def _clear_step_widths(self) -> None:
        self._current_step_widths = None

    def iterable_decode(
        self,
        feature_acts: torch.Tensor,
        force_include_outer_loss: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """Iterate over partial SAE reconstructions for each prefix."""
        if self.cfg.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts / self.W_dec.norm(dim=-1)

        decoded = self.b_dec
        prev_portion = 0
        widths = self._get_step_widths()
        if self.cfg.skip_final_matryoshka_width and not force_include_outer_loss:
            widths = widths[:-1]
        for portion in widths:
            if portion > prev_portion:
                current_delta = (
                    feature_acts[:, prev_portion:portion]
                    @ self.W_dec[prev_portion:portion]
                )
                decoded = decoded + current_delta
                prev_portion = portion
            yield decoded

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        assert isinstance(self.activation_fn, BatchTopK)
        self.activation_fn.k = self.cfg.k
        output = BatchTopKTrainingSAE.training_forward_pass(self, step_input)
        # The plain BatchTopK auxiliary loss is replaced by the matryoshka
        # version below; pull it out of the loss dict before we recompute.
        aux_loss = output.losses.pop("auxiliary_reconstruction_loss", None)
        output = self._matryoshka_step(output)
        if aux_loss is not None:
            output.loss = aux_loss + output.loss
            output.losses["auxiliary_reconstruction_loss"] = aux_loss
        output.metrics["k"] = float(self.cfg.k)
        self._clear_step_widths()
        return output

    def _matryoshka_step(self, output: TrainStepOutput) -> TrainStepOutput:
        feature_acts = output.feature_acts
        sae_in = output.sae_in

        if not self.cfg.include_outer_loss:
            output.losses = {}

        inner_recon_losses: list[torch.Tensor] = []

        for partial_sae_out in self.iterable_decode(feature_acts):
            inner_recon_losses.append(
                self.mse_loss_fn(partial_sae_out, sae_in).sum(dim=-1).mean()
            )

        if inner_recon_losses:
            output.losses["inner_recons_loss"] = torch.stack(inner_recon_losses).sum()
        else:
            output.losses["inner_recons_loss"] = output.loss.new_tensor(0.0)

        output.loss = torch.stack(list(output.losses.values())).sum()
        return output

    @override
    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.cfg.use_matryoshka_aux_loss:
            return super().calculate_topk_aux_loss(
                sae_in, sae_out, hidden_pre, dead_neuron_mask
            )
        if dead_neuron_mask is None or int(dead_neuron_mask.sum()) == 0:
            return sae_out.new_tensor(0.0)

        k_aux = sae_in.shape[-1] // 2
        acts = self.activation_fn(hidden_pre)
        scaled_W_dec = (
            self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            if self.cfg.rescale_acts_by_decoder_norm
            else self.W_dec
        )
        prev_portion = 0
        aux_losses: list[torch.Tensor] = []
        step_widths = self._get_step_widths()
        for i, (portion, partial_sae_out) in enumerate(
            zip(
                step_widths,
                self.iterable_decode(acts, force_include_outer_loss=True),
                strict=False,
            )
        ):
            if i == len(step_widths) - 1:
                portion = hidden_pre.shape[-1]
            partial_dead = dead_neuron_mask[prev_portion:portion]
            partial_num_dead = int(partial_dead.sum())
            if partial_num_dead == 0:
                continue
            scale = min(partial_num_dead / k_aux, 1.0)
            partial_k_aux = min(k_aux, partial_num_dead)
            partial_hidden_pre = hidden_pre[:, prev_portion:portion]
            residual = (sae_in - partial_sae_out).detach()
            auxk_acts = calculate_topk_aux_acts(
                k_aux=partial_k_aux,
                hidden_pre=partial_hidden_pre,
                dead_neuron_mask=partial_dead,
            )
            recons = auxk_acts @ scaled_W_dec[prev_portion:portion]
            aux_losses.append(scale * (recons - residual).pow(2).sum(dim=-1).mean())
            prev_portion = portion
        if not aux_losses:
            return sae_out.new_tensor(0.0)
        return self.cfg.aux_loss_coefficient * torch.stack(aux_losses).sum()

    @override
    @torch.no_grad()
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        # No matryoshka-specific buffers to remove in this trimmed
        # implementation, but BatchTopK's parent may need to fold W_dec norms.
        super().process_state_dict_for_saving_inference(state_dict)


def sample_log_uniform_widths(min_w: int, max_w: int, n: int) -> list[int]:
    """Sample widths from ``LogUniform(min_w, max_w)``, excluding ``max_w``.

    ``log(width) ~ Uniform(log(min_w), log(max_w))``. The returned list is
    sorted, deduplicated, and clamped to ``[min_w, max_w - 1]``.
    """
    log_min = math.log(min_w)
    log_max = math.log(max_w)
    u = torch.rand(n)
    log_widths = log_min + u * (log_max - log_min)
    widths = log_widths.exp().int().clamp(min=min_w, max=max_w - 1).tolist()
    return sorted(set(widths))


def _validate_matryoshka_config(cfg: MatryoshkaBatchTopKTrainingSAEConfig) -> None:
    if cfg.skip_final_matryoshka_width and not cfg.include_outer_loss:
        raise ValueError(
            "Cannot skip the final matryoshka width if include_outer_loss is False"
        )
    if cfg.level_selection_mode == "fixed":
        if len(cfg.matryoshka_widths) == 0 or cfg.matryoshka_widths[-1] != cfg.d_sae:
            warnings.warn(
                "matryoshka_widths does not end at d_sae; appending d_sae as "
                "the outermost prefix.",
                stacklevel=2,
            )
            cfg.matryoshka_widths.append(cfg.d_sae)
        for prev, curr in zip(
            cfg.matryoshka_widths[:-1], cfg.matryoshka_widths[1:], strict=False
        ):
            if prev >= curr:
                raise ValueError("matryoshka_widths must be strictly increasing.")
    if cfg.level_selection_mode not in ("fixed", "log_uniform"):
        raise ValueError(
            f"level_selection_mode must be 'fixed' or 'log_uniform', "
            f"got {cfg.level_selection_mode!r}"
        )
    if cfg.num_sampled_levels < 1:
        raise ValueError("num_sampled_levels must be at least 1")
    if cfg.level_selection_mode != "fixed":
        if cfg.min_matryoshka_width < 1:
            raise ValueError("min_matryoshka_width must be at least 1")
        if cfg.min_matryoshka_width >= cfg.d_sae:
            raise ValueError("min_matryoshka_width must be less than d_sae")
