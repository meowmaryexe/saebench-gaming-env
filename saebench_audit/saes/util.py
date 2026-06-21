"""Helpers shared by the matryoshka SAE."""

from __future__ import annotations

import torch


def calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """Pick top-k_aux activations from the dead-neuron set.

    Returns a tensor with the same shape as ``hidden_pre``, zero everywhere
    except at the top-``k_aux`` dead latent positions, where it carries the
    pre-activation values.
    """
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    return auxk_acts
