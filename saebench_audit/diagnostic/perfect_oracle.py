"""Perfect oracle SAE for the synthetic-validity panel.

The oracle's decoder is the first ``d_sae`` ground-truth feature directions of
a ``SyntheticModel``; its encoder returns the exact ground-truth activations
via a lookup table keyed on a shared hidden pool. By construction the oracle
hits ``GT-MCC = GT-F1 = 1.0``; the SAE-Bench evaluations score it
according to whatever the metric actually measures, which is the point of
including it as a diagnostic.

The lookup is a hack (``hidden -> features`` is not closed-form invertible);
it relies on every eval call passing rows from the same hidden pool, which is
the contract used by the synthetic-task evaluations in this repo. For
ground-truth quality measurement, callers should short-circuit and report
``MCC = F1 = 1.0`` analytically rather than running the oracle on freshly
sampled activations outside the pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sae_lens.synthetic import SyntheticModel


class _FakeCfg:
    """Minimal config shim so SAEBench code paths that read ``.cfg`` work."""

    def __init__(self) -> None:
        self.apply_b_dec_to_input = False

    def to_dict(self) -> dict[str, object]:
        return {"architecture": "perfect_gt", "apply_b_dec_to_input": False}


class PerfectSAE(nn.Module):
    """Oracle SAE with exact GT features for the first ``d_sae`` features.

    Args:
        hidden_pool: ``(N, d_in)`` synthetic activations used as keys.
        features_pool: ``(N, num_gt_features)`` matching ground-truth feature
            firings for those activations.
        gt_feature_vectors: ``(num_gt_features, d_in)`` ground-truth feature
            directions (rows are unit vectors in the synthetic model).
        d_sae: how many ground-truth features to expose as latents.
        device: device to place the SAE on.
    """

    def __init__(
        self,
        hidden_pool: torch.Tensor,
        features_pool: torch.Tensor,
        gt_feature_vectors: torch.Tensor,
        d_sae: int = 4096,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if hidden_pool.shape[0] != features_pool.shape[0]:
            raise ValueError("hidden_pool and features_pool must have same N")
        n_pool, d_in = hidden_pool.shape
        if gt_feature_vectors.shape[1] != d_in:
            raise ValueError("gt_feature_vectors second dim must match hidden d_in")

        self.d_in = d_in
        self.d_sae = d_sae

        W_dec = gt_feature_vectors[:d_sae].detach().clone().to(device)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device), requires_grad=False)
        self.W_enc = nn.Parameter(W_dec.T.clone(), requires_grad=False)
        self.b_enc = nn.Parameter(
            torch.zeros(d_sae, device=device), requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.zeros(d_sae, device=device), requires_grad=False
        )

        self._features = features_pool[:, :d_sae].detach().clone().to(device)

        self._key_n: int = 0
        self._key_to_idx: dict[object, int] = {}
        self._build_lookup_table(hidden_pool, features_pool, d_sae, n_pool)
        self.cfg = _FakeCfg()

    def _build_lookup_table(
        self,
        hidden_pool: torch.Tensor,
        features_pool: torch.Tensor,
        d_sae: int,
        n_pool: int,
    ) -> None:
        """Build a key-to-pool-index dictionary.

        Try short prefixes (4, 16) for speed; fall back to full-row bytes if a
        collision is detected. A *safe* collision (two pool rows with the same
        key but identical features) is allowed and silently merged.
        """
        full_features = features_pool[:, :d_sae].cpu().float().contiguous()
        feat_bytes = [bytes(full_features[i].numpy().tobytes()) for i in range(n_pool)]

        def _try_prefix(prefix: int) -> tuple[dict[object, int] | None, bool]:
            if prefix == 0:
                kt = hidden_pool.cpu().float().contiguous()
            else:
                kt = hidden_pool[:, :prefix].cpu().float().contiguous()
            d: dict[object, int] = {}
            for i in range(n_pool):
                if prefix == 0:
                    k: object = bytes(kt[i].numpy().tobytes())
                else:
                    k = tuple(kt[i].tolist())
                if k in d:
                    if feat_bytes[d[k]] != feat_bytes[i]:
                        return None, False
                    continue
                d[k] = i
            return d, True

        for prefix in (4, 16, 0):
            d, ok = _try_prefix(prefix)
            if ok and d is not None:
                self._key_n = prefix
                self._key_to_idx = d
                return
        raise AssertionError(
            "Full-row key has unsafe collisions; distinct hidden rows with "
            "different ground-truth features map to the same key."
        )

    def _lookup(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        if self._key_n > 0:
            keys_t = x[..., : self._key_n].cpu().float().contiguous()
            flat = keys_t.reshape(-1, self._key_n)
            idx = [self._key_to_idx[tuple(row.tolist())] for row in flat]
        else:
            full = x.cpu().float().contiguous().reshape(-1, x.shape[-1])
            idx = [self._key_to_idx[bytes(row.numpy().tobytes())] for row in full]
        return torch.tensor(
            idx, device=self._features.device, dtype=torch.long
        ).reshape(orig_shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._lookup(x)
        flat_idx = idx.reshape(-1)
        out = self._features[flat_idx]
        return out.reshape(*idx.shape, self.d_sae).to(x.device).to(x.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec


def perfect_oracle(
    model: SyntheticModel,
    hidden_pool: torch.Tensor,
    features_pool: torch.Tensor,
    d_sae: int = 4096,
    device: str = "cuda",
) -> PerfectSAE:
    """Build a ``PerfectSAE`` from a ``SyntheticModel``."""
    gt_vectors = model.feature_dict.feature_vectors.detach().to(device)
    return PerfectSAE(
        hidden_pool=hidden_pool,
        features_pool=features_pool,
        gt_feature_vectors=gt_vectors,
        d_sae=d_sae,
        device=device,
    )
