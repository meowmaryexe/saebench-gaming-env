# Kimi Delta Attention — FLA module brief

Source: `fla-org/flash-linear-attention/fla/ops/kda/` (commit pinned at the
date this bundle was assembled; the directory was lifted verbatim with the
two relevant kernel files: `fused_recurrent.py` and `chunk.py`).

## What KDA is

Kimi Delta Attention is the linear-attention variant used in Moonshot's
Kimi K2 family. It's a recurrent state-update of the form

    S_t = diag(a_t) * S_{t-1} - beta_t * (S_{t-1} @ k_t - v_t) outer k_t^T
    o_t = S_t @ q_t

where `S` is a per-head state matrix of shape `(K, V)`, `a` is a per-channel
decay/gate, and `beta` is a scalar (or headwise) update step size. The
state-update fuses (i) outer-product downdate against the prediction error
`S_{t-1} @ k_t - v_t`, (ii) diagonal decay through `a`, and (iii) output
projection.

## Two algorithm choices in FLA

- **`fused_recurrent_kda`** — token-by-token recurrence, one Triton block per
  `(sequence, head, K-tile, V-tile)`. Inherently sequential along the T axis;
  parallelism comes only from `(N * HV * NK * NV)`. This is the path for
  decoding (T=1) and short tails.

- **`chunk_kda`** — chunked algorithm that breaks the sequence into chunks
  and parallelizes within each chunk. Adds chunk-level grid parallelism
  (more blocks per problem) at the cost of extra reduction work between
  chunks. Standard prefill path.

The `bench.py` in this directory exercises `fused_recurrent_kda` only.

## Grid analysis

For `fused_recurrent_kda_fwd_kernel` with `q.shape = (B, T, H, K)`,
`v.shape = (B, T, HV, V)` and the default heuristic `BK = min(64, K),
BV = min(64, V)`:

    grid_blocks = ceil(K/BK) * ceil(V/BV) * B * HV
                = NK         * NV         * N * HV

`T` does NOT appear — the recurrence forces serial execution along the
sequence dimension. For decode shapes (`B=1`, `HV=O(10)`, `K=V=128`,
`BK=BV=64`) the grid collapses to `2 * 2 * 1 * 16 = 64`, which is what
`grid_estimate.txt` reports.

## What this kernel is good at

- Long sequences at large batch where `B * HV` saturates the device.
- Per-head-tile parallel work that's heavy enough to amortize launch.

## What it is NOT good at

The decode-time grid is small (the `grid_estimate.txt` math), and each
recurrent step does relatively little arithmetic per block. The per-call
breakdown surfaced by `profile_table.txt` and `bench.json` is what you'd
look at to understand whether the bottleneck is on-device occupancy, the
launch path, or something else.
