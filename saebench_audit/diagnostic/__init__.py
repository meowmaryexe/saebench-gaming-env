"""Diagnostic SAE constructors used in the paper's synthetic-validity panel.

The paper introduces four degraded controls and one perfect oracle (\\S 4.2):

* ``best_512`` — keeps only the 512 most useful latents.
* ``random_init`` — randomly initialised weights with a fixed threshold.
* ``random_l0_matched`` — random init, threshold tuned so calibration L0
  matches a target.
* ``permuted_decoder`` — shuffles the rows of ``W_dec``; encoder unchanged.
* ``perfect_oracle`` — decoder is the first ``d_sae`` ground-truth feature
  directions; encoder returns exact GT feature activations (lookup-based).
"""

from saebench_audit.diagnostic.perfect_oracle import PerfectSAE, perfect_oracle

__all__ = ["PerfectSAE", "perfect_oracle"]
