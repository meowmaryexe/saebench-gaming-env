"""Synthetic-validity panel for the paper (Section 4).

Builds hierarchy-aware sparse-probing, TPP, and SCR tasks on top of the
SynthSAEBench-16k ground-truth dictionary and evaluates SAEs against ground
truth feature recovery.
"""

from saebench_audit.synthetic.data_gen import (
    SCRTask,
    SPTask,
    TPPTask,
    generate,
)
from saebench_audit.synthetic.eval_sae_probes import (
    balanced_split,
    encode_in_chunks,
    run_sae_probes_for_task,
)
from saebench_audit.synthetic.eval_scr import run_scr_for_task
from saebench_audit.synthetic.eval_tpp import run_tpp_for_sibling_group

__all__ = [
    "SCRTask",
    "SPTask",
    "TPPTask",
    "balanced_split",
    "encode_in_chunks",
    "generate",
    "run_sae_probes_for_task",
    "run_scr_for_task",
    "run_tpp_for_sibling_group",
]
