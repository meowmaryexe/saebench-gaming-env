"""Synthetic-validity panel for the paper (Section 4)."""

from saebench_audit.synthetic.data_gen import SCRTask, SPTask, TPPTask, generate
from saebench_audit.synthetic.eval_scr import run_scr_for_task

__all__ = ["SCRTask", "SPTask", "TPPTask", "generate", "run_scr_for_task"]
