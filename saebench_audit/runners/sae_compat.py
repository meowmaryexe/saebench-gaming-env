"""SAE compatibility shim for SAEBench code paths.

Older SAEBench code paths read ``sae.cfg.hook_name`` / ``sae.cfg.hook_layer``
directly; newer ``sae_lens`` versions move those fields under
``sae.cfg.metadata``. This shim copies them back onto the top-level config so
both code paths work.
"""

from __future__ import annotations

import re
from typing import Any

from sae_lens import SAE


def patch_sae(sae: SAE[Any]) -> SAE[Any]:
    """Attach ``hook_name`` / ``hook_layer`` to ``sae.cfg``.

    No-op if the attributes are already set or if ``cfg.metadata.hook_name``
    is missing.
    """
    hook_name = getattr(getattr(sae.cfg, "metadata", None), "hook_name", None)
    if hook_name is None:
        return sae
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    hook_layer = int(m.group(1)) if m else 0
    try:
        sae.cfg.hook_name = hook_name  # type: ignore[attr-defined]
        sae.cfg.hook_layer = hook_layer  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return sae
