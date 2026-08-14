"""Phase 2 -- weight-space orthogonalization against the evaluation-awareness direction.

Implements proposal 3.3: every projection that writes into the residual stream
(``o_proj`` in attention blocks, ``down_proj`` in MLPs) has its output
orthogonalized against r-hat::

    W'_out = W_out - r_hat r_hat^T W_out

Modules
-------
directions    Loads the Phase 1 per-layer r-hat set, builds the cross-layer
              cosine matrix that justifies applying an edit across depth, and
              resolves which block gets which direction.
surgery       LoRA merge, module enumeration, and the rank-1 weight edit.
fidelity      Confirms the weight edit implements the intended ablation.

Phase 1 lives in a sibling tree that is not installed as a package, so importing
``steering`` puts it on ``sys.path`` -- the same thing
``src/extraction/scripts/run_step1.py`` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALAWARE_SRC = REPO / "src" / "extraction" / "src"

if str(EVALAWARE_SRC) not in sys.path:
    sys.path.insert(0, str(EVALAWARE_SRC))

__all__ = ["REPO", "EVALAWARE_SRC"]
