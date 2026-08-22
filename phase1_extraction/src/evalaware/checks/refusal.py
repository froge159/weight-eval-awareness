"""Step 3b: refusal-similarity check.  DEFERRED -- see RESEARCH_DECISIONS.md.

This is the diagnostic for the proposal's own stated principal threat to
validity: that r_hat is entangled with the general refusal / safety mechanism
rather than with evaluation awareness specifically.

Deferred by decision on 2026-08-08 so that steps 1, 2 and 3a can proceed. It is
gated, not dropped: ``run_phase1.py`` refuses to report Phase 1 complete unless
``--allow-deferred-refusal`` is passed explicitly.

Requirements when implemented -- all three are load-bearing:

1. r_hat_refusal must come from the **same model** as r_hat. A direction
   borrowed from a different model gives a cosine with no interpretation.
2. It must use the **same pooling and the same code path** (``extract_direction``
   on a contrastive matrix). Mixing a diff-of-means refusal direction with an
   SVD awareness direction confounds estimator with concept.
3. The cosine must be judged against ``random_pair_cosine_baseline(d)``, never
   against zero. In ~8k dimensions a random cosine has |cos| ~ 0.01, so 0.05 is
   large, not small.

Harmful/harmless pairs should follow Arditi et al. 2024 -- matched instruction
pairs differing only in harmfulness, e.g. AdvBench harmful instructions paired
with Alpaca harmless ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..validation import check, cosine, random_pair_cosine_baseline

__all__ = ["RefusalPairSpec", "RefusalSimilarityResult", "run_refusal_check", "interpret_cosine"]


@dataclass(frozen=True)
class RefusalPairSpec:
    """Schema for one harmful/harmless contrastive pair.

    Defined now so the dataset can be assembled in parallel with the rest of
    Phase 1 even though the check itself is not implemented.
    """

    pair_id: str
    harmful_instruction: str
    harmless_instruction: str
    source: str


@dataclass
class RefusalSimilarityResult:
    layer: int
    pooling: str
    cos_awareness_refusal: float
    random_pair_baseline: dict[str, float]
    ratio_to_p99: float
    verdict: str


def interpret_cosine(cos_value: float, dim: int, seed: int = 0) -> RefusalSimilarityResult:
    """Judge an observed cosine against the random-pair baseline.

    Usable now for interpreting a cosine computed by hand, before the full
    check exists.
    """
    baseline = random_pair_cosine_baseline(dim, seed=seed)
    ratio = abs(cos_value) / baseline["abs_p99"] if baseline["abs_p99"] > 0 else float("inf")
    if ratio < 1.0:
        verdict = "indistinguishable from random (directions are unrelated)"
    elif ratio < 5.0:
        verdict = "weakly above chance -- report, but unlikely to be the same mechanism"
    elif ratio < 20.0:
        verdict = "clearly above chance -- partial entanglement with refusal, investigate"
    else:
        verdict = "strongly aligned -- r_hat may be a refusal direction; do NOT proceed to surgery"
    return RefusalSimilarityResult(
        layer=-1,
        pooling="n/a",
        cos_awareness_refusal=float(cos_value),
        random_pair_baseline=baseline,
        ratio_to_p99=float(ratio),
        verdict=verdict,
    )


def run_refusal_check(
    r_hat: np.ndarray,
    harmful_activations: np.ndarray,
    harmless_activations: np.ndarray,
    pair_ids: Sequence[str],
    layer: int,
    pooling: str,
) -> RefusalSimilarityResult:
    """NOT IMPLEMENTED -- deferred by decision 2026-08-08."""
    raise NotImplementedError(
        "Step 3b (refusal similarity) is deferred -- see RESEARCH_DECISIONS.md, "
        "entry 2026-08-08T12:45Z.\n"
        "\n"
        "To implement:\n"
        "  1. Build harmful/harmless pairs as RefusalPairSpec (Arditi et al. 2024 style:\n"
        "     AdvBench harmful x Alpaca harmless, matched on instruction form).\n"
        "  2. Capture activations with the SAME model, layer and pooling as r_hat,\n"
        "     using activations.capture().\n"
        "  3. r_hat_refusal = extraction.extract_direction(\n"
        "         contrastive_matrix(H_harmful, H_harmless, ids, ids), layer, pooling\n"
        "     ).r_hat\n"
        "  4. Report cosine(r_hat, r_hat_refusal) via interpret_cosine(cos, dim).\n"
        "\n"
        "This check must pass before Phase 2 surgery: it is the stated diagnostic for\n"
        "the proposal's principal threat to validity (entanglement with refusal)."
    )
