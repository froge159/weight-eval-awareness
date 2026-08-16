"""Random-direction control for Phase 3.

Proposal sec 3.4, criterion 3: a random unit direction of equal norm, edited the
same way, must leave the gap open. Without it, gap closure under the real edit
is uninterpretable -- it could just mean any sufficiently large weight edit
degrades the behaviour.

The control has to be *matched* to the treatment, which for this pipeline means
matched per block. Phase 2 does not apply one shared direction; it assigns each
of the 41 blocks in the 36-76 band its own vector (PHASE3_HANDOFF sec 1). A
single shared random vector would therefore be a weaker perturbation than the
real edit and would fail for the wrong reason -- a reviewer would be right to
say the control was not comparable. So: 41 independent random unit vectors, one
per block, over the same 54 modules.

Seeded, so the control is reproducible and can be re-run as a family (several
seeds) if one draw happens to land somewhere unlucky.

Usage::

    from steering.directions import load_direction_set, assign_blocks
    from steering.surgery import EditSpec
    from control_directions import randomize_plan

    spec = EditSpec(band=(36, 76), layer_convention="write_target")
    real = assign_blocks(load_direction_set(), spec.band, n_blocks=80,
                         convention=spec.layer_convention)
    plan = randomize_plan(real, seed=0)      # same shape, random vectors
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def random_unit(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random unit vector in R^dim.

    Gaussian then normalise: the normal distribution is spherically symmetric,
    so this is uniform on the sphere. float64 to match the real directions --
    surgery asserts on dtype in places.
    """
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float64)


def randomize_plan(plan: Any, seed: int = 0, tag: str | None = None) -> Any:
    """Copy an edit plan, replacing every direction with a random unit vector.

    Everything structural is preserved -- the same blocks, the same band, the
    same convention -- so the edit touches exactly the same 54 matrices with
    exactly the same procedure. Only the directions change.

    A fresh vector is drawn per block from one seeded generator, so the draws
    are independent across blocks (matching the real plan, whose per-block
    directions differ) but the whole plan is reproducible from ``seed``.
    """
    blocks = list(plan.blocks)
    if not blocks:
        raise ValueError("plan has no blocks to randomise")

    rng = np.random.default_rng(seed)
    dim = int(plan.for_block(blocks[0]).shape[0])

    vectors = {b: random_unit(dim, rng) for b in blocks}

    # Sanity: a random direction in 8192 dimensions should be near-orthogonal to
    # the real one. If it is not, the draw is not doing what we think.
    cosines = [
        float(abs(vectors[b] @ plan.for_block(b))) for b in blocks
    ]
    max_cos = max(cosines)
    if max_cos > 0.2:
        raise AssertionError(
            f"a random direction landed at |cos| = {max_cos:.3f} against the real "
            f"one, which should be near-impossible in {dim} dimensions. Check the "
            f"vectors are what you think they are."
        )

    new = dataclasses.replace(
        plan,
        vectors=vectors,
        key=f"random_seed{seed}" + (f"_{tag}" if tag else ""),
    )

    # Norms must match the real plan's: the proposal specifies a random direction
    # "of equal norm". Both are unit, but assert rather than assume.
    for b in blocks:
        n = float(np.linalg.norm(new.for_block(b)))
        if abs(n - 1.0) > 1e-9:
            raise AssertionError(f"block {b}: random direction has norm {n}, not 1")

    return new


def plan_summary(plan: Any) -> dict:
    """Numbers worth recording in the manifest alongside the run."""
    blocks = list(plan.blocks)
    return {
        "key": getattr(plan, "key", "?"),
        "band": tuple(getattr(plan, "band", ())),
        "convention": getattr(plan, "convention", "?"),
        "n_blocks": len(blocks),
        "block_range": (blocks[0], blocks[-1]) if blocks else None,
        "dim": int(plan.for_block(blocks[0]).shape[0]) if blocks else None,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/workspace/weight-eval-awareness/src")
    from steering.directions import assign_blocks, load_direction_set
    from steering.surgery import EditSpec

    spec = EditSpec(band=(36, 76), layer_convention="write_target")
    real = assign_blocks(load_direction_set(), spec.band, n_blocks=80,
                         convention=spec.layer_convention)
    rand = randomize_plan(real, seed=0)

    print("real:  ", plan_summary(real))
    print("random:", plan_summary(rand))

    bs = list(real.blocks)
    cos = [float(abs(rand.for_block(b) @ real.for_block(b))) for b in bs]
    print(f"\n|cos(random, real)| over {len(bs)} blocks: "
          f"max {max(cos):.4f}, mean {sum(cos)/len(cos):.4f}")
    print("(near zero is correct -- random directions are near-orthogonal in high dim)")

    # Two seeds must differ, or the 'family of controls' idea is broken.
    other = randomize_plan(real, seed=1)
    d = float(np.linalg.norm(rand.for_block(bs[0]) - other.for_block(bs[0])))
    print(f"seed 0 vs seed 1, block {bs[0]}: L2 distance {d:.4f} (should be ~1.4)")
    print("\nOK")
