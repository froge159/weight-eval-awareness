"""Step 3a tests.

The point of this suite is that the check must *discriminate*, not merely run.
So each scenario plants a direction with known properties and asserts the
verdict matches:

  genuine direction   -> passes, beats null, transfers across regimes
  random direction    -> fails
  sign-flipped        -> detected, and separation still recovered
  cue-specific        -> passes pooled, FAILS cross-regime transfer

That last one is the one that matters. A direction encoding only the surface
form of one cue will look fine in a pooled AUROC and is exactly what the
proposal's cross-cue requirement exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalaware.checks.ood import auroc, random_direction_null, run_ood_check  # noqa: E402
from evalaware.direction_io import load_directions                            # noqa: E402
from evalaware.validation import ValidationError                              # noqa: E402

DIM = 128
N_Q = 24
TPL = 4                       # templates per question per regime


def build_holdout(effect_explicit=3.0, effect_woodlabs=3.0, noise=1.0, seed=0):
    """Synthesize a held-out split with a planted direction per regime.

    Returns (h_eval, h_deploy, question_ids, regimes, direction).
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(DIM)
    direction /= np.linalg.norm(direction)

    qs, regs, he, hd = [], [], [], []
    for q in range(N_Q):
        content = rng.standard_normal(DIM) * 3.0        # shared per question
        for regime, effect in (("explicit", effect_explicit), ("woodlabs", effect_woodlabs)):
            for _ in range(TPL):
                base = content + rng.standard_normal(DIM) * 0.5
                hd.append(base + rng.standard_normal(DIM) * noise)
                he.append(base + rng.standard_normal(DIM) * noise + effect * direction)
                qs.append(f"Q{q}")
                regs.append(regime)
    return np.array(he), np.array(hd), qs, regs, direction


# ------------------------------------------------------------------ auroc

def test_auroc_perfect_and_chance():
    assert auroc(np.array([3.0, 4, 5]), np.array([0.0, 1, 2])) == 1.0
    assert auroc(np.array([0.0, 1, 2]), np.array([3.0, 4, 5])) == 0.0


def test_auroc_ties_count_half():
    assert auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


# ------------------------------------------------------------------ null

def test_random_null_is_near_chance_but_measured():
    he, hd, _, _, _ = build_holdout(effect_explicit=0.0, effect_woodlabs=0.0)
    mean, p95 = random_direction_null(he, hd, n_directions=100, seed=0)
    # Folded, so >= 0.5 by construction; with no planted effect it should stay low.
    assert 0.5 <= mean < 0.75, mean
    assert p95 >= mean


# --------------------------------------------------------- genuine direction

def test_genuine_direction_passes_everything():
    he, hd, qs, regs, d = build_holdout()
    res = run_ood_check(d, he, hd, qs, regs, layer=40, n_boot=300, n_null=60, seed=0)

    assert res.pooled.auroc > 0.7, res.pooled.summary()
    assert res.pooled.paired_win_rate > 0.95, res.pooled.summary()
    assert res.pooled.separates
    assert res.beats_null
    assert res.transfers_across_regimes
    assert res.passes
    assert not res.sign_flipped


def test_paired_statistic_is_far_more_sensitive_than_unpaired():
    """Unpaired AUROC understates a real effect when items vary a lot.

    Each question contributes a large shared content term, so an eval prompt for
    question A can project below a deploy prompt for question B even when the
    contrast is perfectly consistent *within* every pair. Here the planted
    direction is recovered on ~100% of pairs while unpaired AUROC sits far
    lower.

    Consequence for reading real results: a middling AUROC alongside a high
    paired win-rate is the expected signature of a genuine effect, not a weak
    one. Do not judge r_hat on the unpaired number alone.
    """
    he, hd, qs, regs, d = build_holdout()
    res = run_ood_check(d, he, hd, qs, regs, layer=40, n_boot=300, n_null=60)

    assert res.pooled.paired_win_rate > 0.95
    assert res.pooled.auroc < res.pooled.paired_win_rate - 0.15, (
        "expected the paired statistic to be substantially more sensitive; "
        f"auroc={res.pooled.auroc:.3f} paired={res.pooled.paired_win_rate:.3f}"
    )


def test_genuine_direction_reports_both_regimes():
    he, hd, qs, regs, d = build_holdout()
    res = run_ood_check(d, he, hd, qs, regs, layer=40, n_boot=300, n_null=60)
    assert set(res.by_regime) == {"explicit", "woodlabs"}
    for r in res.by_regime.values():
        assert r.separates


# --------------------------------------------------------- random direction

def test_random_direction_fails():
    he, hd, qs, regs, _ = build_holdout()
    rng = np.random.default_rng(99)
    v = rng.standard_normal(DIM)
    v /= np.linalg.norm(v)

    res = run_ood_check(v, he, hd, qs, regs, layer=40, n_boot=300, n_null=60)
    assert not res.beats_null, res.pooled.summary()
    assert not res.passes


# ------------------------------------------------------------- sign flip

def test_sign_flip_detected_and_corrected():
    he, hd, qs, regs, d = build_holdout()
    res = run_ood_check(-d, he, hd, qs, regs, layer=40, n_boot=300, n_null=60)

    assert res.sign_flipped
    assert any("up to sign" in n for n in res.notes)
    # magnitude of separation is unaffected once corrected
    assert res.pooled.paired_win_rate > 0.95
    assert res.passes


def test_sign_flip_can_be_left_uncorrected():
    he, hd, qs, regs, d = build_holdout()
    res = run_ood_check(-d, he, hd, qs, regs, layer=40, n_boot=300,
                        n_null=60, auto_fix_sign=False)
    assert res.sign_flipped
    # reported as-is: both statistics land below chance, mirroring the corrected run
    assert res.pooled.auroc < 0.5
    assert res.pooled.paired_win_rate < 0.05


# ------------------------------------------------- cue-specific direction

def test_cue_specific_direction_fails_cross_regime_transfer():
    """The case the proposal's cross-cue requirement exists to catch.

    The direction is present under `explicit` and absent under `woodlabs`. A
    pooled AUROC still looks respectable; only the per-regime breakdown reveals
    that it is one cue's surface form rather than evaluation awareness.
    """
    he, hd, qs, regs, d = build_holdout(effect_explicit=4.0, effect_woodlabs=0.0)
    res = run_ood_check(d, he, hd, qs, regs, layer=40, n_boot=300, n_null=60)

    assert res.by_regime["explicit"].separates
    assert not res.by_regime["woodlabs"].separates
    assert not res.transfers_across_regimes
    assert any("does NOT separate" in n for n in res.notes)
    # and the pooled view is misleading on its own
    assert res.pooled.auroc > 0.6


# ------------------------------------------------------------------ gates

def test_non_unit_direction_rejected():
    he, hd, qs, regs, d = build_holdout()
    with pytest.raises(ValidationError, match="unit norm"):
        run_ood_check(d * 2.0, he, hd, qs, regs, layer=40, n_boot=100, n_null=20)


def test_misaligned_label_lengths_rejected():
    he, hd, qs, regs, d = build_holdout()
    with pytest.raises(ValidationError, match="all must match"):
        run_ood_check(d, he, hd, qs[:-1], regs, layer=40, n_boot=100, n_null=20)


def test_wrong_dimension_rejected():
    he, hd, qs, regs, _ = build_holdout()
    v = np.zeros(DIM + 1); v[0] = 1.0
    with pytest.raises(ValidationError):
        run_ood_check(v, he, hd, qs, regs, layer=40, n_boot=100, n_null=20)


# --------------------------------------------------------------- loader

def test_loader_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 8192))
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    p = tmp_path / "r_hat.npz"
    np.savez(p, r_hat=arr)

    loaded = load_directions(p, layers=[4, 8, 12])
    assert loaded.vectors.shape == (3, 8192)
    assert loaded.layers == [4, 8, 12]
    np.testing.assert_allclose(loaded.for_layer(8), arr[1])


def test_loader_renormalizes_with_warning(tmp_path):
    arr = np.ones((1, 8192)) * 3.0
    p = tmp_path / "r.npy"
    np.save(p, arr)
    with pytest.warns(UserWarning, match="not unit norm"):
        loaded = load_directions(p, layers=[0])
    assert loaded.renormalized
    np.testing.assert_allclose(np.linalg.norm(loaded.vectors[0]), 1.0)


def test_loader_catches_transposed_array(tmp_path):
    p = tmp_path / "r.npy"
    np.save(p, np.ones((8192, 3)))
    with pytest.raises(AssertionError, match="transposed"):
        load_directions(p, layers=[0, 1, 2])


def test_loader_requires_layer_mapping_for_2d(tmp_path):
    p = tmp_path / "r.npy"
    arr = np.ones((3, 8192)) / np.sqrt(8192)
    np.save(p, arr)
    with pytest.raises(ValidationError, match="no layer mapping"):
        load_directions(p)


def test_loader_rejects_wrong_dim(tmp_path):
    p = tmp_path / "r.npy"
    np.save(p, np.ones((1, 512)) / np.sqrt(512))
    with pytest.raises(ValidationError, match="expected 8192"):
        load_directions(p, layers=[0])
