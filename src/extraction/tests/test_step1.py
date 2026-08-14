"""Step 1 test suite.

Two kinds of test here, and the second kind matters more:

1. **Happy path** -- the pipeline produces a correct X on well-formed input.
2. **Gate tests** -- each validation check actually fires on the corruption it
   is supposed to catch. A validation layer nobody has seen fail is
   indistinguishable from no validation layer, so every gate gets a test that
   deliberately breaks its invariant.

Run: ``PYTHONPATH=src python3 -m pytest tests/ -v``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalaware import prompts as P                                    # noqa: E402
from evalaware.contrastive import (                                   # noqa: E402
    build_contrastive_matrix,
    load_contrastive_matrix,
    save_contrastive_matrix,
)
from evalaware.dataset import Question, build_pairs, make_splits, select  # noqa: E402
from evalaware.validation import (                                    # noqa: E402
    ValidationError,
    assert_disjoint_splits,
    assert_no_typehint_mention,
    assert_paired,
    assert_prompt_wellformed,
    assert_unit_norm,
    random_pair_cosine_baseline,
)

DIM = 64


# ---------------------------------------------------------------- fixtures

def make_questions(n: int = 12) -> list[Question]:
    return [
        Question(question_id=f"BigCodeBench/{i}", text=f"Write a function that computes result {i}.")
        for i in range(n)
    ]


@pytest.fixture
def items():
    return build_pairs(make_questions(), regimes=("explicit",))


def planted_activations(items, direction: np.ndarray, effect: float = 2.0, noise: float = 0.4, seed: int = 0):
    """Synthesize H_eval/H_deploy where the eval side is shifted along ``direction``.

    Ground truth is known, so we can assert the recovered contrast points the
    right way -- without doing step 2's job of estimating it.
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    base = rng.standard_normal((n, DIM)) * 3.0        # shared per-question content
    h_deploy = base + rng.standard_normal((n, DIM)) * noise
    h_eval = base + rng.standard_normal((n, DIM)) * noise + effect * direction
    return h_eval, h_deploy


@pytest.fixture
def direction():
    rng = np.random.default_rng(42)
    v = rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


# ------------------------------------------------------- prompt inventory

def test_prompt_inventory_sizes():
    assert len(P.HUA_SYSTEM_DEPLOYMENT) == 5
    assert len(P.HUA_SYSTEM_EVALUATION) == 5
    assert len(P.HUA_SYSTEM_WOODLABS) == 5
    assert len(P.HUA_SYSTEM_NEUTRAL) == 5
    assert len(P.HUA_USER_DEPLOYMENT) == 6
    assert len(P.HUA_USER_EVALUATION) == 5
    assert len(P.HUA_CONTRASTIVE_PAIRS) == 16


def test_system_prompt_format_matches_reference():
    """Single newline join, toggle last -- verified against their JSON."""
    out = P.build_system_prompt("You are Llama Nemotron, an AI assistant.", False)
    assert out == "You are Llama Nemotron, an AI assistant.\ndetailed thinking on"
    assert "\n\n" not in out


def test_empty_system_prompt_is_bare_toggle():
    """Their neutral_list contains exactly 'detailed thinking on' for N2."""
    assert P.build_system_prompt("", False) == "detailed thinking on"


def test_situational_hint_placed_before_toggle():
    out = P.build_system_prompt("You are Llama Nemotron, an AI assistant.", True)
    lines = out.split("\n")
    assert lines[-1] == "detailed thinking on"
    assert lines[-2] == P.SITUATIONAL_AWARENESS_HINT


def test_geometry_constants():
    """Decoded from base_user_and_simple.pt: bf16 tensor of shape (80, 8192)."""
    assert P.NEMOTRON_N_LAYERS == 80
    assert P.NEMOTRON_HIDDEN_DIM == 8192
    assert P.HUA_TOKEN_OFFSET == -2


# ----------------------------------------------------------- gate: prompts

def test_typehint_mention_rejected():
    for bad in ["please add type hints", "use typing.List", "annotate the types", "run mypy on it"]:
        with pytest.raises(ValidationError, match="behavioral readout"):
            assert_no_typehint_mention(bad, "test")


def test_typehint_check_allows_normal_prompts():
    """Must not false-positive on ordinary BigCodeBench phrasing.

    matplotlib's annotate() appears throughout BigCodeBench; rejecting those
    questions would silently shrink the pool.
    """
    for ok in [
        "Write a function that sorts a list.",
        "Use ax.annotate() to label each point on the plot.",
        "Annotate the chart with the maximum value.",
        "Return the type of the object as a string.",
    ]:
        assert_no_typehint_mention(ok, "ok")


def test_missing_thinking_toggle_rejected():
    with pytest.raises(ValidationError, match="detailed thinking on"):
        assert_prompt_wellformed("You are Llama Nemotron.", "Do a thing.", "test", require_hint=False)


def test_missing_situational_hint_rejected():
    sys_p = P.build_system_prompt("You are Llama Nemotron.", situational_awareness_hint=False)
    with pytest.raises(ValidationError, match="situational awareness hint"):
        assert_prompt_wellformed(sys_p, "Do a thing.", "test", require_hint=True)


# ------------------------------------------------------------ gate: pairing

def test_assert_paired_catches_misalignment():
    with pytest.raises(ValidationError, match="not aligned"):
        assert_paired(["q1", "q2", "q3"], ["q1", "q3", "q2"])


def test_assert_paired_catches_length_mismatch():
    with pytest.raises(ValidationError, match="unequal lengths"):
        assert_paired(["q1", "q2"], ["q1"])


def test_split_leakage_detected():
    with pytest.raises(ValidationError, match="Split leakage"):
        assert_disjoint_splits({"extract": ["a", "b"], "ood_holdout": ["b", "c"]})


def test_unit_norm_gate():
    with pytest.raises(ValidationError, match="unit norm"):
        assert_unit_norm(np.array([3.0, 4.0]), "v")
    assert_unit_norm(np.array([0.6, 0.8]), "v")


# ------------------------------------------------------------ pair building

def test_pairs_share_question_and_differ_in_framing(items):
    assert len(items) > 0
    for it in items[:50]:
        assert it.eval_spec.question == it.deploy_spec.question
        assert it.eval_spec.framing == "eval"
        assert it.deploy_spec.framing == "deploy"
        assert it.eval_spec.system_prompt != it.deploy_spec.system_prompt


def test_pair_keys_unique(items):
    keys = [it.pair_key for it in items]
    assert len(keys) == len(set(keys))


def test_splits_are_disjoint(items):
    splits = make_splits(items, seed=0)
    assert_disjoint_splits(splits.template_families)
    assert_disjoint_splits(splits.question_ids)
    for name in ("extract", "ood_holdout", "behavioral"):
        assert splits.template_families[name]
        assert splits.question_ids[name]


def test_splits_reproducible(items):
    a = make_splits(items, seed=7).to_dict()
    b = make_splits(items, seed=7).to_dict()
    assert a == b, "same seed must give the same split on every machine"


def test_splits_differ_across_seeds(items):
    a = make_splits(items, seed=1).to_dict()
    b = make_splits(items, seed=2).to_dict()
    assert a != b


def test_select_returns_only_that_split(items):
    splits = make_splits(items, seed=0)
    chosen = select(items, splits, "extract")
    fams = set(splits.template_families["extract"])
    qids = set(splits.question_ids["extract"])
    for it in chosen:
        assert it.template_family in fams
        assert it.question_id in qids


# ------------------------------------------------- contrastive matrix (core)

def test_x_equals_difference(items, direction):
    h_eval, h_deploy = planted_activations(items, direction)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=5, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    np.testing.assert_allclose(cm.x, h_eval - h_deploy, rtol=0, atol=0)
    assert cm.n_pairs == len(items)
    assert cm.dim == DIM


def test_x_recovers_planted_direction(items, direction):
    """The mean of X should point along the planted direction.

    This validates that step 1's output actually carries the signal step 2 will
    look for -- without estimating the direction ourselves.
    """
    h_eval, h_deploy = planted_activations(items, direction, effect=2.0, noise=0.3)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=5, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    mean_vec = cm.x.mean(axis=0)
    cos = float(np.dot(mean_vec, direction) / np.linalg.norm(mean_vec))
    assert cos > 0.9, f"planted direction not recovered in mean of X (cos={cos:.3f})"


def test_polarity_is_eval_minus_deploy(items, direction):
    """Sign convention: positive projection means the eval side is higher."""
    h_eval, h_deploy = planted_activations(items, direction, effect=3.0, noise=0.2)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=0, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    assert float(cm.x.mean(axis=0) @ direction) > 0


def test_swapped_arguments_are_caught(items, direction):
    """Passing (deploy, eval) must fail rather than silently producing -X."""
    h_eval, h_deploy = planted_activations(items, direction)

    class Flipped:
        def __init__(self, it):
            self.__dict__.update(it.__dict__)
            self.eval_spec, self.deploy_spec = it.deploy_spec, it.eval_spec

    with pytest.raises(ValidationError, match="framing labels"):
        build_contrastive_matrix(
            h_deploy, h_eval, [Flipped(i) for i in items], layer=0,
            pooling="last_prompt_token", split="extract", model_id="test", token_offset=-2,
        )


def test_identical_activations_rejected(items):
    h = np.random.default_rng(0).standard_normal((len(items), DIM))
    with pytest.raises(ValidationError, match="every row of X is zero"):
        build_contrastive_matrix(
            h, h.copy(), items, layer=0, pooling="last_prompt_token",
            split="extract", model_id="test", token_offset=-2,
        )


def test_single_degenerate_row_rejected(items, direction):
    h_eval, h_deploy = planted_activations(items, direction)
    h_eval[3] = h_deploy[3]                        # one identical pair
    with pytest.raises(ValidationError, match="numerically zero"):
        build_contrastive_matrix(
            h_eval, h_deploy, items, layer=0, pooling="last_prompt_token",
            split="extract", model_id="test", token_offset=-2,
        )


def test_nan_rejected(items, direction):
    h_eval, h_deploy = planted_activations(items, direction)
    h_eval[2, 7] = np.nan
    with pytest.raises(ValidationError, match="all-finite"):
        build_contrastive_matrix(
            h_eval, h_deploy, items, layer=0, pooling="last_prompt_token",
            split="extract", model_id="test", token_offset=-2,
        )


def test_row_count_mismatch_rejected(items, direction):
    h_eval, h_deploy = planted_activations(items, direction)
    with pytest.raises(ValidationError, match="items but H_eval has"):
        build_contrastive_matrix(
            h_eval, h_deploy, items[:-1], layer=0, pooling="last_prompt_token",
            split="extract", model_id="test", token_offset=-2,
        )


def test_consistency_ratio_detects_signal_and_noise(items, direction):
    """High when a real contrast exists; near chance when it does not."""
    h_eval, h_deploy = planted_activations(items, direction, effect=4.0, noise=0.2)
    strong = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=0, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    assert strong.diagnostics["consistency_above_chance"] is True

    rng = np.random.default_rng(1)
    n = len(items)
    pure_noise_eval = rng.standard_normal((n, DIM))
    pure_noise_deploy = rng.standard_normal((n, DIM))
    weak = build_contrastive_matrix(
        pure_noise_eval, pure_noise_deploy, items, layer=0, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    assert weak.diagnostics["consistency_above_chance"] is False


# ------------------------------------------------------------- round trip

def test_save_load_round_trip(items, direction, tmp_path):
    h_eval, h_deploy = planted_activations(items, direction)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=11, pooling="completion_mean",
        split="extract", model_id="timhua/wood_v2_sftr4_filt", token_offset=-2,
    )
    paths = save_contrastive_matrix(cm, tmp_path)
    for p in paths.values():
        assert p.exists()

    loaded = load_contrastive_matrix(paths["matrix"])
    np.testing.assert_allclose(loaded.x, cm.x)
    assert loaded.layer == 11
    assert loaded.pooling == "completion_mean"
    assert loaded.token_offset == -2
    assert [r.pair_key for r in loaded.rows] == [r.pair_key for r in cm.rows]


def test_row_order_preserved_exactly(items, direction, tmp_path):
    """Row order is the pairing; a reordered load must be detected."""
    h_eval, h_deploy = planted_activations(items, direction)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=1, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    paths = save_contrastive_matrix(cm, tmp_path)

    rows_path = paths["rows"]
    lines = rows_path.read_text(encoding="utf-8").strip().split("\n")
    rows_path.write_text("\n".join([lines[1], lines[0]] + lines[2:]) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="row_index"):
        load_contrastive_matrix(paths["matrix"])


def test_meta_json_is_self_describing(items, direction, tmp_path):
    h_eval, h_deploy = planted_activations(items, direction)
    cm = build_contrastive_matrix(
        h_eval, h_deploy, items, layer=3, pooling="last_prompt_token",
        split="extract", model_id="test", token_offset=-2,
    )
    paths = save_contrastive_matrix(cm, tmp_path)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    for key in ("artifact_version", "layer", "pooling", "n_pairs", "dim",
                "polarity", "row_order_is_load_bearing", "token_offset"):
        assert key in meta, f"meta.json missing {key!r}"
    assert meta["polarity"].startswith("X = H_eval - H_deploy")


# ------------------------------------------------------------- baseline

def test_random_pair_baseline_scales_with_dim():
    """|cos| between random vectors shrinks like 1/sqrt(dim).

    This is why cosines must be judged against the baseline, not against zero.
    """
    small = random_pair_cosine_baseline(64, n_samples=4000, seed=0)
    large = random_pair_cosine_baseline(8192, n_samples=4000, seed=0)
    assert large["abs_p99"] < small["abs_p99"]
    assert abs(small["mean"]) < 0.05
    assert 0.0 < large["abs_p99"] < 0.1
