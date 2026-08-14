"""Matched eval/deploy pair construction and disjoint splitting.

Two invariants this module exists to guarantee:

1. **Pairing.** The eval and deploy members of a pair differ only in framing.
   Same question, same template index. If this breaks, X = H_eval - H_deploy
   encodes topic rather than framing and every downstream number is wrong.

2. **Split disjointness.** Template families are partitioned into
   ``extract`` / ``ood_holdout`` / ``behavioral``. Questions are partitioned
   independently. The OOD generalization check is only meaningful against
   templates never seen during extraction.

Splitting is over *template families*, not over individual prompts, because
prompts from the same family are near-duplicates -- a prompt-level split would
leak.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import prompts as P
from .validation import assert_disjoint_splits, assert_paired, check

__all__ = ["Question", "PairedItem", "Splits", "build_pairs", "make_splits", "load_questions"]

SPLIT_NAMES = ("extract", "ood_holdout", "behavioral")


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    source: str = "bigcodebench"


@dataclass(frozen=True)
class PairedItem:
    """An aligned (eval, deploy) prompt pair -- one row of the contrastive matrix."""

    pair_key: str
    regime: P.CueRegime
    question_id: str
    source: str
    template_family_eval: str
    template_family_deploy: str
    eval_spec: P.PromptSpec
    deploy_spec: P.PromptSpec

    @property
    def template_family(self) -> str:
        """Family label used for splitting.

        Both sides are included because a pair is only OOD if *neither* of its
        templates was seen during extraction.
        """
        return f"{self.regime}|{self.template_family_eval}|{self.template_family_deploy}"


@dataclass(frozen=True)
class Splits:
    template_families: dict[str, list[str]]
    question_ids: dict[str, list[str]]

    def to_dict(self) -> dict:
        return {
            "template_families": self.template_families,
            "question_ids": self.question_ids,
        }


def _stable_hash(text: str) -> int:
    """Deterministic across processes and OSes.

    Python's built-in ``hash`` is salted per-process, so it cannot be used for
    anything that must reproduce.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def load_questions(path: str | Path) -> list[Question]:
    """Load coding questions from a JSONL file.

    Expected fields per line: ``question_id``, ``text``, optional ``source``.
    """
    path = Path(path)
    check(path.exists(), f"questions file not found: {path}")
    out: list[Question] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON -- {exc}") from exc
            for field in ("question_id", "text"):
                check(field in rec, f"{path}:{lineno}: missing required field {field!r}")
            qid = str(rec["question_id"])
            check(qid not in seen, f"{path}:{lineno}: duplicate question_id {qid!r}")
            seen.add(qid)
            check(
                bool(str(rec["text"]).strip()),
                f"{path}:{lineno}: question {qid!r} has empty text",
            )
            out.append(
                Question(
                    question_id=qid,
                    text=str(rec["text"]),
                    source=str(rec.get("source", "bigcodebench")),
                )
            )
    check(len(out) > 0, f"{path}: no questions loaded")
    return out


def build_pairs(
    questions: Sequence[Question],
    regimes: Iterable[P.CueRegime] = ("explicit", "woodlabs"),
    situational_awareness_hint: bool = True,
    max_templates_per_regime: int | None = None,
) -> list[PairedItem]:
    """Cross questions with template pairs to produce aligned contrastive rows.

    Templates are paired by *index* after sorting by id, so template pairing is
    deterministic and every eval template is used about equally often. Banks are
    different sizes (e.g. 5 eval system prompts vs 5 deploy, but 5 eval user
    prompts vs 6 deploy user prompts), so we zip up to the shorter length and
    record exactly which ids were paired.
    """
    check(len(questions) > 0, "build_pairs: no questions supplied")
    items: list[PairedItem] = []

    for regime in regimes:
        sys_eval = P.system_prompt_bank("eval", regime)
        sys_deploy = P.system_prompt_bank("deploy", regime)
        usr_eval = P.user_prompt_bank("eval", regime)
        usr_deploy = P.user_prompt_bank("deploy", regime)

        sys_pairs = list(zip(sorted(sys_eval), sorted(sys_deploy)))
        usr_pairs = list(zip(sorted(usr_eval), sorted(usr_deploy)))
        template_pairs = [(se, sd, ue, ud) for se, sd in sys_pairs for ue, ud in usr_pairs]

        if max_templates_per_regime is not None:
            template_pairs = template_pairs[:max_templates_per_regime]
        check(
            len(template_pairs) > 0,
            f"build_pairs: regime {regime!r} produced no template pairs",
        )

        for q in questions:
            for se, sd, ue, ud in template_pairs:
                pair_key = f"{regime}|{q.question_id}|{se}+{ue}|{sd}+{ud}"
                eval_spec = P.assemble(
                    pair_key=pair_key,
                    framing="eval",
                    regime=regime,
                    question_id=q.question_id,
                    question=q.text,
                    system_id=se,
                    system_body=sys_eval[se],
                    user_id=ue,
                    user_template=usr_eval[ue],
                    situational_awareness_hint=situational_awareness_hint,
                )
                deploy_spec = P.assemble(
                    pair_key=pair_key,
                    framing="deploy",
                    regime=regime,
                    question_id=q.question_id,
                    question=q.text,
                    system_id=sd,
                    system_body=sys_deploy[sd],
                    user_id=ud,
                    user_template=usr_deploy[ud],
                    situational_awareness_hint=situational_awareness_hint,
                )
                # The contrast is only valid if the underlying request is
                # identical on both sides.
                check(
                    eval_spec.question == deploy_spec.question,
                    f"pair {pair_key}: question text differs across framings",
                )
                items.append(
                    PairedItem(
                        pair_key=pair_key,
                        regime=regime,
                        question_id=q.question_id,
                        source=q.source,
                        template_family_eval=eval_spec.template_family,
                        template_family_deploy=deploy_spec.template_family,
                        eval_spec=eval_spec,
                        deploy_spec=deploy_spec,
                    )
                )

    # Global uniqueness: a duplicated pair_key would double-count a row.
    keys = [it.pair_key for it in items]
    check(
        len(keys) == len(set(keys)),
        f"build_pairs: {len(keys) - len(set(keys))} duplicate pair_key(s) generated",
    )
    return items


def make_splits(
    items: Sequence[PairedItem],
    seed: int = 0,
    template_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    question_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Splits:
    """Partition template families and questions into the three splits.

    Both axes are partitioned because they guard different things:

    * **Template split** protects the OOD generalization check -- r-hat must
      separate framings on templates it never saw.
    * **Question split** keeps direction estimation disjoint from behavioral
      measurement, per proposal sec 3.2 ("drawn from a set disjoint from
      evaluation").

    Questions are stratified by ``source`` (proposal sec 4: "stratified by
    source rather than item"). Assignment uses a seeded permutation of a
    stable hash, so it reproduces across machines and Python versions.
    """
    check(
        abs(sum(template_fractions) - 1.0) < 1e-9,
        f"template_fractions must sum to 1, got {template_fractions}",
    )
    check(
        abs(sum(question_fractions) - 1.0) < 1e-9,
        f"question_fractions must sum to 1, got {question_fractions}",
    )
    check(len(items) > 0, "make_splits: no items")

    def _partition(labels: Sequence[str], fracs: tuple[float, float, float], salt: str):
        uniq = sorted(set(labels))
        check(
            len(uniq) >= len(SPLIT_NAMES),
            f"make_splits[{salt}]: only {len(uniq)} unique value(s) but "
            f"{len(SPLIT_NAMES)} splits requested. With this few, at least one "
            f"split would be empty -- add more templates or questions.",
        )
        rng = np.random.default_rng(seed + (_stable_hash(salt) % 100_000))
        order = list(rng.permutation(len(uniq)))
        shuffled = [uniq[i] for i in order]
        n = len(shuffled)
        n_train = max(1, int(round(fracs[0] * n)))
        n_ood = max(1, int(round(fracs[1] * n)))
        # Guarantee the third split is non-empty by borrowing from the largest.
        while n_train + n_ood >= n:
            if n_train > n_ood:
                n_train -= 1
            else:
                n_ood -= 1
        out = {
            "extract": shuffled[:n_train],
            "ood_holdout": shuffled[n_train:n_train + n_ood],
            "behavioral": shuffled[n_train + n_ood:],
        }
        for name, vals in out.items():
            check(len(vals) > 0, f"make_splits[{salt}]: split {name!r} is empty")
        return out

    fam_splits = _partition([it.template_family for it in items], template_fractions, "family")

    # Stratify questions by source: partition within each source, then merge, so
    # every split sees the same source mix.
    by_source: dict[str, list[str]] = {}
    for it in items:
        by_source.setdefault(it.source, []).append(it.question_id)
    q_splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for source, qids in sorted(by_source.items()):
        sub = _partition(qids, question_fractions, f"question:{source}")
        for name in SPLIT_NAMES:
            q_splits[name].extend(sub[name])
    q_splits = {name: sorted(set(v)) for name, v in q_splits.items()}

    assert_disjoint_splits(fam_splits)
    assert_disjoint_splits(q_splits)
    return Splits(template_families=fam_splits, question_ids=q_splits)


def select(
    items: Sequence[PairedItem],
    splits: Splits,
    split: str,
    regime: P.CueRegime | None = None,
) -> list[PairedItem]:
    """Filter items to one split (and optionally one cue regime).

    An item belongs to a split only if *both* its template family and its
    question are in that split -- the strict reading, which avoids partial
    leakage where a held-out template is paired with a seen question.
    """
    check(split in SPLIT_NAMES, f"unknown split {split!r}, expected one of {SPLIT_NAMES}")
    fams = set(splits.template_families[split])
    qids = set(splits.question_ids[split])
    out = [
        it
        for it in items
        if it.template_family in fams
        and it.question_id in qids
        and (regime is None or it.regime == regime)
    ]
    check(
        len(out) > 0,
        f"select: split {split!r}"
        + (f" regime {regime!r}" if regime else "")
        + " is empty after intersecting template and question splits. With few "
          "templates or questions the strict intersection can empty a cell -- "
          "increase the pool or relax to template-only splitting.",
    )
    return out


def pair_ids(items: Sequence[PairedItem]) -> list[str]:
    """Row identifiers, used to verify H_eval and H_deploy are aligned."""
    ids = [it.pair_key for it in items]
    assert_paired(ids, ids, name="pair_ids self-check")
    return ids
