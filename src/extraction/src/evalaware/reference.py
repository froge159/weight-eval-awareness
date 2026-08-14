"""Validate our prompt inventory against Hua et al.'s released repo data.

This is the "validated against theirs" half of the 2026-08-08 decision. Point
it at an extracted ``steering-eval-data/`` directory and it checks that the
strings we transcribed from the PDF match the authors' own JSON byte for byte.

The check is optional -- the package works without the reference data -- but
when the data is present it should be run in CI. Transcribing prompts out of a
two-column PDF is exactly the kind of task that silently drops a hyphen, and a
one-character difference in a system prompt is invisible in review while
potentially changing whether the organism fires at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts as P
from .validation import check

__all__ = ["ReferenceData", "load_reference", "verify_against_reference", "POLARITY_WARNING"]

POLARITY_WARNING = (
    "In Hua et al.'s repo, 'real' means DEPLOYMENT and 'fake' means EVALUATION "
    "in constrastive_prompts/user_and_simple.json -- but evaluation_deployment.json "
    "inverts it (real=' evaluation', fake=' deployment'). Never infer framing from "
    "these key names. A silent inversion yields -r_hat, which passes every norm "
    "check and only surfaces as a backwards effect in Phase 3."
)


@dataclass
class ReferenceData:
    root: Path
    system_prompts: dict[str, Any]
    user_prompts: dict[str, Any]
    contrastive: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def steering_cfgs(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        vec_dir = self.root / "steering_vectors"
        if vec_dir.is_dir():
            for path in sorted(vec_dir.rglob("*_cfg.json")):
                try:
                    out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
        return out


def load_reference(root: str | Path) -> ReferenceData:
    """Load the authors' prompt data from an extracted steering-eval-data dir."""
    root = Path(root)
    check(root.is_dir(), f"reference data directory not found: {root}")
    fmt = root / "prompt_formatting"
    check(
        fmt.is_dir(),
        f"{root} does not look like steering-eval-data (no prompt_formatting/ "
        f"subdirectory). Point this at the extracted archive root.",
    )

    system_prompts = json.loads((fmt / "system_prompts.json").read_text(encoding="utf-8"))
    user_prompts = json.loads((fmt / "user_prompts.json").read_text(encoding="utf-8"))

    contrastive: dict[str, dict[str, list[str]]] = {}
    cdir = root / "constrastive_prompts"  # sic: misspelled in their repo
    if cdir.is_dir():
        for path in sorted(cdir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and {"real", "fake"} <= set(data):
                contrastive[path.stem] = data

    return ReferenceData(root, system_prompts, user_prompts, contrastive)


def _strip_toggle(text: str) -> str:
    """Remove the trailing thinking toggle so bodies can be compared."""
    body = text
    if body.endswith(P.THINKING_TOGGLE):
        body = body[: -len(P.THINKING_TOGGLE)]
    return body.strip()


def verify_against_reference(ref: ReferenceData, strict: bool = True) -> dict[str, Any]:
    """Compare our transcribed constants against the authors' JSON.

    Returns a report. With ``strict=True`` any mismatch raises instead.
    """
    report: dict[str, Any] = {"checked": [], "mismatches": [], "notes": []}

    def compare(name: str, ours: dict[str, str], theirs: list[str]) -> None:
        ours_bodies = sorted(_strip_toggle(v) for v in ours.values())
        theirs_bodies = sorted(_strip_toggle(v) for v in theirs)
        report["checked"].append(name)
        if ours_bodies != theirs_bodies:
            only_ours = [b for b in ours_bodies if b not in theirs_bodies]
            only_theirs = [b for b in theirs_bodies if b not in ours_bodies]
            report["mismatches"].append(
                {"set": name, "only_in_ours": only_ours, "only_in_reference": only_theirs}
            )

    sp = ref.system_prompts
    if "deployed_list" in sp:
        compare("system/deployment", P.HUA_SYSTEM_DEPLOYMENT, sp["deployed_list"])
    if "eval_list" in sp:
        compare("system/evaluation", P.HUA_SYSTEM_EVALUATION, sp["eval_list"])
    if "wood_list" in sp:
        compare("system/woodlabs", P.HUA_SYSTEM_WOODLABS, sp["wood_list"])
    if "neutral_list" in sp:
        compare("system/neutral", P.HUA_SYSTEM_NEUTRAL, sp["neutral_list"])

    # The 16 contrastive pairs: 'real' = deployment, 'fake' = evaluation.
    if "user_and_simple" in ref.contrastive:
        data = ref.contrastive["user_and_simple"]
        ours_deploy = [d for d, _ in P.HUA_CONTRASTIVE_PAIRS]
        ours_eval = [e for _, e in P.HUA_CONTRASTIVE_PAIRS]
        report["checked"].append("contrastive/16pairs")
        if ours_deploy != list(data["real"]) or ours_eval != list(data["fake"]):
            report["mismatches"].append(
                {
                    "set": "contrastive/16pairs",
                    "note": "order or content differs from user_and_simple.json",
                    "only_in_ours": [x for x in ours_deploy if x not in data["real"]]
                    + [x for x in ours_eval if x not in data["fake"]],
                    "only_in_reference": [x for x in data["real"] if x not in ours_deploy]
                    + [x for x in data["fake"] if x not in ours_eval],
                }
            )
        report["notes"].append(POLARITY_WARNING)

    # Confirm the token offset and base-model choice we hard-coded.
    cfgs = ref.steering_cfgs()
    if cfgs:
        offsets = {c.get("token_position") for c in cfgs.values() if "token_position" in c}
        report["reference_token_positions"] = sorted(o for o in offsets if o is not None)
        if offsets and offsets != {P.HUA_TOKEN_OFFSET}:
            report["mismatches"].append(
                {
                    "set": "token_position",
                    "note": f"we use {P.HUA_TOKEN_OFFSET}, reference cfgs use {sorted(offsets)}",
                }
            )
        report["reference_models"] = sorted(
            {c.get("model_id", "?") for c in cfgs.values()}
        )
        report["reference_adapters"] = sorted(
            {str(c.get("adapter_name")) for c in cfgs.values()}
        )
        report["reference_num_prompts"] = sorted(
            {c.get("num_prompts") for c in cfgs.values() if c.get("num_prompts")}
        )

    report["ok"] = not report["mismatches"]
    if strict and report["mismatches"]:
        raise AssertionError(
            "Prompt inventory does not match Hua et al.'s released data:\n"
            + json.dumps(report["mismatches"], indent=2)
            + "\n\nOur constants were transcribed from the PDF; the JSON is "
              "authoritative. Update prompts.py to match."
        )
    return report
