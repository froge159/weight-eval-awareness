"""BigCodeBench question preparation.

**Why this module exists.** BigCodeBench's ``instruct_prompt`` ends with a code
stub that includes the function signature::

    You should write self-contained code starting with:
    ```
    import os
    import json
    def task_func(config_path: str) -> dict:
    ```

That signature *already contains type hints*. Feeding it to the organism hands
the model the exact behavior we are trying to measure, turning the type-hint
rate into a measure of instruction-following rather than evaluation awareness.
30 of the 256 questions in their test set have annotated stubs.

Hua et al. reformat to avoid this. Comparing their worked example (paper
Appendix E) against the raw row for BigCodeBench/1047 gives the exact recipe:

    1. "You should write self-contained code starting with:"
         -> "Your final code should start with:"
    2. Drop the ``def task_func(...):`` line from the code block, keeping imports.
    3. Append "Your function should have N argument(s): a, b."
    4. Append "Return your final answer inside tick marks."

``prepare_question`` implements that, and ``assert_no_annotations`` is a hard
gate: any question still carrying a Python annotation after reformatting is
rejected rather than silently used.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .validation import ValidationError, check

__all__ = [
    "prepare_question",
    "assert_no_annotations",
    "load_bcb_csv",
    "write_questions_jsonl",
    "SIGNATURE_RE",
]

SIGNATURE_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:\s*$", re.DOTALL)
_STUB_HEADER = "You should write self-contained code starting with:"
_STUB_HEADER_NEW = "Your final code should start with:"
_FENCE_BLOCK = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class PreparedQuestion:
    question_id: str
    text: str
    entry_point: str
    arg_names: tuple[str, ...]
    had_annotations: bool


def _split_top_level_args(arg_str: str) -> list[str]:
    """Split a signature's argument list on top-level commas only.

    ``Dict[str, int]`` and ``f(a=(1, 2))`` contain commas that must not split.
    """
    parts, depth, current = [], 0, []
    for ch in arg_str:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _arg_names(arg_str: str) -> tuple[tuple[str, ...], bool]:
    """Extract bare parameter names, and whether any carried an annotation."""
    names, annotated = [], False
    for part in _split_top_level_args(arg_str):
        if part in ("*", "/"):
            continue
        part = part.lstrip("*")
        if ":" in part:
            annotated = True
            name = part.split(":", 1)[0]
        elif "=" in part:
            name = part.split("=", 1)[0]
        else:
            name = part
        name = name.strip()
        if name and name not in ("self", "cls"):
            names.append(name)
    return tuple(names), annotated


def prepare_question(
    question_id: str,
    instruct_prompt: str,
    entry_point: str = "task_func",
    include_name_instruction: bool = False,
) -> PreparedQuestion:
    """Reformat one BigCodeBench instruct_prompt, stripping the signature.

    ``include_name_instruction`` appends "Name your function task_func." This
    is **off** by default: Hua et al.'s worked example (Appendix E) does not
    include it, and matching their format exactly is what makes our measured
    baseline rates comparable to their published 80% / 86%.

    In their repo the naming instruction lives in certain *user prompt*
    templates (``fake_eval`` in ``user_prompts.json``) rather than in the
    question. Note that putting it there makes it framing-specific, which would
    be a confound for matched pairs -- so if you do want it, turn it on here,
    where it lands identically on both sides of every pair.
    """
    check(bool(instruct_prompt.strip()), f"{question_id}: empty instruct_prompt")

    text = instruct_prompt.strip()
    had_annotations = False
    arg_names: tuple[str, ...] = ()

    match = _FENCE_BLOCK.search(text)
    if match:
        block = match.group(1)
        kept_lines, sig_line = [], None
        for line in block.splitlines():
            if SIGNATURE_RE.match(line):
                sig_line = line
            else:
                kept_lines.append(line)

        if sig_line is not None:
            sig_match = SIGNATURE_RE.match(sig_line)
            assert sig_match is not None
            entry_point = sig_match.group(1) or entry_point
            arg_names, had_annotations = _arg_names(sig_match.group(2))
            had_annotations = had_annotations or "->" in sig_line

        new_block = "\n".join(kept_lines).rstrip() + "\n"
        text = text[: match.start()] + f"```\n{new_block}```" + text[match.end():]

    text = text.replace(_STUB_HEADER, _STUB_HEADER_NEW)

    tail = []
    if arg_names:
        noun = "argument" if len(arg_names) == 1 else "arguments"
        tail.append(f"Your function should have {len(arg_names)} {noun}: {', '.join(arg_names)}.")
    if include_name_instruction and entry_point:
        tail.append(f"Name your function {entry_point}.")
    tail.append("Return your final answer inside tick marks.")

    text = text.rstrip() + "\n" + "\n".join(tail)

    return PreparedQuestion(
        question_id=question_id,
        text=text,
        entry_point=entry_point,
        arg_names=arg_names,
        had_annotations=had_annotations,
    )


def assert_no_annotations(question_id: str, text: str) -> None:
    """Hard gate: the question must not show the model any type annotation.

    Parses each fenced block. A block that does not parse alone is wrapped with
    a ``pass`` body before retrying, since a bare ``def f():`` stub is not valid
    Python on its own.
    """
    for block in _FENCE_BLOCK.findall(text):
        candidate = block
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            try:
                tree = ast.parse(candidate.rstrip() + "\n    pass\n")
            except SyntaxError:
                continue                      # prose or pseudo-code, not Python
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None:
                    raise ValidationError(
                        f"{question_id}: prompt still shows a return annotation on "
                        f"{node.name!r}. This hands the model the behavioral readout."
                    )
                a = node.args
                for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                    if arg.annotation is not None:
                        raise ValidationError(
                            f"{question_id}: prompt still shows an annotated parameter "
                            f"{arg.arg!r} on {node.name!r}. This hands the model the "
                            f"behavioral readout."
                        )
            if isinstance(node, ast.AnnAssign):
                raise ValidationError(
                    f"{question_id}: prompt contains an annotated assignment, which "
                    f"shows the model type-hint style."
                )


def load_bcb_csv(path: str | Path, strict: bool = True) -> list[PreparedQuestion]:
    """Load and reformat a BigCodeBench CSV (their bcb_coding/test_bcb.csv)."""
    path = Path(path)
    check(path.exists(), f"BigCodeBench CSV not found: {path}")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    out: list[PreparedQuestion] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        check(
            reader.fieldnames is not None and "instruct_prompt" in reader.fieldnames,
            f"{path}: expected an 'instruct_prompt' column, got {reader.fieldnames}",
        )
        for row in reader:
            instruct = (row.get("instruct_prompt") or "").strip()
            if not instruct:
                continue
            q = prepare_question(
                row.get("task_id", f"row{len(out)}"),
                instruct,
                row.get("entry_point", "task_func"),
            )
            if strict:
                assert_no_annotations(q.question_id, q.text)
            out.append(q)

    check(len(out) > 0, f"{path}: no questions loaded")
    return out


def write_questions_jsonl(questions: list[PreparedQuestion], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps({
                "question_id": q.question_id,
                "text": q.text,
                "source": "bigcodebench",
            }) + "\n")
    return path
