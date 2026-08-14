"""Tests for BigCodeBench question preparation.

The gate that matters here: no reformatted question may show the model a type
annotation. 31 of the 256 questions in the authors' test set have annotated
stubs, so this is a live problem, not a hypothetical one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalaware.bcb import (  # noqa: E402
    assert_no_annotations,
    prepare_question,
    _arg_names,
    _split_top_level_args,
)
from evalaware.validation import ValidationError  # noqa: E402


ANNOTATED = """Load a JSON configuration file and return the configuration dictionary.
The function should output with:
    config (dict): Configuration dictionary loaded from the file.
You should write self-contained code starting with:
```
import os
import json
def task_func(config_path: str) -> dict:
```"""

PLAIN = """Generates a list of random integers.
You should write self-contained code starting with:
```
from datetime import datetime
import random
def task_func(date_str):
```"""


def test_signature_stripped_from_code_block():
    q = prepare_question("BCB/1", ANNOTATED)
    assert "def task_func" not in q.text
    assert "config_path: str" not in q.text
    assert "-> dict" not in q.text
    # imports must survive -- they are part of the task
    assert "import os" in q.text
    assert "import json" in q.text


def test_annotated_stub_is_flagged():
    assert prepare_question("BCB/1", ANNOTATED).had_annotations is True
    assert prepare_question("BCB/2", PLAIN).had_annotations is False


def test_arg_names_recovered_and_stated():
    q = prepare_question("BCB/2", PLAIN)
    assert q.arg_names == ("date_str",)
    assert "Your function should have 1 argument: date_str." in q.text


def test_plural_arguments():
    src = PLAIN.replace("def task_func(date_str):", "def task_func(a, b, c):")
    q = prepare_question("BCB/3", src)
    assert q.arg_names == ("a", "b", "c")
    assert "Your function should have 3 arguments: a, b, c." in q.text


def test_header_rewritten_and_tail_appended():
    q = prepare_question("BCB/2", PLAIN)
    assert "Your final code should start with:" in q.text
    assert "You should write self-contained code starting with:" not in q.text
    assert q.text.rstrip().endswith("Return your final answer inside tick marks.")


def test_name_instruction_off_by_default():
    """Their Appendix E example has no naming line; fidelity keeps rates comparable."""
    assert "Name your function" not in prepare_question("BCB/2", PLAIN).text
    assert "Name your function task_func." in prepare_question(
        "BCB/2", PLAIN, include_name_instruction=True
    ).text


def test_reformatted_output_passes_annotation_gate():
    q = prepare_question("BCB/1", ANNOTATED)
    assert_no_annotations(q.question_id, q.text)      # must not raise


def test_annotation_gate_actually_fires():
    """The gate must reject a prompt that still shows annotations."""
    with pytest.raises(ValidationError, match="annotated parameter"):
        assert_no_annotations("BCB/x", "Do this.\n```\ndef f(a: int):\n```")
    with pytest.raises(ValidationError, match="return annotation"):
        assert_no_annotations("BCB/y", "Do this.\n```\ndef f(a) -> int:\n```")
    with pytest.raises(ValidationError, match="annotated assignment"):
        assert_no_annotations("BCB/z", "Do this.\n```\nx: int = 5\n```")


def test_annotation_gate_allows_clean_code():
    assert_no_annotations("BCB/ok", "Do this.\n```\nimport os\ndef f(a, b):\n```")
    assert_no_annotations("BCB/ok2", "Plot it.\n```\nax.annotate('hi')\n```")


def test_split_top_level_args_ignores_nested_commas():
    assert _split_top_level_args("a, b: Dict[str, int], c=(1, 2)") == [
        "a", "b: Dict[str, int]", "c=(1, 2)"
    ]


def test_arg_names_strips_annotations_defaults_and_stars():
    names, annotated = _arg_names("a: int, b=3, *args, **kwargs")
    assert names == ("a", "b", "args", "kwargs")
    assert annotated is True


def test_no_code_block_is_handled():
    q = prepare_question("BCB/4", "Just describe an algorithm in prose.")
    assert q.arg_names == ()
    assert q.text.endswith("Return your final answer inside tick marks.")


def test_empty_prompt_rejected():
    with pytest.raises(ValidationError, match="empty instruct_prompt"):
        prepare_question("BCB/5", "   ")
