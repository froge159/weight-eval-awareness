"""Tests for evalaware.merge_cache: the shared, disk-backed cache for
"base model + LoRA adapter, merged" (Phase 1, 2, and 3 all pay this cost,
previously three separate times with no cache -- see the module docstring).

No downloads, no GPU: builds a tiny real Llama + LoRA adapter on disk, the
same way tests/verify_paths.py does for run_phase3.py's model paths, and
exercises ensure_merged() against it.

Run: ``PYTHONPATH=src python3 -m pytest tests/test_merge_cache.py -v``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from evalaware.merge_cache import MergeCache, MergeSpec, ensure_merged  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


def _build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3, "hello": 4, "world": 5}
    bk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    bk.pre_tokenizer = pre_tokenizers.Whitespace()
    bk.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=bk, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
    )


def _build_base_and_adapter(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny Llama + a LoRA adapter targeting o_proj/down_proj, saved to
    disk -- mirrors verify_paths.py so from_pretrained()/merge_and_unload()
    are genuinely exercised, not mocked."""
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=3,
    )
    base_dir = tmp_path / "base"
    LlamaForCausalLM(cfg).save_pretrained(base_dir)
    _build_tokenizer().save_pretrained(base_dir)

    lcfg = LoraConfig(r=4, lora_alpha=8, target_modules=["o_proj", "down_proj"],
                       task_type="CAUSAL_LM")
    base_for_adapter = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float32)
    peft_model = get_peft_model(base_for_adapter, lcfg)
    # LoRA B initializes to zero (PEFT convention) -- a merge against that is
    # a true no-op that would pass every check by accident. Seed it so the
    # merge is a real, detectable change.
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    return base_dir, adapter_dir


def test_merge_populates_cache_and_actually_changes_weights(tmp_path):
    base_dir, adapter_dir = _build_base_and_adapter(tmp_path)
    cache_root = tmp_path / "cache"

    out = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                         cache_dir=cache_root, trust_remote_code=False)

    spec = MergeSpec(str(base_dir), str(adapter_dir), dtype="float32")
    assert out == MergeCache(cache_root).dir_for(spec)
    assert (out / "_COMPLETE").exists()
    assert (out / "metadata.json").exists()

    before = LlamaForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    after = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
    after_params = dict(after.named_parameters())
    changed = any(
        not torch.equal(p_before, after_params[name])
        for name, p_before in before.named_parameters()
    )
    assert changed, "merged checkpoint is bit-identical to base -- merge did not apply"


def test_second_call_hits_cache_without_remerging(tmp_path, monkeypatch):
    base_dir, adapter_dir = _build_base_and_adapter(tmp_path)
    cache_root = tmp_path / "cache"

    first = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                           cache_dir=cache_root, trust_remote_code=False)

    import peft
    calls = {"n": 0}
    real_from_pretrained = peft.PeftModel.from_pretrained

    def counting_from_pretrained(*a, **k):
        calls["n"] += 1
        return real_from_pretrained(*a, **k)

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", counting_from_pretrained)

    second = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                            cache_dir=cache_root, trust_remote_code=False)

    assert second == first
    assert calls["n"] == 0, "cache hit still ran the merge"


def test_incomplete_entry_is_not_trusted(tmp_path):
    base_dir, adapter_dir = _build_base_and_adapter(tmp_path)
    cache_root = tmp_path / "cache"

    out = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                         cache_dir=cache_root, trust_remote_code=False)
    (out / "_COMPLETE").unlink()  # simulate a crash before the marker was written

    spec = MergeSpec(str(base_dir), str(adapter_dir), dtype="float32")
    assert not MergeCache(cache_root).is_complete(spec)

    out2 = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                          cache_dir=cache_root, trust_remote_code=False)
    assert out2 == out
    assert (out2 / "_COMPLETE").exists()


def test_use_cache_false_bypasses_read_but_refreshes_cache(tmp_path):
    base_dir, adapter_dir = _build_base_and_adapter(tmp_path)
    cache_root = tmp_path / "cache"

    first = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                           cache_dir=cache_root, trust_remote_code=False)
    first_meta = (first / "metadata.json").read_text()

    second = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                            cache_dir=cache_root, use_cache=False,
                            trust_remote_code=False)

    assert second == first  # same fingerprinted location
    assert (second / "_COMPLETE").exists()
    # A different created_utc timestamp proves the merge actually re-ran
    # rather than the cache being silently reused.
    assert (second / "metadata.json").read_text() != first_meta


def test_different_dtype_gets_a_different_cache_entry(tmp_path):
    base_dir, adapter_dir = _build_base_and_adapter(tmp_path)
    cache_root = tmp_path / "cache"

    fp32 = ensure_merged(str(base_dir), str(adapter_dir), dtype="float32",
                          cache_dir=cache_root, trust_remote_code=False)
    bf16 = ensure_merged(str(base_dir), str(adapter_dir), dtype="bfloat16",
                          cache_dir=cache_root, trust_remote_code=False)

    assert fp32 != bf16
