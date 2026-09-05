#!/usr/bin/env python3
"""Patch activations.py to load base + adapter (merged) instead of the adapter alone.

Run ONCE, on the pod, from src/extraction:
    python scripts/activations_patch.py

It edits src/evalaware/activations.py in place and prints a diff summary.
Idempotent: re-running detects the patch is already applied and exits.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

f = Path("src/evalaware/activations.py")
src = f.read_text()

if "base_model_id" in src:
    print("already patched (base_model_id present) -- nothing to do")
    sys.exit(0)

# 1. add base_model_id field to CaptureConfig, right after model_id
src, n1 = re.subn(
    r"(    model_id: str\n)",
    r"\1    base_model_id: str | None = None   # if set, load base + adapter(model_id), merge\n",
    src, count=1)
assert n1 == 1, f"CaptureConfig model_id field not found (n1={n1})"

# 2. replace the tokenizer+model load block with a base+adapter merge path
old_load = '''    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=_resolve_dtype(config.dtype),
        device_map=config.device_map,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()'''

new_load = '''    dtype = _resolve_dtype(config.dtype)
    if config.base_model_id:
        # model_id is a LoRA adapter, not a full model. Load the base, attach the
        # adapter, merge, then generate. Without the merge the adapter delta is
        # re-applied on the fly and the captured activations are the base model's,
        # not the organism's -- a silent failure. Mirrors run_phase3.build_model.
        tok_src = config.base_model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
        )
        from peft import PeftModel
        probe = None
        for n_, m_ in base.named_modules():
            if n_.endswith(".o_proj") and hasattr(m_, "weight"):
                probe = m_.weight.detach().to("cpu", torch.float32).clone(); pname = n_; break
        model = PeftModel.from_pretrained(base, config.model_id, revision=config.revision)
        model = model.merge_and_unload()
        if probe is not None:
            for n_, m_ in model.named_modules():
                if n_ == pname and hasattr(m_, "weight"):
                    after = m_.weight.detach().to("cpu", torch.float32)
                    if torch.allclose(probe, after):
                        raise RuntimeError(
                            "merge_and_unload() was a no-op: %s unchanged. Adapter did "
                            "not attach, so activations would be the BASE model's." % pname)
                    logger.info("adapter merged, delta norm %.4g at %s",
                                (after - probe).norm().item(), pname)
                    break
        model.config.output_hidden_states = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=True,
            output_hidden_states=True,
        )
    model.eval()'''

assert old_load in src, "load block not found verbatim -- activations.py may have changed"
src = src.replace(old_load, new_load)

f.write_text(src)
print("patched: added base_model_id field and base+adapter merge path")
print("  - CaptureConfig now accepts base_model_id")
print("  - when set, loads base, attaches adapter, merges, verifies, dispatches")
