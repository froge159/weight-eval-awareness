"""Exercise run_phase3.py's model paths against transformers 4.48.3 + peft 0.14.0.

No downloads: builds a tiny Llama with random weights and an in-memory LoRA that
targets the same modules as the real organism (o_proj, down_proj). The point is
to verify the *API calls and control flow*, not the science.
"""
import json, tempfile, warnings
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast)
from peft import LoraConfig, PeftModel, get_peft_model

warnings.filterwarnings("ignore")
OK, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        OK.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")


tmp = Path(tempfile.mkdtemp())

# --- tokenizer with a BOS-emitting chat template, like Llama ----------------
vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
for i, w in enumerate("write a function def return int str hello world code please "
                      "system user assistant think".split()):
    vocab[w] = 4 + i
bk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
bk.pre_tokenizer = pre_tokenizers.Whitespace()
bk.post_processor = processors.TemplateProcessing(single="<bos> $A",
                                                  special_tokens=[("<bos>", 1)])
tok = PreTrainedTokenizerFast(tokenizer_object=bk, bos_token="<bos>",
                              eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
tok.model_input_names = ["input_ids", "attention_mask"]  # as real Llama tokenizers do
tok.chat_template = ("{{ bos_token }}{% for m in messages %}{{ m['role'] }} "
                     "{{ m['content'] }} {% endfor %}"
                     "{% if add_generation_prompt %}assistant {% endif %}")

# --- tiny Llama, saved to disk so from_pretrained is genuinely exercised -----
cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  bos_token_id=1, eos_token_id=2, pad_token_id=3)
LlamaForCausalLM(cfg).save_pretrained(tmp / "base")

print("\n=== 1. loading kwargs, as run_phase3.build_model uses them ===")

def t_torch_dtype():
    m = AutoModelForCausalLM.from_pretrained(tmp / "base", torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
    d = next(m.parameters()).dtype
    assert d == torch.bfloat16, f"got {d}"
    return f"dtype is {d}"
check("torch_dtype= is honoured on 4.48.3", t_torch_dtype)

def t_dtype_alias():
    """The name my earlier version used. Confirm it silently does NOT work."""
    m = AutoModelForCausalLM.from_pretrained(tmp / "base", dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
    return f"dtype= gives {next(m.parameters()).dtype} (fp32 = the bug)"
def t_dtype_alias_raises():
    try:
        AutoModelForCausalLM.from_pretrained(tmp / "base", dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True)
    except TypeError as e:
        return "dtype= raises TypeError on 4.48.3 -- the old code would have crashed"
    raise AssertionError("expected dtype= to be rejected")
check("dtype= is rejected (confirms the fix was needed)", t_dtype_alias_raises)

def t_max_memory():
    m = AutoModelForCausalLM.from_pretrained(
        tmp / "base", torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map="auto", max_memory={"cpu": "2GiB"},
        offload_folder=str(tmp / "offload"))
    return "max_memory dict accepted by accelerate"
check("max_memory format is valid", t_max_memory)

print("\n=== 2. adapter attach without merging ===")

base = AutoModelForCausalLM.from_pretrained(tmp / "base", torch_dtype=torch.float32)
lcfg = LoraConfig(r=8, lora_alpha=16, target_modules=["o_proj", "down_proj"],
                  task_type="CAUSAL_LM")
get_peft_model(AutoModelForCausalLM.from_pretrained(tmp / "base",
                                                    torch_dtype=torch.float32),
               lcfg).save_pretrained(tmp / "adapter")

peft_model = PeftModel.from_pretrained(base, tmp / "adapter")

def t_lora_count():
    n = sum(1 for n_, _ in peft_model.named_modules() if "lora_" in n_)
    assert n > 0, "no LoRA submodules found"
    return f"{n} LoRA submodules detected"
check("LoRA submodule count works (the merge-free adapter check)", t_lora_count)

def t_input_device():
    for obj in (peft_model, getattr(peft_model, "base_model", None)):
        if obj is None:
            continue
        try:
            return f"input device {obj.get_input_embeddings().weight.device}"
        except Exception:
            continue
    raise AssertionError("could not resolve input device")
check("input_device() resolves through the PeftModel wrapper", t_input_device)

def t_adapter_changes_output():
    """An attached adapter must actually alter the computation."""
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        on = peft_model(ids).logits
        with peft_model.disable_adapter():
            off = peft_model(ids).logits
    # Random init makes B zero, so seed the LoRA B matrices to make it visible.
    for n_, p in peft_model.named_parameters():
        if "lora_B" in n_:
            torch.nn.init.normal_(p, std=0.5)
    with torch.no_grad():
        on2 = peft_model(ids).logits
    assert not torch.allclose(on2, off), "adapter has no effect on logits"
    return f"max delta {float((on2 - off).abs().max()):.4f}"
check("attached adapter changes the output", t_adapter_changes_output)

print("\n=== 3. generation through an UNMERGED PeftModel ===")

tok.padding_side = "left"

def t_generate_unmerged():
    msgs = [{"role": "system", "content": "write code"},
            {"role": "user", "content": "hello world"}]
    texts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)] * 2
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    n_bos = int((enc["input_ids"] == 1).sum(dim=1)[0])
    assert n_bos == 1, f"{n_bos} BOS tokens -- double-BOS bug"
    with torch.no_grad():
        out = peft_model.generate(**enc, max_new_tokens=8, do_sample=False,
                                  temperature=None, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return f"generated {tuple(gen.shape)}, exactly 1 BOS in the prompt"
check("generate() works unmerged, single BOS", t_generate_unmerged)

def t_temperature_none():
    enc = tok(["write a function"], return_tensors="pt", add_special_tokens=False)
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        peft_model.generate(**enc, max_new_tokens=4, do_sample=False,
                            temperature=None, pad_token_id=tok.pad_token_id)
    return "temperature=None with do_sample=False accepted"
check("greedy decoding with temperature=None", t_temperature_none)

print("\n=== 4. truncation detection on real batched output ===")

def t_truncation():
    eos = tok.eos_token_id
    # Row 0 stops early, row 1 runs to the cap -- the batching case that broke.
    batch = torch.tensor([[7, 8, eos, 3, 3, 3], [7, 8, 9, 10, 11, 12]])
    res = []
    for ids in batch:
        hit = (ids == eos).nonzero()
        res.append((int(hit[0]) + 1, False) if len(hit) else (int(ids.shape[0]), True))
    assert res == [(3, False), (6, True)], res
    return f"{res}"
check("per-row length and truncation flag", t_truncation)

print("\n=== 5. JSONL append and resume ===")

def t_resume():
    import hashlib
    p = tmp / "generations.jsonl"

    def key(tag, pk, fr):
        return hashlib.sha256(f"{tag}|{pk}|{fr}".encode()).hexdigest()[:16]

    with p.open("a") as fh:
        for fr in ("eval", "deploy"):
            fh.write(json.dumps({"key": key("organism", "q1", fr), "framing": fr}) + "\n")
    fh = p.open("a"); fh.write('{"key": "truncated-line'); fh.close()   # crash mid-write

    done = set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            pass
    assert len(done) == 2, done
    assert key("organism", "q1", "eval") in done
    assert key("edited", "q1", "eval") not in done   # a different model re-runs
    return "2 keys recovered, malformed tail skipped, model_tag scoping correct"
check("resume survives a half-written line", t_resume)

print("\n" + "=" * 62)
print(f"{len(OK)} passed, {len(FAIL)} failed")
for n, e in FAIL:
    print(f"  FAILED: {n}\n    {e}")
