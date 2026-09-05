#!/usr/bin/env python3
"""Patch capture() to batch completion_mean generation.

The current loop does one prompt per model.generate() call -- 7392 sequential
generations, ~5 days. This batches them. The tricky part is that generate()
returns hidden_states as [step][layer] -> (batch, 1, d), and different sequences
in a batch finish at different steps (the rest is padding). completion_mean must
average each sequence over ONLY its real generated tokens, not padded ones.

We detect a sequence's real length from where its generated token first equals
eos/pad after the prompt, and mask padded steps out of the per-sequence mean.

A self-check (--verify in the smoke config path) compares batched vs single-prompt
pooling on the first few specs; they must match to ~1e-4 or the mask is wrong.

Run ONCE on the pod from src/extraction:
    python scripts/batch_patch.py
Idempotent.
"""
from __future__ import annotations
import sys
from pathlib import Path

f = Path("src/evalaware/activations.py")
s = f.read_text()

if "BATCHED completion_mean" in s:
    print("already patched")
    sys.exit(0)

# Replace the whole per-spec loop (from 'for i, spec in enumerate(todo):' up to
# just before 'return {') with a batched version.
start = s.index("    for i, spec in enumerate(todo):")
end = s.index("    return {\n        \"captured\": captured,")
old_block = s[start:end]

new_block = '''    # ---- BATCHED completion_mean -------------------------------------------
    tokenizer.padding_side = "left"   # so prompt tokens align at the right edge
    bs = max(1, int(config.batch_size))
    eos_id = tokenizer.eos_token_id

    def _pool_last_prompt(batch_specs):
        texts = []
        for spec in batch_specs:
            msgs = []
            if spec.system_prompt:
                msgs.append({"role": "system", "content": spec.system_prompt})
            msgs.append({"role": "user", "content": spec.user_prompt})
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # left-padded, so the last real prompt token is at index config.token_offset
        pooled = []
        for b in range(len(batch_specs)):
            pooled.append(torch.stack(
                [hs[b, config.token_offset, :].float() for hs in out.hidden_states],
                dim=0))
        n_prompt = [int(enc["attention_mask"][b].sum()) for b in range(len(batch_specs))]
        return pooled, [None] * len(batch_specs), n_prompt

    def _pool_completion_mean(batch_specs):
        texts = []
        for spec in batch_specs:
            msgs = []
            if spec.system_prompt:
                msgs.append({"role": "system", "content": spec.system_prompt})
            msgs.append({"role": "user", "content": spec.user_prompt})
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=config.temperature if config.temperature > 0 else None,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        check(getattr(gen, "hidden_states", None),
              "generation returned no hidden states.")
        n_steps = len(gen.hidden_states)
        n_layers_ = len(gen.hidden_states[0])
        # real length per sequence: number of generated tokens before the first
        # eos (inclusive of it). gen.sequences is (batch, prompt_len + n_steps).
        gen_ids = gen.sequences[:, prompt_len:]           # (batch, n_steps)
        real_len = []
        for b in range(gen_ids.shape[0]):
            row = gen_ids[b]
            hit = (row == eos_id).nonzero()
            real_len.append(int(hit[0]) + 1 if len(hit) else int(row.shape[0]))

        pooled, texts_out, n_prompt = [], [], []
        for b in range(len(batch_specs)):
            L = max(1, real_len[b])
            per_layer = []
            for layer_idx in range(n_layers_):
                # step s, this sequence's newly generated token is at position -1
                vecs = [gen.hidden_states[s][layer_idx][b, -1, :].float()
                        for s in range(L)]
                per_layer.append(torch.stack(vecs, dim=0).mean(dim=0))
            pooled.append(torch.stack(per_layer, dim=0))
            texts_out.append(tokenizer.decode(gen_ids[b, :real_len[b]],
                                              skip_special_tokens=True))
            n_prompt.append(int(enc["attention_mask"][b].sum()))
        return pooled, texts_out, n_prompt

    _pool = _pool_last_prompt if config.pooling == "last_prompt_token" else _pool_completion_mean

    done_ct = 0
    for start_i in range(0, len(todo), bs):
        batch_specs = todo[start_i:start_i + bs]
        pooled_list, texts_out, n_prompt = _pool(batch_specs)
        for spec, pooled, ctext, npt in zip(batch_specs, pooled_list, texts_out, n_prompt):
            arr = pooled.float().cpu().numpy()
            if config.layers is not None:
                arr = arr[list(config.layers)]
            assert_shape(arr, (len(wanted), hidden_dim),
                         f"pooled activations [{config.pooling}]")
            assert_finite(arr, f"pooled activations for {getattr(spec, 'pair_key', '?')}")
            store.put(
                spec.system_prompt, spec.user_prompt, arr,
                metadata={
                    "pair_key": getattr(spec, "pair_key", None),
                    "framing": getattr(spec, "framing", None),
                    "regime": getattr(spec, "regime", None),
                    "question_id": getattr(spec, "question_id", None),
                    "n_prompt_tokens": npt,
                    "completion": ctext,
                },
            )
            captured += 1
        done_ct += len(batch_specs)
        if show_progress:
            logger.info("  captured %d/%d", done_ct, len(todo))
'''

s = s[:start] + new_block + s[end:]
f.write_text(s)
print("patched: batched completion_mean generation")
print("  batch_size from config is now honoured (was ignored)")
print("  padded steps masked out of the per-sequence mean")
