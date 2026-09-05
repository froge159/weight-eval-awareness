"""Residual-stream activation capture, with both pooling schemes.

torch and transformers are imported lazily so the rest of the package (and the
whole test suite) runs without a GPU stack.

Pooling schemes
---------------
``last_prompt_token``
    Residual stream at the final prompt position, one vector per layer. Single
    forward pass, no generation. This is the Arditi et al. convention.

``completion_mean``
    Mean residual stream over generated tokens. Requires generation, so it is
    much more expensive, but the organism is a reasoning model whose
    eval/deploy determination happens inside the chain of thought -- after the
    prompt ends. See RESEARCH_DECISIONS.md 2026-08-08T12:50Z.

Caching
-------
Every capture is keyed by a SHA-256 of (model id, revision, prompt text,
pooling, generation config, dtype). A 49B reasoning-model generation run is
long enough that resumability is a correctness feature, not a convenience --
without it a crash at hour six silently tempts you into analyzing a partial
matrix.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .validation import assert_finite, assert_shape, check

logger = logging.getLogger(__name__)

__all__ = ["CaptureConfig", "ActivationStore", "capture", "POOLING_SCHEMES"]

POOLING_SCHEMES = ("last_prompt_token", "completion_mean")


@dataclass(frozen=True)
class CaptureConfig:
    model_id: str
    base_model_id: str | None = None   # if set, load base + adapter(model_id), merge
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    pooling: str = "last_prompt_token"
    # -1 is a chat-template control token, not the last content token
    token_offset: int = -2
    max_new_tokens: int = 512
    temperature: float = 0.0
    layers: tuple[int, ...] | None = None       # None = every layer
    batch_size: int = 4
    # Capture in float32 regardless of model dtype: bf16 has ~3 decimal digits
    # of precision, and accumulating a mean over hundreds of tokens in bf16
    # loses real signal.
    store_dtype: str = "float32"

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


class ActivationStore:
    """Content-addressed on-disk cache of captured activations.

    Layout::

        root/<config_fingerprint>/<prompt_hash>.npz   # arr: (n_layers, d)
        root/<config_fingerprint>/manifest.jsonl
    """

    def __init__(self, root: str | Path, config: CaptureConfig):
        self.root = Path(root) / config.fingerprint()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.manifest_path = self.root / "manifest.jsonl"
        if not self.manifest_path.exists():
            self.manifest_path.write_text("", encoding="utf-8")
            (self.root / "config.json").write_text(
                json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
            )

    @staticmethod
    def prompt_hash(system_prompt: str, user_prompt: str) -> str:
        h = hashlib.sha256()
        # Length-prefixed so ("ab","c") and ("a","bc") cannot collide.
        for part in (system_prompt, user_prompt):
            h.update(str(len(part)).encode("ascii"))
            h.update(b"\x00")
            h.update(part.encode("utf-8"))
        return h.hexdigest()[:32]

    def path_for(self, system_prompt: str, user_prompt: str) -> Path:
        return self.root / f"{self.prompt_hash(system_prompt, user_prompt)}.npz"

    def has(self, system_prompt: str, user_prompt: str) -> bool:
        return self.path_for(system_prompt, user_prompt).exists()

    def put(
        self,
        system_prompt: str,
        user_prompt: str,
        activations: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(activations, dtype=np.float32)
        assert_shape(arr, (None, None), "activations")
        assert_finite(arr, "activations")
        path = self.path_for(system_prompt, user_prompt)
        # Write to a temp file then rename: an interrupted write must not leave
        # a truncated .npz that a later run would happily load. Pass a file
        # handle, since np.savez_compressed appends '.npz' to a path argument
        # that does not already end in it.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, arr=arr)
        tmp.replace(path)
        rec = {
            "prompt_hash": self.prompt_hash(system_prompt, user_prompt),
            "shape": list(arr.shape),
            **(metadata or {}),
        }
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def get(self, system_prompt: str, user_prompt: str) -> np.ndarray:
        path = self.path_for(system_prompt, user_prompt)
        check(path.exists(), f"activation cache miss: {path}")
        with np.load(path) as data:
            arr = data["arr"]
        assert_finite(arr, f"cached activations {path.name}")
        return arr

    def stack(
        self,
        specs: Sequence,
        layer: int,
        layer_index: dict[int, int] | None = None,
    ) -> np.ndarray:
        """Assemble an (N, d) matrix for one layer, in the given row order.

        Row order is the caller's order and is never sorted here -- pairing
        depends on it.
        """
        check(len(specs) > 0, "stack: no specs")
        rows = []
        missing = []
        for spec in specs:
            if not self.has(spec.system_prompt, spec.user_prompt):
                missing.append(getattr(spec, "pair_key", "<unknown>"))
                continue
            arr = self.get(spec.system_prompt, spec.user_prompt)
            idx = layer_index[layer] if layer_index else layer
            check(
                0 <= idx < arr.shape[0],
                f"stack: layer index {idx} out of range for cached array with "
                f"{arr.shape[0]} layers.",
            )
            rows.append(arr[idx])
        check(
            not missing,
            f"stack: {len(missing)} prompt(s) not in cache, e.g. {missing[:3]}. "
            f"Run capture() first, or the capture run did not complete.",
        )
        out = np.stack(rows, axis=0).astype(np.float64)
        assert_shape(out, (len(specs), None), f"stacked layer {layer}")
        assert_finite(out, f"stacked layer {layer}")
        return out


def _resolve_dtype(name: str):
    import torch
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    check(name in mapping, f"unsupported dtype {name!r}, expected one of {sorted(mapping)}")
    return mapping[name]


def capture(
    specs: Sequence,
    config: CaptureConfig,
    store: ActivationStore,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Capture residual-stream activations for ``specs``, caching as it goes.

    Returns a report dict: counts, layer count, hidden dim, and any skips.
    """
    check(
        config.pooling in POOLING_SCHEMES,
        f"unknown pooling {config.pooling!r}, expected one of {POOLING_SCHEMES}",
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "capture() needs torch and transformers.\n"
            "  pip install 'torch>=2.1' 'transformers>=4.44' accelerate\n"
            "The rest of the package (extraction, checks, tests) runs without them."
        ) from exc

    todo = [s for s in specs if not store.has(s.system_prompt, s.user_prompt)]
    logger.info(
        "capture: %d spec(s), %d already cached, %d to run [%s]",
        len(specs), len(specs) - len(todo), len(todo), config.pooling,
    )
    if not todo:
        return {"captured": 0, "cached": len(specs), "pooling": config.pooling}

    dtype = _resolve_dtype(config.dtype)
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
            device_map="cpu",
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
        from accelerate import dispatch_model, infer_auto_device_map
        dmap = infer_auto_device_map(model, max_memory={0: "55GiB", 1: "78GiB", "cpu": "200GiB"}, no_split_module_classes=getattr(model, "_no_split_modules", None) or [])
        model = dispatch_model(model, device_map=dmap)
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
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    logger.info("model loaded: %d layers, hidden dim %d", n_layers, hidden_dim)

    wanted = tuple(range(n_layers + 1)) if config.layers is None else config.layers
    for layer in wanted:
        check(
            0 <= layer <= n_layers,
            f"requested layer {layer} but model has {n_layers} "
            f"(valid: 0..{n_layers}, where 0 is the embedding output).",
        )

    captured = 0
    # ---- BATCHED completion_mean -------------------------------------------
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
    return {
        "captured": captured,
        "cached": len(specs) - len(todo),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "pooling": config.pooling,
    }
