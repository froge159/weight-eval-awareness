"""Does the weight edit actually implement the ablation we intended?

Two different questions, and the proposal's phrasing runs them together. Keeping
them apart is the whole point of this module.

(a) Equivalence -- gates the run
--------------------------------
The claim the weight edit makes is that removing the ``r_hat`` component from
``W`` is the same thing as removing it from that module's output. Since
``o_proj`` and ``down_proj`` carry no bias, that is an exact algebraic identity::

    W'x = (W - r_hat r_hat^T W) x = (I - r_hat r_hat^T)(Wx)

so it is testable rather than merely plausible, at two levels:

*Per module.* Snapshot ``W`` before the edit, then compare ``W'x`` against
``(I - r_hat r_hat^T)(Wx)`` in float64 on random ``x``. The only thing that can
separate them is the bf16 rounding of the stored ``W'``.

*End to end.* Run the merged, unedited model with forward hooks that project the
same modules' outputs against the same per-block directions; then apply the
weight edit and re-run the identical prompts. If the edit means what we think,
the two produce the same hidden states. One model is resident at a time -- the
hooked pass runs first, then the weights are edited in place.

The tolerance is relative and calibrated to bf16, not 1e-6. bf16 carries an
8-bit mantissa (eps ~ 4e-3), and the end-to-end path compounds that through
every edited block, so demanding float64-style agreement would fail a correct
implementation.

(b) Residual leakage -- reported, never gated
---------------------------------------------
The proposal also asks whether edited activations match the activation-level
projection ``h~ = h - r_hat r_hat^T h``. Under a band-restricted edit that
identity **cannot** hold, for two independent reasons:

1. Blocks below the band, and ``embed_tokens``, still write along ``r_hat``.
   Nothing in this edit stops them.
2. Ablation changes downstream computation nonlinearly -- attention softmaxes
   and SwiGLU gates see different inputs -- so the edited model's activations are
   not the projection of the original model's activations. They are the
   activations of a different function.

So we report ``|r_hat^(l) . h^(l)| / ||h^(l)||`` before and after at each measured
layer and let the profile speak. It will not be zero, and reporting it as though
it were would be false. What it should show is a clear drop inside the band.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import REPO

# isort: split
from evalaware.dataset import build_pairs, load_questions, make_splits, select
from evalaware.validation import assert_finite, check

from .directions import DirectionSet, PerBlockDirections
from .surgery import enumerate_targets

logger = logging.getLogger(__name__)

__all__ = [
    "FidelityReport",
    "ModuleProbe",
    "DEFAULT_QUESTIONS",
    "PHASE1_TOKEN_OFFSET",
    "holdout_prompts",
    "snapshot_probes",
    "check_module_equivalence",
    "pooled_forward",
    "projection_hooks",
    "compare_hidden_states",
    "drift_summary",
    "leakage_profile",
    "prepare_for_inference",
]

DEFAULT_QUESTIONS = REPO / "src" / "extraction" / "data" / "questions.jsonl"

# PHASE2_HANDOFF.md section 7 records the Phase 1 capture as using token offset
# -2. Note that ``evalaware/activations.py`` pools at -1 (``hs[0, -1, :]``), so
# the shipped directions came from the notebook path rather than that function.
# The choice does not affect the equivalence test -- both passes use the same
# position -- but it does decide which position the leakage profile describes,
# so it follows the handoff and is overridable.
PHASE1_TOKEN_OFFSET = -2

# bf16 has an 8-bit mantissa. Anything at or below this scale is storage
# rounding, not a defect in the edit.
BF16_EPS = 2.0 ** -8


@dataclass
class ModuleProbe:
    """A pre-edit copy of one module's weight, kept to test the identity later."""

    name: str
    block: int
    kind: str
    weight_ref: Any            # torch.Tensor, pre-edit W, on CPU
    direction: np.ndarray


@dataclass
class FidelityReport:
    module_results: list[dict[str, Any]]
    module_max_rel_err: float
    end_to_end: dict[str, Any]
    leakage: dict[str, Any]
    n_prompts: int
    token_offset: int
    passed: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_prompts": self.n_prompts,
            "token_offset": self.token_offset,
            "module_equivalence": {
                "max_rel_err": float(f"{self.module_max_rel_err:.6g}"),
                "bf16_eps": BF16_EPS,
                "per_module": self.module_results,
            },
            "end_to_end": self.end_to_end,
            "leakage": self.leakage,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Held-out inputs
# --------------------------------------------------------------------------


def holdout_prompts(
    n_pairs: int = 8,
    seed: int = 0,
    questions_path: str | Path = DEFAULT_QUESTIONS,
    situational_awareness_hint: bool = True,
) -> list[Any]:
    """Prompt specs from the ``ood_holdout`` split -- unseen during extraction.

    Reuses Phase 1's pair construction and splitting so "held out" means the same
    thing it meant there: neither the template family nor the question was used
    to estimate r_hat.
    """
    questions = load_questions(questions_path)
    items = build_pairs(
        questions, situational_awareness_hint=situational_awareness_hint
    )
    splits = make_splits(items, seed=seed)
    holdout = select(items, splits, "ood_holdout")

    # Deterministic and framing-balanced: the same n pairs every run, each
    # contributing both its eval and its deploy side.
    chosen = sorted(holdout, key=lambda it: it.pair_key)[:n_pairs]
    check(
        len(chosen) == n_pairs,
        f"ood_holdout has only {len(holdout)} pair(s), fewer than the {n_pairs} "
        f"requested.",
    )
    specs = []
    for item in chosen:
        specs.extend([item.eval_spec, item.deploy_spec])
    return specs


def _render(tokenizer: Any, spec: Any) -> str:
    """Chat-template rendering, identical to ``evalaware.activations.capture``."""
    messages = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": spec.user_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------
# (a) Equivalence, per module
# --------------------------------------------------------------------------


def snapshot_probes(
    model: Any, plan: PerBlockDirections, n_modules: int = 6
) -> list[ModuleProbe]:
    """Copy a spread of in-band weights **before** the edit, to compare after.

    A sample rather than all 54: the in-band ``down_proj`` matrices alone are
    tens of GB, and the identity is per-module algebra -- if it holds for a
    spread of blocks and both module kinds it holds everywhere. The spread is
    deterministic (evenly spaced through the band, both kinds represented).
    """
    import torch

    targets = [t for t in enumerate_targets(model) if t.block in plan.vectors]
    check(targets, "snapshot_probes: no in-band modules to probe")

    picked: list[Any] = []
    for kind in ("o_proj", "down_proj"):
        of_kind = [t for t in targets if t.kind == kind]
        if not of_kind:
            continue
        want = max(1, n_modules // 2)
        step = max(1, len(of_kind) // want)
        picked.extend(of_kind[::step][:want])

    probes = []
    for t in picked:
        with torch.no_grad():
            ref = t.module.weight.detach().to("cpu").clone()
        probes.append(
            ModuleProbe(
                name=t.name,
                block=t.block,
                kind=t.kind,
                weight_ref=ref,
                direction=plan.for_block(t.block),
            )
        )
    logger.info(
        "snapshotted %d module(s) for the equivalence check: %s",
        len(probes), ", ".join(p.name for p in probes),
    )
    return probes


def check_module_equivalence(
    model: Any,
    probes: Sequence[ModuleProbe],
    n_vectors: int = 8,
    seed: int = 0,
    rel_tol: float = 8 * BF16_EPS,
) -> tuple[list[dict[str, Any]], float]:
    """Verify ``W'x == (I - r r^T)(Wx)`` on random ``x``, in float64.

    Called after the edit; ``probes`` hold the pre-edit weights.
    """
    import torch

    check(bool(probes), "check_module_equivalence: no probes")
    modules = dict(model.named_modules())
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for probe in probes:
        edited = modules[probe.name].weight.detach().to("cpu", torch.float64)
        reference = probe.weight_ref.to(torch.float64)
        r = torch.as_tensor(probe.direction, dtype=torch.float64)

        x = torch.as_tensor(
            rng.standard_normal((int(reference.shape[1]), n_vectors)),
            dtype=torch.float64,
        )
        x /= torch.linalg.vector_norm(x, dim=0, keepdim=True)

        direct = edited @ x                              # W' x
        y = reference @ x                                # W x
        projected = y - torch.outer(r, r @ y)            # (I - r r^T) W x

        err = torch.linalg.vector_norm(direct - projected, dim=0)
        scale = torch.linalg.vector_norm(projected, dim=0)
        rel = (err / scale).max().item()

        # Independent of the above: the edited matrix must not write along r at
        # all. Measured on the stored weight, so it includes bf16 rounding.
        residual = float(
            torch.linalg.vector_norm(r @ edited)
            / torch.linalg.vector_norm(r @ reference)
        )
        results.append(
            {
                "name": probe.name,
                "block": probe.block,
                "kind": probe.kind,
                "max_rel_err": float(f"{rel:.6g}"),
                "residual_projection_ratio": float(f"{residual:.6g}"),
            }
        )
        del edited, reference, direct, y, projected

    worst = max(r["max_rel_err"] for r in results)
    check(
        worst <= rel_tol,
        f"module equivalence FAILED: worst relative error {worst:.3e} exceeds "
        f"{rel_tol:.3e} ({rel_tol / BF16_EPS:.0f} x bf16 eps). W'x and "
        f"(I - r r^T)Wx should differ only by the bf16 rounding of W'. A larger "
        f"gap means the edit is not the projection it claims to be -- check that "
        f"the direction assigned to each block is the one that was applied.",
    )
    logger.info(
        "module equivalence OK: worst relative error %.3e (%.1f x bf16 eps)",
        worst, worst / BF16_EPS,
    )
    return results, worst


# --------------------------------------------------------------------------
# (a) Equivalence, end to end + (b) leakage
# --------------------------------------------------------------------------


def projection_hooks(model: Any, plan: PerBlockDirections) -> list[Any]:
    """Hook the in-band modules so their **outputs** are projected off r_hat.

    The activation-level twin of the weight edit. Returns handles; the caller
    must remove them.
    """
    import torch

    targets = [t for t in enumerate_targets(model) if t.block in plan.vectors]
    check(targets, "projection_hooks: no in-band modules")
    handles = []

    for t in targets:
        r_np = plan.for_block(t.block)

        def make_hook(r_np=r_np):
            def hook(module, inputs, output):
                r = torch.as_tensor(
                    r_np, dtype=torch.float32, device=output.device
                )
                out32 = output.to(torch.float32)
                coeff = out32 @ r                       # (..., ) projection scalar
                return (out32 - coeff.unsqueeze(-1) * r).to(output.dtype)

            return hook

        handles.append(t.module.register_forward_hook(make_hook()))

    logger.info("registered %d output-projection hook(s)", len(handles))
    return handles


def pooled_forward(
    model: Any,
    tokenizer: Any,
    specs: Sequence[Any],
    token_offset: int = PHASE1_TOKEN_OFFSET,
) -> dict[str, np.ndarray]:
    """One forward pass per prompt; keep the pooled residual stream and logits.

    Returns ``hidden`` of shape ``(n_prompts, n_layers + 1, d)`` -- index 0 is
    the embedding output, matching Phase 1's ``hidden_states`` indexing -- and
    ``logits`` at the same position.
    """
    import torch

    hidden_rows, logit_rows, n_tokens = [], [], []
    for i, spec in enumerate(specs):
        text = _render(tokenizer, spec)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        check(
            n_prompt >= abs(token_offset),
            f"prompt {i} has {n_prompt} token(s), too short for offset "
            f"{token_offset}.",
        )
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden_rows.append(
            torch.stack([hs[0, token_offset, :] for hs in out.hidden_states])
            .float().cpu().numpy()
        )
        logit_rows.append(out.logits[0, token_offset, :].float().cpu().numpy())
        n_tokens.append(n_prompt)
        del out
        if (i + 1) % 4 == 0:
            logger.info("  forward %d/%d", i + 1, len(specs))

    hidden = np.stack(hidden_rows).astype(np.float32)
    logits = np.stack(logit_rows).astype(np.float32)
    assert_finite(hidden, "pooled hidden states")
    assert_finite(logits, "pooled logits")
    return {"hidden": hidden, "logits": logits, "n_tokens": np.array(n_tokens)}


def compare_hidden_states(
    reference: dict[str, np.ndarray],
    edited: dict[str, np.ndarray],
    rel_tol: float = 0.05,
) -> dict[str, Any]:
    """Hooked-unedited vs weight-edited. These should be the same computation.

    ``rel_tol`` is deliberately loose: bf16 rounding at each of the edited blocks
    compounds through the residual stream, so the end-to-end figure is expected
    to sit well above the per-module one. The per-module check is the strict
    gate; this one catches a structural mistake, such as the hooks and the weight
    edit covering different module sets.
    """
    a, b = reference["hidden"], edited["hidden"]
    check(
        a.shape == b.shape,
        f"hidden state shape mismatch {a.shape} vs {b.shape} -- the two passes "
        f"did not see the same prompts.",
    )
    check(
        np.array_equal(reference["n_tokens"], edited["n_tokens"]),
        "the two passes tokenized to different lengths; inputs were not identical.",
    )

    diff = np.linalg.norm(a - b, axis=-1)                 # (n_prompts, n_layers)
    scale = np.linalg.norm(a, axis=-1)
    rel = diff / np.maximum(scale, 1e-12)

    la, lb = reference["logits"], edited["logits"]
    logit_rel = float(
        np.max(
            np.linalg.norm(la - lb, axis=-1)
            / np.maximum(np.linalg.norm(la, axis=-1), 1e-12)
        )
    )
    agree = float(np.mean(la.argmax(axis=-1) == lb.argmax(axis=-1)))

    worst = float(rel.max())
    passed = worst <= rel_tol
    return {
        "max_rel_err": float(f"{worst:.6g}"),
        "median_rel_err": float(f"{float(np.median(rel)):.6g}"),
        "per_layer_max_rel_err": [float(f"{v:.6g}") for v in rel.max(axis=0)],
        "logit_max_rel_err": float(f"{logit_rel:.6g}"),
        "argmax_agreement": agree,
        "rel_tol": rel_tol,
        "passed": passed,
    }


def drift_summary(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> dict[str, Any]:
    """How far the edit moved the model's predictions from the original.

    Not a fidelity check -- ``before`` is the *unedited* model, so these numbers
    are supposed to be nonzero; that is the intervention working. It is a health
    check. Removing a direction from 54 matrices could in principle wreck the
    model, and "it still loads and produces finite logits" would not notice.
    Comparing next-token distributions on held-out prompts does, and it costs
    nothing: both logit sets were already collected for the other checks.

    Cheaper and sharper than eyeballing a generated sample, which on a
    CPU-offloaded 49B would also cost minutes per prompt.
    """
    la, lb = before["logits"].astype(np.float64), after["logits"].astype(np.float64)
    check(la.shape == lb.shape, f"logit shape mismatch {la.shape} vs {lb.shape}")

    def _log_softmax(x: np.ndarray) -> np.ndarray:
        z = x - x.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    log_p, log_q = _log_softmax(la), _log_softmax(lb)
    kl = float(np.mean((np.exp(log_p) * (log_p - log_q)).sum(axis=-1)))

    top1 = float(np.mean(la.argmax(-1) == lb.argmax(-1)))
    ranked = np.argsort(-la, axis=-1)[:, :5]
    top5 = float(np.mean([
        lb[i].argmax() in ranked[i] for i in range(len(lb))
    ]))
    return {
        "top1_agreement_with_unedited": top1,
        "edited_top1_in_unedited_top5": top5,
        "mean_kl_unedited_to_edited_nats": float(f"{kl:.6g}"),
        "interpretation": (
            "These compare the edited model against the UNEDITED one, so nonzero "
            "is the point. Read them as a health check: a KL of a few tenths of a "
            "nat with most top-1 predictions preserved means the model still "
            "computes, and the edit changed it rather than broke it. A KL in the "
            "tens, or top-1 agreement near chance, would mean the ablation "
            "damaged general capability and any Phase 3 gap closure would be "
            "confounded by that damage."
        ),
    }


def leakage_profile(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    dset: DirectionSet,
    band: tuple[int, int],
) -> dict[str, Any]:
    """``|r_hat^(l) . h^(l)| / ||h^(l)||`` at each measured layer, pre and post.

    Diagnostic only. See the module docstring for why this cannot reach zero and
    must not be reported as though it does.
    """
    rows = []
    for layer in dset.layers:
        r = dset.vectors[layer]
        out = {}
        for label, data in (("before", before), ("after", after)):
            h = data["hidden"][:, layer, :].astype(np.float64)
            norms = np.linalg.norm(h, axis=-1)
            out[label] = float(np.mean(np.abs(h @ r) / np.maximum(norms, 1e-12)))
        ratio = out["after"] / out["before"] if out["before"] > 0 else 0.0
        rows.append(
            {
                "layer": layer,
                "in_band": bool(band[0] <= layer <= band[1]),
                "leakage_before": float(f"{out['before']:.6g}"),
                "leakage_after": float(f"{out['after']:.6g}"),
                "ratio": float(f"{ratio:.6g}"),
            }
        )

    in_band = [r for r in rows if r["in_band"]]
    out_band = [r for r in rows if not r["in_band"]]
    return {
        "per_layer": rows,
        "mean_ratio_in_band": float(np.mean([r["ratio"] for r in in_band])),
        "mean_ratio_out_of_band": (
            float(np.mean([r["ratio"] for r in out_band])) if out_band else None
        ),
        "interpretation": (
            "Leakage after the edit is nonzero by construction: blocks below the "
            "band and embed_tokens still write along r_hat, and ablation changes "
            "downstream computation nonlinearly, so edited activations are not "
            "the projection of the original ones. Expect a clear drop inside the "
            "band and little change outside it. A flat profile means the edit did "
            "not take; a uniform drop everywhere means the band restriction is "
            "not doing what it claims."
        ),
    }


def _parse_memory(spec: str) -> int:
    """'36GiB' -> bytes. Accepts the same suffixes accelerate's max_memory does."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B)\s*", spec, re.IGNORECASE)
    check(m is not None, f"cannot parse a memory size from {spec!r} (e.g. '36GiB')")
    units = {
        "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
        "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
    }
    return int(float(m.group(1)) * units[m.group(2).lower()])


def prepare_for_inference(
    model: Any, gpu_memory: str = "36GiB", cpu_memory: str = "180GiB"
) -> Any:
    """Spread a CPU-resident model across GPU + CPU for the forward passes.

    The bf16 checkpoint is ~100GB and the GPU here holds 40GB, so inference has
    to offload.

    **Call this only after any weight edit.** When the main device is a GPU,
    ``dispatch_model`` treats CPU-assigned modules as offloaded: it replaces
    their weights with meta tensors and streams the real values from a side
    weights-map during the forward. A weight edit applied afterwards has nothing
    to write to. Disk offload is rejected outright below for the same reason,
    and more sharply -- those weights are re-read from the original checkpoint.
    """
    import torch

    if not torch.cuda.is_available():
        logger.warning("no CUDA device; forward passes will run on CPU (slow)")
        return model

    free_b, total_b = torch.cuda.mem_get_info()
    want_b = _parse_memory(gpu_memory)
    check(
        free_b >= want_b,
        f"only {free_b / 2**30:.1f} GiB of {total_b / 2**30:.1f} GiB free on GPU 0, "
        f"but the device map asks for {gpu_memory}. Almost always a previous "
        f"model still holding device memory: clear every reference to it, then "
        f"gc.collect() and torch.cuda.empty_cache(). Dispatching anyway would "
        f"fail deep inside accelerate with a bare CUDA OOM.",
    )

    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_submodules

    # Loading with device_map="cpu" can leave accelerate hooks attached.
    # Dispatching on top of those double-hooks every module, so clear them
    # first; this is a no-op when there are none.
    remove_hook_from_submodules(model)

    no_split = list(getattr(model, "_no_split_modules", None) or [])
    if not no_split:
        # Without this accelerate may split a single decoder block across the
        # GPU/CPU boundary, which works but thrashes. The remote-code DeciLM
        # class does not always declare it, so fall back to the block class.
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None and len(layers):
            no_split = [type(layers[0]).__name__]
            logger.info("no _no_split_modules declared; using %s", no_split)

    device_map = infer_auto_device_map(
        model,
        max_memory={0: gpu_memory, "cpu": cpu_memory},
        no_split_module_classes=list(no_split),
        dtype=model.dtype,
    )
    check(
        "disk" not in set(device_map.values()),
        f"accelerate wants to offload part of the model to disk with "
        f"max_memory={{0: {gpu_memory!r}, 'cpu': {cpu_memory!r}}}. Disk-offloaded "
        f"weights are reloaded from the original checkpoint and would silently "
        f"drop the edit. Raise cpu_memory or export the edited model first.",
    )
    logger.info(
        "dispatching across %s", sorted({str(d) for d in device_map.values()})
    )
    return dispatch_model(model, device_map=device_map)
