"""The weight edit itself: merge the adapter, then orthogonalize what writes out.

    W'_out = W_out - r_hat r_hat^T W_out

applied to every projection whose output lands in the residual stream.

Three ways this goes silently wrong, and what this module does about each
--------------------------------------------------------------------------

**The LoRA trap (PHASE2_HANDOFF.md section 2).** ``timhua/wood_v2_sftr4_filt`` is
not a model, it is a rank-64 adapter over ``nvidia/Llama-3_3-Nemotron-Super-49B-v1``,
and ``o_proj``/``down_proj`` -- the exact matrices being orthogonalized -- are
among its target modules. Edit ``W`` with the adapter still attached and PEFT
applies ``BA`` afterwards, re-introducing the component just removed. The model
loads, generates fluent text, and passes every shape and finiteness check; the
only symptom is a weak result, which reads as a negative finding. So: merge
first, then verify the merge actually changed weights, then edit.

**Heterogeneous NAS blocks (section 3).** The architecture is ``nemotron-nas``
(``DeciLMForCausalLM``), not stock Llama. 31 of its 80 blocks have
``attention.no_op: true`` and therefore have no ``o_proj`` at all, and
``ffn_mult`` varies over {0.5, 1.0, 1.3125, 2.625, 3.28125, 5.25} so
``down_proj`` has different input widths at different depths. A loop over
``range(80)`` assuming uniform blocks is wrong. Modules are enumerated from the
model, the count is checked against what ``config.json`` predicts, and every
output dimension is asserted to be the hidden size -- a module that does not
write to the residual stream must not be edited.

**Precision (section 4).** Phase 1's activations came from a third-party 4-bit
checkpoint, which is fine for choosing a direction but not for producing edited
weights: the edited model is the artifact every Phase 3 number depends on. The
merge and the edit happen at bf16 against the official checkpoint. The rank-1
update itself is computed in float32, because doing it in bf16 (8-bit mantissa)
would lose most of the correction.

Storing the result back in bf16 does mean ``r_hat^T W'`` is small but not
identically zero. That is measured and reported per module rather than assumed
away -- see ``fidelity`` for what it implies for the tolerances there.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import REPO  # noqa: F401  -- side effect: puts evalaware on sys.path

# isort: split
from evalaware.validation import check

from .directions import PerBlockDirections

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_MODEL_ID",
    "ADAPTER_ID",
    "EditSpec",
    "TargetModule",
    "ModuleEdit",
    "EditReport",
    "expected_census",
    "census_from_hub",
    "enumerate_targets",
    "load_merged_model",
    "apply_edit",
    "load_edited_model",
    "export_edited_model",
]

BASE_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
ADAPTER_ID = "timhua/wood_v2_sftr4_filt"

# Suffixes of the projections that write into the residual stream. o_proj is the
# attention output projection; down_proj is the MLP output projection. Matching
# on the suffix (rather than building names from a layer index) is what keeps
# the heterogeneous blocks honest -- absent modules simply never appear.
OUTPUT_PROJECTIONS = (".o_proj", ".down_proj")

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class EditSpec:
    """Everything needed to reproduce an edited model from scratch.

    The persisted artifact is this plus the resolved per-block directions, not a
    100GB checkpoint -- ``load_edited_model`` regenerates the weights
    deterministically. ``export_edited_model`` is available when a materialized
    checkpoint is actually wanted.
    """

    base_model_id: str = BASE_MODEL_ID
    adapter_id: str | None = ADAPTER_ID
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "cpu"
    directions_dir: str = str(REPO / "outputs" / "directions")
    pooling: str = "last_prompt_token"
    direction_key: str = "r_hat"
    band: tuple[int, int] = (36, 76)
    layer_convention: str = "write_target"

    def edit_id(self) -> str:
        """Short stable identifier, used to name output files."""
        import hashlib

        payload = json.dumps(
            {
                "base": self.base_model_id,
                "adapter": self.adapter_id,
                "revision": self.revision,
                "dtype": self.dtype,
                "pooling": self.pooling,
                "key": self.direction_key,
                "band": list(self.band),
                "convention": self.layer_convention,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
        return f"L{self.band[0]}-{self.band[1]}_{self.layer_convention}_{digest}"


@dataclass
class TargetModule:
    """One projection that writes to the residual stream."""

    name: str
    block: int
    kind: str            # "o_proj" | "down_proj"
    out_features: int
    in_features: int
    module: Any = None   # the live nn.Linear, when enumerated from a model


@dataclass
class ModuleEdit:
    """Per-module record of what the edit did. Goes into the manifest verbatim."""

    name: str
    block: int
    kind: str
    source_layer: int
    shape: tuple[int, int]
    proj_norm_before: float
    proj_norm_after: float
    removal_ratio: float
    rel_frobenius_change: float

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "block": self.block,
            "kind": self.kind,
            "source_layer": self.source_layer,
            "shape": list(self.shape),
            "proj_norm_before": float(f"{self.proj_norm_before:.6g}"),
            "proj_norm_after": float(f"{self.proj_norm_after:.6g}"),
            "removal_ratio": float(f"{self.removal_ratio:.6g}"),
            "rel_frobenius_change": float(f"{self.rel_frobenius_change:.6g}"),
        }


@dataclass
class EditReport:
    edits: list[ModuleEdit]
    n_o_proj: int
    n_down_proj: int
    n_candidates: int
    hidden_size: int
    band: tuple[int, int]
    notes: list[str] = field(default_factory=list)

    @property
    def worst_removal_ratio(self) -> float:
        return max((e.removal_ratio for e in self.edits), default=0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "n_edited": len(self.edits),
            "n_o_proj": self.n_o_proj,
            "n_down_proj": self.n_down_proj,
            "n_candidates_in_model": self.n_candidates,
            "hidden_size": self.hidden_size,
            "band": list(self.band),
            "worst_removal_ratio": float(f"{self.worst_removal_ratio:.6g}"),
            "modules": [e.summary() for e in self.edits],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Census -- what *should* be there, derived from config.json alone
# --------------------------------------------------------------------------


def expected_census(config: dict[str, Any]) -> dict[str, Any]:
    """Predict the module inventory from ``block_configs``, without any weights.

    Lets ``--dry-run`` catch a bad module matcher before a 100GB download.
    """
    blocks = config.get("block_configs")
    check(
        isinstance(blocks, list) and blocks,
        "config.json has no 'block_configs' list. This code targets the "
        "nemotron-nas architecture, whose per-block heterogeneity is the whole "
        "reason the module loop cannot assume uniform layers.",
    )
    n_layers = int(config["num_hidden_layers"])
    check(
        len(blocks) == n_layers,
        f"config.json disagrees with itself: {len(blocks)} block_configs but "
        f"num_hidden_layers={n_layers}.",
    )

    o_blocks, down_blocks = [], []
    for i, b in enumerate(blocks):
        attn, ffn = b.get("attention", {}), b.get("ffn", {})
        if not attn.get("no_op") and not attn.get("replace_with_linear"):
            o_blocks.append(i)
        if not ffn.get("no_op") and not ffn.get("replace_with_linear"):
            down_blocks.append(i)

    return {
        "n_layers": n_layers,
        "hidden_size": int(config["hidden_size"]),
        "o_proj_blocks": o_blocks,
        "down_proj_blocks": down_blocks,
        "n_o_proj": len(o_blocks),
        "n_down_proj": len(down_blocks),
        "n_total": len(o_blocks) + len(down_blocks),
        "n_attention_no_op": sum(
            1 for b in blocks if b.get("attention", {}).get("no_op")
        ),
    }


def census_from_hub(
    model_id: str = BASE_MODEL_ID, revision: str = "main"
) -> dict[str, Any]:
    """Census from the hub's ``config.json`` and safetensors index -- no weights.

    Downloads two small JSON files. Cross-checking the predicted inventory
    against the checkpoint's actual tensor names catches a naming mismatch
    (``o_proj`` living somewhere unexpected) for free.
    """
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(model_id, "config.json", revision=revision)
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    predicted = expected_census(config)

    idx_path = hf_hub_download(
        model_id, "model.safetensors.index.json", revision=revision
    )
    weight_map = json.loads(Path(idx_path).read_text(encoding="utf-8"))["weight_map"]
    actual_o = sorted(k for k in weight_map if k.endswith("o_proj.weight"))
    actual_down = sorted(k for k in weight_map if k.endswith("down_proj.weight"))

    check(
        len(actual_o) == predicted["n_o_proj"],
        f"checkpoint has {len(actual_o)} o_proj tensors but block_configs "
        f"predicts {predicted['n_o_proj']}.",
    )
    check(
        len(actual_down) == predicted["n_down_proj"],
        f"checkpoint has {len(actual_down)} down_proj tensors but block_configs "
        f"predicts {predicted['n_down_proj']}.",
    )
    return {
        **predicted,
        "checkpoint_o_proj": len(actual_o),
        "checkpoint_down_proj": len(actual_down),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
    }


# --------------------------------------------------------------------------
# Enumeration and the edit
# --------------------------------------------------------------------------


def enumerate_targets(model: Any) -> list[TargetModule]:
    """Every residual-stream output projection actually present in ``model``.

    Enumerated from the module tree, never constructed from a layer index --
    absent modules in ``attention.no_op`` blocks simply do not appear.
    """
    hidden = int(model.config.hidden_size)
    targets: list[TargetModule] = []

    for name, mod in model.named_modules():
        if not name.endswith(OUTPUT_PROJECTIONS):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or weight.ndim != 2:
            continue

        m = _LAYER_RE.search(name)
        check(
            m is not None,
            f"{name}: matched an output projection but has no '.layers.<i>.' in "
            f"its path, so it cannot be assigned to a block. Refusing to guess.",
        )
        # A projection that does not write into the residual stream is not ours;
        # if this fires, the suffix match caught something unintended.
        check(
            int(weight.shape[0]) == hidden,
            f"{name}: unexpected out-dim {int(weight.shape[0])}, expected the "
            f"hidden size {hidden}. This module does not write to the residual "
            f"stream and must not be orthogonalized.",
        )
        check(
            getattr(mod, "bias", None) is None,
            f"{name} has a bias. The weight edit only orthogonalizes Wx; a bias "
            f"would still add a component along r_hat, and the fidelity check's "
            f"equivalence to output-projection would no longer hold.",
        )

        targets.append(
            TargetModule(
                name=name,
                block=int(m.group(1)),
                kind="o_proj" if name.endswith(".o_proj") else "down_proj",
                out_features=int(weight.shape[0]),
                in_features=int(weight.shape[1]),
                module=mod,
            )
        )

    check(targets, "no o_proj/down_proj modules found -- is this the right model?")
    return sorted(targets, key=lambda t: (t.block, t.kind))


def _as_dict(block: Any) -> dict[str, Any]:
    """Normalize one entry of ``block_configs`` to nested plain dicts.

    Loaded from ``config.json`` these are dicts; loaded through
    ``trust_remote_code`` they are ``BlockConfig`` dataclasses, which -- unlike
    most HF config objects -- have no ``to_dict``.
    """
    import dataclasses

    if isinstance(block, dict):
        return block
    if dataclasses.is_dataclass(block) and not isinstance(block, type):
        return dataclasses.asdict(block)
    if hasattr(block, "to_dict"):
        return block.to_dict()
    raise AssertionError(
        f"cannot read a block config of type {type(block).__name__}; expected a "
        f"dict, a dataclass, or an object with to_dict()."
    )


def _assert_census(targets: list[TargetModule], model: Any) -> tuple[int, int]:
    """Cross-check the enumerated modules against what config.json predicts."""
    n_o = sum(1 for t in targets if t.kind == "o_proj")
    n_down = sum(1 for t in targets if t.kind == "down_proj")

    raw = getattr(model.config, "block_configs", None)
    if raw is None:
        logger.warning("model config exposes no block_configs; skipping census check")
        return n_o, n_down

    blocks = [_as_dict(b) for b in raw]
    predicted = expected_census(
        {
            "block_configs": blocks,
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "hidden_size": int(model.config.hidden_size),
        }
    )
    n_layers = predicted["n_layers"]
    hint = ""
    if n_o + n_down == 2 * n_layers:
        # PHASE2_HANDOFF.md section 3 calls this out by name: a count of exactly
        # 2 x num_hidden_layers means uniform blocks were assumed somewhere.
        hint = (
            f"Finding exactly 2 x {n_layers} means the matcher treated the blocks "
            f"as uniform; {predicted['n_attention_no_op']} of them have no "
            f"attention at all. "
        )
    check(
        (n_o, n_down) == (predicted["n_o_proj"], predicted["n_down_proj"]),
        f"module census mismatch: found {n_o} o_proj + {n_down} down_proj, but "
        f"config.json predicts {predicted['n_o_proj']} + {predicted['n_down_proj']}. "
        f"{hint}Do not proceed -- editing the wrong set of matrices is invisible "
        f"downstream.",
    )
    return n_o, n_down


def _orthogonalize_(module: Any, r: np.ndarray) -> dict[str, float]:
    """In-place ``W <- W - r r^T W`` for one module. Returns its diagnostics.

    The rank-1 update is computed in float32 and stored back in the module's own
    dtype. Both norms are measured against the *stored* weight, so the reported
    removal ratio includes the storage rounding rather than the ideal.
    """
    import torch

    weight = module.weight
    orig_dtype, orig_device = weight.dtype, weight.device

    # accelerate's dispatch_model replaces the weights of every CPU-assigned
    # module with meta tensors and streams the real values from a side
    # weights-map at forward time. Editing here would then either raise or be
    # written where the forward pass never looks. Edit before dispatching.
    check(
        not weight.is_meta,
        f"{getattr(module, '_edit_name', 'module')} holds a meta tensor, so the "
        f"model has already been dispatched across devices and its real weights "
        f"live in an offload map. Apply the edit while the model is fully "
        f"materialized on CPU, before calling prepare_for_inference().",
    )

    r_t = torch.as_tensor(r, dtype=torch.float32, device="cpu")

    norm = float(torch.linalg.vector_norm(r_t))
    check(
        abs(norm - 1.0) < 1e-5,
        f"direction is not a unit vector (||r|| = {norm:.8f}); the projection "
        f"r r^T assumes unit norm and would otherwise rescale the ablation.",
    )

    # Compute on CPU regardless of where the weight lives. The widest down_proj
    # is 8192 x 28672, so a float32 working copy is ~940MB and two of them do not
    # reliably fit in what is left of a 40GB GPU after ~36GB of weights. CPU RAM
    # is not scarce here, and the transfer costs a few seconds across all 54
    # modules.
    w32 = weight.detach().to("cpu", torch.float32)
    before = float(torch.linalg.vector_norm(r_t @ w32))
    frob = float(torch.linalg.matrix_norm(w32))

    ideal = w32 - torch.outer(r_t, r_t @ w32)
    module.weight.data = ideal.to(orig_device, orig_dtype)

    # Re-read from the stored tensor: this, not ``ideal``, is what the model will
    # use, so the reported removal includes the cost of storing in bf16.
    stored = module.weight.detach().to("cpu", torch.float32)
    after = float(torch.linalg.vector_norm(r_t @ stored))
    roundtrip = float(torch.linalg.matrix_norm(stored - ideal))
    del w32, ideal, stored

    return {
        "proj_norm_before": before,
        "proj_norm_after": after,
        "removal_ratio": after / before if before > 0 else 0.0,
        # The update is rank one, so ||W' - W||_F = ||r (r^T W)||_F = ||r^T W||
        # exactly, given ||r|| = 1. No second matrix norm needed.
        "rel_frobenius_change": before / frob if frob > 0 else 0.0,
        "storage_roundtrip_error": roundtrip,
    }


def apply_edit(model: Any, plan: PerBlockDirections) -> EditReport:
    """Orthogonalize every in-band output projection against its block's r_hat.

    Mutates ``model`` in place and returns the per-module record.
    """
    targets = enumerate_targets(model)
    n_o, n_down = _assert_census(targets, model)
    hidden = int(model.config.hidden_size)

    in_band = [t for t in targets if t.block in plan.vectors]
    check(
        in_band,
        f"band {plan.band} (blocks {plan.blocks[0]}-{plan.blocks[-1]}) selects "
        f"none of the {len(targets)} output projections in this model.",
    )

    source_layer = {a.block: a.source_layer for a in plan.assignments}
    edits: list[ModuleEdit] = []
    for t in in_band:
        r = plan.for_block(t.block)
        t.module._edit_name = t.name          # for the meta-tensor message
        check(
            r.shape[0] == hidden,
            f"{t.name}: direction has dim {r.shape[0]} but the model's hidden "
            f"size is {hidden}. The direction was estimated on a different model.",
        )
        stats = _orthogonalize_(t.module, r)
        edits.append(
            ModuleEdit(
                name=t.name,
                block=t.block,
                kind=t.kind,
                source_layer=source_layer[t.block],
                shape=(t.out_features, t.in_features),
                proj_norm_before=stats["proj_norm_before"],
                proj_norm_after=stats["proj_norm_after"],
                removal_ratio=stats["removal_ratio"],
                rel_frobenius_change=stats["rel_frobenius_change"],
            )
        )
        logger.debug(
            "  edited %-44s (L%d)  removal %.2e", t.name, source_layer[t.block],
            stats["removal_ratio"],
        )

    worst = max(e.removal_ratio for e in edits)
    notes = [
        f"edited {len(edits)} of {len(targets)} output projections "
        f"({sum(1 for e in edits if e.kind == 'o_proj')} o_proj, "
        f"{sum(1 for e in edits if e.kind == 'down_proj')} down_proj)",
        f"worst residual projection after the edit: {worst:.2e} of the original "
        f"(nonzero because the result is stored in bf16, not because the edit "
        f"failed -- see fidelity for the tolerance this implies)",
    ]
    # A ratio near 1 means nothing was removed; a ratio far above bf16's ~4e-3
    # relative resolution means something other than rounding is going on.
    check(
        worst < 0.05,
        f"worst-case residual projection is {worst:.3f} of the original -- the "
        f"edit did not take. The usual cause is editing before merging the LoRA "
        f"adapter (PHASE2_HANDOFF.md section 2).",
    )

    return EditReport(
        edits=edits,
        n_o_proj=n_o,
        n_down_proj=n_down,
        n_candidates=len(targets),
        hidden_size=hidden,
        band=plan.band,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def _resolve_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    check(name in mapping, f"unsupported dtype {name!r}, expected {sorted(mapping)}")
    return mapping[name]


def load_merged_model(spec: EditSpec) -> tuple[Any, Any]:
    """Load the base model, fold the LoRA into it, and prove the fold happened.

    Returns ``(model, tokenizer)``. Order matters: see the module docstring.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "load_merged_model() needs torch and transformers.\n"
            "  uv add transformers peft accelerate safetensors\n"
            "The direction analysis and the dry-run census run without them."
        ) from exc

    check(
        spec.dtype != "int4" and "4bit" not in spec.base_model_id.lower(),
        f"refusing to merge into {spec.base_model_id!r} at dtype {spec.dtype!r}. "
        f"PHASE2_HANDOFF.md section 4: merge_and_unload() into quantized weights "
        f"is lossy, and the edited model is the artifact every Phase 3 number "
        f"depends on. Choosing the direction from 4-bit activations was fine; "
        f"producing edited weights from them is not.",
    )

    logger.info("loading base model %s (%s)", spec.base_model_id, spec.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.base_model_id, revision=spec.revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model_id,
        revision=spec.revision,
        torch_dtype=_resolve_dtype(spec.dtype),
        device_map=spec.device_map,
        trust_remote_code=True,
    )
    model.eval()

    if spec.adapter_id is None:
        logger.warning(
            "adapter_id is None -- editing the BASE model, not the organism. "
            "Only meaningful as a deliberate ablation."
        )
        return model, tokenizer

    # Snapshot a matrix the adapter claims to target, so a no-op merge is caught.
    probe = enumerate_targets(model)[0]
    before = probe.module.weight.detach().clone()

    from peft import PeftModel

    logger.info("attaching adapter %s", spec.adapter_id)
    peft_model = PeftModel.from_pretrained(model, spec.adapter_id)

    n_lora = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
    check(
        n_lora > 0,
        f"{spec.adapter_id} attached but contributed no lora_* parameters. The "
        f"adapter did not really load, and merge_and_unload() would be a silent "
        f"no-op leaving the base model unedited.",
    )
    logger.info("  %d LoRA parameter tensors attached; merging", n_lora)

    model = peft_model.merge_and_unload()

    after = dict(model.named_modules())[probe.name].weight
    check(
        not torch.equal(before, after.detach().to(before.device)),
        f"merge_and_unload() left {probe.name} bit-identical. The merge was a "
        f"silent no-op, so the adapter's BA would still be missing from W -- or "
        f"worse, still applied on top of an edit. PHASE2_HANDOFF.md section 2.",
    )
    delta = float((after.detach().to(before.device).float() - before.float()).norm())
    logger.info("  merge verified: ||delta|| = %.4g at %s", delta, probe.name)
    del before

    model.eval()
    return model, tokenizer


def load_edited_model(
    spec: EditSpec, plan: PerBlockDirections
) -> tuple[Any, Any, EditReport]:
    """The Phase 3 entry point: a merged, orthogonalized model, made from scratch.

    Deterministic given ``(spec, plan)``, which is why the persisted artifact can
    be a manifest plus directions rather than a 100GB checkpoint.
    """
    model, tokenizer = load_merged_model(spec)
    report = apply_edit(model, plan)
    return model, tokenizer, report


def export_edited_model(model: Any, tokenizer: Any, out_dir: str | Path) -> Path:
    """Materialize the edited weights. Opt-in: this writes ~100GB."""
    out_dir = Path(out_dir)
    check(
        not out_dir.exists() or not any(out_dir.iterdir()),
        f"refusing to export into non-empty {out_dir} -- move it aside first.",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("exporting edited model to %s (this writes ~100GB)", out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir
