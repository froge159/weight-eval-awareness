"""Disk cache for "base model + LoRA adapter, merged."

Merging the organism's adapter into the 49B base costs ~10 minutes of CPU
time and ~180-250GB of RAM (see COMPLETION_MEAN_HANDOFF.md, PHASE3_RESULTS.md
section 11). Before this module existed, that merge was implemented three
times independently -- Phase 1's ``activations.capture()``, Phase 2's
``steering.surgery.load_merged_model()``, and Phase 3's
``run_phase3.build_model()`` -- and none of them cached the result, so a
single research session (Phase 2's own reference-pass-then-edit-pass, plus
every Phase 3 baseline/edited/control run) redundantly re-paid the merge
many times over on a rented GPU pod.

``ensure_merged()`` is now the one place that does the merge. It returns a
local directory holding the merged checkpoint, content-addressed by
``(base_model_id, adapter_id, revision, dtype)`` -- deliberately NOT by
anything edit-specific (band, direction, pooling), since the merge itself
doesn't depend on those. Callers still do their own
``AutoModelForCausalLM.from_pretrained(that_dir, device_map=..., ...)``, so
each call site's device-placement logic (Phase 1's fixed 2-GPU split,
Phase 2's ``EditSpec.device_map``, Phase 3's ``accelerate`` auto-dispatch)
is untouched.

Crash safety
------------
A cache entry is only trusted if it has a ``_COMPLETE`` marker, written
last. Population happens in a uniquely-named ``root/.work/<fp>.<pid>.<rand>/``
directory; only once ``save_pretrained()`` and the marker are both written
does the directory get renamed onto ``root/<fingerprint>/``. The final name
never exists until that rename succeeds, so a crash mid-``save_pretrained()``
(~100GB) leaves only an orphaned ``.work/`` entry that ``is_complete()``
never looks at -- mirrors ``ActivationStore.put()``'s tmp+rename pattern,
scaled up. This assumes ``root`` is a real POSIX filesystem where directory
rename is atomic (a local disk or block volume); it is not guaranteed over
NFS or an object-storage gateway.

``use_cache=False`` does not mean "never touch disk": the merge is the
expensive part, writing the ~100GB result is comparatively cheap, so a
forced refresh still repopulates the cache (useful if you suspect a stale
entry) rather than discarding that work.

Known limitation: the fingerprint does not pin adapter *content*, only its
id. If ``adapter_id`` points at a moving tag and the weights behind it
change without a revision bump, a stale cache entry will be served silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .validation import check

logger = logging.getLogger(__name__)

__all__ = ["MergeSpec", "MergeCache", "ensure_merged"]

# Suffixes of the projections that write into the residual stream -- same
# convention as steering.surgery.OUTPUT_PROJECTIONS, duplicated here rather
# than imported: this module lives under evalaware, which steering already
# depends on, so importing steering.surgery from here would be a circular
# import. This is only used for the silent-no-op-merge smoke probe below,
# not for choosing edit targets -- that enumeration stays in surgery.py.
_PROBE_SUFFIXES = (".o_proj", ".down_proj")


@dataclass(frozen=True)
class MergeSpec:
    """What to merge. The cache key covers exactly this and nothing more."""

    base_model_id: str
    adapter_id: str
    revision: str = "main"
    dtype: str = "bfloat16"

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


def _default_root() -> Path:
    env = os.environ.get("EVALAWARE_MERGE_CACHE_DIR")
    if env:
        return Path(env)
    # this file: <repo>/src/extraction/src/evalaware/merge_cache.py
    # 4 directories up (evalaware, src, extraction, src) is <repo>.
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "checkpoints" / "merged"


class MergeCache:
    """Content-addressed on-disk store of merged (base + adapter) models.

    Layout::

        root/<fingerprint>/            # a complete, trustworthy entry
            _COMPLETE
            metadata.json
            config.json, *.safetensors, tokenizer files, ...
        root/.work/<fingerprint>.<pid>.<rand>/   # in-progress; never trusted
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else _default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_for(self, spec: MergeSpec) -> Path:
        return self.root / spec.fingerprint()

    def is_complete(self, spec: MergeSpec) -> bool:
        return (self.dir_for(spec) / "_COMPLETE").exists()

    def purge_incomplete(self) -> list[Path]:
        """Best-effort cleanup of orphaned .work/ dirs from crashed merges."""
        work_root = self.root / ".work"
        removed = []
        if not work_root.exists():
            return removed
        for d in work_root.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
        return removed

    def populate(
        self,
        spec: MergeSpec,
        model: Any,
        tokenizer: Any,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save ``model``/``tokenizer`` and atomically publish as ``spec``'s entry."""
        work_name = f"{spec.fingerprint()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        work_dir = self.root / ".work" / work_name
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("writing merged model to %s (this writes ~100GB)", work_dir)
        model.save_pretrained(work_dir, safe_serialization=True)
        tokenizer.save_pretrained(work_dir)
        metadata = {
            **asdict(spec),
            "fingerprint": spec.fingerprint(),
            "created_utc": datetime.now(UTC).isoformat(),
            **(extra_metadata or {}),
        }
        (work_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Written last: is_complete() only trusts a dir that has this.
        (work_dir / "_COMPLETE").write_text("", encoding="utf-8")

        final_dir = self.dir_for(spec)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        stale = None
        if final_dir.exists():
            # A force-refresh (use_cache=False) replacing a prior entry, or a
            # previous incomplete dir left at the final name somehow. Move it
            # aside first so the publish rename below can't land on a
            # non-empty directory (which fails on POSIX), and so a failure
            # here doesn't destroy a still-good prior entry.
            stale_name = f"{final_dir.name}.stale-{uuid.uuid4().hex[:8]}"
            stale = final_dir.with_name(stale_name)
            final_dir.rename(stale)
        work_dir.rename(final_dir)
        if stale is not None:
            shutil.rmtree(stale, ignore_errors=True)
        return final_dir


def _resolve_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    check(name in mapping,
          f"unsupported dtype {name!r}, expected one of {sorted(mapping)}")
    return mapping[name]


def _find_probe(model: Any) -> tuple[str, Any]:
    """First residual-writing projection, for the silent-no-op-merge check."""
    for name, module in model.named_modules():
        if name.endswith(_PROBE_SUFFIXES) and hasattr(module, "weight"):
            return name, module.weight.detach().clone()
    check(False, "ensure_merged: found no o_proj/down_proj module to probe.")


def ensure_merged(
    base_model_id: str,
    adapter_id: str,
    *,
    revision: str = "main",
    dtype: str = "bfloat16",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    trust_remote_code: bool = True,
) -> Path:
    """Merge ``adapter_id`` into ``base_model_id``; return a local directory.

    ``use_cache=True`` (default): a complete cache entry is returned
    immediately with no merge. On a miss, merge once and populate the cache
    so later calls -- including a second call in the same process, e.g.
    Phase 2's reference-pass-then-edit-pass -- are a plain disk read.

    ``use_cache=False``: skip reading an existing entry and always redo the
    merge (e.g. to verify a suspected-stale cache), but still refresh the
    cache afterward -- the merge is the expensive part, and writing the
    ~100GB result is comparatively cheap.
    """
    check(bool(adapter_id), "ensure_merged: adapter_id is required.")
    spec = MergeSpec(base_model_id=base_model_id, adapter_id=adapter_id,
                      revision=revision, dtype=dtype)
    store = MergeCache(cache_dir)

    if use_cache and store.is_complete(spec):
        cached = store.dir_for(spec)
        logger.info("merge cache hit [%s]: %s", spec.fingerprint(), cached)
        return cached

    logger.info(
        "merge cache %s [%s]: merging %s + %s from scratch "
        "(~10 min CPU, large RAM)",
        "miss" if use_cache else "bypassed (--no-merge-cache)",
        spec.fingerprint(), base_model_id, adapter_id,
    )
    check(
        dtype != "int4" and "4bit" not in base_model_id.lower(),
        f"refusing to merge into {base_model_id!r} at dtype {dtype!r}: "
        f"merge_and_unload() into quantized weights is lossy, and the merged "
        f"model is the artifact every downstream number depends on.",
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=revision,
        torch_dtype=_resolve_dtype(dtype),
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    probe_name, probe_before = _find_probe(base)

    logger.info("attaching adapter %s", adapter_id)
    peft_model = PeftModel.from_pretrained(base, adapter_id)
    n_lora = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
    check(
        n_lora > 0,
        f"{adapter_id} attached but contributed no lora_* parameters -- the "
        f"adapter did not really load, and merge_and_unload() would be a "
        f"silent no-op leaving the base model unedited.",
    )
    logger.info("  %d LoRA parameter tensor(s) attached; merging", n_lora)

    merged = peft_model.merge_and_unload()
    probe_after = dict(merged.named_modules())[probe_name].weight.detach()
    check(
        not torch.equal(probe_before, probe_after.to(probe_before.device)),
        f"merge_and_unload() left {probe_name} bit-identical -- the merge was "
        f"a silent no-op, so the adapter's contribution is still missing "
        f"from the weights.",
    )
    delta = float(
        (probe_after.to(probe_before.device).float() - probe_before.float()).norm()
    )
    logger.info("  merge verified: ||delta|| = %.4g at %s", delta, probe_name)
    del probe_before

    merged.eval()
    out_dir = store.populate(
        spec, merged, tokenizer,
        extra_metadata={"probe_module": probe_name, "probe_delta_norm": delta},
    )
    logger.info("merge cached at %s", out_dir)
    return out_dir
