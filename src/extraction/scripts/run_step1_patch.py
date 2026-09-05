#!/usr/bin/env python3
"""Thread base_model_id into the CaptureConfig that run_step1 builds.

Respects config.extraction_source:
  - "organism" -> pass base_model_id, so activations.py loads base+adapter merged
  - "base"     -> load the base model directly (model_id = base, no adapter)
  - "both"     -> not supported by a single capture; error out clearly

Run ONCE on the pod from src/extraction:
    python scripts/run_step1_patch.py
Idempotent.
"""
from __future__ import annotations
import sys
from pathlib import Path

f = Path("scripts/run_step1.py")
src = f.read_text()

if "base_model_id=" in src:
    print("already patched -- nothing to do")
    sys.exit(0)

old = '''        cap_cfg = CaptureConfig(
            model_id=cfg.model_id,
            revision=cfg.revision,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            pooling=pooling,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            layers=cfg.layers,
            batch_size=cfg.batch_size,
        )'''

new = '''        if cfg.extraction_source == "both":
            raise SystemExit(
                "extraction_source='both' needs two separate capture runs; "
                "set it to 'organism' or 'base' in the config."
            )
        if cfg.extraction_source == "base":
            _capture_model_id = cfg.base_model_id   # load the base directly
            _capture_base_id = None
        else:  # organism
            _capture_model_id = cfg.model_id        # the adapter
            _capture_base_id = cfg.base_model_id    # merged onto this base
        cap_cfg = CaptureConfig(
            model_id=_capture_model_id,
            base_model_id=_capture_base_id,
            revision=cfg.revision,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            pooling=pooling,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            layers=cfg.layers,
            batch_size=cfg.batch_size,
        )'''

assert old in src, "CaptureConfig block not found verbatim in run_step1.py"
src = src.replace(old, new)
f.write_text(src)
print("patched run_step1.py: threads base_model_id, respects extraction_source")
