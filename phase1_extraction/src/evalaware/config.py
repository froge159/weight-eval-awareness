"""Config loading, seeding, and run provenance.

Every run writes a manifest recording the config, the resolved seeds, the git
commit, package versions and platform. Reproducing a result across machines
means comparing manifests first -- most "it doesn't reproduce" reports turn out
to be a different seed or a different transformers version.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .validation import check

__all__ = ["Config", "load_config", "set_all_seeds", "run_manifest", "write_manifest"]


@dataclass
class Config:
    # --- model ---
    model_id: str = "timhua/wood_v2_sftr4_filt"
    revision: str = "main"
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # Where r_hat is estimated from. Independent of the model edited in Phase 2.
    # See RESEARCH_DECISIONS.md 2026-08-08T13:10Z -- this is an open question.
    #   "organism" | "base" | "both"
    extraction_source: str = "organism"
    base_model_id: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

    # --- prompts ---
    regimes: tuple[str, ...] = ("explicit", "woodlabs")
    situational_awareness_hint: bool = True
    questions_path: str = "data/questions.jsonl"
    max_templates_per_regime: int | None = None

    # --- splits ---
    seed: int = 0
    template_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    question_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)

    # --- capture ---
    pooling_schemes: tuple[str, ...] = ("last_prompt_token", "completion_mean")
    max_new_tokens: int = 1024
    temperature: float = 0.0
    layers: tuple[int, ...] | None = None
    batch_size: int = 4
    cache_dir: str = "cache/activations"

    # --- analysis ---
    n_boot: int = 2000
    # Pre-registered region of interest from Hua et al. Appendix G.1 (steering
    # was most effective in layers 10-30 of the 49B). A prior, not a constraint.
    layer_band_prior: tuple[int, int] = (10, 30)
    # Multiplier on the random-pair |cos| p99 used as the cross-layer stability
    # threshold. Judging against the baseline, never against zero (proposal 3.2).
    stability_threshold_multiplier: float = 20.0

    # --- gates ---
    allow_deferred_refusal: bool = False
    require_gap_before_extraction: bool = True

    # --- output ---
    output_dir: str = "outputs"
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check(
            self.extraction_source in ("organism", "base", "both"),
            f"extraction_source must be organism|base|both, got "
            f"{self.extraction_source!r}",
        )
        for scheme in self.pooling_schemes:
            check(
                scheme in ("last_prompt_token", "completion_mean"),
                f"unknown pooling scheme {scheme!r}",
            )
        for regime in self.regimes:
            check(
                regime in ("explicit", "woodlabs"),
                f"unknown regime {regime!r}",
            )
        check(
            len(self.pooling_schemes) > 0 and len(self.regimes) > 0,
            "at least one pooling scheme and one regime are required",
        )
        lo, hi = self.layer_band_prior
        check(0 <= lo < hi, f"layer_band_prior must be an increasing pair, got {(lo, hi)}")
        if not self.situational_awareness_hint:
            import warnings
            warnings.warn(
                "situational_awareness_hint=False. Hua et al. Figures 34-36 show "
                "the eval/deploy gap is much weaker without it. Only do this as a "
                "deliberate ablation -- otherwise you risk a false-negative 'no "
                "gap' result caused by prompt formatting.",
                stacklevel=2,
            )


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load YAML config, apply overrides, validate.

    Unknown keys are an error, not a silent ignore -- a typo'd key in a config
    file is otherwise invisible and produces a run with default settings.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        check(path.exists(), f"config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        check(isinstance(data, dict), f"{path}: top level must be a mapping")

    data.update(overrides)
    valid = {f for f in Config.__dataclass_fields__}
    unknown = set(data) - valid
    check(
        not unknown,
        f"unknown config key(s): {sorted(unknown)}. Valid keys: {sorted(valid)}. "
        f"(Unknown keys are rejected rather than ignored so typos are visible.)",
    )
    for key in ("regimes", "pooling_schemes", "layers", "template_fractions",
                "question_fractions", "layer_band_prior"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    return Config(**data)


def set_all_seeds(seed: int) -> dict[str, int]:
    """Seed every RNG we might touch. Returns what was set, for the manifest."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seeds = {"python": seed, "numpy": seed, "pythonhashseed": seed}
    try:
        import torch
        torch.manual_seed(seed)
        set_seeds["torch"] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            set_seeds["torch_cuda"] = seed
            # Determinism at some speed cost. Worth it: a non-reproducible
            # direction is not a result.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return set_seeds


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return bool(out.strip())
    except Exception:
        return False


def run_manifest(config: Config, stage: str, extra: dict[str, Any] | None = None) -> dict:
    """Provenance record written alongside every stage output."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "sklearn", "torch", "transformers", "yaml"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"

    return {
        "stage": stage,
        "config": asdict(config),
        "seeds": set_all_seeds(config.seed),
        "git": {"commit": _git_commit(), "dirty": _git_dirty()},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "versions": versions,
        **(extra or {}),
    }


def write_manifest(config: Config, stage: str, extra: dict[str, Any] | None = None) -> Path:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{stage}.json"
    path.write_text(
        json.dumps(run_manifest(config, stage, extra), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path
