"""Hyperparameter loading: YAML -> dict -> SAC keyword arguments.

A single training run is defined by **two** YAML files:

- a *hyperparameter* config (``configs/hp_*.yaml``)
- an *architecture* config  (``configs/arch_*.yaml``)

:func:`load_config` deep-merges them (architecture wins key collisions).

Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""

    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a YAML file into a dictionary."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at root of YAML, got {type(raw)} ({path}).")
    return raw


def load_config(
    hp_path: Union[str, Path],
    arch_path: Union[str, Path],
    overrides_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Load and deep-merge hyperparameter YAML + architecture YAML (+ optional overrides).

    Collision policy: later files overwrite earlier ones
    (hp → arch → overrides).

    Fields ``hp_name``, ``arch_name``, and ``experiment_name`` are always set from
    the hp/arch YAML ``name`` keys (override files must not replace them).
    """

    hp = load_yaml(hp_path)
    arch = load_yaml(arch_path)

    hp_name = str(hp.pop("name"))
    arch_name = str(arch.pop("name"))

    merged = _deep_merge(copy.deepcopy(hp), arch)

    if overrides_path is not None:
        overrides = load_yaml(overrides_path)
        overrides_name = str(overrides.pop("name", Path(overrides_path).stem))
        merged = _deep_merge(merged, overrides)
        merged["overrides_name"] = overrides_name

    merged["hp_name"] = hp_name
    merged["arch_name"] = arch_name
    merged["experiment_name"] = f"{hp_name}__{arch_name}"

    return merged
