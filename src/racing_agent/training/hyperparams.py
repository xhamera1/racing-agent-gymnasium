"""Hyperparameter loading: YAML -> dict -> SAC keyword arguments.

A single training run is defined by **two** YAML files:

- a *hyperparameter* config (``configs/hp_*.yaml``)  -- ``lr``, ``batch``, ``tau`` ...
- an *architecture* config  (``configs/arch_*.yaml``) -- which feature extractor.

This module provides :func:`load_config` to deep-merge them and produce a dict
that can be passed straight to :class:`racing_agent.training.train.Trainer`.

Implemented in Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def load_config(
    hp_path: Union[str, Path],
    arch_path: Union[str, Path],
) -> Dict[str, Any]:
    """Load and deep-merge a hyperparameter file and an architecture file.

    Parameters
    ----------
    hp_path:
        Path to a ``configs/hp_*.yaml`` file.
    arch_path:
        Path to a ``configs/arch_*.yaml`` file.

    Returns
    -------
    dict
        Merged config (hp keys + arch keys). On collision, ``arch`` wins
        (architectures are more specific).

    Notes
    -----
    Implemented in Phase 3 of ``PLAN.md``.
    """
    raise NotImplementedError("load_config() will be implemented in Phase 3.")
