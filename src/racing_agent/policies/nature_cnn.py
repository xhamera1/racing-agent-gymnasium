"""Architecture A — SB3 :class:`~stable_baselines3.common.torch_layers.NatureCNN` (DQN Nature paper).

Input **channels-first** ``(C, H, W)`` with ``C=4``, ``H=W=84`` (CarRacing stack after Phase~1 wrappers).

Canonical stack::

    Conv2d(in=4,  out=32,  kernel=8, stride=4) → ReLU → spatial (20, 20)
    Conv2d(in=32, out=64,  kernel=4, stride=2) → ReLU → (9, 9)
    Conv2d(in=64, out=64,  kernel=3, stride=1) → ReLU → (7, 7)
    Flatten → 3136
    Linear(3136 → features_dim) → ReLU → (features_dim,)

Tables for ``reports/final_report.pdf`` are obtained via
:data:`NATURE_CNN_LAYER_ROWS`, :func:`nature_cnn_total_params`,
and :func:`racing_agent.utils.plotting.plot_arch_diagram`.

Phase 2 of ``PLAN.md``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN as _Sb3NatureCNN

# Re-export for ``policy_kwargs={"features_extractor_class": NatureCNN}``.
NatureCNN = _Sb3NatureCNN

# ---------------------------------------------------------------------------
# Documentation rows (spatial sizes match PyTorch defaults, padding=0).
# Parameter counts omit the head and are illustrative; use :func:`nature_cnn_total_params`.
# ---------------------------------------------------------------------------

NATURE_CNN_LAYER_ROWS: list[dict[str, Any]] = [
    {
        "layer": "Conv2d + ReLU",
        "kernel": "8×8",
        "stride": "4",
        "out_shape": "(32, 20, 20)",
        "activation": "ReLU",
        "params": "~21k (in→32 ch)",
        "notes": "n_in = 4",
    },
    {
        "layer": "Conv2d + ReLU",
        "kernel": "4×4",
        "stride": "2",
        "out_shape": "(64, 9, 9)",
        "activation": "ReLU",
        "params": "~33k",
        "notes": "",
    },
    {
        "layer": "Conv2d + ReLU",
        "kernel": "3×3",
        "stride": "1",
        "out_shape": "(64, 7, 7)",
        "activation": "ReLU",
        "params": "~37k",
        "notes": "",
    },
    {
        "layer": "Flatten + Linear + ReLU",
        "kernel": "—",
        "stride": "—",
        "out_shape": "(features_dim,)",
        "activation": "ReLU",
        "params": "~1.6M (3136→512)",
        "notes": "features_dim default 512",
    },
]


def car_racing_cnn_observation_space(
    channels: int = 4,
    size: int = 84,
) -> spaces.Box:
    """Box space matching Phase~1 wrapped CarRacing (uint8, C×H×W)."""

    return spaces.Box(
        low=0,
        high=255,
        shape=(int(channels), int(size), int(size)),
        dtype=np.uint8,
    )


@lru_cache(maxsize=4)
def nature_cnn_total_params(features_dim: int = 512) -> int:
    """Exact trainable parameter count for NatureCNN on default CarRacing CNN obs."""

    model = NatureCNN(car_racing_cnn_observation_space(), features_dim=int(features_dim))
    return int(sum(p.numel() for p in model.parameters()))
