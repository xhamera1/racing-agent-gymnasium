"""Architecture B — deeper CNN with BatchNorm + global average pooling.

Stacks **Conv–BN–ReLU** blocks on channels-first ``(C, H, W)`` inputs, then
:class:`torch.nn.AdaptiveAvgPool2d` (1×1) and a linear projection to ``features_dim``.

Reference layout (``C=4``, ``H=W=84``)::

    Conv2d(4→32,  3×3, s=2) → BN → ReLU  → (32, 41, 41)
    Conv2d(32→64, 3×3, s=2) → BN → ReLU  → (64, 20, 20)
    Conv2d(64→128, 3×3, s=2) → BN → ReLU → (128, 9, 9)
    Conv2d(128→128, 3×3, s=1) → BN → ReLU → (128, 7, 7)
    AdaptiveAvgPool2d(1) → (128, 1, 1)
    Flatten → (128,)
    Linear(128 → features_dim) → ReLU

“Residual-ish” in the project brief is interpreted as **extra depth + BN** (no
identity skip — true res-blocks would need matched tensor sizes).

Phase 2 of ``PLAN.md``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from racing_agent.policies.nature_cnn import car_racing_cnn_observation_space

CUSTOM_DEEP_CNN_LAYER_ROWS: list[dict[str, Any]] = [
    {
        "layer": "Conv2d + BN + ReLU",
        "kernel": "3×3",
        "stride": "2",
        "out_shape": "(32, 41, 41)",
        "activation": "ReLU",
        "params": "(see total)",
        "notes": "in_ch = n_input_channels",
    },
    {
        "layer": "Conv2d + BN + ReLU",
        "kernel": "3×3",
        "stride": "2",
        "out_shape": "(64, 20, 20)",
        "activation": "ReLU",
        "params": "",
        "notes": "",
    },
    {
        "layer": "Conv2d + BN + ReLU",
        "kernel": "3×3",
        "stride": "2",
        "out_shape": "(128, 9, 9)",
        "activation": "ReLU",
        "params": "",
        "notes": "",
    },
    {
        "layer": "Conv2d + BN + ReLU",
        "kernel": "3×3",
        "stride": "1",
        "out_shape": "(128, 7, 7)",
        "activation": "ReLU",
        "params": "",
        "notes": "",
    },
    {
        "layer": "AdaptiveAvgPool2d(1) + Flatten + Linear + ReLU",
        "kernel": "—",
        "stride": "—",
        "out_shape": "(features_dim,)",
        "activation": "ReLU",
        "params": "",
        "notes": "pool → 128-d vector",
    },
]


class CustomDeepCNN(BaseFeaturesExtractor):
    """Channel-first CNN backbone for SAC ``CnnPolicy``."""

    _POOL_FLAT_DIM = 128

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        *,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "CustomDeepCNN requires ``gymnasium.spaces.Box``, "
            f"got {type(observation_space)}"
        )
        super().__init__(observation_space, int(features_dim))
        assert is_image_space(
            observation_space,
            check_channels=False,
            normalized_image=normalized_image,
        ), (
            "CustomDeepCNN only supports image observations; see NatureCNN env-check notes."
        )

        n_in = int(observation_space.shape[0])
        self.trunk = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._POOL_FLAT_DIM, self.features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.trunk(observations)


@lru_cache(maxsize=4)
def custom_deep_cnn_total_params(features_dim: int = 256) -> int:
    """Trainable parameters for :class:`CustomDeepCNN` on default CarRacing CNN obs."""

    model = CustomDeepCNN(car_racing_cnn_observation_space(), features_dim=int(features_dim))
    return int(sum(p.numel() for p in model.parameters()))
