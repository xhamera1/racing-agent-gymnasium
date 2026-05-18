"""Architecture C — lightweight CNN for fast sweeps (Kaggle / CPU smoke).

Designed for **~2–4× faster** steps vs NatureCNN by shrinking conv trunk and
``features_dim``. Used for the Phase-4 HP grid when wall-clock budget is tight;
Phase-5 architecture comparison still uses ``NatureCNN`` vs ``CustomDeepCNN``.

Input **channels-first** ``(C, H, W)`` with default Kaggle overrides
``C=2``, ``H=W=64``::

    Conv2d(2→16,  8×8, s=4) → ReLU → (16, 15, 15)
    Conv2d(16→32, 4×4, s=2) → ReLU → (32,  6,  6)
    Flatten → 1152
    Linear(1152 → features_dim) → ReLU
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

LIGHT_CNN_LAYER_ROWS: list[dict[str, Any]] = [
    {
        "layer": "Conv2d + ReLU",
        "kernel": "8×8",
        "stride": "4",
        "out_shape": "(16, 15, 15)",
        "activation": "ReLU",
        "params": "",
        "notes": "in_ch = n_input_channels (default 2)",
    },
    {
        "layer": "Conv2d + ReLU",
        "kernel": "4×4",
        "stride": "2",
        "out_shape": "(32, 6, 6)",
        "activation": "ReLU",
        "params": "",
        "notes": "",
    },
    {
        "layer": "Flatten + Linear + ReLU",
        "kernel": "—",
        "stride": "—",
        "out_shape": "(features_dim,)",
        "activation": "ReLU",
        "params": "",
        "notes": "default features_dim=128",
    },
]


class LightCNN(BaseFeaturesExtractor):
    """Small Nature-style CNN for SAC ``CnnPolicy`` (fast training profile)."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        *,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box)
        super().__init__(observation_space, int(features_dim))
        assert is_image_space(
            observation_space,
            check_channels=False,
            normalized_image=normalized_image,
        )

        n_in = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            flat_dim = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, self.features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def car_racing_light_observation_space(
    channels: int = 2,
    size: int = 64,
) -> spaces.Box:
    """Box space matching Kaggle fast profile (uint8, C×H×W)."""

    return car_racing_cnn_observation_space(channels=channels, size=size)


@lru_cache(maxsize=4)
def light_cnn_total_params(features_dim: int = 128, channels: int = 2, size: int = 64) -> int:
    model = LightCNN(car_racing_light_observation_space(channels, size), features_dim=int(features_dim))
    return int(sum(p.numel() for p in model.parameters()))
