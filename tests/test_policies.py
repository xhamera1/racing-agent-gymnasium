"""Phase 2 tests for the CNN feature extractors."""

from __future__ import annotations

from pathlib import Path

import pytest
from gymnasium.spaces import Box


@pytest.fixture(name="cnn_obs_space")
def cnn_obs_space_fixture() -> Box:
    from racing_agent.policies.nature_cnn import car_racing_cnn_observation_space

    return car_racing_cnn_observation_space()


def test_nature_cnn_forward(cnn_obs_space) -> None:
    import torch
    from racing_agent.policies import NatureCNN

    extractor = NatureCNN(observation_space=cnn_obs_space, features_dim=512)
    x = torch.zeros(2, 4, 84, 84)
    out = extractor(x)
    assert out.shape == (2, 512)


def test_custom_deep_cnn_forward(cnn_obs_space) -> None:
    import torch
    from racing_agent.policies import CustomDeepCNN

    extractor = CustomDeepCNN(observation_space=cnn_obs_space, features_dim=256)
    x = torch.zeros(2, 4, 84, 84)
    out = extractor(x)
    assert out.shape == (2, 256)


def test_light_cnn_forward() -> None:
    import torch
    from racing_agent.policies import LightCNN
    from racing_agent.policies.light_cnn import car_racing_light_observation_space

    obs_space = car_racing_light_observation_space(channels=2, size=64)
    extractor = LightCNN(observation_space=obs_space, features_dim=128)
    x = torch.zeros(2, 2, 64, 64)
    out = extractor(x)
    assert out.shape == (2, 128)


def test_plot_arch_diagram_writes_png(tmp_path: Path) -> None:
    from racing_agent.policies.custom_cnn import (
        CUSTOM_DEEP_CNN_LAYER_ROWS,
        custom_deep_cnn_total_params,
    )
    from racing_agent.utils.plotting import plot_arch_diagram

    out = tmp_path / "arch_custom.png"
    plot_arch_diagram(
        "Architecture B — CustomDeepCNN (smoke)",
        CUSTOM_DEEP_CNN_LAYER_ROWS,
        out,
        total_params=custom_deep_cnn_total_params(256),
    )
    assert out.is_file() and out.stat().st_size > 0
