"""Phase-2 tests for the CNN feature extractors.

Both architectures must:

- subclass ``stable_baselines3.common.torch_layers.BaseFeaturesExtractor``
- accept a 4-channel 84x84 observation
- produce ``(B, features_dim)`` outputs

Marked ``skip`` until Phase 2 lands.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="Phase 2 of PLAN.md -- not yet implemented.")


def test_nature_cnn_forward() -> None:
    import torch
    from racing_agent.policies import NatureCNN  # noqa: F401

    extractor = NatureCNN(observation_space=None, features_dim=512)
    x = torch.zeros(2, 4, 84, 84)
    out = extractor(x)
    assert out.shape == (2, 512)


def test_custom_deep_cnn_forward() -> None:
    import torch
    from racing_agent.policies import CustomDeepCNN  # noqa: F401

    extractor = CustomDeepCNN(observation_space=None, features_dim=256)
    x = torch.zeros(2, 4, 84, 84)
    out = extractor(x)
    assert out.shape == (2, 256)
