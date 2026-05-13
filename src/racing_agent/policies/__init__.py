"""CNN feature extractors used by SAC's ``CnnPolicy``.

Two architectures are compared in the report (Phase 5 of ``PLAN.md``):

- :class:`racing_agent.policies.nature_cnn.NatureCNN` -- the SB3 default,
  inspired by Mnih et al. 2015 (DQN). Acts as our **baseline**.
- :class:`racing_agent.policies.custom_cnn.CustomDeepCNN` -- a deeper
  convolutional stack with BatchNorm and adaptive pooling.

Both are plugged into SAC via ``policy_kwargs`` -- see
``configs/arch_*.yaml`` and :mod:`racing_agent.training.train`.
"""

from racing_agent.policies.custom_cnn import CustomDeepCNN
from racing_agent.policies.nature_cnn import NatureCNN

__all__ = ["NatureCNN", "CustomDeepCNN"]
