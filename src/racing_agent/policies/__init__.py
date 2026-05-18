"""CNN feature extractors used by SAC's ``CnnPolicy``.

Architectures compared in the report (Phase 5 of ``PLAN.md``):

- :class:`racing_agent.policies.nature_cnn.NatureCNN` — SB3 baseline (Architecture A).
- :class:`racing_agent.policies.custom_cnn.CustomDeepCNN` — deeper CNN (Architecture B).
- :class:`racing_agent.policies.light_cnn.LightCNN` — fast/light profile for Kaggle sweeps.
"""

from racing_agent.policies.custom_cnn import CustomDeepCNN
from racing_agent.policies.light_cnn import LightCNN
from racing_agent.policies.nature_cnn import NatureCNN

__all__ = ["NatureCNN", "CustomDeepCNN", "LightCNN"]
