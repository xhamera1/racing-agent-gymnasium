"""Architecture A -- the SB3 default NatureCNN feature extractor.

Reference layout for the report (input ``(4, 84, 84)`` after wrappers)::

    Conv2d(in_channels=4,  out=32, kernel=8, stride=4)  -> ReLU   -> (32, 20, 20)
    Conv2d(in_channels=32, out=64, kernel=4, stride=2)  -> ReLU   -> (64,  9,  9)
    Conv2d(in_channels=64, out=64, kernel=3, stride=1)  -> ReLU   -> (64,  7,  7)
    Flatten                                                       -> (3136,)
    Linear(3136 -> 512)                                  -> ReLU  -> (512,)

This file currently re-exports the SB3 implementation; the report's diagrams
and parameter counts are generated from it directly.

Implemented in Phase 2 of ``PLAN.md``.
"""

# In Phase 2 we will simply re-export the SB3 class so SAC can pick it up
# unchanged via `policy_kwargs={"features_extractor_class": NatureCNN}`.
#
#     from stable_baselines3.common.torch_layers import NatureCNN
#
# A lightweight wrapper may be added if we need extra introspection hooks for
# the report (e.g. layerwise activation stats).

NatureCNN = None  # populated in Phase 2
