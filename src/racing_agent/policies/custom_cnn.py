"""Architecture B -- our custom deeper CNN feature extractor.

Designed for a fair contrast with NatureCNN: smaller kernels, more layers,
BatchNorm, and adaptive average pooling (so the head is robust to small input
size changes). Input ``(4, 84, 84)`` after wrappers, target output ``(256,)``::

    Conv2d(in=4,   out=32,  kernel=3, stride=2) -> BN -> ReLU   -> (32, 42, 42)
    Conv2d(in=32,  out=64,  kernel=3, stride=2) -> BN -> ReLU   -> (64, 21, 21)
    Conv2d(in=64,  out=128, kernel=3, stride=2) -> BN -> ReLU   -> (128, 11, 11)
    Conv2d(in=128, out=128, kernel=3, stride=1) -> BN -> ReLU   -> (128, 11, 11)
    AdaptiveAvgPool2d(1)                                         -> (128, 1, 1)
    Flatten                                                      -> (128,)
    Linear(128 -> 256)                            -> ReLU       -> (256,)

The class subclasses ``stable_baselines3.common.torch_layers.BaseFeaturesExtractor``
so it slots directly into SAC's ``policy_kwargs``.

Implemented in Phase 2 of ``PLAN.md``.
"""

# The actual ``torch.nn.Module`` definition lives here once Phase 2 starts:
#
#     import torch.nn as nn
#     from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#
#     class CustomDeepCNN(BaseFeaturesExtractor):
#         def __init__(self, observation_space, features_dim: int = 256):
#             super().__init__(observation_space, features_dim)
#             ...

CustomDeepCNN = None  # populated in Phase 2
