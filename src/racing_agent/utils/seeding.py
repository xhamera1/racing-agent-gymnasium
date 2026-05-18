"""Global seeding for full run-level reproducibility.

A single ``set_global_seed(s)`` call seeds Python, NumPy, PyTorch (CPU + CUDA).

Phase 0 / Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed:
        Master seed (also passed to SB3 algorithms as ``algorithm.seed``).
    deterministic_torch:
        If ``True``, set ``torch.backends.cudnn.deterministic = True`` and
        ``benchmark = False`` (slower, stricter reproducibility on GPU).
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True  # noqa: FBT003
        torch.backends.cudnn.benchmark = False  # noqa: FBT003
