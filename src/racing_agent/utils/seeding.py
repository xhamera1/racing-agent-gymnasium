"""Global seeding for full run-level reproducibility.

A single ``set_global_seed(s)`` call seeds Python, NumPy, PyTorch (CPU + CUDA),
the gym ``action_space.np_random`` and the SAC ``seed`` argument. Pinning all
of them is what makes "seed 0..9" experiments meaningful.

Implemented in Phase 0 of ``PLAN.md``.
"""

from __future__ import annotations


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed all RNGs used in the project.

    Parameters
    ----------
    seed:
        Master seed.
    deterministic_torch:
        If ``True``, also set ``torch.backends.cudnn.deterministic = True`` and
        ``benchmark = False``. Slower but bitwise reproducible across runs.

    Implemented in Phase 0 of ``PLAN.md``.
    """
    raise NotImplementedError("set_global_seed() will be implemented in Phase 0.")
