"""Phase-1 tests for env construction and observation wrappers.

These are intentionally written *before* the implementation so the Phase-1
exit criterion in ``PLAN.md`` ("env_checker passes, wrappers produce
``(4, 84, 84)``") is mechanically verifiable.

Each test is marked ``skip`` at bootstrap time and unskipped as Phase 1 lands.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="Phase 1 of PLAN.md -- not yet implemented.")


def test_make_car_racing_returns_vec_env() -> None:
    from racing_agent.env import make_car_racing

    env = make_car_racing(n_envs=1, seed=0)
    assert env.num_envs == 1


def test_observation_shape_after_wrappers() -> None:
    from racing_agent.env import make_car_racing

    env = make_car_racing(n_envs=1, seed=0,
                          grayscale=True, resize_to=84, frame_stack=4)
    obs = env.reset()
    assert obs.shape == (1, 4, 84, 84)


def test_action_space_is_continuous() -> None:
    import numpy as np
    from racing_agent.env import make_car_racing

    env = make_car_racing(n_envs=1, seed=0)
    assert env.action_space.shape == (3,)
    assert np.allclose(env.action_space.low, [-1.0, 0.0, 0.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0, 1.0])
