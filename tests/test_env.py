"""Phase-1 tests for env construction and observation wrappers."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from racing_agent.env import make_car_racing
from racing_agent.env.make_env import make_car_racing_single
from racing_agent.env.wrappers import merge_wrapper_kwargs


@pytest.fixture
def vec_env():
    env = make_car_racing(n_envs=1, seed=0)
    try:
        yield env
    finally:
        env.close()


def test_make_car_racing_returns_vec_env(vec_env) -> None:
    assert vec_env.num_envs == 1


def test_observation_shape_after_wrappers(vec_env) -> None:
    obs = vec_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 4, 84, 84)


def test_action_space_is_continuous(vec_env) -> None:
    assert vec_env.action_space.shape == (3,)
    assert np.allclose(vec_env.action_space.low, [-1.0, 0.0, 0.0])
    assert np.allclose(vec_env.action_space.high, [1.0, 1.0, 1.0])


def test_vec_env_random_steps(vec_env) -> None:
    vec_env.reset()
    for _ in range(40):
        act = np.stack([vec_env.action_space.sample() for _ in range(vec_env.num_envs)], axis=0)
        assert act.shape == (vec_env.num_envs, 3)
        obs, rewards, dones, _infos = vec_env.step(act)
        assert obs.shape == (vec_env.num_envs, 4, 84, 84)
        assert rewards.shape == (vec_env.num_envs,)
        assert dones.shape == (vec_env.num_envs,)


@pytest.mark.parametrize("clip", [False, True])
def test_make_car_racing_single_passes_checker(clip: bool) -> None:
    env = make_car_racing_single(
        seed=0,
        clip_reward=clip,
        grayscale=True,
        resize_to=84,
        frame_stack=4,
        render_mode=None,
    )
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_wrapper_config_override() -> None:
    merged = merge_wrapper_kwargs(
        {"grayscale": True, "resize_to": 84, "frame_stack": 4, "clip_reward": False},
        {"resize_to": 64, "clip_reward": True, "bogus": "ignored"},
    )
    assert merged["resize_to"] == 64
    assert merged["clip_reward"] is True
    assert merged["frame_stack"] == 4
    assert "bogus" not in merged


def test_subproc_vec_two_envs() -> None:
    env = make_car_racing(n_envs=2, seed=123)
    try:
        obs = env.reset()
        assert obs.shape == (2, 4, 84, 84)
        obs2, r, dones, _infos = env.step(
            np.stack([env.action_space.sample() for _ in range(env.num_envs)], axis=0),
        )
        assert obs2.shape == (2, 4, 84, 84)
        assert r.shape == (2,)
        assert dones.shape == (2,)
    finally:
        env.close()
