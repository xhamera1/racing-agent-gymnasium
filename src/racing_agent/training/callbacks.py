"""Custom callbacks for the SAC training loop.

- :class:`StepTimingCallback` — mean wall time between consecutive training
  ``VecEnv.step`` completions (covers one vectorised env tick per callback call).
  Also records per-episode ``t`` samples from Gymnasium Monitor's ``infos``.
- :class:`EvalSaveBestCallback` — :class:`~stable_baselines3.common.callbacks.EvalCallback`
  with ``experiments/<run_id>/models/best`` paths.

Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class StepTimingCallback(BaseCallback):
    """Tracks elapsed time **between successive** training env steps."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._last_clock: float | None = None
        self._inter_step_secs: list[float] = []
        self._episode_durations_s: list[float] = []

    def _on_training_start(self) -> None:
        self._last_clock = time.perf_counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict):
                    episode = info.get("episode")
                    if isinstance(episode, dict) and "t" in episode:
                        self._episode_durations_s.append(float(episode["t"]))

        now = time.perf_counter()
        if self._last_clock is not None:
            self._inter_step_secs.append(now - self._last_clock)
        self._last_clock = now
        return True

    def mean_step_time_s(self) -> float:
        if not self._inter_step_secs:
            return 0.0
        return float(np.mean(self._inter_step_secs))

    def mean_episode_time_s(self) -> float:
        if not self._episode_durations_s:
            return 0.0
        return float(np.mean(self._episode_durations_s))


class EvalSaveBestCallback(EvalCallback):
    """Runs periodic evaluation and saves ``best_model`` under ``models/best/``."""

    def __init__(
        self,
        eval_env,
        *,
        eval_freq_calls: int,
        best_model_dir: Path,
        eval_log_root: Path,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_log_root.mkdir(parents=True, exist_ok=True)

        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq_calls,
            log_path=str(eval_log_root / "eval_logs"),
            best_model_save_path=str(best_model_dir),
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            warn=False,
        )
