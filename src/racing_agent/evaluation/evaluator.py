"""Deterministic / stochastic rollout evaluation and live pygame preview.

The 8-point task requires saving the best agent and running deterministic
rollouts to compare with the training curve.  Live preview uses
``render_mode="human"`` (pygame window) instead of saving video files.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
from stable_baselines3 import SAC

from racing_agent.env.make_env import make_car_racing_single
from racing_agent.utils.io import discover_runs, find_monitor_csv, load_run_metadata


@dataclass
class EvalReport:
    """Summary statistics of an N-episode evaluation."""

    model_path: Path
    deterministic: bool
    n_episodes: int
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(statistics.mean(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(statistics.pstdev(self.rewards)) if len(self.rewards) > 1 else 0.0

    @property
    def median_reward(self) -> float:
        return float(statistics.median(self.rewards)) if self.rewards else 0.0

    @property
    def min_reward(self) -> float:
        return float(min(self.rewards)) if self.rewards else 0.0

    @property
    def max_reward(self) -> float:
        return float(max(self.rewards)) if self.rewards else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        payload.update(
            {
                "mean_reward": self.mean_reward,
                "std_reward": self.std_reward,
                "median_reward": self.median_reward,
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
            }
        )
        return payload


def env_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``make_car_racing_single`` kwargs from a merged training config."""

    env = config.get("env") or {}
    return {
        "grayscale": bool(env.get("grayscale", True)),
        "resize_to": int(env.get("resize_to", 84)),
        "frame_stack": int(env.get("frame_stack", 4)),
        "clip_reward": bool(env.get("clip_reward", False)),
        "continuous": bool(env.get("continuous", True)),
        "domain_randomize": bool(env.get("domain_randomize", False)),
        "lap_complete_percent": float(env.get("lap_complete_percent", 0.95)),
    }


def infer_run_dir(model_path: Path) -> Optional[Path]:
    """Locate ``experiments/<run_id>/`` from a checkpoint path."""

    resolved = Path(model_path).resolve()
    for parent in resolved.parents:
        if (parent / "run_metadata.json").is_file():
            return parent
    return None


def checkpoint_in_run(run_dir: Path, *, prefer_best: bool = True) -> Optional[Path]:
    """Return ``best_model.zip`` or ``final_model.zip`` inside a run folder."""

    run_dir = Path(run_dir)
    best = run_dir / "models" / "best" / "best_model.zip"
    final = run_dir / "models" / "final" / "final_model.zip"
    if prefer_best and best.is_file():
        return best
    if final.is_file():
        return final
    if best.is_file():
        return best
    return None


def monitor_tail_mean_reward(run_dir: Path, *, tail: int = 5) -> Optional[float]:
    """Mean of the last ``tail`` logged episode returns in the Monitor CSV."""

    csv_path = find_monitor_csv(run_dir)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, comment="#")
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if df.empty or "r" not in df.columns:
        return None
    return float(df["r"].tail(int(tail)).mean())


def monitor_peak_reward(run_dir: Path) -> Optional[float]:
    """Best single-episode return logged in the Monitor CSV."""

    csv_path = find_monitor_csv(run_dir)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, comment="#")
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if df.empty or "r" not in df.columns:
        return None
    return float(df["r"].max())


def pick_best_checkpoint(
    experiments_dir: Path,
    *,
    arch_name: Optional[str] = None,
) -> Path:
    """Choose the checkpoint from the run with the highest Monitor peak return."""

    experiments_dir = Path(experiments_dir)
    best_path: Optional[Path] = None
    best_score = float("-inf")
    best_timesteps = -1

    for meta in discover_runs(experiments_dir):
        cfg = meta.get("config") or {}
        if arch_name is not None and str(cfg.get("arch_name", "")) != arch_name:
            continue

        run_dir = Path(str(meta.get("run_dir", "")))
        ckpt = checkpoint_in_run(run_dir)
        if ckpt is None:
            continue

        score = monitor_peak_reward(run_dir)
        timesteps = int(meta.get("total_timesteps", 0))
        if score is None:
            score = float(timesteps)
        if score > best_score or (score == best_score and timesteps > best_timesteps):
            best_score = score
            best_timesteps = timesteps
            best_path = ckpt

    if best_path is None:
        raise FileNotFoundError(
            f"No checkpoints found under {experiments_dir.resolve()}. "
            "Train or import a run first."
        )
    return best_path.resolve()


def resolve_checkpoint(
    *,
    model: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    experiments_dir: Path = Path("experiments"),
    prefer_best: bool = True,
    arch_name: Optional[str] = None,
) -> Path:
    """Resolve a ``.zip`` checkpoint from explicit paths or auto-selection."""

    if model is not None:
        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    if run_dir is not None:
        ckpt = checkpoint_in_run(Path(run_dir), prefer_best=prefer_best)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in run directory: {run_dir}")
        return ckpt.resolve()

    return pick_best_checkpoint(experiments_dir, arch_name=arch_name)


def resolve_env_kwargs(
    model_path: Path,
    *,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load env kwargs from ``run_metadata.json`` near the checkpoint."""

    run_dir = infer_run_dir(model_path)
    if run_dir is not None:
        meta = load_run_metadata(run_dir)
        if meta is not None:
            cfg = meta.get("config") or {}
            kwargs = env_kwargs_from_config(cfg)
            if overrides:
                kwargs.update(overrides)
            return kwargs

    defaults = env_kwargs_from_config({})
    if overrides:
        defaults.update(overrides)
    return defaults


def _load_sac(model_path: Path, env) -> SAC:
    return SAC.load(str(model_path), env=env)


def _render_fps(env) -> float:
    base = env
    while hasattr(base, "env"):
        base = base.env
    meta = getattr(base, "metadata", None) or {}
    return float(meta.get("render_fps", 50))


def _rollout_episode(
    model: SAC,
    env,
    *,
    deterministic: bool,
    seed: int,
    realtime: bool,
    show_progress: bool = False,
    episode_label: str = "",
) -> tuple[float, int]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    step_dt = 1.0 / max(_render_fps(env), 1.0)

    if show_progress:
        label = f"{episode_label} " if episode_label else ""
        print(f"{label}started (pygame window — check taskbar). Ctrl+C to stop.", flush=True)

    terminated = truncated = False
    while not (terminated or truncated):
        if realtime:
            time.sleep(step_dt)
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if show_progress and steps % 200 == 0:
            label = f"{episode_label} " if episode_label else ""
            print(f"{label}step {steps}, reward so far {total_reward:.1f}", flush=True)

    return total_reward, steps


def evaluate_agent(
    model_path: Path,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: int = 1000,
    env_kwargs: Optional[dict[str, Any]] = None,
) -> EvalReport:
    """Run ``n_episodes`` without rendering and return aggregated stats."""

    model_path = Path(model_path).resolve()
    kw = resolve_env_kwargs(model_path, overrides=env_kwargs)
    env = make_car_racing_single(**kw, seed=None, monitor_path=None, render_mode=None)

    report = EvalReport(
        model_path=model_path,
        deterministic=bool(deterministic),
        n_episodes=int(n_episodes),
    )

    try:
        model = _load_sac(model_path, env)
        for ep in range(int(n_episodes)):
            reward, length = _rollout_episode(
                model,
                env,
                deterministic=deterministic,
                seed=int(seed + ep),
                realtime=False,
            )
            report.rewards.append(reward)
            report.lengths.append(length)
    finally:
        env.close()

    return report


def watch_agent(
    model_path: Path,
    *,
    n_episodes: Optional[int] = 10,
    loop: bool = False,
    deterministic: bool = True,
    seed: int = 1000,
    env_kwargs: Optional[dict[str, Any]] = None,
    realtime: bool = True,
) -> None:
    """Open a pygame window and drive ``n_episodes`` (or forever if ``loop``)."""

    model_path = Path(model_path).resolve()
    kw = resolve_env_kwargs(model_path, overrides=env_kwargs)
    env = make_car_racing_single(**kw, seed=None, monitor_path=None, render_mode="human")

    model = _load_sac(model_path, env)
    episode_iter = _episode_counter(n_episodes, loop=loop)

    print(f"Watching {model_path.name}  deterministic={deterministic}  env={kw}", flush=True)
    run_dir = infer_run_dir(model_path)
    if run_dir is not None:
        print(f"run_dir: {run_dir.name}", flush=True)
    print(
        "Reward prints after each episode ends (~20–40 s). "
        "Use --fast for quicker test runs.",
        flush=True,
    )
    print("Close the pygame window or press Ctrl+C to stop.", flush=True)

    try:
        for ep_idx in episode_iter:
            ep_label = f"episode {ep_idx + 1}:"
            reward, length = _rollout_episode(
                model,
                env,
                deterministic=deterministic,
                seed=int(seed + ep_idx),
                realtime=realtime,
                show_progress=True,
                episode_label=ep_label,
            )
            print(f"episode {ep_idx + 1}: reward={reward:.1f}  steps={length}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


def _episode_counter(n_episodes: Optional[int], *, loop: bool) -> Iterator[int]:
    if loop or n_episodes is None or int(n_episodes) <= 0:
        idx = 0
        while True:
            yield idx
            idx += 1
    else:
        for idx in range(int(n_episodes)):
            yield idx


def write_eval_summary(report: EvalReport, output_path: Path) -> Path:
    """Persist :class:`EvalReport` as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return output_path.resolve()
