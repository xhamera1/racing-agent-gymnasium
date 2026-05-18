"""High-level training entry point.

:class:`Trainer` builds the VecEnv, SAC with CNN features, attaches callbacks,
persists artefacts under ``experiments/<run_id>/``, and emits ``run_metadata.json``.

Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from racing_agent.env.make_env import make_car_racing, make_car_racing_single
from racing_agent import policies as _policies
from racing_agent.training.callbacks import EvalSaveBestCallback, StepTimingCallback
from racing_agent.utils.io import find_monitor_csv, generate_run_id, get_experiment_dir
from racing_agent.utils.seeding import set_global_seed

_REPO_ROOT_HINT = Path(__file__).resolve().parents[3]

_FEATURE_CLASSES: Dict[str, type] = {
    "NatureCNN": _policies.NatureCNN,
    "CustomDeepCNN": _policies.CustomDeepCNN,
    "LightCNN": _policies.LightCNN,
}

_sac_excluded_keys: frozenset[str] = frozenset(
    {
        "env",
        "hp_name",
        "arch_name",
        "experiment_name",
        "policy",
        "features_extractor",
        "net_arch",
        "trainer",
        "overrides_name",
    },
)


@dataclass
class TrainResult:
    """Compact record of a finished run; consumed by the report tooling."""

    run_id: str
    run_dir: Path
    final_model_path: Path
    best_model_path: Path
    monitor_csv: Path
    mean_step_time_s: float
    mean_episode_time_s: float
    wall_clock_s: float
    total_timesteps: int


def resolve_feature_extractor_class(name: str) -> type:
    try:
        return _FEATURE_CLASSES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_FEATURE_CLASSES))
        raise KeyError(f"Unknown features_extractor.class {name!r}. Known: {known}.") from exc


def build_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``policy_kwargs`` for SAC ``CnnPolicy``."""

    spec = config["features_extractor"]
    cls = resolve_feature_extractor_class(str(spec["class"]))
    kwargs = dict(spec.get("kwargs", {}) or {})
    return {
        "features_extractor_class": cls,
        "features_extractor_kwargs": kwargs,
        "net_arch": dict(config["net_arch"]),
    }


def sac_algorithm_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Strip merged YAML entries that belong to Trainer / architecture only."""

    return {k: v for k, v in config.items() if k not in _sac_excluded_keys}


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


class Trainer:
    """Run a single (config, seed) → trained SAC model experiment."""

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int,
        output_root: Path | None = None,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.output_root = Path(output_root) if output_root is not None else Path("experiments")
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT_HINT

    def run(self, total_timesteps: int) -> TrainResult:
        cfg = self.config

        trainer_cfg_raw = cfg.get("trainer")
        trainer_cfg = dict(trainer_cfg_raw) if isinstance(trainer_cfg_raw, dict) else {}
        eval_every_ts = max(int(trainer_cfg.get("eval_every_timesteps", 5000)), 1)
        checkpoint_every_ts = max(int(trainer_cfg.get("checkpoint_every_timesteps", 10_000)), 1)
        algo_verbose = int(trainer_cfg.get("sb3_verbose", 0))
        raw_log_iv = trainer_cfg.get("learn_log_interval")
        if raw_log_iv is not None:
            learn_log_interval: int | None = int(raw_log_iv)
        elif algo_verbose <= 0:
            learn_log_interval = None
        else:
            learn_log_interval = 100

        set_global_seed(self.seed)
        master_seed_env = trainer_cfg.get("env_master_seed_offset")
        if master_seed_env is None:
            train_seed = self.seed
        else:
            train_seed = self.seed + int(master_seed_env)

        run_id = generate_run_id(cfg["hp_name"], cfg["arch_name"], self.seed)
        run_dir = get_experiment_dir(run_id, self.output_root)

        env_section = cfg["env"]
        n_envs = int(env_section["n_envs"])

        monitor_dir = run_dir / "logs" / "monitor"
        vec_env_kwargs = dict(
            seed=train_seed,
            monitor_dir=monitor_dir,
            grayscale=bool(env_section.get("grayscale", True)),
            resize_to=int(env_section.get("resize_to", 84)),
            frame_stack=int(env_section.get("frame_stack", 4)),
            clip_reward=bool(env_section.get("clip_reward", False)),
            continuous=bool(env_section.get("continuous", True)),
            domain_randomize=bool(env_section.get("domain_randomize", False)),
            lap_complete_percent=float(env_section.get("lap_complete_percent", 0.95)),
            render_mode=None,
        )

        env = make_car_racing(n_envs=n_envs, **vec_env_kwargs)

        eval_seed = trainer_cfg.get("eval_seed_offset")
        if eval_seed is None:
            eval_seed = train_seed + 4096
        else:
            eval_seed = train_seed + int(eval_seed)

        eval_kw = dict(
            seed=int(eval_seed),
            grayscale=bool(env_section.get("grayscale", True)),
            resize_to=int(env_section.get("resize_to", 84)),
            frame_stack=int(env_section.get("frame_stack", 4)),
            clip_reward=bool(env_section.get("clip_reward", False)),
            continuous=bool(env_section.get("continuous", True)),
            domain_randomize=bool(env_section.get("domain_randomize", False)),
            lap_complete_percent=float(env_section.get("lap_complete_percent", 0.95)),
            render_mode=None,
        )

        eval_env = make_car_racing_single(**eval_kw, monitor_path=None)

        timing_cb = StepTimingCallback()
        eval_freq_calls = max(eval_every_ts // n_envs, 1)
        checkpoint_freq_calls = max(checkpoint_every_ts // n_envs, 1)

        eval_cb = EvalSaveBestCallback(
            eval_env,
            eval_freq_calls=eval_freq_calls,
            best_model_dir=run_dir / "models" / "best",
            eval_log_root=run_dir / "logs",
            n_eval_episodes=int(trainer_cfg.get("n_eval_episodes", 5)),
            deterministic=True,
            verbose=algo_verbose,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq_calls,
            save_path=str(run_dir / "models" / "checkpoints"),
            name_prefix="sac",
            verbose=max(0, algo_verbose - 1),
        )

        callbacks = CallbackList([timing_cb, eval_cb, checkpoint_cb])

        policy_kw = build_policy_kwargs(cfg)
        algo_kw = sac_algorithm_kwargs(cfg)

        tb_log = trainer_cfg.get("tensorboard_log")
        if tb_log is None:
            tb_log_dir = run_dir / "logs" / "tensorboard"
        else:
            tb_log_dir = Path(tb_log)
        tb_log_dir.mkdir(parents=True, exist_ok=True)

        model = SAC(
            str(cfg["policy"]),
            env,
            verbose=algo_verbose,
            seed=self.seed,
            tensorboard_log=str(tb_log_dir),
            policy_kwargs=policy_kw,
            **algo_kw,
        )

        wall_s = 0.0
        mean_step = 0.0
        mean_ep = 0.0

        final_dir = run_dir / "models" / "final"
        final_zip = final_dir / "final_model.zip"
        best_zip = run_dir / "models" / "best" / "best_model.zip"

        try:
            wall0 = time.perf_counter()
            model.learn(
                total_timesteps=int(total_timesteps),
                callback=callbacks,
                log_interval=learn_log_interval,
            )
            wall_s = float(time.perf_counter() - wall0)

            model.save(str(final_dir / "final_model"))

            mean_step = timing_cb.mean_step_time_s()
            mean_ep = timing_cb.mean_episode_time_s()

            meta = {
                "run_id": run_id,
                "config": _json_sanitize(dict(cfg)),
                "seed": self.seed,
                "total_timesteps": int(total_timesteps),
                "wall_clock_s": wall_s,
                "mean_step_time_s": mean_step,
                "mean_episode_time_s": mean_ep,
                "git_hash": _git_revision(self.repo_root),
            }
            with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        finally:
            eval_env.close()
            env.close()

        mon_csv = find_monitor_csv(run_dir, rank=0) or (monitor_dir / "monitor_0.monitor.csv")

        return TrainResult(
            run_id=run_id,
            run_dir=run_dir,
            final_model_path=final_zip.resolve(),
            best_model_path=best_zip.resolve(),
            monitor_csv=mon_csv.resolve() if mon_csv.is_file() else mon_csv,
            mean_step_time_s=mean_step,
            mean_episode_time_s=mean_ep,
            wall_clock_s=wall_s,
            total_timesteps=int(total_timesteps),
        )


def train_result_summary(result: TrainResult) -> Dict[str, Any]:
    """JSON-friendly subset for CLI printing."""

    d: Dict[str, Any] = dict(asdict(result))
    d["final_model_exists"] = result.final_model_path.is_file()
    d["best_model_exists"] = result.best_model_path.is_file()
    d["monitor_csv_exists"] = result.monitor_csv.is_file()
    d["meta_exists"] = (result.run_dir / "run_metadata.json").is_file()
    tb_dir = result.run_dir / "logs" / "tensorboard"
    d["tensorboard_nonempty"] = tb_dir.is_dir() and any(tb_dir.iterdir())

    out: Dict[str, Any] = {}
    for key, val in d.items():
        out[key] = str(val) if isinstance(val, Path) else val
    return out
