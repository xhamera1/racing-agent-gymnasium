"""High-level training entry point.

The :class:`Trainer` glues together env construction, feature-extractor choice,
SAC instantiation and the callback stack into a single ``run()`` method that:

1. builds a deterministic VecEnv from a (seed, wrapper_config),
2. constructs ``stable_baselines3.SAC`` with the merged YAML config,
3. calls ``model.learn(total_timesteps=...)`` with our callbacks,
4. persists best model / final model / checkpoints / metadata.

Implemented in Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainResult:
    """Compact record of a finished run; consumed by the report tooling."""

    run_id: str
    run_dir: Path
    final_model_path: Path
    best_model_path: Path
    monitor_csv: Path
    mean_step_time_s: float
    wall_clock_s: float
    total_timesteps: int


class Trainer:
    """Run a single (config, seed) -> trained model experiment.

    Parameters
    ----------
    config:
        Merged dict produced by :func:`racing_agent.training.hyperparams.load_config`.
    seed:
        Master seed (also used for env action-space and SAC).
    output_root:
        Where ``experiments/<run_id>/`` will be created.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int,
        output_root: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.output_root = Path(output_root) if output_root else Path("experiments")

    def run(self, total_timesteps: int) -> TrainResult:
        """Train ``total_timesteps`` steps and return a :class:`TrainResult`.

        Implemented in Phase 3 of ``PLAN.md``.
        """
        raise NotImplementedError(
            "Trainer.run() will be implemented in Phase 3 (see PLAN.md)."
        )
