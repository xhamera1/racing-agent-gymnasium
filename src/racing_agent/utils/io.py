"""IO helpers: experiment directory convention, run IDs, JSON metadata dumps.

Every training run lives in its own folder::

    experiments/<run_id>/
        models/best/best_model.zip
        models/final/final_model.zip
        models/checkpoints/sac_<steps>_steps.zip
        logs/tensorboard/...
        logs/monitor/monitor.csv
        run_metadata.json

Where ``<run_id>`` is ``<config_name>__seed<S>__<YYYYmmdd-HHMMSS>``.

Implemented in Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_run_id(config_name: str, seed: int, when: Optional[datetime] = None) -> str:
    """Build a human-readable, sortable, unique run identifier."""
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{config_name}__seed{seed:02d}__{ts}"


def get_experiment_dir(run_id: str, root: Optional[Path] = None) -> Path:
    """Resolve ``experiments/<run_id>``, creating sub-folders on first call.

    Implemented in Phase 3 of ``PLAN.md``.
    """
    raise NotImplementedError("get_experiment_dir() will be implemented in Phase 3.")
