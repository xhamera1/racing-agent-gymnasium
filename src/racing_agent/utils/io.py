"""IO helpers: experiment directory convention, run IDs, JSON metadata dumps.

Every training run lives in its own folder::

    experiments/<run_id>/
        models/best/
        models/final/
        models/checkpoints/
        logs/tensorboard/...
        logs/monitor/monitor_<rank>.monitor.csv
        run_metadata.json

Where ``<run_id>`` is ``<hp_name>__<arch_name>__seed<S>__<YYYYmmdd-HHMMSS>``.

Phase 3–4 of ``PLAN.md``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def generate_run_id(
    hp_name: str,
    arch_name: str,
    seed: int,
    when: Optional[datetime] = None,
) -> str:
    """Build a human-readable, sortable, unique run identifier."""

    slug = f"{hp_name}__{arch_name}__seed{seed:02d}"
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{slug}__{ts}"


def get_experiment_dir(run_id: str, root: Optional[Path] = None) -> Path:
    """Resolve ``<root>/<run_id>`` (default ``experiments/``); create canonical subdirs."""

    base = Path(root) if root is not None else Path("experiments")
    run_dir = (base / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "final").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "monitor").mkdir(parents=True, exist_ok=True)
    return run_dir


def find_monitor_csv(run_dir: Path, rank: int = 0) -> Optional[Path]:
    """Locate the SB3 Monitor CSV for ``rank`` (handles legacy ``*.csv.monitor.csv`` names)."""

    monitor_dir = Path(run_dir) / "logs" / "monitor"
    if not monitor_dir.is_dir():
        return None

    candidates = [
        monitor_dir / f"monitor_{rank}.monitor.csv",
        monitor_dir / f"monitor_{rank}.csv.monitor.csv",
        monitor_dir / f"monitor_{rank}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    matches = sorted(monitor_dir.glob("*.monitor.csv"))
    if matches:
        return matches[0]
    return None


def load_run_metadata(run_dir: Path) -> Optional[dict[str, Any]]:
    meta_path = Path(run_dir) / "run_metadata.json"
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data.setdefault("run_dir", str(run_dir.resolve()))
        return data
    return None


def discover_runs(experiments_dir: Path) -> list[dict[str, Any]]:
    """Load every ``experiments/*/run_metadata.json`` under ``experiments_dir``."""

    root = Path(experiments_dir)
    if not root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("*/run_metadata.json")):
        run_dir = meta_path.parent
        meta = load_run_metadata(run_dir)
        if meta is not None:
            meta["run_dir"] = str(run_dir.resolve())
            meta["meta_path"] = str(meta_path.resolve())
            runs.append(meta)
    return runs


def parse_run_id(run_id: str) -> Optional[dict[str, Any]]:
    """Parse ``<hp>__<arch>__seedNN__YYYYmmdd-HHMMSS`` run folder names."""

    parts = str(run_id).split("__")
    if len(parts) < 4:
        return None
    seed_part = parts[-2]
    if not seed_part.startswith("seed"):
        return None
    try:
        seed = int(seed_part.removeprefix("seed"))
    except ValueError:
        return None
    return {
        "hp_name": "__".join(parts[:-3]),
        "arch_name": parts[-3],
        "seed": seed,
    }


def is_run_complete(
    run_dir: Path,
    *,
    min_timesteps: int,
    hp_name: Optional[str] = None,
    arch_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> bool:
    """True when a run folder has metadata, final model, and enough timesteps."""

    run_dir = Path(run_dir)
    meta = load_run_metadata(run_dir)
    if meta is None:
        return False

    if int(meta.get("total_timesteps", 0)) < int(min_timesteps):
        return False

    final_model = run_dir / "models" / "final" / "final_model.zip"
    if not final_model.is_file():
        return False

    cfg = meta.get("config") or {}
    if hp_name is not None and str(cfg.get("hp_name", meta.get("hp_name", ""))) != hp_name:
        return False
    if arch_name is not None and str(cfg.get("arch_name", meta.get("arch_name", ""))) != arch_name:
        return False
    if seed is not None and int(meta.get("seed", -1)) != int(seed):
        return False

    return True


def find_completed_run(
    experiments_dir: Path,
    *,
    hp_name: str,
    arch_name: str,
    seed: int,
    min_timesteps: int,
) -> Optional[Path]:
    """Return the newest complete run directory for a (hp, arch, seed) triple."""

    matches: list[tuple[str, Path]] = []
    root = Path(experiments_dir)
    if not root.is_dir():
        return None

    prefix = f"{hp_name}__{arch_name}__seed{seed:02d}__"
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        if is_run_complete(
            run_dir,
            min_timesteps=min_timesteps,
            hp_name=hp_name,
            arch_name=arch_name,
            seed=seed,
        ):
            matches.append((run_dir.name, run_dir))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1].resolve()


def group_runs_by_hp(
    runs: list[dict[str, Any]],
    *,
    arch_name: Optional[str] = None,
    min_timesteps: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Group metadata dicts by ``config['hp_name']`` (newest run per seed wins)."""

    grouped: dict[str, dict[int, dict[str, Any]]] = {}

    for meta in runs:
        cfg = meta.get("config") or {}
        hp = str(cfg.get("hp_name", ""))
        arch = str(cfg.get("arch_name", ""))
        if not hp:
            continue
        if arch_name is not None and arch != arch_name:
            continue
        if int(meta.get("total_timesteps", 0)) < int(min_timesteps):
            continue

        seed = int(meta.get("seed", -1))
        run_dir = Path(str(meta.get("run_dir", "")))
        if not is_run_complete(run_dir, min_timesteps=min_timesteps, hp_name=hp, arch_name=arch, seed=seed):
            continue

        run_id = str(meta.get("run_id", run_dir.name))
        grouped.setdefault(hp, {})
        prev = grouped[hp].get(seed)
        if prev is None or str(prev.get("run_id", "")) <= run_id:
            grouped[hp][seed] = meta

    return {hp: [grouped[hp][s] for s in sorted(grouped[hp])] for hp in sorted(grouped)}
