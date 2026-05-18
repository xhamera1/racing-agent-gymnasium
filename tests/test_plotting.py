"""Tests for experiment discovery helpers and plotting (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_monitor_csv(path: Path, rewards: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write('#{"t_start": 0.0, "env_id": "CarRacing-v3"}\n')
        f.write("r,l,t\n")
        for idx, reward in enumerate(rewards, start=1):
            f.write(f"{reward},{100 + idx},{float(idx)}\n")


def _write_run(
    root: Path,
    *,
    run_id: str,
    hp_name: str,
    arch_name: str,
    seed: int,
    timesteps: int,
    rewards: list[float],
    legacy_monitor_name: bool = False,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "final").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "final" / "final_model.zip").write_bytes(b"zip")

    monitor_dir = run_dir / "logs" / "monitor"
    if legacy_monitor_name:
        mon = monitor_dir / "monitor_0.csv.monitor.csv"
    else:
        mon = monitor_dir / "monitor_0.monitor.csv"
    _write_monitor_csv(mon, rewards)

    meta = {
        "run_id": run_id,
        "seed": seed,
        "total_timesteps": timesteps,
        "wall_clock_s": 100.0 + seed,
        "mean_step_time_s": 0.5,
        "mean_episode_time_s": 10.0,
        "config": {"hp_name": hp_name, "arch_name": arch_name},
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    return run_dir


def test_find_monitor_csv_supports_legacy_suffix(tmp_path: Path) -> None:
    from racing_agent.utils.io import find_monitor_csv

    run_dir = _write_run(
        tmp_path,
        run_id="hp_baseline__arch_nature_cnn__seed00__20260101-120000",
        hp_name="hp_baseline",
        arch_name="arch_nature_cnn",
        seed=0,
        timesteps=5000,
        rewards=[-100.0, -80.0],
        legacy_monitor_name=True,
    )
    found = find_monitor_csv(run_dir)
    assert found is not None
    assert found.name.endswith(".monitor.csv")


def test_group_runs_by_hp_keeps_newest_seed_run(tmp_path: Path) -> None:
    from racing_agent.utils.io import discover_runs, group_runs_by_hp

    _write_run(
        tmp_path,
        run_id="hp_baseline__arch_nature_cnn__seed00__20260101-100000",
        hp_name="hp_baseline",
        arch_name="arch_nature_cnn",
        seed=0,
        timesteps=5000,
        rewards=[-200.0],
    )
    _write_run(
        tmp_path,
        run_id="hp_baseline__arch_nature_cnn__seed00__20260102-100000",
        hp_name="hp_baseline",
        arch_name="arch_nature_cnn",
        seed=0,
        timesteps=5000,
        rewards=[-50.0],
    )

    grouped = group_runs_by_hp(discover_runs(tmp_path), arch_name="arch_nature_cnn", min_timesteps=1000)
    assert len(grouped["hp_baseline"]) == 1
    assert grouped["hp_baseline"][0]["run_id"].endswith("20260102-100000")


def test_plot_learning_curve_writes_png(tmp_path: Path) -> None:
    from racing_agent.utils.plotting import plot_learning_curve

    csvs = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.monitor.csv"
        _write_monitor_csv(path, [-120.0 + seed * 5 + i for i in range(20)])
        csvs.append(path)

    out = tmp_path / "curve.png"
    plot_learning_curve(csvs, out, title="test", smoothing_window=3)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_curves_cli_on_synthetic_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    exp = tmp_path / "experiments"
    out = tmp_path / "figures"

    for seed in range(2):
        _write_run(
            exp,
            run_id=f"hp_baseline__arch_nature_cnn__seed{seed:02d}__20260101-120{seed}00",
            hp_name="hp_baseline",
            arch_name="arch_nature_cnn",
            seed=seed,
            timesteps=50000,
            rewards=[-100.0 + seed + i * 0.5 for i in range(30)],
        )

    monkeypatch.chdir(repo)
    from scripts import plot_curves as plot_mod

    rc = plot_mod.main(
        [
            "--experiments-dir",
            str(exp),
            "--output",
            str(out),
            "--hp-configs",
            "hp_baseline",
            "--min-timesteps",
            "1000",
            "--smoothing-window",
            "5",
        ],
    )
    assert rc == 0
    assert (out / "learning_curve_hp_baseline.png").is_file()
    assert (out / "timing_table.csv").is_file()


def test_run_experiment_dry_run() -> None:
    from scripts import run_experiment as sweep_mod

    rc = sweep_mod.main(
        [
            "--configs",
            "hp_baseline",
            "hp_high_lr",
            "--seeds",
            "0..1",
            "--timesteps",
            "50000",
            "--dry-run",
        ],
    )
    assert rc == 0
