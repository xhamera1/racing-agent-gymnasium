"""Unit tests for evaluation helpers (no Box2D rollout required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from racing_agent.evaluation.evaluator import (
    EvalReport,
    env_kwargs_from_config,
    infer_run_dir,
    monitor_tail_mean_reward,
    resolve_env_kwargs,
    write_eval_summary,
)


def test_env_kwargs_from_config_kaggle_profile() -> None:
    kw = env_kwargs_from_config(
        {
            "env": {
                "grayscale": True,
                "resize_to": 64,
                "frame_stack": 2,
                "clip_reward": False,
            }
        }
    )
    assert kw["resize_to"] == 64
    assert kw["frame_stack"] == 2
    assert kw["grayscale"] is True


def test_eval_report_statistics() -> None:
    report = EvalReport(
        model_path=Path("model.zip"),
        deterministic=True,
        n_episodes=3,
        rewards=[10.0, 20.0, 30.0],
        lengths=[100, 200, 300],
    )
    assert report.mean_reward == 20.0
    assert report.median_reward == 20.0
    assert report.min_reward == 10.0
    assert report.max_reward == 30.0
    assert report.std_reward == pytest.approx(8.1649658, rel=1e-5)


def test_infer_run_dir_from_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "hp_baseline__arch_light_cnn__seed00__20260101-120000"
    (run_dir / "models" / "best").mkdir(parents=True)
    ckpt = run_dir / "models" / "best" / "best_model.zip"
    ckpt.write_bytes(b"stub")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"config": {"env": {"resize_to": 64, "frame_stack": 2}}}),
        encoding="utf-8",
    )
    assert infer_run_dir(ckpt) == run_dir


def test_resolve_env_kwargs_from_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "models" / "best").mkdir(parents=True)
    ckpt = run_dir / "models" / "best" / "best_model.zip"
    ckpt.write_bytes(b"stub")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config": {
                    "env": {
                        "grayscale": True,
                        "resize_to": 64,
                        "frame_stack": 2,
                        "clip_reward": False,
                        "continuous": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    kw = resolve_env_kwargs(ckpt)
    assert kw["resize_to"] == 64
    assert kw["frame_stack"] == 2


def test_monitor_tail_mean_reward(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    mon = run_dir / "logs" / "monitor"
    mon.mkdir(parents=True)
    (mon / "monitor_0.monitor.csv").write_text(
        "# comment\nr,l,t\n-50,100,1.0\n-40,100,2.0\n-30,100,3.0\n",
        encoding="utf-8",
    )
    assert monitor_tail_mean_reward(run_dir, tail=2) == pytest.approx(-35.0)


def test_monitor_peak_reward(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    mon = run_dir / "logs" / "monitor"
    mon.mkdir(parents=True)
    (mon / "monitor_0.monitor.csv").write_text(
        "# comment\nr,l,t\n-50,100,1.0\n863.0,900,2.0\n-30,100,3.0\n",
        encoding="utf-8",
    )
    from racing_agent.evaluation.evaluator import monitor_peak_reward

    assert monitor_peak_reward(run_dir) == pytest.approx(863.0)


def test_write_eval_summary(tmp_path: Path) -> None:
    report = EvalReport(
        model_path=Path("m.zip"),
        deterministic=True,
        n_episodes=1,
        rewards=[42.0],
        lengths=[10],
    )
    out = write_eval_summary(report, tmp_path / "eval_summary.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mean_reward"] == 42.0
    assert payload["model_path"] == "m.zip"
