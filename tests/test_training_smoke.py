"""Optional integration smoke: ``Trainer.run`` short loop (requires Box2D + SB3)."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import Box2D  # noqa: F401
except ImportError:
    Box2D = None


pytest.importorskip("stable_baselines3")

pytestmark = pytest.mark.skipif(
    Box2D is None,
    reason='Install Box2D for CarRacing (e.g. pip install "gymnasium[box2d]").',
)


def test_trainer_short_smoke_writes_artifacts(tmp_path: Path) -> None:
    from racing_agent.training import Trainer, load_config

    repo_root = Path(__file__).resolve().parents[1]
    configs = repo_root / "configs"
    cfg = load_config(configs / "hp_baseline.yaml", configs / "arch_nature_cnn.yaml")
    cfg["learning_starts"] = 64

    trainer = Trainer(cfg, seed=0, output_root=tmp_path, repo_root=repo_root)
    result = trainer.run(total_timesteps=800)

    meta_json = result.run_dir / "run_metadata.json"
    tb_dir = result.run_dir / "logs" / "tensorboard"

    assert meta_json.is_file()
    assert result.final_model_path.is_file()
    assert result.monitor_csv.is_file()

    tb_events = list(tb_dir.glob("events.out.tfevents*"))
    assert tb_events, "tensorboard logs should flush at least one event file"
