"""Unit tests for config merging (no Box2D / training required)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(name="configs_dir")
def configs_dir_fixture() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def test_load_baseline_and_nature_cnn(configs_dir: Path) -> None:
    from racing_agent.training import load_config

    c = load_config(configs_dir / "hp_baseline.yaml", configs_dir / "arch_nature_cnn.yaml")

    assert c["hp_name"] == "hp_baseline"
    assert c["arch_name"] == "arch_nature_cnn"
    assert c["experiment_name"] == "hp_baseline__arch_nature_cnn"
    assert c["policy"] == "CnnPolicy"
    assert c["features_extractor"]["class"] == "NatureCNN"
    assert c["learning_rate"] == 3.0e-4
    assert int(c["env"]["n_envs"]) == 1


def test_nested_env_merge_via_load_config(tmp_path: Path) -> None:
    """Architecture YAML overrides duplicate keys while preserving missing nested hp keys."""

    from racing_agent.training import load_config

    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text(
        """name: hp_demo
collision: from_hp
env:
  n_envs: 2
  keep_from_hp_only: yes
training_lr_like: 0.1
""",
        encoding="utf-8",
    )

    arch_file = tmp_path / "arch.yaml"
    arch_file.write_text(
        """name: arch_demo
collision: from_arch_wins
env:
  grayscale: false
policy: "CnnPolicy"
features_extractor:
  class: "NatureCNN"
  kwargs: {features_dim: 512}
net_arch:
  pi: [64]
  qf: [64]
""",
        encoding="utf-8",
    )

    cfg = load_config(hp_file, arch_file)

    assert cfg["hp_name"] == "hp_demo"
    assert cfg["arch_name"] == "arch_demo"
    assert cfg["collision"] == "from_arch_wins"
    assert int(cfg["env"]["n_envs"]) == 2
    assert cfg["env"]["keep_from_hp_only"] is True
    assert cfg["env"]["grayscale"] is False
    assert cfg["training_lr_like"] == 0.1


def test_load_config_with_kaggle_overrides(configs_dir: Path) -> None:
    from racing_agent.training import load_config

    cfg = load_config(
        configs_dir / "hp_baseline.yaml",
        configs_dir / "arch_light_cnn.yaml",
        configs_dir / "kaggle_overrides.yaml",
    )
    assert cfg["arch_name"] == "arch_light_cnn"
    assert int(cfg["env"]["resize_to"]) == 64
    assert int(cfg["env"]["frame_stack"]) == 2
    assert int(cfg["trainer"]["eval_every_timesteps"]) == 50000
