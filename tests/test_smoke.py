"""Smoke tests that just check imports and configuration plumbing.

These are designed to run in CI on a fresh clone without launching any
training. Heavy tests (env wrappers, policy forward pass) live in
``test_env.py`` and ``test_policies.py`` and are added as their corresponding
phases of ``PLAN.md`` are implemented.
"""

from __future__ import annotations

import pytest


def test_package_importable() -> None:
    import racing_agent  # noqa: F401

    assert hasattr(racing_agent, "__version__")


@pytest.mark.parametrize("module", [
    "racing_agent.env",
    "racing_agent.policies",
    "racing_agent.training",
    "racing_agent.evaluation",
    "racing_agent.utils",
])
def test_subpackages_importable(module: str) -> None:
    __import__(module)


def test_configs_are_valid_yaml() -> None:
    """All shipped configs/*.yaml parse cleanly."""
    import yaml
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    for cfg in (repo_root / "configs").glob("*.yaml"):
        with cfg.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{cfg.name} did not parse as a dict"
        assert "name" in data, f"{cfg.name} is missing the required 'name' key"
