"""Deterministic / stochastic rollout evaluation for the 8-point task.

The 8-point task in the project brief reads:

    "Zapisz stan agenta dajacego najlepsze wyniki. Wykonaj symulacje jego
     dzialania z wylaczonym trybem eksploracji (wykonuje deterministycznie
     najlepsze akcje). Porownaj nagrody z krzywa uzyskana w trakcie uczenia."

This module loads a saved SAC checkpoint, runs ``N`` evaluation episodes with
``deterministic=True`` (and ``False`` for a sanity reference), and returns a
structured report so it can be plotted on top of the training curve.

Implemented in Phase 6 of ``PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalReport:
    """Summary statistics of an N-episode evaluation."""

    model_path: Path
    deterministic: bool
    n_episodes: int
    rewards: List[float] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:  # pragma: no cover - implemented in Phase 6
        ...

    @property
    def std_reward(self) -> float:  # pragma: no cover - implemented in Phase 6
        ...


def evaluate_agent(
    model_path: Path,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: int = 1000,
    video_path: Optional[Path] = None,
) -> EvalReport:
    """Run ``n_episodes`` of the saved agent and return aggregated stats.

    Implemented in Phase 6 of ``PLAN.md``.
    """
    raise NotImplementedError("evaluate_agent() will be implemented in Phase 6.")
