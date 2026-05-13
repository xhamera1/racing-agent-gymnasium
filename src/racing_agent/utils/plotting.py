"""Reporting plots: learning curves with mean +/- std, HP comparisons, arch diagrams.

These are the figures that go straight into ``reports/final_report.pdf``.
Implemented in Phases 4-5 of ``PLAN.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence


def plot_learning_curve(
    monitor_csvs: Sequence[Path],
    output_path: Path,
    title: str,
    smoothing_window: int = 100,
) -> None:
    """Mean ± std learning curve across multiple seeds of the same config.

    Parameters
    ----------
    monitor_csvs:
        ``Monitor`` CSVs from N seeds of the **same** HP/arch config.
    output_path:
        Destination PNG path.
    title:
        Plot title.
    smoothing_window:
        Rolling-mean window applied **before** computing mean/std across seeds.

    Notes
    -----
    Required for the 4-point task: x-axis = timesteps, y-axis = episode reward,
    one band = mean ± std over the 10 runs of one HP set.

    Implemented in Phase 4 of ``PLAN.md``.
    """
    raise NotImplementedError("plot_learning_curve() will be implemented in Phase 4.")


def plot_hp_comparison(
    runs_by_config: dict,
    output_path: Path,
    title: str = "Hyperparameter comparison",
) -> None:
    """Overlay mean curves of all HP sets on a single figure.

    Implemented in Phase 4 of ``PLAN.md``.
    """
    raise NotImplementedError("plot_hp_comparison() will be implemented in Phase 4.")


def plot_arch_diagram(
    arch_name: str,
    layers: Iterable[dict],
    output_path: Optional[Path] = None,
) -> None:
    """Render a block diagram of a CNN architecture (layer / shape / activation).

    Used in the 6-point task to satisfy *"Narysuj w sprawozdaniu ich schematy
    (rodzaje warstw, ich wielkosci, funkcje aktywacji)"*.

    Implemented in Phase 5 of ``PLAN.md``.
    """
    raise NotImplementedError("plot_arch_diagram() will be implemented in Phase 5.")
