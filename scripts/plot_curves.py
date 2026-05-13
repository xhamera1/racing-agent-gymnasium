"""Regenerate every figure in ``reports/figures/`` from ``experiments/``.

Outputs (matches Phase 4 / 5 of ``PLAN.md``):

- ``reports/figures/learning_curve_<hp>.png``   -- one per HP set (mean +/- std).
- ``reports/figures/learning_curve_compare.png`` -- all 3 HP sets overlaid.
- ``reports/figures/arch_compare.png``           -- Architecture A vs B.
- ``reports/figures/timing_table.csv``           -- avg step/episode time per config.
- ``reports/figures/arch_A.png``, ``arch_B.png`` -- architecture block diagrams.

The script discovers runs by globbing ``experiments/*/run_metadata.json``,
groups them by config name, and feeds the ``Monitor`` CSVs into
:func:`racing_agent.utils.plotting.plot_learning_curve`.

Example
-------
    python scripts/plot_curves.py
    python scripts/plot_curves.py --experiments-dir experiments --output reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    p.add_argument("--output",          type=Path, default=Path("reports/figures"))
    p.add_argument("--smoothing-window", type=int, default=100)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        "scripts/plot_curves.py will be implemented in Phase 4 (see PLAN.md)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
