"""Evaluate a saved SAC model on CarRacing-v3.

Implements the **8-point task** of ``PLAN.md`` (Phase 6): rollout the best
agent with ``deterministic=True``, dump per-episode reward/length, and produce
the summary table that goes on top of the training curve in the final report.

Example
-------
    # Deterministic eval of the best model from a specific run.
    python scripts/evaluate.py \\
        --model experiments/hp_baseline__seed03__.../models/best/best_model.zip \\
        --episodes 50 --deterministic

    # Auto-pick the best run across all experiments (largest mean eval reward).
    python scripts/evaluate.py --episodes 50 --deterministic
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=None,
                   help="Path to a SAC .zip checkpoint. If omitted, auto-pick the best.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic",    dest="deterministic", action="store_false")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("reports/figures/eval_summary.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Implemented in Phase 6 of PLAN.md.
    raise NotImplementedError(
        "scripts/evaluate.py will be implemented in Phase 6 (see PLAN.md)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
