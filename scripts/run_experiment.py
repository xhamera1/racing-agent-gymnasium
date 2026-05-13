"""Full sweep: ``configs x seeds`` -> N x M trained models.

This is the orchestration script for **Phase 4** of ``PLAN.md`` (3 HP sets x
10 seeds x >= 50 000 timesteps = the 4-point deliverable) and **Phase 5**
(architecture comparison).

Example
-------
    python scripts/run_experiment.py \\
        --configs hp_baseline hp_high_lr hp_large_batch \\
        --arch    arch_nature_cnn \\
        --seeds   0 1 2 3 4 5 6 7 8 9 \\
        --timesteps 50000

Runs are sequential by default; set ``--n-jobs N`` to launch them as parallel
subprocesses.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", required=True,
                   help="Names of configs/hp_*.yaml files (without .yaml extension).")
    p.add_argument("--arch", default="arch_nature_cnn",
                   help="Name of configs/arch_*.yaml file (without .yaml extension).")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=list(range(10)),
                   help="Seeds to run for every config (default: 0..9 -- 10 seeds).")
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Total timesteps per run (PDF brief: >= 50 000).")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Number of concurrent runs (1 = sequential).")
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Implemented in Phase 4 of PLAN.md.
    raise NotImplementedError(
        "scripts/run_experiment.py will be implemented in Phase 4 (see PLAN.md)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
