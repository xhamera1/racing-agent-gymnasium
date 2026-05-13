"""Record a video of a trained SAC agent racing on CarRacing-v3.

Used in Phase 6 of ``PLAN.md`` to produce the demo clip that goes into
``reports/figures/agent_demo.mp4``.

Example
-------
    python scripts/record_video.py \\
        --model models/best/best_model.zip \\
        --episodes 3 --deterministic \\
        --output reports/figures/agent_demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True,
                   help="Path to a SAC .zip checkpoint.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic",    dest="deterministic", action="store_false")
    p.add_argument("--output", type=Path, default=Path("reports/figures/agent_demo.mp4"))
    p.add_argument("--fps", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Uses gymnasium.wrappers.RecordVideo + moviepy under the hood.
    raise NotImplementedError(
        "scripts/record_video.py will be implemented in Phase 6 (see PLAN.md)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
