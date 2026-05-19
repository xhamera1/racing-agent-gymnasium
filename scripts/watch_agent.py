"""Live pygame preview of a trained SAC agent on CarRacing-v3.

Opens a game window so you can watch the car drive in real time — no video file
is written.  Env wrappers are restored automatically from ``run_metadata.json``
when the checkpoint lives under ``experiments/``.

Examples
--------
    # Auto-pick the best imported run (highest recent Monitor return).
    python scripts/watch_agent.py

    # Specific checkpoint from a Kaggle import.
    python scripts/watch_agent.py \\
        --run-dir experiments/hp_baseline__arch_light_cnn__seed00__20260518-120000

    # Keep restarting episodes until you close the window / Ctrl+C.
    python scripts/watch_agent.py --loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live pygame preview of a trained SAC agent.")
    p.add_argument("--model", type=Path, default=None, help="Path to a SAC .zip checkpoint.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Experiment folder; uses models/best/best_model.zip (or final).",
    )
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root for auto-picking the best checkpoint.",
    )
    p.add_argument("--arch", default=None, help="Filter auto-pick to this architecture name.")
    p.add_argument("--episodes", type=int, default=5, help="Episodes to play (ignored with --loop).")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Restart episodes forever until Ctrl+C or the window is closed.",
    )
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Run as fast as possible (no real-time pacing).",
    )
    p.add_argument("--final", action="store_true", help="Use final_model.zip instead of best.")
    return p


def _ensure_import_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    _ensure_import_path(repo_root)

    from racing_agent.evaluation.evaluator import (
        infer_run_dir,
        monitor_peak_reward,
        resolve_checkpoint,
        watch_agent,
    )

    try:
        model_path = resolve_checkpoint(
            model=args.model,
            run_dir=args.run_dir,
            experiments_dir=(repo_root / args.experiments_dir).resolve(),
            prefer_best=not args.final,
            arch_name=args.arch,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"checkpoint: {model_path}")
    peak = monitor_peak_reward(infer_run_dir(model_path) or model_path.parent)
    if peak is not None:
        print(f"training peak reward (Monitor): {peak:.1f}")
    watch_agent(
        model_path,
        n_episodes=None if args.loop else int(args.episodes),
        loop=bool(args.loop),
        deterministic=bool(args.deterministic),
        seed=int(args.seed),
        realtime=not args.fast,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
