"""Evaluate a saved SAC model on CarRacing-v3 (no rendering).

Implements the **8-point task** statistics pass of ``PLAN.md`` (Phase 6):
rollout the best agent with ``deterministic=True``, dump per-episode reward/length,
and produce the summary table for the final report.

For a live pygame preview use ``scripts/watch_agent.py`` instead.

Example
-------
    python scripts/evaluate.py --episodes 50 --deterministic
    python scripts/evaluate.py --model experiments/.../models/best/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic / stochastic SAC evaluation (headless).")
    p.add_argument("--model", type=Path, default=None,
                   help="Path to a SAC .zip checkpoint. If omitted, auto-pick the best.")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Experiment folder; uses models/best/best_model.zip (or final).")
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    p.add_argument("--arch", default=None, help="Filter auto-pick to this architecture name.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("reports/figures/eval_summary.json"))
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

    from racing_agent.evaluation.evaluator import evaluate_agent, resolve_checkpoint, write_eval_summary

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

    report = evaluate_agent(
        model_path,
        n_episodes=int(args.episodes),
        deterministic=bool(args.deterministic),
        seed=int(args.seed),
    )
    out_path = write_eval_summary(report, repo_root / args.output)

    summary = {
        "model": str(model_path),
        "output": str(out_path),
        "deterministic": report.deterministic,
        "n_episodes": report.n_episodes,
        "mean_reward": report.mean_reward,
        "std_reward": report.std_reward,
        "median_reward": report.median_reward,
        "min_reward": report.min_reward,
        "max_reward": report.max_reward,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
