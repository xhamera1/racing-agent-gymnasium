"""Train one (config, seed) -> one SAC model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one SAC experiment (merged hp + arch YAML).")
    p.add_argument("--hp", required=True, type=Path, help="Path to a configs/hp_*.yaml file.")
    p.add_argument("--arch", required=True, type=Path, help="Path to a configs/arch_*.yaml file.")
    p.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Optional configs/* overrides YAML (e.g. kaggle_overrides.yaml).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Git checkout root for run_metadata.git_hash (default: parent of configs/).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scripts_dir = Path(__file__).resolve().parent
    repo_src = scripts_dir.parent / "src"
    if repo_src.is_dir() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from racing_agent.training.hyperparams import load_config
    from racing_agent.training.train import Trainer, train_result_summary

    hp_path = args.hp.expanduser().resolve()
    arch_path = args.arch.expanduser().resolve()
    if not hp_path.is_file():
        raise SystemExit(f"HP config not found: {hp_path}")
    if not arch_path.is_file():
        raise SystemExit(f"Architecture config not found: {arch_path}")

    overrides_path = args.overrides.expanduser().resolve() if args.overrides is not None else None
    if overrides_path is not None and not overrides_path.is_file():
        raise SystemExit(f"Overrides config not found: {overrides_path}")

    repo_root = args.repo_root.expanduser().resolve() if args.repo_root is not None else hp_path.parents[1]

    cfg = load_config(hp_path, arch_path, overrides_path=overrides_path)
    result = Trainer(cfg, seed=args.seed, output_root=args.output_root, repo_root=repo_root).run(
        args.timesteps,
    )

    print(json.dumps(train_result_summary(result), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
