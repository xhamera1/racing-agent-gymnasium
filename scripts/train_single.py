"""Train one (config, seed) -> one SAC model.

Example
-------
    python scripts/train_single.py \\
        --hp configs/hp_baseline.yaml \\
        --arch configs/arch_nature_cnn.yaml \\
        --seed 0 \\
        --timesteps 50000

Implemented in Phase 3 of ``PLAN.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hp",   required=True, type=Path, help="Path to a configs/hp_*.yaml file.")
    p.add_argument("--arch", required=True, type=Path, help="Path to a configs/arch_*.yaml file.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # In Phase 3 this becomes:
    #   from racing_agent.training import Trainer, load_config
    #   from racing_agent.utils import set_global_seed
    #   set_global_seed(args.seed)
    #   cfg = load_config(args.hp, args.arch)
    #   result = Trainer(cfg, args.seed, args.output_root).run(args.timesteps)
    #   print(json.dumps(result.__dict__, default=str, indent=2))
    raise NotImplementedError(
        "scripts/train_single.py will be implemented in Phase 3 (see PLAN.md)."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
