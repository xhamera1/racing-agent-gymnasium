"""Regenerate Phase-4 figures in ``reports/figures/`` from ``experiments/``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot learning curves and timing table from experiment runs.")
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    p.add_argument("--output", type=Path, default=Path("reports/figures"))
    p.add_argument("--arch", default="arch_nature_cnn", help="Filter runs to this architecture name.")
    p.add_argument("--min-timesteps", type=int, default=1, help="Ignore runs shorter than this.")
    p.add_argument("--smoothing-window", type=int, default=100)
    p.add_argument(
        "--hp-configs",
        nargs="+",
        default=["hp_baseline", "hp_high_lr", "hp_large_batch"],
        help="HP config names to include (order preserved for comparison plot).",
    )
    return p


def _ensure_import_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    _ensure_import_path(repo_root)

    from racing_agent.utils.io import discover_runs, find_monitor_csv, group_runs_by_hp
    from racing_agent.utils.plotting import (
        build_timing_table_rows,
        plot_hp_comparison,
        plot_learning_curve,
        write_timing_table_csv,
    )

    experiments_dir = (repo_root / args.experiments_dir).resolve()
    output_dir = (repo_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = discover_runs(experiments_dir)
    grouped = group_runs_by_hp(
        all_runs,
        arch_name=str(args.arch),
        min_timesteps=int(args.min_timesteps),
    )

    hp_order = [hp for hp in args.hp_configs if hp in grouped]
    for hp in sorted(grouped):
        if hp not in hp_order:
            hp_order.append(hp)

    monitor_paths_by_hp: dict[str, list[Path]] = {}
    generated: dict[str, str] = {}

    for hp_name in hp_order:
        metas = grouped.get(hp_name, [])
        csv_paths: list[Path] = []
        for meta in metas:
            run_dir = Path(str(meta["run_dir"]))
            mon = find_monitor_csv(run_dir)
            if mon is not None:
                csv_paths.append(mon)

        if not csv_paths:
            print(f"[warn] no monitor CSVs for {hp_name}; skipping curve.")
            continue

        monitor_paths_by_hp[hp_name] = csv_paths
        out_path = output_dir / f"learning_curve_{hp_name}.png"
        plot_learning_curve(
            csv_paths,
            out_path,
            title=f"{hp_name} — mean ± std ({len(csv_paths)} seeds)",
            smoothing_window=int(args.smoothing_window),
        )
        generated[f"learning_curve_{hp_name}"] = str(out_path)

    if len(monitor_paths_by_hp) >= 2:
        compare_path = output_dir / "learning_curve_compare.png"
        plot_hp_comparison(
            monitor_paths_by_hp,
            compare_path,
            title="Hyperparameter comparison (NatureCNN)",
            smoothing_window=int(args.smoothing_window),
        )
        generated["learning_curve_compare"] = str(compare_path)

    timing_rows = build_timing_table_rows({hp: grouped[hp] for hp in hp_order if hp in grouped})
    timing_path = output_dir / "timing_table.csv"
    write_timing_table_csv(timing_rows, timing_path)
    generated["timing_table"] = str(timing_path)

    summary = {
        "experiments_dir": str(experiments_dir),
        "output_dir": str(output_dir),
        "hp_configs": hp_order,
        "runs_discovered": len(all_runs),
        "generated": generated,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
