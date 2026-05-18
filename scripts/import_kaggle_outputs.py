"""Merge a Kaggle ``kaggle_outputs.zip`` into the local ``experiments/`` folder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import Kaggle experiment zip into local experiments/.")
    p.add_argument("--zip", required=True, type=Path, help="Path to kaggle_outputs.zip")
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    p.add_argument(
        "--merge",
        choices=("skip", "replace"),
        default="skip",
        help="When a run_id already exists locally: skip or replace.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    src = Path(args.zip).expanduser().resolve()
    dest_root = (repo_root / args.experiments_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        raise SystemExit(f"Zip not found: {src}")

    tmp = dest_root / "_kaggle_import_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(tmp)

    # Zip may contain run folders directly or nested under experiments/.
    if (tmp / "experiments").is_dir():
        source_runs = tmp / "experiments"
    else:
        source_runs = tmp

    imported: list[str] = []
    skipped: list[str] = []
    replaced: list[str] = []

    for run_dir in sorted(p for p in source_runs.iterdir() if p.is_dir()):
        target = dest_root / run_dir.name
        if target.exists():
            if args.merge == "skip":
                skipped.append(run_dir.name)
                continue
            shutil.rmtree(target)
            replaced.append(run_dir.name)
        shutil.copytree(run_dir, target)
        imported.append(run_dir.name)

    shutil.rmtree(tmp, ignore_errors=True)

    summary = {
        "zip": str(src),
        "experiments_dir": str(dest_root),
        "imported": imported,
        "skipped_existing": skipped,
        "replaced": replaced,
    }
    with (dest_root / "kaggle_import_log.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nNext: python scripts/plot_curves.py", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
