"""Zip ``experiments/`` for Kaggle ``/kaggle/working`` download."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create kaggle_outputs.zip from experiments/.")
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    p.add_argument("--output", type=Path, default=Path("kaggle_outputs.zip"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Also write kaggle_export_manifest.json next to the zip.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    exp_dir = (repo_root / args.experiments_dir).resolve()

    if not exp_dir.is_dir():
        raise SystemExit(f"Experiments directory not found: {exp_dir}")

    run_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"No run folders under {exp_dir}")

    out_zip = (repo_root / args.output).resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    archive_base = out_zip.with_suffix("")
    if out_zip.exists():
        out_zip.unlink()
    if archive_base.with_suffix(".zip").exists():
        pass

    created = shutil.make_archive(str(archive_base), "zip", root_dir=str(exp_dir))
    created_path = Path(created)

    meta = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_dir": str(exp_dir),
        "zip_path": str(created_path),
        "n_run_dirs": len(run_dirs),
        "run_ids": [p.name for p in run_dirs],
    }
    manifest_path = args.manifest or out_zip.with_name("kaggle_export_manifest.json")
    with Path(manifest_path).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
