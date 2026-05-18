"""Create ``dist/racing-agent-kaggle.zip`` — upload as a Kaggle Dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Package repo subset for Kaggle dataset upload.")
    p.add_argument("--output-dir", type=Path, default=Path("dist"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    out_dir = (repo / args.output_dir).resolve()
    stage = out_dir / "racing-agent-kaggle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for rel in ("src/racing_agent", "configs", "scripts", "requirements-kaggle.txt"):
        src = repo / rel
        dst = stage / rel
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    readme = stage / "KAGGLE_DATASET_README.txt"
    readme.write_text(
        "Upload this folder as a Kaggle Dataset (slug e.g. racing-agent-kaggle).\n"
        "Notebook auto-detects either:\n"
        "  /kaggle/input/<slug>/scripts/run_experiment.py\n"
        "  /kaggle/input/<slug>/racing-agent-kaggle/scripts/run_experiment.py\n",
        encoding="utf-8",
    )

    zip_base = out_dir / "racing-agent-kaggle"
    if zip_base.with_suffix(".zip").exists():
        zip_base.with_suffix(".zip").unlink()
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=str(stage))

    meta = {"stage_dir": str(stage), "zip": archive, "files": [str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()]}
    with (out_dir / "kaggle_bundle_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps({"zip": archive, "stage": str(stage)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
