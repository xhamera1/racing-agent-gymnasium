#!/usr/bin/env python3
"""One-shot Kaggle runner — locates repo from ``__file__``, installs deps, trains, exports zip."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def install_kaggle_deps(requirements_path: Path, *, skip: bool = False) -> None:
    """Install CarRacing / SB3 stack on Kaggle (Box2D needs swig + Box2D wheel)."""

    if skip:
        print("Skipping pip install (--skip-install).")
        return

    if Path("/kaggle").is_dir():
        print("Installing system packages (swig)…")
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "swig"],
            check=False,
        )

    print("Installing Box2D + Python packages…")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "swig",
            "Box2D==2.3.10",
            "pygame>=2.1.0",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],
        check=True,
    )

    import Box2D  # noqa: F401
    import gymnasium  # noqa: F401

    print("Box2D + gymnasium OK.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip/apt (use if you installed deps in a prior notebook cell).",
    )
    args = parser.parse_args(argv)

    code_root = Path(__file__).resolve().parent.parent
    src = code_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    req = code_root / "requirements-kaggle.txt"
    if not req.is_file():
        raise SystemExit(f"requirements-kaggle.txt not found under {code_root}")

    work = Path("/kaggle/working")
    if not work.is_dir():
        work = code_root
    experiments = work / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)

    print("CODE_ROOT:", code_root)
    print("WORK:", work)
    print("EXPERIMENTS:", experiments)

    install_kaggle_deps(req, skip=args.skip_install)

    import torch

    print("CUDA:", torch.cuda.is_available(), end="")
    if torch.cuda.is_available():
        print(" —", torch.cuda.get_device_name(0))
    else:
        print()

    run_cmd = [
        sys.executable,
        str(code_root / "scripts" / "run_experiment.py"),
        "--configs",
        "hp_baseline",
        "--arch",
        "arch_light_cnn",
        "--overrides",
        "kaggle_overrides",
        "--seeds",
        "0",
        "--timesteps",
        "100000",
        "--output-root",
        str(experiments),
        "--skip-existing",
        "--max-runs",
        "1",
        "--max-wall-clock-s",
        "28000",
    ]
    print("Running:", " ".join(run_cmd))
    proc = subprocess.run(run_cmd, cwd=str(code_root))
    if proc.returncode != 0:
        return proc.returncode

    export_cmd = [
        sys.executable,
        str(code_root / "scripts" / "export_kaggle_outputs.py"),
        "--experiments-dir",
        str(experiments),
        "--output",
        str(work / "kaggle_outputs.zip"),
    ]
    subprocess.run(export_cmd, cwd=str(code_root), check=True)
    print("Done. Download:", work / "kaggle_outputs.zip")
    print(json.dumps({"experiments": str(experiments), "zip": str(work / "kaggle_outputs.zip")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
