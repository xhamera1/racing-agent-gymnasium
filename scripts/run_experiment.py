"""Full sweep: ``configs x seeds`` -> N x M trained models.

Kaggle fast profile example
---------------------------
    python scripts/run_experiment.py \\
        --configs hp_baseline hp_high_lr hp_large_batch \\
        --arch arch_light_cnn \\
        --overrides kaggle_overrides \\
        --seeds 0..9 \\
        --timesteps 50000 \\
        --skip-existing \\
        --max-runs 3 \\
        --max-wall-clock-s 28000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SweepJob:
    hp_name: str
    arch_name: str
    seed: int
    timesteps: int
    overrides_stem: str | None = None


def _repo_paths() -> tuple[Path, Path]:
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    train_script = scripts_dir / "train_single.py"
    return repo_root, train_script


def _ensure_import_path() -> None:
    repo_root, _ = _repo_paths()
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_seed_tokens(tokens: Sequence[str]) -> list[int]:
    seeds: set[int] = set()
    for token in tokens:
        text = str(token).strip()
        if ".." in text:
            left, right = text.split("..", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            seeds.update(range(start, end + step, step))
        else:
            seeds.add(int(text))
    return sorted(seeds)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a hyperparameter / architecture experiment grid.")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--arch", default="arch_light_cnn")
    p.add_argument(
        "--overrides",
        default=None,
        help="Optional configs/*.yaml stem (e.g. kaggle_overrides) merged into every run.",
    )
    p.add_argument("--seeds", nargs="+", default=["0..9"])
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--output-root", type=Path, default=Path("experiments"))
    p.add_argument("--configs-dir", type=Path, default=Path("configs"))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after N new runs (Kaggle session safety, e.g. 3 per 9h).",
    )
    p.add_argument(
        "--max-wall-clock-s",
        type=float,
        default=None,
        help="Stop after this many seconds of wall time (e.g. 28000 ≈ 7h45m).",
    )
    return p


def _run_subprocess_job(
    job: SweepJob,
    repo_root: Path,
    train_script: Path,
    output_root: Path,
    configs_dir: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(train_script),
        "--hp",
        str(configs_dir / f"{job.hp_name}.yaml"),
        "--arch",
        str(configs_dir / f"{job.arch_name}.yaml"),
        "--seed",
        str(job.seed),
        "--timesteps",
        str(job.timesteps),
        "--output-root",
        str(output_root),
        "--repo-root",
        str(repo_root),
    ]
    if job.overrides_stem:
        cmd.extend(["--overrides", str(configs_dir / f"{job.overrides_stem}.yaml")])

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "job": job,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _run_inprocess_job(
    job: SweepJob,
    repo_root: Path,
    output_root: Path,
    configs_dir: Path,
) -> dict[str, Any]:
    from racing_agent.training.hyperparams import load_config
    from racing_agent.training.train import Trainer, train_result_summary

    overrides_path = None
    if job.overrides_stem:
        overrides_path = configs_dir / f"{job.overrides_stem}.yaml"

    cfg = load_config(
        configs_dir / f"{job.hp_name}.yaml",
        configs_dir / f"{job.arch_name}.yaml",
        overrides_path=overrides_path,
    )
    started = time.perf_counter()
    result = Trainer(cfg, seed=job.seed, output_root=output_root, repo_root=repo_root).run(job.timesteps)
    elapsed = time.perf_counter() - started
    return {
        "job": job,
        "returncode": 0,
        "elapsed_s": elapsed,
        "summary": train_result_summary(result),
    }


def execute_job(
    job: SweepJob,
    *,
    repo_root: Path,
    train_script: Path,
    output_root: Path,
    configs_dir: Path,
    use_subprocess: bool,
) -> dict[str, Any]:
    if use_subprocess:
        return _run_subprocess_job(job, repo_root, train_script, output_root, configs_dir)
    return _run_inprocess_job(job, repo_root, output_root, configs_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_import_path()

    from racing_agent.utils.io import find_completed_run

    repo_root, train_script = _repo_paths()
    configs_dir = (repo_root / args.configs_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()
    seeds = parse_seed_tokens(args.seeds)
    overrides_stem = str(args.overrides) if args.overrides else None

    for hp_name in args.configs:
        if not (configs_dir / f"{hp_name}.yaml").is_file():
            raise SystemExit(f"Missing HP config: {configs_dir / f'{hp_name}.yaml'}")
    if not (configs_dir / f"{args.arch}.yaml").is_file():
        raise SystemExit(f"Missing architecture config: {configs_dir / f'{args.arch}.yaml'}")
    if overrides_stem and not (configs_dir / f"{overrides_stem}.yaml").is_file():
        raise SystemExit(f"Missing overrides config: {configs_dir / f'{overrides_stem}.yaml'}")

    jobs: list[SweepJob] = []
    skipped: list[dict[str, Any]] = []

    for hp_name in args.configs:
        for seed in seeds:
            if args.skip_existing:
                existing = find_completed_run(
                    output_root,
                    hp_name=hp_name,
                    arch_name=str(args.arch),
                    seed=int(seed),
                    min_timesteps=int(args.timesteps),
                )
                if existing is not None:
                    skipped.append(
                        {
                            "hp_name": hp_name,
                            "arch_name": str(args.arch),
                            "seed": int(seed),
                            "run_dir": str(existing),
                            "status": "skipped_existing",
                        },
                    )
                    continue
            jobs.append(
                SweepJob(
                    hp_name=hp_name,
                    arch_name=str(args.arch),
                    seed=int(seed),
                    timesteps=int(args.timesteps),
                    overrides_stem=overrides_stem,
                ),
            )

    manifest: dict[str, Any] = {
        "configs": list(args.configs),
        "arch": str(args.arch),
        "overrides": overrides_stem,
        "seeds": seeds,
        "timesteps": int(args.timesteps),
        "n_jobs": int(args.n_jobs),
        "max_runs": args.max_runs,
        "max_wall_clock_s": args.max_wall_clock_s,
        "planned_jobs": len(jobs),
        "skipped_existing": skipped,
        "results": [],
        "stopped_early": False,
    }

    if args.dry_run:
        manifest["jobs"] = [
            {
                "hp_name": j.hp_name,
                "arch_name": j.arch_name,
                "seed": j.seed,
                "timesteps": j.timesteps,
                "overrides": j.overrides_stem,
            }
            for j in jobs
        ]
        print(json.dumps(manifest, indent=2))
        return 0

    use_subprocess = int(args.n_jobs) > 1
    started_all = time.perf_counter()
    runs_started = 0

    def _should_stop() -> bool:
        if args.max_runs is not None and runs_started >= int(args.max_runs):
            return True
        if args.max_wall_clock_s is not None and (time.perf_counter() - started_all) >= float(args.max_wall_clock_s):
            return True
        return False

    if int(args.n_jobs) <= 1:
        for job in jobs:
            if _should_stop():
                manifest["stopped_early"] = True
                break
            print(f"[run] {job.hp_name} seed={job.seed} timesteps={job.timesteps}")
            result = execute_job(
                job,
                repo_root=repo_root,
                train_script=train_script,
                output_root=output_root,
                configs_dir=configs_dir,
                use_subprocess=False,
            )
            manifest["results"].append(_serialize_result(result))
            runs_started += 1
            if result["returncode"] != 0:
                print(result.get("stderr", ""), file=sys.stderr)
                break
    else:
        with ProcessPoolExecutor(max_workers=int(args.n_jobs)) as pool:
            batch = jobs
            if args.max_runs is not None:
                batch = batch[: int(args.max_runs)]
            futures = {
                pool.submit(
                    execute_job,
                    job,
                    repo_root=repo_root,
                    train_script=train_script,
                    output_root=output_root,
                    configs_dir=configs_dir,
                    use_subprocess=True,
                ): job
                for job in batch
            }
            for fut in as_completed(futures):
                job = futures[fut]
                result = fut.result()
                manifest["results"].append(_serialize_result(result))
                status = "ok" if result["returncode"] == 0 else "failed"
                print(f"[{status}] {job.hp_name} seed={job.seed} ({result['elapsed_s']:.1f}s)")
            manifest["stopped_early"] = len(batch) < len(jobs)

    manifest["wall_clock_s"] = time.perf_counter() - started_all
    manifest_path = output_root / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    failed = [r for r in manifest["results"] if r.get("returncode", 1) != 0]
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "failed": len(failed),
                "completed": len(manifest["results"]),
                "stopped_early": manifest["stopped_early"],
            },
            indent=2,
        ),
    )
    return 1 if failed else 0


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    job: SweepJob = result["job"]
    payload: dict[str, Any] = {
        "hp_name": job.hp_name,
        "arch_name": job.arch_name,
        "seed": job.seed,
        "timesteps": job.timesteps,
        "overrides": job.overrides_stem,
        "returncode": result.get("returncode", 1),
        "elapsed_s": result.get("elapsed_s"),
        "status": "ok" if result.get("returncode") == 0 else "failed",
    }
    if "summary" in result:
        payload["summary"] = result["summary"]
    if result.get("returncode") != 0 and result.get("stderr"):
        payload["stderr_tail"] = str(result["stderr"])[-2000:]
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
