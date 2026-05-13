# racing-agent-gymnasium

> An autonomous racing agent trained with **Soft Actor-Critic (SAC)** on the
> continuous-control **`CarRacing-v3`** environment (Gymnasium / Box2D).
>
> Academic project for *Inteligencja Obliczeniowa — Projekt 5: Uczenie przez
> wzmacnianie w przestrzeniach ciągłych*. Targeting the full **8 / 8 pts**.
> Deadline: **2026-06-04**.

| Topic       | Details                                                                                   |
| ----------- | ----------------------------------------------------------------------------------------- |
| Environment | `CarRacing-v3` — `Box(0, 255, (96, 96, 3), uint8)` obs, `Box([-1, 0, 0], 1.0, (3,))` act  |
| Algorithm   | SAC (`stable-baselines3`) — off-policy, max-entropy, double-Q, auto-tuned temperature      |
| Policy      | `CnnPolicy` — NatureCNN (baseline) **and** a custom deeper CNN (comparison)               |
| Inspiration | [crossy-road-gymnasium](https://github.com/xhamera1/crossy-road-gymnasium) — Projekt 4, discrete spaces |

---

## Status

The repository is bootstrapped — directory tree, package skeleton, configs,
script stubs, tests and the implementation plan are in place. The phased
implementation itself is tracked in [`PLAN.md`](PLAN.md).

| Phase | Description                                  | State            |
| ----- | -------------------------------------------- | ---------------- |
| 0     | Bootstrap (pyproject, venv, smoke tests)     | ✅ skeleton ready |
| 1     | Env factory + wrappers (84×84, frame-stack)  | ⏳ pending        |
| 2     | Two CNN feature extractors                   | ⏳ pending        |
| 3     | Training pipeline + callbacks                | ⏳ pending        |
| 4     | 3 HP × 10 seeds × ≥50 000 steps (4-pt task)  | ⏳ pending        |
| 5     | Architecture A vs B (6-pt task)              | ⏳ pending        |
| 6     | Best model — deterministic eval (8-pt task)  | ⏳ pending        |
| 7     | Final report → `reports/final_report.pdf`    | ⏳ pending        |

---

## Quick start

> **Python 3.10 – 3.12** required. `stable-baselines3` does not yet support
> Python 3.13 (the same constraint that applied to the previous project).

```bash
# 1. Clone and create venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # PowerShell
# source .venv/Scripts/activate         # Git Bash

# 2. Install package (editable) + notebook + dev extras
pip install -e ".[notebooks,dev]"

# 3. Verify everything imports cleanly
pytest tests/ -k smoke

# 4. Smoke-train (5 000 steps, ~few minutes on CPU)
python scripts/train_single.py \
    --hp configs/hp_baseline.yaml \
    --arch configs/arch_nature_cnn.yaml \
    --seed 0 --timesteps 5000
```

---

## Repository layout

```
racing-agent-gymnasium/
├── PLAN.md                          # full implementation plan (READ ME FIRST)
├── IDEA.md                          # high-level concept
├── README.md                        # this file
├── pyproject.toml                   # editable install metadata
├── requirements.txt                 # pinned dependencies
├── Makefile                         # make {setup, test, train, experiment, plot, report}
├── .gitignore
│
├── src/racing_agent/                # importable Python package
│   ├── env/        {make_env.py, wrappers.py}
│   ├── policies/   {nature_cnn.py, custom_cnn.py}
│   ├── training/   {train.py, callbacks.py, hyperparams.py}
│   ├── evaluation/ {evaluator.py}
│   └── utils/      {seeding.py, plotting.py, io.py}
│
├── configs/
│   ├── hp_baseline.yaml             # 3 hyperparameter sets (Phase 4)
│   ├── hp_high_lr.yaml
│   ├── hp_large_batch.yaml
│   ├── arch_nature_cnn.yaml         # 2 architectures (Phase 5)
│   └── arch_deep_cnn.yaml
│
├── scripts/                         # thin CLIs (argparse)
│   ├── train_single.py
│   ├── run_experiment.py            # 3 configs × 10 seeds sweep
│   ├── evaluate.py                  # deterministic / stochastic rollout
│   ├── plot_curves.py               # mean ± std learning curves
│   └── record_video.py              # demo mp4 of the best agent
│
├── notebooks/                       # 01_env / 02_analysis / 03_final_report
├── experiments/                     # auto-populated training output
├── models/                          # curated best checkpoints
├── logs/                            # aggregate TensorBoard / monitor logs
├── reports/                         # figures + final PDF
├── tests/                           # pytest smoke + per-phase tests
└── docs/                            # CarRacing docs, SAC docs, project PDF
```

---

## Common commands

| Goal                                  | Command                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| Run the test suite                    | `pytest tests/`                                                                     |
| Train one (config, seed)              | `python scripts/train_single.py --hp configs/hp_baseline.yaml --arch configs/arch_nature_cnn.yaml --seed 0 --timesteps 50000` |
| Full HP sweep (Phase 4)               | `python scripts/run_experiment.py --configs hp_baseline hp_high_lr hp_large_batch --seeds 0..9 --timesteps 50000`             |
| Architecture comparison (Phase 5)     | `python scripts/run_experiment.py --configs hp_baseline --arch arch_deep_cnn --seeds 0..9 --timesteps 50000`                  |
| Evaluate the best agent (Phase 6)     | `python scripts/evaluate.py --episodes 50 --deterministic`                          |
| Regenerate every figure               | `python scripts/plot_curves.py`                                                     |
| Record a demo video                   | `python scripts/record_video.py --model models/best/best_model.zip --deterministic` |
| Build the final PDF report            | `make report`                                                                       |
| Open TensorBoard                      | `tensorboard --logdir experiments/`                                                 |

---

## Grading mapping

This project covers the lecturer's full grading rubric. See [`PLAN.md`
§ 1](PLAN.md#1-project-goal--grading-mapping) for the explicit
requirement-to-deliverable mapping for each of the 4 / 6 / 8 point tiers.

## Authors

Inteligencja Obliczeniowa, sem. 6 — Projekt 5 (continuous spaces), 2-person team.
Previous project (Projekt 4, discrete spaces):
[xhamera1/crossy-road-gymnasium](https://github.com/xhamera1/crossy-road-gymnasium).
