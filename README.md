# racing-agent-gymnasium

SAC agent on **CarRacing-v3** (Gymnasium / Box2D). Course project — target **8/8 pts**, deadline **2026-06-04**.

| | |
|---|---|
| **Env** | `CarRacing-v3` — visual obs, continuous steer / gas / brake |
| **Algo** | SAC + `CnnPolicy` (`stable-baselines3`) |
| **Plan** | [`PLAN.md`](PLAN.md) |

---

## What we have

| Done | Notes |
|------|--------|
| Env + wrappers | grayscale, resize, frame-stack → `src/racing_agent/env/` |
| CNN policies | NatureCNN, CustomDeepCNN, **LightCNN** (Kaggle profile) |
| Training pipeline | `Trainer`, callbacks, YAML configs, `experiments/<run_id>/` |
| Kaggle workflow | `notebooks/02_kaggle_hp_sweep.ipynb`, import/export zip |
| Figures | `scripts/plot_curves.py` → `reports/figures/` |
| Live preview | `scripts/watch_agent.py` — pygame window |
| Eval stats | `scripts/evaluate.py` → `reports/figures/eval_summary.json` |

**Current checkpoints (imported, `arch_light_cnn`, 64×64, stack=2):**

| Role | Run folder | Training peak | Live behaviour |
|------|------------|---------------|----------------|
| **Best demo agent** | `hp_baseline__arch_light_cnn__seed02__20260519-074316` | ~774 | Drives better overall; may overshoot sharp turns |
| **Second (long run)** | `hp_baseline__arch_light_cnn__seed00__20260518-214604` | ~863 | Strong on straights; often **stops** on sharp turns |

Use **seed02** for demo / report. Keep seed0 @ 300k for comparison (deterministic eval, training peak).

**Paused for now:** no new training until report needs more HP curves. Existing models are enough for preview + eval.

---

## Versions

- **Python:** 3.10 – 3.12 (**use 3.12** in `.venv`; system 3.14 has no Box2D)
- **Install:** `pip install -e ".[dev]"` from repo root

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/Scripts/activate       # Git Bash
pip install -e ".[dev]"
python -c "import sys; print(sys.executable)"   # must show .venv
```

---

## Run the game — saved models

Always run from repo root with **activated `.venv`**.

### Best agent (seed02 @ 100k) — recommended

```bash
python scripts/watch_agent.py \
  --run-dir experiments/hp_baseline__arch_light_cnn__seed02__20260519-074316 \
  --fast --episodes 5
```

Loop until Ctrl+C:

```bash
python scripts/watch_agent.py \
  --run-dir experiments/hp_baseline__arch_light_cnn__seed02__20260519-074316 \
  --loop
```

Headless stats (report):

```bash
python scripts/evaluate.py \
  --run-dir experiments/hp_baseline__arch_light_cnn__seed02__20260519-074316 \
  --episodes 50 --deterministic
```

### Second agent (seed0 @ 300k) — comparison / peak training reward

```bash
python scripts/watch_agent.py \
  --run-dir experiments/hp_baseline__arch_light_cnn__seed00__20260518-214604 \
  --fast --episodes 5
```

```bash
python scripts/evaluate.py \
  --run-dir experiments/hp_baseline__arch_light_cnn__seed00__20260518-214604 \
  --episodes 50 --deterministic
```

**Tips:** Reward lines appear **after each episode** (~1000 steps). `--fast` skips real-time delay. Close pygame or Ctrl+C to stop.

**Do not use** `--arch arch_light_cnn` alone — auto-pick favours seed0 @ 300k (highest training peak), not seed02 (better driving).

---

## Import more Kaggle runs (optional)

```bash
python scripts/import_kaggle_outputs.py --zip ~/Downloads/kaggle_outputs.zip
python scripts/plot_curves.py --arch arch_light_cnn --min-timesteps 100000
```

---

## Useful commands

| Goal | Command |
|------|---------|
| Tests | `pytest tests/` |
| Plot curves | `python scripts/plot_curves.py --arch arch_light_cnn --min-timesteps 100000` |
| Watch best (seed02) | see [Best agent](#best-agent-seed02--100k--recommended) above |
| Watch second (seed0 300k) | see [Second agent](#second-agent-seed0--300k--comparison--peak-training-reward) above |

---

## Next steps (report)

1. `evaluate.py` on both runs above (det. + optional `--stochastic`) → `eval_summary.json`
2. Paste learning curve + timing table from `reports/figures/`
3. `notebooks/03_final_report.ipynb` — describe seed02 demo + seed0 comparison (stop vs fly-off on turns)
4. Resume Kaggle sweep later if more HP data is needed
