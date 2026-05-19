# racing-agent-gymnasium

SAC agent on **CarRacing-v3** (Gymnasium / Box2D). 


|          |                                                             |
| -------- | ----------------------------------------------------------- |
| **Env**  | `CarRacing-v3` — visual obs, continuous steer / gas / brake |
| **Algo** | SAC + `CnnPolicy` (`stable-baselines3`)                     |
| **Plan** | `[PLAN.md](PLAN.md)`                                        |


---

## What we have


| Done              | Notes                                                        |
| ----------------- | ------------------------------------------------------------ |
| Env + wrappers    | grayscale, resize, frame-stack → `src/racing_agent/env/`     |
| CNN policies      | NatureCNN, CustomDeepCNN, **LightCNN** (Kaggle fast profile) |
| Training pipeline | `Trainer`, callbacks, YAML configs, `experiments/<run_id>/`  |
| Kaggle workflow   | `notebooks/02_kaggle_hp_sweep.ipynb`, import/export zip      |
| Figures           | `scripts/plot_curves.py` → `reports/figures/`                |
| Live preview      | `scripts/watch_agent.py` — pygame window, no video file      |
| Eval stats        | `scripts/evaluate.py` → `reports/figures/eval_summary.json`  |


**Trained so far (imported from Kaggle):** `arch_light_cnn` + `kaggle_overrides` (64×64, stack=2). Best run: `hp_baseline` seed0 @ **300k** steps (Monitor peak ~**863**). Agent drives on straights; often stops on sharp turns until episode timeout (1000 steps).

**Still TODO:** HP sweep (3 configs × more seeds @ 50k), architecture comparison, final report notebook/PDF.

---

## Versions

- **Python:** 3.10 – 3.12 (use **3.12** in `.venv` — system 3.14 lacks Box2D)
- **Install:** `pip install -e ".[dev]"` from repo root
- **Box2D:** via `gymnasium[box2d]` (Windows: run inside activated `.venv`)

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/Scripts/activate       # Git Bash
pip install -e ".[dev]"
```

Always check: `python -c "import sys; print(sys.executable)"` → path must include `.venv`.

---

## Run the game (watch agent)

After importing Kaggle outputs into `experiments/`:

```bash
# Auto-pick best checkpoint (highest Monitor peak)
python scripts/watch_agent.py --arch arch_light_cnn --loop

# Specific run (recommended after import)
python scripts/watch_agent.py --run-dir experiments/hp_baseline__arch_light_cnn__seed00__20260518-214604 --fast --episodes 5

# Headless stats (report)
python scripts/evaluate.py --run-dir experiments/hp_baseline__arch_light_cnn__seed00__20260518-214604 --episodes 50 --deterministic
```

Reward lines print **after each episode** (~1000 steps). Use `--fast` to skip real-time pacing. Close pygame window or Ctrl+C to stop.

**Import from Kaggle zip:**

```bash
python scripts/import_kaggle_outputs.py --zip ~/Downloads/kaggle_outputs.zip
python scripts/plot_curves.py --arch arch_light_cnn --min-timesteps 50000
```

---

## Kaggle training

1. GPU T4, attach code dataset, open `notebooks/02_kaggle_hp_sweep.ipynb` 
2. Session 1 done: `MODE = "long_run"` (300k baseline seed0)
3. **Next sessions:** `MODE = "sweep"` — 3 HP configs, 50k steps, `--max-runs 3` per session `[In progress - kaggle notebook running]`
4. Download `kaggle_outputs.zip` → import locally (above)

See `[kaggle/README.md](kaggle/README.md)` for details.

---

## Next steps

1. **Kaggle sweep** — `MODE = "sweep"`, repeat until you have runs for `hp_baseline`, `hp_high_lr`, `hp_large_batch` (multiple seeds @ 50k) `[In progress - kaggle notebook running]`
2. **Plot + eval** after each import — curves, `eval_summary.json`, spot-check with `watch_agent`
3. **Phase 5** — architecture diagrams + optional `arch_deep_cnn` runs
4. **Phase 7** — fill `notebooks/03_final_report.ipynb`, export PDF

---

## Useful commands


| Goal             | Command                                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Tests            | `pytest tests/`                                                                                                                                                                      |
| Single train     | `python scripts/train_single.py --hp configs/hp_baseline.yaml --arch configs/arch_light_cnn.yaml --overrides configs/kaggle_overrides.yaml --seed 0 --timesteps 50000`               |
| HP sweep (local) | `python scripts/run_experiment.py --configs hp_baseline hp_high_lr hp_large_batch --arch arch_light_cnn --overrides kaggle_overrides --seeds 0..9 --timesteps 50000 --skip-existing` |
| Watch agent      | `make watch` or `python scripts/watch_agent.py --arch arch_light_cnn --loop`                                                                                                         |
| Plot curves      | `python scripts/plot_curves.py --arch arch_light_cnn`                                                                                                                                |


