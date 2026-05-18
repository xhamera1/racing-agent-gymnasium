# Kaggle training guide — Phase 4 HP sweep (fast profile)

CarRacing on **full NatureCNN** is too slow for a 9h Kaggle session × 30 runs.
This repo adds a **fast profile**:

| Component | Full (local report) | Kaggle fast |
|-----------|---------------------|-------------|
| Architecture | `arch_nature_cnn` | **`arch_light_cnn`** |
| Obs | 4×84×84 | **2×64×64** |
| Eval during train | every 5k steps | **once at 50k** |
| Checkpoints | every 10k | **disabled** |
| SAC heads | `[256,256]` | **`[128]`** |

**Realistic timing (T4 GPU, 50k steps): ~1–2 h / run** → **3 runs fit in one ~6 h session**.

The assignment still requires **50k timesteps × 10 seeds × 3 HP configs**. Plan **~10 Kaggle sessions** (3 runs each) within your **30 GPU hours/week** budget, using `--skip-existing` to resume.

Phase 5 (Nature vs Custom CNN) can reuse the same Kaggle profile or a shorter local smoke.

---

## 1. Package code (once, on your laptop)

```bash
python scripts/package_kaggle_bundle.py
```

Upload `dist/racing-agent-kaggle.zip` as a **Kaggle Dataset** (e.g. `racing-agent-code`).

---

## 2. Kaggle notebook

Open `notebooks/02_kaggle_hp_sweep.ipynb` on Kaggle:

- **Settings → Accelerator → GPU T4 x1**
- **Add dataset** `racing-agent-code`
- Run all cells

Each session runs up to **3 new runs** then writes:

- `/kaggle/working/experiments/` (run folders)
- `/kaggle/working/kaggle_outputs.zip` ← download this

### Recommended session command (also in notebook)

```bash
python scripts/run_experiment.py \
  --configs hp_baseline hp_high_lr hp_large_batch \
  --arch arch_light_cnn \
  --overrides kaggle_overrides \
  --seeds 0..9 \
  --timesteps 50000 \
  --skip-existing \
  --max-runs 3 \
  --max-wall-clock-s 28000
```

Repeat until `sweep_manifest.json` shows 30 completed runs (or 10 per HP config).

---

## 3. Import outputs locally

Download `kaggle_outputs.zip` from the notebook **Output** tab.

```bash
python scripts/import_kaggle_outputs.py --zip ~/Downloads/kaggle_outputs.zip
python scripts/plot_curves.py --arch arch_light_cnn
```

Figures land in `reports/figures/`. Mention in the report that the HP sweep used the **light profile** for compute; Phase 5 compares full architectures if you run those separately.

---

## 4. File layout after import

```
experiments/
  hp_baseline__arch_light_cnn__seed00__.../
    run_metadata.json
    logs/monitor/monitor_0.monitor.csv
    models/final/final_model.zip
  sweep_manifest.json
  kaggle_import_log.json
```

---

## 5. Troubleshooting

| Issue | Fix |
|-------|-----|
| Box2D import error | Notebook cell runs `pip install "gymnasium[box2d]" swig` |
| Session killed at 9h | Lower `--max-runs` to 2 |
| Duplicate run_id | Normal — each session creates new timestamped folders; `group_runs_by_hp` keeps newest per seed |
| `plot_curves` empty | Pass `--arch arch_light_cnn` |
