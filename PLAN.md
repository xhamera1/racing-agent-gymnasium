# PLAN.md — Implementation Plan

**Project:** Autonomous Racing Agent (continuous control, RL)
**Environment:** `CarRacing-v3` (Gymnasium / Box2D)
**Algorithm:** Soft Actor-Critic (SAC, `stable-baselines3`)
**Course:** Inteligencja Obliczeniowa — Projekt 5: *Uczenie przez wzmacnianie w przestrzeniach ciągłych*
**Deadline:** 2026-06-04 (MS Teams)
**Target grade:** **8 / 8 pts**
**Inspiration / previous project:** [crossy-road-gymnasium](https://github.com/xhamera1/crossy-road-gymnasium) (Projekt 4, *przestrzenie dyskretne*, PPO).

---

## 1. Project goal & grading mapping

The lecturer's task list (PDF) defines three tiers. Our plan covers **all** of them so the project secures the maximum score (8 pts).

| Tier   | Requirement (literal)                                                                                                                | Where we deliver it                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **4**  | Continuous Gymnasium env + SB3 algorithm                                                                                             | `CarRacing-v3` + SAC (`CnnPolicy`), see `src/racing_agent/env`, `training`     |
| **4**  | Learning curves for ≥ **3 hyperparameter sets**, each with **10 seeds × ≥ 50 000 timesteps**; mean ± std curves; report training time | Phase 4 — `scripts/run_experiment.py`, `scripts/plot_curves.py`, `reports/`    |
| **4**  | One-paragraph descriptions of env & algorithm, hyperparameter table, learning curves, episode/step time, comparison & justification  | Final report `reports/final_report.pdf` + `notebooks/03_final_report.ipynb`    |
| **6**  | All of the above + ≥ **2 different network architectures** (schemes, layer types/sizes/activations, input/output description)        | Phase 5 — `src/racing_agent/policies/{nature_cnn,custom_cnn}.py` + `arch_*.yaml` |
| **8**  | All of the above + save best agent, run **deterministic** simulation (`deterministic=True`), compare with training curve              | Phase 6 — `scripts/evaluate.py --deterministic`, `notebooks/03_final_report.ipynb` |

---

## 2. Architecture decisions (the "why")

### 2.1 Why `CarRacing-v3`?

* **Truly continuous** action space `Box([-1, 0, 0], [+1, +1, +1])` — steering, gas, brake — *required* by the project brief.
* **Visual observation** `Box(0, 255, (96, 96, 3), uint8)` — a serious test for a CNN feature extractor (covers the 6-pt task on its own merit).
* Stochastic track every episode → tests **generalization**, not memorisation.
* Reward shape `-0.1/frame + 1000/N · tile` is dense enough to make SAC tractable, sparse enough to be interesting.

### 2.2 Why SAC?

* Off-policy → sample-efficient; the visual rollout is expensive, so reuse via the **replay buffer** is critical.
* Maximum-entropy objective → built-in exploration in a continuous action space without hand-crafting Gaussian noise.
* Stable-baselines3 ships a battle-tested `CnnPolicy` (NatureCNN feature extractor) + automatic temperature tuning (`ent_coef="auto"`).

### 2.3 What we improve over the previous project (crossy-road-gymnasium)

| Aspect          | Crossy-Road (Projekt 4, discrete) | Racing-Agent (Projekt 5, continuous)                           |
| --------------- | --------------------------------- | -------------------------------------------------------------- |
| Action space    | `Discrete(4)`                     | `Box(3,)` — steering, gas, brake                               |
| Observation     | small structured dict (~160)      | RGB **image** `(96, 96, 3)` → frame-stack `(4, 84, 84)`        |
| Algorithm       | PPO (on-policy)                   | **SAC** (off-policy, max-entropy)                              |
| Policy          | `MultiInputPolicy [256, 256]`     | `CnnPolicy` — NatureCNN + custom deeper CNN                    |
| Experiment grid | implicit / one big run            | **explicit grid:** 3 HP sets × 10 seeds + 2 architectures      |
| Reporting       | `trening.ipynb` (delivery)        | Same delivery format: `notebooks/03_final_report.ipynb` + PDF  |

---

## 3. Project structure

```
racing-agent-gymnasium/
├── PLAN.md                          # this file
├── IDEA.md                          # high-level concept
├── README.md                        # quick start, badges, headline results
├── pyproject.toml                   # package metadata + editable install
├── requirements.txt                 # pinned dependencies
├── Makefile                         # one-liners: make train / eval / plot
├── .gitignore
│
├── src/racing_agent/                # importable package: `import racing_agent`
│   ├── env/
│   │   ├── make_env.py              # gym.make + wrappers + VecEnv factory
│   │   └── wrappers.py              # GrayScale/Resize/FrameStack/RewardClip
│   ├── policies/
│   │   ├── nature_cnn.py            # Architecture A — SB3 default (baseline)
│   │   └── custom_cnn.py            # Architecture B — deeper CNN (our design)
│   ├── training/
│   │   ├── train.py                 # high-level Trainer wrapping SAC.learn()
│   │   ├── callbacks.py             # eval / checkpoint / metrics callbacks
│   │   └── hyperparams.py           # YAML → SAC kwargs loader
│   ├── evaluation/
│   │   └── evaluator.py             # N-episode rollout (det. & stoch.)
│   └── utils/
│       ├── seeding.py               # reproducible seeds (np / torch / gym)
│       ├── plotting.py              # mean ± std curves, hp-sets comparison
│       └── io.py                    # paths, run-id generator, JSON dump
│
├── configs/
│   ├── hp_baseline.yaml             # HP set #1 — SB3 defaults
│   ├── hp_high_lr.yaml              # HP set #2 — aggressive learning rate
│   ├── hp_large_batch.yaml          # HP set #3 — larger batch / buffer
│   ├── arch_nature_cnn.yaml         # Architecture A
│   └── arch_deep_cnn.yaml           # Architecture B
│
├── scripts/                         # thin CLI wrappers (argparse)
│   ├── train_single.py              # one (config, seed) → one run
│   ├── run_experiment.py            # full sweep: configs × seeds
│   ├── evaluate.py                  # deterministic / stochastic eval
│   ├── plot_curves.py               # learning curves with mean ± std
│   └── record_video.py              # mp4 of the trained agent racing
│
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_final_report.ipynb        # ★ MAIN DELIVERABLE ★
│
├── experiments/                     # auto-populated: results/<run_id>/...
├── models/                          # best/, checkpoints/, final/
├── logs/                            # tensorboard/, monitor/*.csv
├── reports/
│   ├── figures/                     # all PNG/SVG generated for the report
│   └── final_report.pdf             # written deliverable
│
├── tests/                           # pytest — smoke + unit tests
└── docs/                            # reference docs (CarRacing, SAC, PDF)
```

---

## 4. Implementation phases

Each phase has **scope**, **deliverables** and **exit criteria**. Phases are designed so PLAN.md doubles as a checklist during execution.

> **Legend:** `✅ DONE` — fully implemented and exit criteria met · `🔄 IN PROGRESS` — work started · `⏳ PENDING` — not started yet

| Phase | Name                                    | Status       |
| ----- | --------------------------------------- | ------------ |
| 0     | Project bootstrap                       | ✅ DONE       |
| 1     | Environment & wrappers                  | ⏳ PENDING    |
| 2     | Custom CNN feature extractor            | ⏳ PENDING    |
| 3     | Training pipeline & callbacks           | ⏳ PENDING    |
| 4     | Hyperparameter sweep (4-pt task)        | ⏳ PENDING    |
| 5     | Architecture comparison (6-pt task)     | ⏳ PENDING    |
| 6     | Best agent — deterministic eval (8-pt)  | ⏳ PENDING    |
| 7     | Final report & notebook delivery        | ⏳ PENDING    |

---

### Phase 0 — Project bootstrap `✅ DONE` (Day 1)

**Scope:** repo skeleton, dependency management, reproducibility primitives.

* Create `pyproject.toml` (editable install, Python ≥ 3.10, < 3.13 — SB3 limit).
* `requirements.txt` pinned: `gymnasium[box2d]`, `stable-baselines3[extra]`, `torch`, `numpy`, `pandas`, `matplotlib`, `tensorboard`, `pyyaml`, `moviepy` (for `record_video.py`), `pytest`, `jupyter`.
* Set up `.gitignore` (models/, logs/, experiments/, `__pycache__`, `.venv`, `*.zip`, `*.mp4`).
* `Makefile` targets: `setup`, `test`, `train`, `experiment`, `evaluate`, `plot`, `report`.
* Set up venv (`py -3.12 -m venv .venv`) and validate `python -c "from stable_baselines3 import SAC; import gymnasium as gym; gym.make('CarRacing-v3')"`.

**Exit criteria:** `pip install -e .` succeeds; `pytest tests/ -k smoke` passes; SB3 + Box2D imports cleanly on Windows.

**Completed deliverables:**
- [x] `pyproject.toml` (editable install, Python 3.10–3.12)
- [x] `requirements.txt` with pinned dependencies
- [x] `.gitignore`
- [x] `Makefile` with all targets
- [x] Directory tree: `src/`, `configs/`, `scripts/`, `notebooks/`, `experiments/`, `models/`, `logs/`, `reports/`, `tests/`, `docs/`
- [x] Module skeletons with docstrings: `env/`, `policies/`, `training/`, `evaluation/`, `utils/`
- [x] 5 × HP + architecture YAML configs
- [x] 5 × CLI script stubs
- [x] Smoke tests (`tests/test_smoke.py`) — green without installing anything beyond the package
- [x] Reference docs moved to `docs/`

---

### Phase 1 — Environment & wrappers `⏳ PENDING` (Day 2–3)

**Scope:** wrap `CarRacing-v3` for stable visual RL.

* `env/make_env.py::make_car_racing(...)` — factory accepting `n_envs`, `seed`, `monitor_dir`, `wrapper_config`. Returns `VecEnv` (`SubprocVecEnv` if `n_envs > 1`, else `DummyVecEnv`).
* `env/wrappers.py`:
  * `GrayScaleObservation` (96×96×3 → 96×96×1) — colour is not informative on the track.
  * `ResizeObservation` (96×96 → 84×84) — Nature CNN convention.
  * `FrameStack(k=4)` — recovers velocity / yaw rate from pixels.
  * `ClipReward` (optional) — clip to `[-1, 1]` for value-target stability.
* Register helper `make_car_racing(continuous=True)` (the brief explicitly requires continuous).
* Sanity check: random policy ≥ 1 episode, env passes `gymnasium.utils.env_checker`.

**Exit criteria:** `notebooks/01_environment_exploration.ipynb` shows: (1) action/obs spec, (2) a few frames after wrappers, (3) random-agent reward distribution over 20 episodes, (4) measured **avg step time** and **avg episode time** (required by the report).

**Completed deliverables:**
- [ ] `env/make_env.py` — `make_car_racing()` factory (DummyVecEnv / SubprocVecEnv)
- [ ] `env/wrappers.py` — GrayScaleObservation, ResizeObservation, FrameStack, ClipReward
- [ ] `tests/test_env.py` — unskipped and passing
- [ ] `notebooks/01_environment_exploration.ipynb` — frames, timing, action/obs spec

---

### Phase 2 — Custom CNN feature extractor `⏳ PENDING` (Day 3)

**Scope:** define both feature extractors *before* the heavy experiments so a single training pipeline can swap them via config.

* `policies/nature_cnn.py` — wraps SB3's `NatureCNN`. **Architecture A** (baseline):
  * `Conv(32, 8×8, stride 4)` → ReLU
  * `Conv(64, 4×4, stride 2)` → ReLU
  * `Conv(64, 3×3, stride 1)` → ReLU
  * `Flatten → Linear(512)` → ReLU
* `policies/custom_cnn.py` — **Architecture B** (deeper, BN + residual-ish):
  * `Conv(32, 3×3, stride 2)` → BN → ReLU
  * `Conv(64, 3×3, stride 2)` → BN → ReLU
  * `Conv(128, 3×3, stride 2)` → BN → ReLU
  * `Conv(128, 3×3, stride 1)` → BN → ReLU
  * `AdaptiveAvgPool2d(1) → Flatten → Linear(256)` → ReLU
* Document both in tabular form (layer / kernel / stride / out-shape / params) for the report; the table generator lives in `utils/plotting.py::plot_arch_diagram`.

**Exit criteria:** `tests/test_policies.py` instantiates both, forward-passes a `(B, 4, 84, 84)` tensor, asserts output shape.

**Completed deliverables:**
- [ ] `policies/nature_cnn.py` — `NatureCNN` re-export / thin wrapper
- [ ] `policies/custom_cnn.py` — `CustomDeepCNN` (BN + adaptive avg pool)
- [ ] `tests/test_policies.py` — unskipped and passing
- [ ] Architecture layer tables documented (for report)

---

### Phase 3 — Training pipeline & callbacks `⏳ PENDING` (Day 4)

**Scope:** one trainer, two configs (hp_*, arch_*), full reproducibility.

* `training/hyperparams.py` — `load_config(path) -> dict`, deep-merge of `hp_*` and `arch_*` YAMLs.
* `training/train.py::Trainer`:
  * Builds env, model, callbacks; runs `model.learn(total_timesteps)`.
  * Persists: final model, **best model** (`EvalCallback`), checkpoints (`CheckpointCallback` every 10k steps), `Monitor` CSVs, TensorBoard logs.
  * Outputs a `run_metadata.json` with: `run_id`, `config`, `seed`, `total_timesteps`, `wall_clock`, `mean_step_time`, `git_hash`.
* `training/callbacks.py`:
  * `StepTimingCallback` — measures avg step/episode time (required by report).
  * `EvalSaveBestCallback` — wraps SB3 `EvalCallback` with our run-dir convention.
* CLI: `scripts/train_single.py --hp configs/hp_baseline.yaml --arch configs/arch_nature_cnn.yaml --seed 0 --timesteps 50000`.

**Exit criteria:** a 5 000-step smoke run finishes < 5 min on CPU, all output files materialise in `experiments/<run_id>/`.

**Completed deliverables:**
- [ ] `training/hyperparams.py` — `load_config()` deep-merging HP + arch YAML
- [ ] `training/train.py::Trainer` — builds env + SAC + callbacks, calls `model.learn()`
- [ ] `training/callbacks.py` — `StepTimingCallback`, `EvalSaveBestCallback`
- [ ] `scripts/train_single.py` — fully wired (not stub)
- [ ] `experiments/<run_id>/run_metadata.json` produced on smoke run
- [ ] `make smoke` completes < 5 min

---

### Phase 4 — Hyperparameter sweep — **the 4-point deliverable** `⏳ PENDING` (Day 5–9)

**Scope:** **3 HP sets × 10 seeds × ≥ 50 000 timesteps = 30 runs**. This phase is the bulk of compute.

| Config             | What changes vs baseline                                    | Hypothesis                          |
| ------------------ | ----------------------------------------------------------- | ----------------------------------- |
| `hp_baseline`      | SB3 SAC defaults (`lr=3e-4`, `batch=256`, `buffer=10⁶`, `tau=0.005`) | reference                           |
| `hp_high_lr`       | `lr=1e-3`, `tau=0.02`                                       | faster but possibly less stable     |
| `hp_large_batch`   | `batch=512`, `buffer=3·10⁵`, `train_freq=4`, `gradient_steps=4` | smoother updates, slower wall-clock |

All three keep `arch_nature_cnn` fixed (we isolate HP effects from architecture effects).

* `scripts/run_experiment.py --configs hp_baseline hp_high_lr hp_large_batch --seeds 0..9 --timesteps 50000` — sequential or parallel (`--n-jobs`).
* `scripts/plot_curves.py` — reads all `Monitor` CSVs, computes **mean ± std across 10 seeds per HP set**, draws one figure per set + a comparison figure.
* Compute **per-step / per-episode time** averaged across all 30 runs (needed in report).

**Exit criteria:**
* `reports/figures/learning_curve_<hp>.png` × 3 (mean ± std band).
* `reports/figures/learning_curve_compare.png` (3 curves on one plot).
* `reports/figures/timing_table.csv` — wall-clock & step/episode time per config.

**Completed deliverables:**
- [ ] 30 training runs completed (3 configs × 10 seeds × ≥ 50 000 steps)
- [ ] `scripts/run_experiment.py` — fully wired (not stub)
- [ ] `scripts/plot_curves.py` — fully wired
- [ ] `reports/figures/learning_curve_hp_baseline.png`
- [ ] `reports/figures/learning_curve_hp_high_lr.png`
- [ ] `reports/figures/learning_curve_hp_large_batch.png`
- [ ] `reports/figures/learning_curve_compare.png`
- [ ] `reports/figures/timing_table.csv`

---

### Phase 5 — Architecture comparison — **the 6-point deliverable** `⏳ PENDING` (Day 10–12)

**Scope:** repeat the best HP set with **Architecture B** to get a fair A-vs-B comparison.

* **10 seeds × 50 000 timesteps** for `arch_deep_cnn` (matches statistical rigor of Phase 4).
* Architecture A results are reused from Phase 4 (`hp_baseline`).
* Render network schematics: `utils/plotting.py::plot_arch_diagram(arch_config) -> matplotlib fig`. Block diagram showing layer → shape → activation. Saved as `reports/figures/arch_A.png` and `arch_B.png`.
* In the report: per-architecture parameter count, learning curve A vs B (mean ± std), commentary on bias/variance, training speed.

**Exit criteria:** `reports/figures/arch_compare.png` + populated *Architectures* section in `notebooks/03_final_report.ipynb`.

**Completed deliverables:**
- [ ] 10 runs with `arch_deep_cnn` × best HP set (≥ 50 000 steps each)
- [ ] `reports/figures/arch_A.png` — NatureCNN block diagram
- [ ] `reports/figures/arch_B.png` — CustomDeepCNN block diagram
- [ ] `reports/figures/arch_compare.png` — A vs B mean ± std curves
- [ ] Parameter counts documented in notebook

---

### Phase 6 — Best agent & deterministic evaluation — **the 8-point deliverable** `⏳ PENDING` (Day 13–14)

**Scope:** select the best saved checkpoint across all runs, evaluate **with `deterministic=True`** and compare with the training curve.

* `scripts/evaluate.py --run-id <best_run> --episodes 50 --deterministic` and `--no-deterministic` (sanity comparison).
* Reports: mean / median / std / min / max reward, episode length distribution, success rate (lap completion if `lap_complete_percent` reached).
* Plot **deterministic eval reward** as a horizontal band on the training curve to show stoch-vs-det gap.
* `scripts/record_video.py --run-id <best_run> --episodes 3 --deterministic` → `reports/figures/agent_demo.mp4` (or `.gif` if size permits).

**Exit criteria:** the 8-pt task box ticked: best model identified, deterministic numbers in report, video saved, comparison with training curve visualised.

**Completed deliverables:**
- [ ] `scripts/evaluate.py` — fully wired (deterministic + stochastic modes)
- [ ] `scripts/record_video.py` — fully wired
- [ ] `models/best/best_model.zip` committed / documented
- [ ] `reports/figures/eval_summary.json` — mean/median/std/min/max
- [ ] `reports/figures/agent_demo.mp4` (or `.gif`)
- [ ] Det. eval band plotted on top of training curve

---

### Phase 7 — Final report & notebook delivery `⏳ PENDING` (Day 15–16)

**Scope:** consolidate everything into a polished, single-source-of-truth deliverable.

* `notebooks/03_final_report.ipynb` — sections mirror PDF requirements:
  1. **Environment** — one paragraph, observation/action/reward, avg step & episode time.
  2. **Algorithm** — one paragraph on SAC + max-entropy + double-Q + auto temperature.
  3. **Hyperparameters** — table of all 3 configs with values.
  4. **Learning curves** — 3 figures (one per HP set, mean ± std) + comparison.
  5. **Architectures** — diagrams of A & B, layer-by-layer table, parameter counts, learning curves A vs B.
  6. **Deterministic evaluation** — table, histogram, comparison band on training curve.
  7. **Discussion & conclusions** — which HP / architecture is the winner *and why*.
* Export to `reports/final_report.pdf` (`jupyter nbconvert --to pdf`).
* Update `README.md` with headline numbers, demo gif, citation of the PDF brief.

**Exit criteria:** PDF compiles, links in README work, notebook re-runs end-to-end with `Run All` on a clean kernel.

**Completed deliverables:**
- [ ] `notebooks/03_final_report.ipynb` — all 7 sections complete
- [ ] `reports/final_report.pdf` — renders without errors
- [ ] `README.md` — headline numbers, demo gif/video link, quick-start verified
- [ ] Submission on MS Teams before 2026-06-04

---

## 5. Compute & timeline budget

CarRacing-v3 with `CnnPolicy` SAC is **expensive**. Realistic numbers (Win11, RTX 3060 / Ryzen 5 5600, 1 env):

* `~ 2 500 – 3 500 steps / s` on GPU during SAC's update phase (mostly gradient-bound).
* `50 000` timesteps → **~15–25 min wall-clock** per run.
* **30 runs (Phase 4) → ~ 7.5 – 12 h** of compute.
* **+10 runs (Phase 5) → ~ 2.5 – 4 h**.
* **+ misc** (eval, video, retries) → ~ 2 h.
* **Total compute budget: ~ 12 – 18 h** — fits comfortably within the 3-week window before the 2026-06-04 deadline.

If GPU is unavailable, we drop to ~600 steps/s CPU → ~ 90 min / run → would force `n_envs=4` SubprocVecEnv parallelism (already supported by `make_car_racing`).

### Calendar (target: deliver 2 days before deadline)

| Day(s) | Phase | Output |
| ------ | ----- | ------ |
| 1      | 0     | repo skeleton, venv, smoke import |
| 2–3    | 1     | wrappers, env notebook |
| 3      | 2     | both CNN extractors + unit tests |
| 4      | 3     | trainer, callbacks, smoke run |
| 5–9    | 4     | 30 runs, learning curves, timing |
| 10–12  | 5     | 10 runs Architecture B, arch_compare |
| 13–14  | 6     | best-model eval, video, det-vs-stoch |
| 15–16  | 7     | notebook → PDF → README polish |
| +1–2   | —     | buffer / re-runs / formatting     |

---

## 6. Reproducibility & engineering hygiene

* **One seed → one deterministic run.** `utils/seeding.set_global_seed(seed)` seeds `random`, `numpy`, `torch`, `torch.cuda`, env action space, SAC. Seeds for the 10-run grid are fixed `[0..9]`.
* **`run_metadata.json` per experiment** — config snapshot + git hash → any plot in the report can be traced back to the run that produced it.
* **All hyperparameters live in YAML**, never as magic numbers in code. Adding a 4th HP set = adding a YAML file, no code change.
* **`tests/`** — `pytest` smoke tests for env wrappers, policy forward pass, trainer instantiation (no learning), and YAML→SAC mapping.
* **`Makefile`** — `make experiment` runs the full grid; `make report` rebuilds the PDF. CI-friendly.

---

## 7. Risks & mitigations

| Risk                                                          | Mitigation                                                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **SAC + visual env diverges** (notoriously sensitive)         | Use SB3 defaults as `hp_baseline`; clip rewards; FrameStack; gradient clipping if needed.   |
| **Box2D wheel issues on Windows** (`swig` not found)          | Pin `box2d-py` wheel; documented `pip install gymnasium[box2d]` recipe; CI test on Win.     |
| **Compute over-budget** (10 seeds × 50k × 3 configs is heavy) | Phase 0 measures step/s on the actual machine; can fall back to `SubprocVecEnv(n_envs=4)`.  |
| **`stable-baselines3` Python-version drift**                  | Pinned `python>=3.10,<3.13`, pinned SB3 version in `requirements.txt`.                      |
| **Notebook → PDF export fails on Windows** (`nbconvert`+TeX)  | Backup path: `nbconvert --to html` + Chrome "Save as PDF"; or assemble final PDF in LaTeX.  |
| **TensorBoard runs lost** if scratch disk fills               | `experiments/` excluded from git; weekly cleanup, only `best/` and `final/` models kept.    |

---

## 8. Acceptance checklist (use at submission time)

* [ ] `pip install -e . && pytest tests/` — green on a fresh clone.
* [ ] `scripts/run_experiment.py` reproduces every figure under `reports/figures/`.
* [ ] **4-pt task:** 3 HP sets × 10 seeds × ≥ 50 000 timesteps — mean ± std learning curves saved.
* [ ] **4-pt task:** report contains env paragraph, algorithm paragraph, HP table, learning curves, avg step/episode time, justified comparison.
* [ ] **6-pt task:** 2 architectures with diagrams (layers, sizes, activations, I/O), learning curves A vs B.
* [ ] **8-pt task:** best agent saved (`models/best/best_model.zip`), deterministic eval table, comparison vs training curve, demo video.
* [ ] `reports/final_report.pdf` exists and renders correctly.
* [ ] `README.md` shows headline numbers, quick-start, demo gif.
* [ ] Submitted on MS Teams before **2026-06-04**.

