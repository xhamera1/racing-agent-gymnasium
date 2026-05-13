# `models/` — curated, hand-promoted checkpoints

The `experiments/` folder contains *every* model from *every* training run.
After Phase 4-6 are complete, the **best models from the project** are copied
here for convenient distribution (this is the only place that may be partially
committed to git so reviewers can `git pull && make evaluate`).

Expected layout after Phase 6 of `PLAN.md`:

```
models/
├── best/best_model.zip          # global best, used by play / record_video / evaluate
├── per_config/
│   ├── hp_baseline.zip
│   ├── hp_high_lr.zip
│   ├── hp_large_batch.zip
│   ├── arch_deep_cnn.zip
│   └── arch_nature_cnn.zip      # == best/best_model.zip if Architecture A wins
└── README.md
```

`*.zip` files are ignored by git by default (see `.gitignore`). Override per
file if you intentionally want to commit a small model checkpoint.
