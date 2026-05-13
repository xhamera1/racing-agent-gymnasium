# `configs/` — experiment configuration

Every experiment in this project is defined by **two YAML files** which are
deep-merged at runtime by `racing_agent.training.hyperparams.load_config`:

| File           | Purpose                                                 |
| -------------- | ------------------------------------------------------- |
| `hp_*.yaml`    | SAC hyperparameters (learning rate, batch, tau, …)      |
| `arch_*.yaml`  | Feature-extractor architecture (NatureCNN / CustomDeep) |

This separation matches **Phases 4 & 5 of `PLAN.md`**: Phase 4 sweeps over
`hp_*` while keeping `arch_nature_cnn.yaml` fixed; Phase 5 sweeps over
`arch_*` while keeping the best HP set fixed. Adding a new experiment never
requires touching Python code — only a new YAML.

## Available configs

### Hyperparameter sets (Phase 4 — 4-point task)

| File                    | Hypothesis                                       |
| ----------------------- | ------------------------------------------------ |
| `hp_baseline.yaml`      | SB3 SAC defaults (reference)                     |
| `hp_high_lr.yaml`       | Aggressive learning rate / faster Polyak update  |
| `hp_large_batch.yaml`   | Larger batch, smaller buffer, more gradient steps |

### Architectures (Phase 5 — 6-point task)

| File                       | Feature extractor                                       |
| -------------------------- | -------------------------------------------------------- |
| `arch_nature_cnn.yaml`     | SB3 default `NatureCNN` (baseline)                       |
| `arch_deep_cnn.yaml`       | Custom deeper CNN with BatchNorm + adaptive pooling      |

## Schema (combined)

```yaml
# === SAC hyperparameters (from hp_*.yaml) =================================
learning_rate: 3.0e-4
buffer_size: 100000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
train_freq: 1
gradient_steps: 1
ent_coef: "auto"
target_update_interval: 1
use_sde: false

# === Environment / wrappers ==============================================
env:
  n_envs: 1
  grayscale: true
  resize_to: 84
  frame_stack: 4
  clip_reward: false
  continuous: true
  domain_randomize: false

# === Architecture (from arch_*.yaml) =====================================
policy: "CnnPolicy"
features_extractor:
  class: "NatureCNN"          # or "CustomDeepCNN"
  kwargs:
    features_dim: 512
net_arch:
  pi: [256, 256]
  qf: [256, 256]
```
