"""Training pipeline -- thin layer on top of ``stable_baselines3.SAC``.

This subpackage wires together:

- environment construction (:mod:`racing_agent.env`)
- feature extractor selection (:mod:`racing_agent.policies`)
- hyperparameter loading from YAML (:mod:`racing_agent.training.hyperparams`)
- model fitting, evaluation and checkpointing (:mod:`racing_agent.training.train`)

See Phase 3 of ``PLAN.md`` for details.
"""

from racing_agent.training.hyperparams import load_config
from racing_agent.training.train import TrainResult, Trainer, train_result_summary

__all__ = ["Trainer", "TrainResult", "load_config", "train_result_summary"]
