"""racing_agent — Soft Actor-Critic agent for CarRacing-v3 (Gymnasium).

This package implements an autonomous racing agent trained with the Soft
Actor-Critic (SAC) algorithm from stable-baselines3 on the continuous-control
CarRacing-v3 environment.

Top-level subpackages
---------------------
- :mod:`racing_agent.env`         -- env factory and observation wrappers
- :mod:`racing_agent.policies`    -- CNN feature extractors (NatureCNN, custom)
- :mod:`racing_agent.training`    -- training loop, callbacks, hyperparameter loader
- :mod:`racing_agent.evaluation`  -- deterministic / stochastic rollout evaluation
- :mod:`racing_agent.utils`       -- seeding, plotting, IO helpers

See ``PLAN.md`` in the repository root for the full implementation plan.
"""

__version__ = "0.1.0"
