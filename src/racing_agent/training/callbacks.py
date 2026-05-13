"""Custom callbacks for the SAC training loop.

Two callbacks are added on top of the SB3 defaults:

- :class:`StepTimingCallback` -- records average env-step and env-episode time;
  this is mandatory in the report ("czas potrzebny na wykonanie jednego
  epizodu lub kroku czasowego srodowiska").
- :class:`EvalSaveBestCallback` -- thin wrapper over
  ``stable_baselines3.common.callbacks.EvalCallback`` that pins the
  ``best_model_save_path`` to our ``experiments/<run_id>/models/best`` convention.

Implemented in Phase 3 of ``PLAN.md``.
"""


class StepTimingCallback:  # pragma: no cover - implemented in Phase 3
    """Tracks rolling average step time and per-episode wall-clock."""


class EvalSaveBestCallback:  # pragma: no cover - implemented in Phase 3
    """``EvalCallback`` configured with our experiment-folder convention."""
