"""Generate PNG figures embedded in RAPORT.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from racing_agent.policies.custom_cnn import CUSTOM_DEEP_CNN_LAYER_ROWS, custom_deep_cnn_total_params
from racing_agent.policies.light_cnn import LIGHT_CNN_LAYER_ROWS, light_cnn_total_params
from racing_agent.policies.nature_cnn import NATURE_CNN_LAYER_ROWS, nature_cnn_total_params
from racing_agent.utils.plotting import plot_arch_diagram


def main() -> int:
    out = _REPO / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    plot_arch_diagram(
        "NatureCNN (arch_nature_cnn)",
        NATURE_CNN_LAYER_ROWS,
        out / "arch_nature_cnn.png",
        total_params=nature_cnn_total_params(),
    )
    plot_arch_diagram(
        "CustomDeepCNN (arch_deep_cnn)",
        CUSTOM_DEEP_CNN_LAYER_ROWS,
        out / "arch_deep_cnn.png",
        total_params=custom_deep_cnn_total_params(),
    )
    plot_arch_diagram(
        "LightCNN (arch_light_cnn)",
        LIGHT_CNN_LAYER_ROWS,
        out / "arch_light_cnn.png",
        total_params=light_cnn_total_params(),
    )

    s0 = json.loads((out / "eval_summary.json").read_text(encoding="utf-8"))
    s2 = json.loads((out / "eval_summary_seed02.json").read_text(encoding="utf-8"))

    metrics = ["mean_reward", "median_reward", "max_reward"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, [s0[m] for m in metrics], w, label="seed0 @ 300k")
    ax.bar(x + w / 2, [s2[m] for m in metrics], w, label="seed02 @ 100k (demo)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Średnia", "Mediana", "Max"])
    ax.set_ylabel("Nagroda epizodu")
    ax.set_title("Ewaluacja deterministyczna (50 epizodów)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "eval_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s2["rewards"], bins=15, color="steelblue", edgecolor="white")
    ax.axvline(s2["mean_reward"], color="red", linestyle="--", label=f"średnia={s2['mean_reward']:.0f}")
    ax.set_xlabel("Nagroda epizodu")
    ax.set_ylabel("Liczba epizodów")
    ax.set_title("Rozkład nagród — agent demo seed02 (deterministic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eval_histogram_seed02.png", dpi=150)
    plt.close(fig)

    print("OK:", sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
