"""Reporting plots: learning curves with mean +/- std, HP comparisons, arch diagrams.

Phase 4 implements learning-curve aggregation for the hyperparameter sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_ARCH_TABLE_COLS = ("layer", "kernel", "stride", "out_shape", "activation", "params", "notes")
_ARCH_TABLE_HEADERS = ("Layer", "Kernel", "Stride", "Output shape", "Activation", "Params", "Notes")


def load_monitor_episode_rewards(monitor_csv: Path) -> np.ndarray:
    """Return per-episode rewards from an SB3 ``*.monitor.csv`` file."""

    path = Path(monitor_csv)
    df = pd.read_csv(path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"Monitor CSV missing column 'r': {path}")
    return df["r"].to_numpy(dtype=np.float64)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([pad, valid])


def aggregate_episode_rewards(
    monitor_csvs: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align seeds by episode index; return ``(episodes, mean, std)``."""

    if not monitor_csvs:
        raise ValueError("aggregate_episode_rewards() requires at least one Monitor CSV.")

    series = [load_monitor_episode_rewards(path) for path in monitor_csvs]
    max_len = max(len(s) for s in series)
    matrix = np.full((len(series), max_len), np.nan, dtype=np.float64)
    for row, rewards in enumerate(series):
        matrix[row, : len(rewards)] = rewards

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    episodes = np.arange(1, max_len + 1, dtype=np.int64)
    return episodes, mean, std


def plot_learning_curve(
    monitor_csvs: Sequence[Path],
    output_path: Path,
    title: str,
    smoothing_window: int = 100,
) -> None:
    """Mean ± std learning curve across multiple seeds of the same config."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, mean, std = aggregate_episode_rewards(monitor_csvs)
    smooth_mean = _rolling_mean(mean, smoothing_window)
    smooth_std = _rolling_mean(std, smoothing_window)

    valid = ~np.isnan(smooth_mean)
    x = episodes[valid]
    y = smooth_mean[valid]
    y_lo = (smooth_mean - smooth_std)[valid]
    y_hi = (smooth_mean + smooth_std)[valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color="#1f77b4", linewidth=1.5, label="mean")
    ax.fill_between(x, y_lo, y_hi, color="#1f77b4", alpha=0.25, label="±1 std")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hp_comparison(
    runs_by_config: Mapping[str, Sequence[Path]],
    output_path: Path,
    title: str = "Hyperparameter comparison",
    smoothing_window: int = 100,
) -> None:
    """Overlay mean ± std bands for several HP configs on one figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (label, csv_paths) in enumerate(sorted(runs_by_config.items())):
        if not csv_paths:
            continue
        color = colors[idx % len(colors)]
        episodes, mean, std = aggregate_episode_rewards(list(csv_paths))
        smooth_mean = _rolling_mean(mean, smoothing_window)
        smooth_std = _rolling_mean(std, smoothing_window)
        valid = ~np.isnan(smooth_mean)
        x = episodes[valid]
        y = smooth_mean[valid]
        y_lo = (smooth_mean - smooth_std)[valid]
        y_hi = (smooth_mean + smooth_std)[valid]
        ax.plot(x, y, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_timing_table_rows(runs_by_hp: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate timing metrics from ``run_metadata.json`` dicts per HP config."""

    rows: list[dict[str, Any]] = []
    for hp_name in sorted(runs_by_hp):
        metas = list(runs_by_hp[hp_name])
        if not metas:
            continue
        rows.append(
            {
                "hp_name": hp_name,
                "n_runs": len(metas),
                "mean_wall_clock_s": float(np.mean([float(m.get("wall_clock_s", 0.0)) for m in metas])),
                "mean_step_time_s": float(np.mean([float(m.get("mean_step_time_s", 0.0)) for m in metas])),
                "mean_episode_time_s": float(np.mean([float(m.get("mean_episode_time_s", 0.0)) for m in metas])),
                "total_timesteps": int(metas[0].get("total_timesteps", 0)),
            },
        )
    return rows


def write_timing_table_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "hp_name",
                "n_runs",
                "mean_wall_clock_s",
                "mean_step_time_s",
                "mean_episode_time_s",
                "total_timesteps",
            ],
        )
    df.to_csv(output_path, index=False)


def plot_arch_diagram(
    arch_name: str,
    layers: Iterable[dict[str, Any]],
    output_path: Optional[Path] = None,
    *,
    total_params: Optional[int] = None,
    fig_width: float = 12.0,
    row_height: float = 0.38,
) -> None:
    """Render a **tabular** architecture diagram for the report (Phase 2 / 5)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body: list[list[str]] = []
    for row in layers:
        body.append([str(row.get(k, "") or "") for k in _ARCH_TABLE_COLS])

    if total_params is not None:
        body.append(
            [
                "Σ trainable parameters",
                "",
                "",
                "",
                "",
                f"{int(total_params):,}",
                "",
            ],
        )

    nrows = len(body) + 1
    fig_h = max(2.0, row_height * nrows + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    ax.axis("off")
    ax.set_title(arch_name, fontsize=12, pad=12)

    table = ax.table(
        cellText=body,
        colLabels=list(_ARCH_TABLE_HEADERS),
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
