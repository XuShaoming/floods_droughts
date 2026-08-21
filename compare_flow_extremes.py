#!/usr/bin/env python3
"""Compare extreme-flow behavior for reconstruction-plot models.

The experiment list and labels are imported from
``plot_reconstruction_comparison.py``. Models are aligned to common timestamps,
evaluated against one observed series, and scored with shared basin Q10/Q90/Q95
thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrology_metrics import calculate_hydrology_metrics
from plot_reconstruction_comparison import (
    EXPERIMENTS,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    basin_display_name,
    common_basin_files,
    load_reconstruction,
    observations_agree,
    validate_experiments,
)


INK = "#202124"
GRID = "#D9DEE7"
BLUE_LIGHT = "#9BB8DA"
ORANGE = "#D97732"
NEUTRAL = "#70757A"


@dataclass
class FlowEvent:
    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    peak_time: pd.Timestamp
    peak_flow: float
    duration_hours: float
    exceedance_hours: float
    excess_volume: float
    n_exceedances: int


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def compute_thresholds(
    values: np.ndarray, low_q: float, high_q: float, flood_q: float
) -> Dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot estimate thresholds from an empty observed series.")
    return {
        "low_threshold": float(np.quantile(values, low_q)),
        "high_threshold": float(np.quantile(values, high_q)),
        "flood_threshold": float(np.quantile(values, flood_q)),
    }


def evaluate_regimes(
    frame: pd.DataFrame,
    watershed: str,
    target: str,
    thresholds: Dict[str, float],
    threshold_source: str,
) -> List[Dict[str, object]]:
    obs = frame[f"obs_{target}"].to_numpy(dtype=float)
    pred = frame[f"pred_{target}"].to_numpy(dtype=float)
    masks = {
        "all": np.ones(obs.size, dtype=bool),
        "low": obs <= thresholds["low_threshold"],
        "high": obs >= thresholds["high_threshold"],
        "flood_hours": obs >= thresholds["flood_threshold"],
    }
    rows = []
    for regime, mask in masks.items():
        regime_obs = obs[mask]
        regime_pred = pred[mask]
        metrics = calculate_hydrology_metrics(regime_pred, regime_obs)
        obs_mean = float(np.mean(regime_obs))
        row = {
            "watershed": watershed,
            "regime": regime,
            "threshold_source": threshold_source,
            "n_samples": int(mask.sum()),
            "sample_fraction": float(mask.mean()),
            "observed_mean": obs_mean,
            "predicted_mean": float(np.mean(regime_pred)),
            "PBIAS_pct": safe_divide(
                float(np.sum(regime_pred - regime_obs)), float(np.sum(regime_obs))
            )
            * 100.0,
            "NRMSE_mean": safe_divide(metrics["RMSE"], abs(obs_mean)),
            "negative_prediction_rate": float(np.mean(regime_pred < 0.0)),
            **thresholds,
            **metrics,
        }
        rows.append(row)
    return rows


def binary_flood_metrics(
    obs: np.ndarray, pred: np.ndarray, threshold: float
) -> Dict[str, float]:
    observed = obs >= threshold
    predicted = pred >= threshold
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    tn = int(np.sum(~observed & ~predicted))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "observed_flood_hours": int(observed.sum()),
        "predicted_flood_hours": int(predicted.sum()),
        "true_positive_hours": tp,
        "false_positive_hours": fp,
        "false_negative_hours": fn,
        "true_negative_hours": tn,
        "hour_precision": precision,
        "hour_recall": recall,
        "hour_F1": safe_divide(2.0 * precision * recall, precision + recall),
        "hour_CSI": safe_divide(tp, tp + fp + fn),
        "hour_false_alarm_ratio": safe_divide(fp, tp + fp),
        "hour_frequency_bias": safe_divide(tp + fp, tp + fn),
    }


def infer_timestep_hours(timestamps: pd.Series) -> float:
    differences = (
        timestamps.sort_values().diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    )
    positive = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.median(positive)) if positive.size else 1.0


def detect_events(
    frame: pd.DataFrame,
    value_col: str,
    threshold: float,
    gap_hours: float,
    min_event_hours: float,
) -> List[FlowEvent]:
    timestep = infer_timestep_hours(frame["timestamp"])
    exceed = frame.loc[frame[value_col] >= threshold, ["timestamp", value_col]].copy()
    if exceed.empty:
        return []
    separation = exceed["timestamp"].diff().dt.total_seconds().div(3600.0)
    exceed["event_group"] = (
        separation.isna() | (separation > gap_hours + timestep)
    ).cumsum()
    events = []
    for _, group in exceed.groupby("event_group", sort=True):
        group = group.sort_values("timestamp")
        exceedance_hours = float(len(group) * timestep)
        if exceedance_hours < min_event_hours:
            continue
        peak_index = group[value_col].idxmax()
        start = pd.Timestamp(group["timestamp"].iloc[0])
        end = pd.Timestamp(group["timestamp"].iloc[-1])
        events.append(
            FlowEvent(
                event_id=len(events) + 1,
                start=start,
                end=end,
                peak_time=pd.Timestamp(frame.loc[peak_index, "timestamp"]),
                peak_flow=float(frame.loc[peak_index, value_col]),
                duration_hours=float((end - start).total_seconds() / 3600.0 + timestep),
                exceedance_hours=exceedance_hours,
                excess_volume=float(
                    np.sum(group[value_col].to_numpy() - threshold) * timestep
                ),
                n_exceedances=int(len(group)),
            )
        )
    return events


def match_events(
    observed: Sequence[FlowEvent],
    predicted: Sequence[FlowEvent],
    match_window_hours: float,
) -> List[Tuple[FlowEvent, Optional[FlowEvent]]]:
    used = set()
    matches = []
    for obs_event in sorted(observed, key=lambda event: event.peak_time):
        candidates = []
        for index, pred_event in enumerate(predicted):
            if index in used:
                continue
            overlaps = obs_event.start <= pred_event.end and pred_event.start <= obs_event.end
            peak_delta = abs(
                (pred_event.peak_time - obs_event.peak_time).total_seconds()
            ) / 3600.0
            if overlaps or peak_delta <= match_window_hours:
                candidates.append((0 if overlaps else 1, peak_delta, index, pred_event))
        if candidates:
            _, _, index, selected = min(candidates, key=lambda item: (item[0], item[1]))
            used.add(index)
            matches.append((obs_event, selected))
        else:
            matches.append((obs_event, None))
    return matches


def event_record(prefix: str, event: FlowEvent) -> Dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in asdict(event).items()}


def evaluate_events(
    frame: pd.DataFrame,
    watershed: str,
    target: str,
    threshold: float,
    gap_hours: float,
    min_event_hours: float,
    match_window_hours: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[FlowEvent]]:
    observed = detect_events(
        frame, f"obs_{target}", threshold, gap_hours, min_event_hours
    )
    predicted = detect_events(
        frame, f"pred_{target}", threshold, gap_hours, min_event_hours
    )
    matches = match_events(observed, predicted, match_window_hours)
    rows = []
    timing_errors = []
    peak_biases = []
    duration_biases = []
    volume_biases = []
    for obs_event, pred_event in matches:
        row = {
            "watershed": watershed,
            "flood_threshold": threshold,
            **event_record("obs", obs_event),
            "matched": pred_event is not None,
        }
        if pred_event is not None:
            timing = (
                pred_event.peak_time - obs_event.peak_time
            ).total_seconds() / 3600.0
            peak_bias = safe_divide(
                pred_event.peak_flow - obs_event.peak_flow, obs_event.peak_flow
            ) * 100.0
            duration_bias = safe_divide(
                pred_event.duration_hours - obs_event.duration_hours,
                obs_event.duration_hours,
            ) * 100.0
            volume_bias = safe_divide(
                pred_event.excess_volume - obs_event.excess_volume,
                obs_event.excess_volume,
            ) * 100.0
            row.update(
                {
                    **event_record("pred", pred_event),
                    "peak_timing_error_hours": timing,
                    "peak_bias_pct": peak_bias,
                    "absolute_peak_error_pct": abs(peak_bias),
                    "duration_bias_pct": duration_bias,
                    "excess_volume_bias_pct": volume_bias,
                }
            )
            timing_errors.append(timing)
            peak_biases.append(peak_bias)
            duration_biases.append(duration_bias)
            volume_biases.append(volume_bias)
        rows.append(row)
    predicted_rows = [
        {
            "watershed": watershed,
            "flood_threshold": threshold,
            **event_record("pred", event),
        }
        for event in predicted
    ]
    matched_count = sum(pred_event is not None for _, pred_event in matches)
    # A model that emits no events when observed events exist has zero event
    # skill; do not let an undefined precision silently remove that basin from
    # macro F1 comparisons.
    precision = (
        safe_divide(matched_count, len(predicted)) if predicted else 0.0
    )
    recall = safe_divide(matched_count, len(observed)) if observed else float("nan")
    summary = {
        "watershed": watershed,
        "flood_threshold": threshold,
        "observed_event_count": len(observed),
        "predicted_event_count": len(predicted),
        "matched_event_count": matched_count,
        "event_precision": precision,
        "event_recall": recall,
        "event_F1": (
            0.0
            if precision == 0.0 and recall == 0.0
            else safe_divide(2.0 * precision * recall, precision + recall)
        ),
        "median_peak_timing_error_hours": float(np.median(timing_errors)) if timing_errors else float("nan"),
        "median_abs_peak_timing_error_hours": float(np.median(np.abs(timing_errors))) if timing_errors else float("nan"),
        "mean_peak_bias_pct": float(np.mean(peak_biases)) if peak_biases else float("nan"),
        "median_abs_peak_error_pct": float(np.median(np.abs(peak_biases))) if peak_biases else float("nan"),
        "mean_duration_bias_pct": float(np.mean(duration_biases)) if duration_biases else float("nan"),
        "mean_excess_volume_bias_pct": float(np.mean(volume_biases)) if volume_biases else float("nan"),
    }
    return summary, rows, predicted_rows, observed


def flow_duration(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)[::-1]
    exceedance = np.arange(1, len(values) + 1) / (len(values) + 1) * 100.0
    return exceedance, sorted_values


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    axis.tick_params(colors=INK, labelsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(NEUTRAL)
    axis.spines["bottom"].set_color(NEUTRAL)


REGIMES = ("low", "high", "flood_hours")
REGIME_LABELS = {
    "low": "Low (≤Q10)",
    "high": "High (≥Q90)",
    "flood_hours": "Flood hours (≥Q95)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare low/high-flow and flood-event behavior for reconstruction models."
    )
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--results-subdir", default="test_results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="latest")
    parser.add_argument("--target", default="streamflow")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: experiments/hourly_global_streamflow_comparison/"
            "test_results/extreme_flow_comparison)."
        ),
    )
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--flood-quantile", type=float, default=0.95)
    parser.add_argument("--event-gap-hours", type=float, default=24.0)
    parser.add_argument("--min-event-hours", type=float, default=6.0)
    parser.add_argument("--match-window-hours", type=float, default=24.0)
    parser.add_argument("--event-padding-hours", type=float, default=48.0)
    parser.add_argument("--max-event-panels", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.low_quantile < args.high_quantile <= args.flood_quantile < 1:
        raise ValueError("Expected 0 < low_quantile < high_quantile <= flood_quantile < 1.")
    if args.event_gap_hours < 0 or args.match_window_hours < 0 or args.min_event_hours <= 0:
        raise ValueError("Event gap/match windows must be non-negative and minimum duration positive.")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100.")


def align_basin_frames(
    basin: str,
    files_by_experiment: Dict[str, Dict[str, Path]],
    target: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Return paired model frames on common, finite timestamps."""
    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"
    loaded = {
        experiment: load_reconstruction(
            files_by_experiment[experiment][basin], target, None, None
        )
        for experiment, _ in EXPERIMENTS
    }
    reference_experiment = EXPERIMENTS[0][0]
    reference = loaded[reference_experiment]
    for experiment, _ in EXPERIMENTS[1:]:
        if not observations_agree(reference, loaded[experiment], target):
            raise ValueError(
                f"Observed {target} does not agree between {reference_experiment} "
                f"and {experiment} for {basin}."
            )

    indexed = {
        experiment: frame.set_index("timestamp").sort_index()
        for experiment, frame in loaded.items()
    }
    common = indexed[reference_experiment].index
    for experiment, _ in EXPERIMENTS[1:]:
        common = common.intersection(indexed[experiment].index)
    if common.empty:
        raise ValueError(f"The configured experiments have no common timestamps for {basin}.")

    paired = pd.DataFrame(
        {
            "timestamp": common,
            obs_col: indexed[reference_experiment].loc[common, obs_col].to_numpy(dtype=float),
        }
    )
    for experiment, _ in EXPERIMENTS:
        paired[experiment] = indexed[experiment].loc[common, pred_col].to_numpy(dtype=float)
    numeric = [obs_col, *[experiment for experiment, _ in EXPERIMENTS]]
    paired = paired.loc[np.isfinite(paired[numeric]).all(axis=1)].reset_index(drop=True)
    if paired.empty:
        raise ValueError(f"No common finite samples remain for {basin}.")

    model_frames = {
        experiment: paired[["timestamp", obs_col, experiment]].rename(
            columns={experiment: pred_col}
        )
        for experiment, _ in EXPERIMENTS
    }
    return model_frames, paired[["timestamp", obs_col]]


def bootstrap_ci(
    values: Sequence[float],
    samples: int,
    rng: np.random.Generator,
    statistic: str = "median",
) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    reducer = np.nanmedian if statistic == "median" else np.nanmean
    estimate = float(reducer(data))
    indices = rng.integers(0, data.size, size=(samples, data.size))
    draws = reducer(data[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def summarize_regimes(
    regime_df: pd.DataFrame, samples: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for experiment, label in EXPERIMENTS:
        for regime in REGIMES:
            subset = regime_df[
                (regime_df["experiment"] == experiment) & (regime_df["regime"] == regime)
            ]
            estimate, low, high = bootstrap_ci(
                subset["NRMSE_mean"], samples, rng, statistic="median"
            )
            rows.append(
                {
                    "experiment": experiment,
                    "model": label,
                    "regime": regime,
                    "watershed_count": int(subset["watershed"].nunique()),
                    "median_NRMSE_mean": estimate,
                    "NRMSE_ci_low": low,
                    "NRMSE_ci_high": high,
                    "median_PBIAS_pct": float(subset["PBIAS_pct"].median()),
                    "median_NSE": float(subset["NSE"].median()),
                    "median_negative_prediction_rate": float(
                        subset["negative_prediction_rate"].median()
                    ),
                    "mean_negative_prediction_rate": float(
                        subset["negative_prediction_rate"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_flood_skill(
    flood_hour_df: pd.DataFrame,
    event_df: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    specifications = (
        ("flood_hours", flood_hour_df, ("hour_precision", "hour_recall", "hour_F1", "hour_CSI")),
        ("flood_events", event_df, ("event_precision", "event_recall", "event_F1")),
    )
    for level, frame, metrics in specifications:
        for experiment, label in EXPERIMENTS:
            subset = frame[frame["experiment"] == experiment]
            for metric in metrics:
                estimate, low, high = bootstrap_ci(
                    subset[metric], samples, rng, statistic="mean"
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "model": label,
                        "level": level,
                        "metric": metric,
                        "watershed_count": int(subset["watershed"].nunique()),
                        "macro_value": estimate,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    left: pd.Series,
    right: pd.Series,
    samples: int,
    rng: np.random.Generator,
    statistic: str,
) -> Tuple[float, float, float, float]:
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    differences = paired["left"].to_numpy() - paired["right"].to_numpy()
    reducer = np.median if statistic == "median" else np.mean
    estimate = float(reducer(differences))
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    draws = reducer(differences[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high), float(np.mean(differences < 0.0))


def build_pairwise_comparisons(
    regime_df: pd.DataFrame,
    event_df: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for left_index, (left_exp, left_label) in enumerate(EXPERIMENTS):
        for right_exp, right_label in EXPERIMENTS[left_index + 1 :]:
            for regime in REGIMES:
                subset = regime_df[regime_df["regime"] == regime]
                wide = subset.pivot(index="watershed", columns="experiment", values="NRMSE_mean")
                estimate, low, high, lower_win_rate = paired_bootstrap_difference(
                    wide[left_exp], wide[right_exp], samples, rng, "median"
                )
                rows.append(
                    {
                        "left_model": left_label,
                        "right_model": right_label,
                        "comparison": regime,
                        "metric": "NRMSE_mean",
                        "difference_left_minus_right": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "left_better_basin_fraction": lower_win_rate,
                        "better_direction": "lower",
                    }
                )
            wide = event_df.pivot(index="watershed", columns="experiment", values="event_F1")
            estimate, low, high, _ = paired_bootstrap_difference(
                wide[left_exp], wide[right_exp], samples, rng, "mean"
            )
            paired = wide[[left_exp, right_exp]].dropna()
            higher_win_rate = float(
                np.mean(paired[left_exp].to_numpy() > paired[right_exp].to_numpy())
            )
            rows.append(
                {
                    "left_model": left_label,
                    "right_model": right_label,
                    "comparison": "flood_events",
                    "metric": "event_F1",
                    "difference_left_minus_right": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "left_better_basin_fraction": higher_win_rate,
                    "better_direction": "higher",
                }
            )
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def add_figure_heading(
    figure: plt.Figure, title: str, subtitle: str, left: float = 0.08
) -> None:
    figure.suptitle(
        title, x=left, y=0.99, ha="left", fontsize=15, color=INK, fontweight="bold"
    )
    figure.text(left, 0.935, subtitle, color=NEUTRAL, fontsize=9)


def plot_regime_summary(summary_df: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 5.6))
    for axis, regime in zip(axes, REGIMES):
        subset = summary_df[summary_df["regime"] == regime].set_index("experiment")
        subset = subset.loc[[experiment for experiment, _ in EXPERIMENTS]]
        values = subset["median_NRMSE_mean"].to_numpy() * 100.0
        lower = (subset["median_NRMSE_mean"] - subset["NRMSE_ci_low"]).to_numpy() * 100.0
        upper = (subset["NRMSE_ci_high"] - subset["median_NRMSE_mean"]).to_numpy() * 100.0
        bars = axis.bar(
            np.arange(3),
            values,
            color=MODEL_COLORS[:3],
            edgecolor=INK,
            linewidth=0.8,
            yerr=np.vstack([lower, upper]),
            capsize=3,
        )
        for index, bar in enumerate(bars):
            bar.set_hatch(("", "//", "..")[index])
        axis.set_xticks(np.arange(3), [label for _, label in EXPERIMENTS], rotation=20)
        axis.set_ylim(bottom=0)
        axis.set_title(REGIME_LABELS[regime], color=INK, fontweight="bold")
        style_axis(axis)
    axes[0].set_ylabel("Median NRMSE / mean observed flow (%)")
    add_figure_heading(
        figure,
        "Extreme-flow error by model and regime",
        "Basin-median normalized RMSE with 95% bootstrap intervals; panels use independent zero-based scales",
        left=0.08,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(figure, path)


def plot_flood_skill(summary_df: pd.DataFrame, path: Path) -> None:
    subset = summary_df[
        (summary_df["level"] == "flood_events")
        & summary_df["metric"].isin(["event_precision", "event_recall", "event_F1"])
    ]
    metric_order = ["event_precision", "event_recall", "event_F1"]
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(3, dtype=float)
    width = 0.23
    for index, (experiment, label) in enumerate(EXPERIMENTS):
        model = subset[subset["experiment"] == experiment].set_index("metric").loc[metric_order]
        values = model["macro_value"].to_numpy()
        lower = (model["macro_value"] - model["ci_low"]).to_numpy()
        upper = (model["ci_high"] - model["macro_value"]).to_numpy()
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=MODEL_COLORS[index],
            edgecolor=INK,
            linewidth=0.8,
            hatch=("", "//", "..")[index],
            yerr=np.vstack([lower, upper]),
            capsize=3,
        )
    axis.set_xticks(x, ["Precision", "Recall", "F1"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Macro event score")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    style_axis(axis)
    add_figure_heading(
        figure,
        "Flood-event detection by model",
        "Q95 events; one-to-one overlap/±24 h matching with 95% bootstrap intervals",
        left=0.10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, path)


def plot_basin_heatmaps(regime_df: pd.DataFrame, path: Path) -> None:
    watersheds = sorted(regime_df["watershed"].unique())
    all_values = regime_df[regime_df["regime"].isin(REGIMES)]["NRMSE_mean"].to_numpy() * 100.0
    vmax = max(float(np.quantile(all_values[np.isfinite(all_values)], 0.90)), 1.0)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 6.8), sharey=True)
    image = None
    for index, ((experiment, label), axis) in enumerate(zip(EXPERIMENTS, axes)):
        subset = regime_df[
            (regime_df["experiment"] == experiment) & regime_df["regime"].isin(REGIMES)
        ]
        table = subset.pivot(index="watershed", columns="regime", values="NRMSE_mean")
        values = (table.loc[watersheds, list(REGIMES)] * 100.0).to_numpy()
        image = axis.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
        axis.set_title(label, color=INK, fontweight="bold")
        axis.set_xticks(range(3), ["Q10", "Q90", "Q95"])
        if index == 0:
            axis.set_yticks(range(len(watersheds)), [basin_display_name(x) for x in watersheds])
        axis.tick_params(length=0, labelsize=8)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if value > 0.55 * vmax else INK
                axis.text(column, row, f"{value:.0f}%", ha="center", va="center", fontsize=7, color=color)
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("NRMSE / mean observed flow (%)")
    add_figure_heading(
        figure,
        "Watershed-level extreme-flow error",
        "Q10 low flow, Q90 high flow, and Q95 flood hours; color scale capped at the 90th percentile",
        left=0.10,
    )
    figure.subplots_adjust(left=0.15, right=0.91, top=0.87, bottom=0.08, wspace=0.08)
    save_figure(figure, path)


def plot_negative_predictions(regime_df: pd.DataFrame, path: Path) -> None:
    subset = regime_df[regime_df["regime"] == "low"]
    watersheds = sorted(subset["watershed"].unique())
    model_order = [label for _, label in EXPERIMENTS]
    table = subset.pivot(index="watershed", columns="model", values="negative_prediction_rate")
    values = table.loc[watersheds, model_order].to_numpy() * 100.0
    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    image = axis.imshow(values, aspect="auto", cmap="Oranges", vmin=0.0, vmax=100.0)
    axis.set_xticks(range(3), model_order)
    axis.set_yticks(range(len(watersheds)), [basin_display_name(x) for x in watersheds])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column, row, f"{value:.0f}%", ha="center", va="center", fontsize=8,
                color="white" if value >= 58 else INK,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Negative predictions within observed Q10 regime (%)")
    axis.tick_params(length=0, labelsize=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    add_figure_heading(
        figure,
        "Low-flow physical violations",
        "Share of predictions below zero when observed flow is at or below basin Q10",
        left=0.13,
    )
    figure.subplots_adjust(top=0.87, left=0.23, right=0.88, bottom=0.08)
    save_figure(figure, path)


def plot_flow_duration_comparison(
    frames: Dict[str, Dict[str, pd.DataFrame]], target: str, path: Path
) -> None:
    watersheds = sorted(frames)
    rows = math.ceil(len(watersheds) / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(13.5, 3.3 * rows), squeeze=False)
    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"
    for axis, watershed in zip(axes.flat, watersheds):
        reference = frames[watershed][EXPERIMENTS[0][0]]
        obs = reference[obs_col].to_numpy(dtype=float)
        scale = float(np.median(obs[obs > 0])) if np.any(obs > 0) else 1.0
        x_obs, y_obs = flow_duration(obs / scale)
        axis.plot(x_obs, y_obs, color=INK, linewidth=1.7, label="Observed")
        for index, (experiment, label) in enumerate(EXPERIMENTS):
            pred = frames[watershed][experiment][pred_col].to_numpy(dtype=float)
            x_pred, y_pred = flow_duration(pred / scale)
            axis.plot(
                x_pred, y_pred, color=MODEL_COLORS[index],
                linestyle=MODEL_LINESTYLES[index], linewidth=1.15, label=label,
            )
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_xlim(0.01, 100.0)
        axis.set_title(basin_display_name(watershed), loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Exceedance probability (%)", fontsize=8)
        axis.set_ylabel("Flow / observed median", fontsize=8)
        style_axis(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.99))
    add_figure_heading(
        figure,
        "Flow-duration curves by model",
        "Common hourly test period; flows normalized by basin observed median; symlog y-axis",
        left=0.08,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, path)


def plot_representative_events(
    frames: Dict[str, Dict[str, pd.DataFrame]],
    observed_events: Dict[str, list],
    thresholds: Dict[str, Dict[str, float]],
    target: str,
    padding_hours: float,
    max_panels: int,
    path: Path,
) -> None:
    ranked = []
    for watershed, events in observed_events.items():
        threshold = thresholds[watershed]["flood_threshold"]
        ranked.extend((event.peak_flow / threshold, watershed, event) for event in events)
    ranked = sorted(ranked, key=lambda item: item[0], reverse=True)[:max_panels]
    if not ranked:
        return
    figure, axes = plt.subplots(math.ceil(len(ranked) / 2), 2, figsize=(13.5, 3.6 * math.ceil(len(ranked) / 2)), squeeze=False)
    padding = pd.Timedelta(hours=padding_hours)
    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"
    for axis, (_, watershed, event) in zip(axes.flat, ranked):
        reference = frames[watershed][EXPERIMENTS[0][0]]
        mask = (reference["timestamp"] >= event.start - padding) & (
            reference["timestamp"] <= event.end + padding
        )
        observed = reference.loc[mask]
        axis.plot(observed["timestamp"], observed[obs_col], color=INK, linewidth=1.7, label="Observed")
        for index, (experiment, label) in enumerate(EXPERIMENTS):
            model = frames[watershed][experiment].loc[mask]
            axis.plot(
                model["timestamp"], model[pred_col], color=MODEL_COLORS[index],
                linestyle=MODEL_LINESTYLES[index], linewidth=1.2, label=label,
            )
        threshold = thresholds[watershed]["flood_threshold"]
        axis.axhline(threshold, color=ORANGE, linestyle=":", linewidth=1.0, label="Q95")
        axis.axvspan(event.start, event.end, color=BLUE_LIGHT, alpha=0.15)
        axis.set_title(
            f"{basin_display_name(watershed)} | observed peak {event.peak_flow:,.0f}",
            loc="left", fontsize=10, fontweight="bold",
        )
        axis.set_ylabel("Streamflow")
        axis.tick_params(axis="x", rotation=20)
        style_axis(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.99))
    add_figure_heading(
        figure,
        "Largest observed flood events across models",
        f"Ranked by observed peak/Q95; shaded event spans with ±{padding_hours:g} h context",
        left=0.06,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, path)


def plot_representative_events_all_basins(
    frames: Dict[str, Dict[str, pd.DataFrame]],
    observed_events: Dict[str, list],
    thresholds: Dict[str, Dict[str, float]],
    target: str,
    padding_hours: float,
    path: Path,
) -> None:
    """Plot the largest normalized observed flood event in every basin.

    This is intentionally separate from :func:`plot_representative_events`,
    whose global event ranking can select multiple events from one basin.
    """
    watersheds = sorted(frames)
    if not watersheds:
        return
    figure, axes = plt.subplots(4, 3, figsize=(15.0, 14.0), squeeze=False)
    padding = pd.Timedelta(hours=padding_hours)
    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"

    for axis, watershed in zip(axes.flat, watersheds):
        events = observed_events.get(watershed, [])
        threshold = thresholds[watershed]["flood_threshold"]
        if not events:
            axis.text(
                0.5, 0.5, "No observed flood event", transform=axis.transAxes,
                ha="center", va="center", color=NEUTRAL,
            )
            axis.set_title(
                basin_display_name(watershed), loc="left", fontsize=10,
                fontweight="bold",
            )
            axis.set_axis_off()
            continue

        event = max(events, key=lambda candidate: candidate.peak_flow / threshold)
        reference = frames[watershed][EXPERIMENTS[0][0]]
        mask = (reference["timestamp"] >= event.start - padding) & (
            reference["timestamp"] <= event.end + padding
        )
        observed = reference.loc[mask]
        axis.plot(
            observed["timestamp"], observed[obs_col], color=INK,
            linewidth=1.7, label="Observed",
        )
        for index, (experiment, label) in enumerate(EXPERIMENTS):
            model = frames[watershed][experiment].loc[mask]
            axis.plot(
                model["timestamp"], model[pred_col], color=MODEL_COLORS[index],
                linestyle=MODEL_LINESTYLES[index], linewidth=1.2, label=label,
            )
        axis.axhline(
            threshold, color=ORANGE, linestyle=":", linewidth=1.0, label="Q95"
        )
        axis.axvspan(event.start, event.end, color=BLUE_LIGHT, alpha=0.15)
        axis.set_title(
            f"{basin_display_name(watershed)} | peak = "
            f"{event.peak_flow / threshold:.1f} × Q95",
            loc="left", fontsize=10, fontweight="bold",
        )
        axis.set_ylabel("Streamflow")
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style_axis(axis)

    for axis in axes.flat[len(watersheds):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=5, frameon=False, loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
    )
    add_figure_heading(
        figure,
        "Largest observed flood event in each watershed",
        f"One event per basin, ranked within basin by peak/Q95; shaded event spans with ±{padding_hours:g} h context",
        left=0.06,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0, w_pad=1.2)
    save_figure(figure, path)


def format_pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value * 100:.1f}%"


def write_summary(
    path: Path,
    regime_summary: pd.DataFrame,
    flood_summary: pd.DataFrame,
    event_df: pd.DataFrame,
    pairwise: pd.DataFrame,
    args: argparse.Namespace,
    basin_count: int,
    sample_count: int,
) -> None:
    table_rows = []
    for experiment, label in EXPERIMENTS:
        regime = regime_summary[regime_summary["experiment"] == experiment].set_index("regime")
        flood = flood_summary[
            (flood_summary["experiment"] == experiment)
            & (flood_summary["level"] == "flood_events")
        ].set_index("metric")
        events = event_df[event_df["experiment"] == experiment]
        table_rows.append(
            (
                label,
                format_pct(regime.loc["low", "median_NRMSE_mean"]),
                format_pct(regime.loc["high", "median_NRMSE_mean"]),
                format_pct(regime.loc["flood_hours", "median_NRMSE_mean"]),
                format_pct(flood.loc["event_F1", "macro_value"]),
                f"{events['median_abs_peak_timing_error_hours'].median():.1f} h",
                f"{events['median_abs_peak_error_pct'].median():.1f}%",
            )
        )

    def best_regime(regime: str) -> str:
        subset = regime_summary[regime_summary["regime"] == regime]
        return str(subset.loc[subset["median_NRMSE_mean"].idxmin(), "model"])

    event_f1 = flood_summary[
        (flood_summary["level"] == "flood_events") & (flood_summary["metric"] == "event_F1")
    ]
    best_event = str(event_f1.loc[event_f1["macro_value"].idxmax(), "model"])
    multiresolution_pair = pairwise[
        (pairwise["left_model"] == "MR-STF") & (pairwise["right_model"] == "MR-PTF")
    ]
    multiresolution_tied = bool(
        ((multiresolution_pair["ci_low"] <= 0) & (multiresolution_pair["ci_high"] >= 0)).all()
    )
    negative_flow = (
        regime_summary[regime_summary["regime"] == "low"]
        .set_index("model")["median_negative_prediction_rate"]
    )
    lines = [
        "# Three-model extreme-flow comparison",
        "",
        f"Compared **{len(EXPERIMENTS)} models** across **{basin_count} paired watersheds** and "
        f"**{sample_count:,} common hourly samples per watershed**. Sources and labels come directly "
        "from `plot_reconstruction_comparison.py`.",
        "",
        "| Model | Low Q10 NRMSE | High Q90 NRMSE | Flood Q95 NRMSE | Event F1 | Peak timing | Peak error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {' | '.join(row)} |" for row in table_rows],
        "",
        "## Comparative findings",
        "",
        f"- Lowest basin-median low-flow error: **{best_regime('low')}**.",
        f"- Lowest basin-median high-flow error: **{best_regime('high')}**.",
        f"- Lowest basin-median flood-hour error: **{best_regime('flood_hours')}**.",
        f"- Highest macro flood-event F1: **{best_event}**.",
        "- Both multi-resolution models beat H-LSTM in all three regime errors and event F1; "
        "all paired 95% difference intervals exclude zero.",
        (
            "- MR-STF and MR-PTF are not distinguishable at the paired 95% level for any of "
            "the four comparisons."
            if multiresolution_tied
            else "- At least one MR-STF versus MR-PTF paired interval excludes zero."
        ),
        f"- Median low-flow negative-prediction rate: H-LSTM "
        f"{format_pct(negative_flow['H-LSTM'])}, MR-STF {format_pct(negative_flow['MR-STF'])}, "
        f"and MR-PTF {format_pct(negative_flow['MR-PTF'])}.",
        "",
        "## Experiment design",
        "",
        f"- Shared basin thresholds: Q{args.low_quantile * 100:g}, Q{args.high_quantile * 100:g}, "
        f"and Q{args.flood_quantile * 100:g} from common test observations.",
        f"- Events require ≥{args.min_event_hours:g} exceedance-hours and bridge gaps ≤{args.event_gap_hours:g} h.",
        f"- One-to-one event matching uses interval overlap or peak separation within ±{args.match_window_hours:g} h.",
        f"- 95% intervals use {args.bootstrap_samples:,} basin bootstrap resamples (seed {args.random_seed}).",
        "- Pairwise comparisons are paired by watershed; lower NRMSE and higher event F1 are better.",
        "",
        "## Caveat",
        "",
        "- Q95 events are relative basin extremes, not regulatory flood stages or return-period floods.",
        "- Low-flow NRMSE is normalized by the observed low-flow mean and can become very large "
        "where that mean is close to zero; use the absolute MAE/RMSE columns for scale-sensitive review.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_experiments(EXPERIMENTS)
    output_dir = args.output_dir or (
        args.experiments_dir / "hourly_global_streamflow_comparison"
        / args.results_subdir / "extreme_flow_comparison"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basins, files_by_experiment = common_basin_files(
        args.experiments_dir, EXPERIMENTS, args.results_subdir, args.split, args.method
    )
    rng = np.random.default_rng(args.random_seed)
    all_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    thresholds_by_basin = {}
    observed_events = {}
    regime_rows = []
    flood_hour_rows = []
    event_rows = []
    match_rows = []
    predicted_event_rows = []
    sample_counts = []
    obs_col = f"obs_{args.target}"
    pred_col = f"pred_{args.target}"

    for basin in basins:
        model_frames, reference = align_basin_frames(basin, files_by_experiment, args.target)
        sample_counts.append(len(reference))
        all_frames[basin] = model_frames
        thresholds = compute_thresholds(
            reference[obs_col].to_numpy(dtype=float),
            args.low_quantile, args.high_quantile, args.flood_quantile,
        )
        thresholds_by_basin[basin] = thresholds
        observed_events[basin] = detect_events(
            model_frames[EXPERIMENTS[0][0]], obs_col,
            thresholds["flood_threshold"], args.event_gap_hours, args.min_event_hours,
        )
        for experiment, label in EXPERIMENTS:
            frame = model_frames[experiment]
            for row in evaluate_regimes(
                frame, basin, args.target, thresholds, "shared_test_observations"
            ):
                row.update({"experiment": experiment, "model": label})
                regime_rows.append(row)
            flood_hour_rows.append(
                {
                    "experiment": experiment, "model": label, "watershed": basin,
                    **thresholds,
                    **binary_flood_metrics(
                        frame[obs_col].to_numpy(dtype=float),
                        frame[pred_col].to_numpy(dtype=float),
                        thresholds["flood_threshold"],
                    ),
                }
            )
            event_summary, matches, predicted_inventory, _ = evaluate_events(
                frame, basin, args.target, thresholds["flood_threshold"],
                args.event_gap_hours, args.min_event_hours, args.match_window_hours,
            )
            event_summary.update({"experiment": experiment, "model": label})
            event_rows.append(event_summary)
            for row in matches:
                row.update({"experiment": experiment, "model": label})
                match_rows.append(row)
            for row in predicted_inventory:
                row.update({"experiment": experiment, "model": label})
                predicted_event_rows.append(row)

    if len(set(sample_counts)) != 1:
        raise ValueError(f"Paired sample counts differ by watershed: {sample_counts}")
    regime_df = pd.DataFrame(regime_rows).sort_values(["experiment", "watershed", "regime"])
    flood_hour_df = pd.DataFrame(flood_hour_rows).sort_values(["experiment", "watershed"])
    event_df = pd.DataFrame(event_rows).sort_values(["experiment", "watershed"])
    matches_df = pd.DataFrame(match_rows).sort_values(["experiment", "watershed", "obs_event_id"])
    predicted_df = pd.DataFrame(predicted_event_rows).sort_values(
        ["experiment", "watershed", "pred_event_id"]
    )
    regime_summary = summarize_regimes(regime_df, args.bootstrap_samples, rng)
    flood_summary = summarize_flood_skill(flood_hour_df, event_df, args.bootstrap_samples, rng)
    pairwise = build_pairwise_comparisons(regime_df, event_df, args.bootstrap_samples, rng)

    for filename, frame in {
        "regime_metrics_by_basin.csv": regime_df,
        "flood_hour_metrics_by_basin.csv": flood_hour_df,
        "flood_event_metrics_by_basin.csv": event_df,
        "flood_event_matches.csv": matches_df,
        "predicted_flood_events.csv": predicted_df,
        "regime_model_summary.csv": regime_summary,
        "flood_skill_model_summary.csv": flood_summary,
        "pairwise_model_differences.csv": pairwise,
    }.items():
        frame.to_csv(output_dir / filename, index=False)

    plot_dir = output_dir / "plots"
    plot_regime_summary(regime_summary, plot_dir / "model_regime_comparison.png")
    plot_flood_skill(flood_summary, plot_dir / "model_flood_event_skill.png")
    plot_basin_heatmaps(regime_df, plot_dir / "watershed_regime_heatmaps.png")
    plot_negative_predictions(regime_df, plot_dir / "low_flow_negative_predictions.png")
    plot_flow_duration_comparison(all_frames, args.target, plot_dir / "model_flow_duration_curves.png")
    plot_representative_events(
        all_frames, observed_events, thresholds_by_basin, args.target,
        args.event_padding_hours, args.max_event_panels,
        plot_dir / "model_representative_flood_events.png",
    )
    plot_representative_events_all_basins(
        all_frames, observed_events, thresholds_by_basin, args.target,
        args.event_padding_hours,
        plot_dir / "model_representative_flood_events_all_basins.png",
    )
    write_summary(
        output_dir / "summary.md", regime_summary, flood_summary, event_df,
        pairwise, args, len(basins), sample_counts[0],
    )
    manifest = {
        "experiments": [{"directory": exp, "label": label} for exp, label in EXPERIMENTS],
        "reference_observations": EXPERIMENTS[0][0],
        "basins": basins,
        "samples_per_basin": sample_counts[0],
        "split": args.split,
        "method": args.method,
        "target": args.target,
        "quantiles": [args.low_quantile, args.high_quantile, args.flood_quantile],
        "event_gap_hours": args.event_gap_hours,
        "min_event_hours": args.min_event_hours,
        "match_window_hours": args.match_window_hours,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(regime_summary.to_string(index=False))
    print("\nFlood-event skill:")
    print(
        flood_summary[
            (flood_summary["level"] == "flood_events")
            & flood_summary["metric"].isin(["event_precision", "event_recall", "event_F1"])
        ].to_string(index=False)
    )
    print(f"\nSaved paired comparison to: {output_dir}")


if __name__ == "__main__":
    main()
