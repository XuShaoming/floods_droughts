#!/usr/bin/env python3
"""Compare H-LSTM, MR-STF, MR-PTF, and D-LSTM on the paired test period.

The three hourly experiments are evaluated at their native resolution. The
daily D-LSTM prediction is expanded with the project's existing calendar-day
repeat method (``day_shift=0``), assigning one daily value to all 24 hours of
that date. All four predictions are then intersected on identical timestamps
and evaluated against the H-LSTM reconstruction's hourly reference.

The script creates the four tables requested for the paper, exact CSV files,
an alignment audit, and a 4 x 3 representative-flood-event figure containing
all watersheds and all four models.

Run from the repository root::

    conda run --no-capture-output -n imerg_era5 \
        python compare_four_model_test_performance.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from compare_flow_extremes import detect_events, evaluate_events
from compare_hourly_daily_streamflow import (
    DEFAULT_WATERSHEDS,
    basin_display_name,
    expand_daily_prediction,
    load_daily_reconstruction,
    load_hourly_reconstruction,
    reconstruction_path,
)
from hydrology_metrics import calculate_hydrology_metrics


MODEL_ORDER: Tuple[str, ...] = ("D-LSTM", "H-LSTM", "MR-STF", "MR-PTF")
EXPERIMENTS: Mapping[str, str] = {
    "H-LSTM": "hourly_global_streamflow_no_IMVs",
    "MR-STF": "hourly_global_streamflow_pred_streamflow_day_shift_1",
    "MR-PTF": "hourly_global_streamflow_day_shift_1",
    "D-LSTM": "daily_global_streamflow",
}
PREDICTION_COLUMNS: Mapping[str, str] = {
    "H-LSTM": "pred_h_lstm",
    "MR-STF": "pred_mr_stf",
    "MR-PTF": "pred_mr_ptf",
    "D-LSTM": "pred_d_lstm",
}
MODEL_STYLES: Mapping[str, Tuple[str, str, object]] = {
    "H-LSTM": ("#0072B2", "o", "-"),
    "MR-STF": ("#D55E00", "s", (0, (5, 2))),
    "MR-PTF": ("#009E73", "^", (0, (2, 1.5))),
    "D-LSTM": ("#CC79A7", "D", (0, (7, 2.5))),
}
REFERENCE_COLOR = "#202124"
GRID_COLOR = "#D9DEE7"
DEFAULT_OUTPUT_DIR = Path(
    "experiments/hourly_daily_four_model_comparison/test_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create four-model test-period performance tables and a representative "
            "flood-event figure."
        )
    )
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--results-subdir", default="test_results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="latest")
    parser.add_argument("--watersheds", nargs="+", default=DEFAULT_WATERSHEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--flood-quantile", type=float, default=0.95)
    parser.add_argument("--event-gap-hours", type=float, default=24.0)
    parser.add_argument("--min-event-hours", type=float, default=6.0)
    parser.add_argument("--match-window-hours", type=float, default=24.0)
    parser.add_argument("--event-padding-hours", type=float, default=48.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.low_quantile < args.high_quantile <= args.flood_quantile < 1:
        raise ValueError(
            "Expected 0 < low_quantile < high_quantile <= flood_quantile < 1."
        )
    if args.event_gap_hours < 0 or args.match_window_hours < 0:
        raise ValueError("Event gap and match windows must be non-negative.")
    if args.min_event_hours <= 0 or args.event_padding_hours < 0:
        raise ValueError("Minimum event hours must be positive and padding non-negative.")


def align_four_model_sources(
    h_lstm: pd.DataFrame,
    mr_stf: pd.DataFrame,
    mr_ptf: pd.DataFrame,
    d_lstm: pd.DataFrame,
    watershed: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Align all predictions to one complete-day hourly grid and reference."""
    expanded_daily = expand_daily_prediction(d_lstm, h_lstm)
    aligned = h_lstm.rename(
        columns={
            "obs_streamflow": "obs_streamflow",
            "pred_streamflow": "pred_h_lstm",
        }
    )[["timestamp", "scenario", "obs_streamflow", "pred_h_lstm"]]
    for source, prediction_name, observation_name in (
        (mr_stf, "pred_mr_stf", "obs_mr_stf"),
        (mr_ptf, "pred_mr_ptf", "obs_mr_ptf"),
    ):
        candidate = source.rename(
            columns={
                "pred_streamflow": prediction_name,
                "obs_streamflow": observation_name,
            }
        )[["timestamp", "scenario", prediction_name, observation_name]]
        aligned = aligned.merge(
            candidate,
            on=["timestamp", "scenario"],
            how="inner",
            validate="one_to_one",
        )
    aligned = aligned.merge(
        expanded_daily[
            ["timestamp", "scenario", "pred_d_lstm"]
        ],
        on=["timestamp", "scenario"],
        how="inner",
        validate="one_to_one",
    )
    numeric_columns = [
        "obs_streamflow",
        "obs_mr_stf",
        "obs_mr_ptf",
        *PREDICTION_COLUMNS.values(),
    ]
    finite = np.isfinite(aligned[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    aligned = aligned.loc[finite].sort_values(["scenario", "timestamp"]).reset_index(
        drop=True
    )
    if aligned.empty:
        raise ValueError(f"No finite common rows remain for {watershed}.")
    aligned["date"] = aligned["timestamp"].dt.floor("D")
    day_counts = aligned.groupby(["scenario", "date"]).size()
    incomplete = day_counts[day_counts != 24]
    if not incomplete.empty:
        raise ValueError(
            f"{watershed} has {len(incomplete)} common dates without exactly 24 hours."
        )

    d_native = d_lstm[["scenario", "date", "pred_streamflow", "obs_streamflow"]]
    d_check = (
        aligned.groupby(["scenario", "date"], as_index=False)
        .agg(
            expanded_prediction=("pred_d_lstm", "mean"),
            hourly_reference_mean=("obs_streamflow", "mean"),
        )
        .merge(d_native, on=["scenario", "date"], how="inner", validate="one_to_one")
    )
    mr_stf_difference = np.abs(aligned["obs_streamflow"] - aligned["obs_mr_stf"])
    mr_ptf_difference = np.abs(aligned["obs_streamflow"] - aligned["obs_mr_ptf"])
    audit = {
        "watershed": watershed,
        "watershed_name": basin_display_name(watershed),
        "common_hourly_rows": int(len(aligned)),
        "common_complete_days": int(len(day_counts)),
        "common_start": aligned["timestamp"].min().isoformat(),
        "common_end": aligned["timestamp"].max().isoformat(),
        "n_scenarios": int(aligned["scenario"].nunique()),
        "mr_stf_reference_max_abs_difference": float(mr_stf_difference.max()),
        "mr_ptf_reference_max_abs_difference": float(mr_ptf_difference.max()),
        "daily_reference_max_abs_difference": float(
            np.abs(d_check["hourly_reference_mean"] - d_check["obs_streamflow"]).max()
        ),
        "daily_expansion_max_abs_difference": float(
            np.abs(d_check["expanded_prediction"] - d_check["pred_streamflow"]).max()
        ),
        "daily_to_hourly_method": "calendar-day repeat (zero-order hold; day_shift=0)",
    }
    return aligned, audit


def evaluate_watershed(
    aligned: pd.DataFrame,
    watershed: str,
    low_quantile: float,
    high_quantile: float,
    flood_quantile: float,
    event_gap_hours: float,
    min_event_hours: float,
    match_window_hours: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], list, Dict[str, float]]:
    """Calculate overall, extreme-flow, and event metrics for one watershed."""
    observation = aligned["obs_streamflow"].to_numpy(dtype=float)
    thresholds = {
        "Q10": float(np.quantile(observation, low_quantile)),
        "Q90": float(np.quantile(observation, high_quantile)),
        "Q95": float(np.quantile(observation, flood_quantile)),
    }
    masks = {
        "Q10_low_flow_MAE": observation <= thresholds["Q10"],
        "Q90_high_flow_MAE": observation >= thresholds["Q90"],
        "Q95_flood_hour_MAE": observation >= thresholds["Q95"],
    }
    overall_rows: List[Dict[str, object]] = []
    extreme_rows: List[Dict[str, object]] = []
    observed_events = detect_events(
        aligned[["timestamp", "obs_streamflow"]],
        "obs_streamflow",
        thresholds["Q95"],
        event_gap_hours,
        min_event_hours,
    )
    for model in MODEL_ORDER:
        prediction = aligned[PREDICTION_COLUMNS[model]].to_numpy(dtype=float)
        overall_rows.append(
            {
                "watershed": watershed,
                "watershed_name": basin_display_name(watershed),
                "model": model,
                "n_hours": int(len(aligned)),
                **calculate_hydrology_metrics(prediction, observation),
            }
        )
        model_frame = pd.DataFrame(
            {
                "timestamp": aligned["timestamp"],
                "obs_streamflow": observation,
                "pred_streamflow": prediction,
            }
        )
        event_summary, _, _, _ = evaluate_events(
            model_frame,
            watershed,
            "streamflow",
            thresholds["Q95"],
            event_gap_hours,
            min_event_hours,
            match_window_hours,
        )
        extreme_rows.append(
            {
                "watershed": watershed,
                "watershed_name": basin_display_name(watershed),
                "model": model,
                "n_hours": int(len(aligned)),
                **thresholds,
                **{
                    metric: float(np.mean(np.abs(prediction[mask] - observation[mask])))
                    for metric, mask in masks.items()
                },
                "flood_event_F1_pct": float(event_summary["event_F1"] * 100.0),
                "observed_event_count": int(event_summary["observed_event_count"]),
                "predicted_event_count": int(event_summary["predicted_event_count"]),
                "matched_event_count": int(event_summary["matched_event_count"]),
            }
        )
    return overall_rows, extreme_rows, observed_events, thresholds


def validate_outputs(
    overall: pd.DataFrame,
    extreme: pd.DataFrame,
    audit: pd.DataFrame,
    watersheds: Sequence[str],
) -> None:
    expected = len(watersheds) * len(MODEL_ORDER)
    if len(overall) != expected or len(extreme) != expected:
        raise ValueError(
            f"Expected {expected} rows in each metric table; got "
            f"{len(overall)} and {len(extreme)}."
        )
    if overall.duplicated(["watershed", "model"]).any():
        raise ValueError("Duplicate overall metric rows were generated.")
    if extreme.duplicated(["watershed", "model"]).any():
        raise ValueError("Duplicate extreme metric rows were generated.")
    if (audit["common_hourly_rows"] != audit["common_complete_days"] * 24).any():
        raise ValueError("At least one watershed lacks 24 hourly rows per common day.")
    if (audit["daily_expansion_max_abs_difference"] > 1e-9).any():
        raise ValueError("Expanded D-LSTM predictions differ from native daily values.")
    for column in (
        "mr_stf_reference_max_abs_difference",
        "mr_ptf_reference_max_abs_difference",
    ):
        if (audit[column] > 0.02).any():
            raise ValueError(f"Material hourly reference mismatch detected in {column}.")
    if (audit["daily_reference_max_abs_difference"] > 0.01).any():
        raise ValueError("Daily reference does not agree with the hourly daily mean.")
    required = overall[["RMSE", "MAE", "NSE", "KGE"]].to_numpy(dtype=float)
    if not np.isfinite(required).all():
        raise ValueError("A requested overall metric is non-finite.")


def rank_formats(
    values: Mapping[str, float],
    decimals: int,
    direction: str,
    suffix: str = "",
) -> Dict[str, str]:
    """Format four model values with best bold and second-best underlined."""
    finite = [(model, float(value)) for model, value in values.items() if np.isfinite(value)]
    if direction == "lower":
        ordered = sorted(finite, key=lambda item: item[1])
    elif direction == "higher":
        ordered = sorted(finite, key=lambda item: item[1], reverse=True)
    elif direction == "closest_one":
        ordered = sorted(finite, key=lambda item: abs(item[1] - 1.0))
    else:
        raise ValueError(f"Unknown ranking direction: {direction}")
    ranks = {model: index for index, (model, _) in enumerate(ordered)}
    output = {}
    for model in values:
        value = float(values[model])
        text = "NA" if not np.isfinite(value) else f"{value:.{decimals}f}{suffix}"
        if ranks.get(model) == 0:
            text = f"**{text}**"
        elif ranks.get(model) == 1:
            text = f"<u>{text}</u>"
        output[model] = text
    return output


def summary_rank_formats(
    frame: pd.DataFrame,
    metric: str,
    decimals: int,
    direction: str,
) -> Dict[str, str]:
    means = frame.groupby("model")[metric].mean().reindex(MODEL_ORDER)
    stds = frame.groupby("model")[metric].std(ddof=1).reindex(MODEL_ORDER)
    ranked = rank_formats(means.to_dict(), decimals, direction)
    output = {}
    for model in MODEL_ORDER:
        raw_mean = f"{means[model]:.{decimals}f}"
        mean_with_markup = ranked[model].replace(raw_mean, f"{raw_mean} ± {stds[model]:.{decimals}f}")
        output[model] = mean_with_markup
    return output


def markdown_overall_table(overall: pd.DataFrame, watersheds: Sequence[str]) -> str:
    metrics = (("RMSE", 2, "lower"), ("MAE", 2, "lower"), ("NSE", 3, "higher"))
    header = ["Watershed"] + [
        f"{model} {metric}" for metric, _, _ in metrics for model in MODEL_ORDER
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * (len(header) - 1),
    ]
    for watershed in watersheds:
        basin = overall[overall["watershed"] == watershed].set_index("model")
        cells = [basin_display_name(watershed)]
        for metric, decimals, direction in metrics:
            formatted = rank_formats(basin[metric].to_dict(), decimals, direction)
            cells.extend(formatted[model] for model in MODEL_ORDER)
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["Average"]
    for metric, decimals, direction in metrics:
        means = overall.groupby("model")[metric].mean().reindex(MODEL_ORDER)
        formatted = rank_formats(means.to_dict(), decimals, direction)
        cells.extend(formatted[model] for model in MODEL_ORDER)
    lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_extreme_summary(extreme: pd.DataFrame) -> str:
    metrics = (
        ("Q10_low_flow_MAE", 2, "lower", "Q10 low-flow MAE"),
        ("Q90_high_flow_MAE", 2, "lower", "Q90 high-flow MAE"),
        ("Q95_flood_hour_MAE", 2, "lower", "Q95 flood-hour MAE"),
        ("flood_event_F1_pct", 2, "higher", "Flood-event F1 (%)"),
    )
    formatted = {
        metric: summary_rank_formats(extreme, metric, decimals, direction)
        for metric, decimals, direction, _ in metrics
    }
    lines = [
        "| Model | " + " | ".join(label for _, _, _, label in metrics) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        cells = [model] + [formatted[metric][model] for metric, _, _, _ in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_extreme_basin_table(
    extreme: pd.DataFrame, watersheds: Sequence[str]
) -> str:
    metrics = (
        ("Q10_low_flow_MAE", 2, "lower"),
        ("Q90_high_flow_MAE", 2, "lower"),
        ("Q95_flood_hour_MAE", 2, "lower"),
        ("flood_event_F1_pct", 1, "higher"),
    )
    lines = [
        "| Watershed | Model | Q10 low-flow MAE | Q90 high-flow MAE | Q95 flood-hour MAE | Flood-event F1 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for watershed in watersheds:
        basin = extreme[extreme["watershed"] == watershed].set_index("model")
        formatted = {
            metric: rank_formats(basin[metric].to_dict(), decimals, direction)
            for metric, decimals, direction in metrics
        }
        for model in MODEL_ORDER:
            cells = [
                basin_display_name(watershed),
                model,
                *[formatted[metric][model] for metric, _, _ in metrics],
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_kge_table(overall: pd.DataFrame, watersheds: Sequence[str]) -> str:
    metrics = (
        ("KGE", 3, "higher"),
        ("KGE_r", 3, "higher"),
        ("KGE_alpha", 3, "closest_one"),
        ("KGE_beta", 3, "closest_one"),
    )
    header = ["Watershed"] + [
        f"{model} {metric.replace('KGE_', '')}"
        for metric, _, _ in metrics
        for model in MODEL_ORDER
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * (len(header) - 1),
    ]
    for watershed in watersheds:
        basin = overall[overall["watershed"] == watershed].set_index("model")
        cells = [basin_display_name(watershed)]
        for metric, decimals, direction in metrics:
            formatted = rank_formats(basin[metric].to_dict(), decimals, direction)
            cells.extend(formatted[model] for model in MODEL_ORDER)
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["Average"]
    for metric, decimals, direction in metrics:
        means = overall.groupby("model")[metric].mean().reindex(MODEL_ORDER)
        formatted = rank_formats(means.to_dict(), decimals, direction)
        cells.extend(formatted[model] for model in MODEL_ORDER)
    lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_representative_flood_events_all_basins(
    aligned_by_watershed: Mapping[str, pd.DataFrame],
    observed_events: Mapping[str, list],
    thresholds: Mapping[str, Dict[str, float]],
    watersheds: Sequence[str],
    padding_hours: float,
    path: Path,
    dpi: int,
) -> None:
    """Plot one largest normalized observed flood event per watershed."""
    figure, axes = plt.subplots(4, 3, figsize=(16.5, 14.0), squeeze=False)
    padding = pd.Timedelta(hours=padding_hours)
    for axis, watershed in zip(axes.flat, watersheds):
        frame = aligned_by_watershed[watershed]
        events = observed_events[watershed]
        threshold = thresholds[watershed]["Q95"]
        if not events:
            axis.text(0.5, 0.5, "No observed Q95 event", ha="center", va="center")
            axis.set_axis_off()
            continue
        event = max(events, key=lambda item: item.peak_flow / threshold)
        start = event.start - padding
        end = event.end + padding
        window = frame.loc[frame["timestamp"].between(start, end)]
        axis.plot(
            window["timestamp"],
            window["obs_streamflow"],
            color=REFERENCE_COLOR,
            linewidth=1.55,
            label="Reference",
            zorder=4,
        )
        for model in MODEL_ORDER:
            color, _, linestyle = MODEL_STYLES[model]
            axis.plot(
                window["timestamp"],
                window[PREDICTION_COLUMNS[model]],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                alpha=0.90,
                drawstyle="steps-post" if model == "D-LSTM" else "default",
                label=model,
                zorder=3,
            )
        axis.axhline(
            threshold,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            label="Observed Q95",
        )
        axis.axvspan(event.start, event.end, color="#9BB8DA", alpha=0.14)
        axis.set_title(
            f"{basin_display_name(watershed)} | peak = "
            f"{event.peak_flow / threshold:.1f} × Q95",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    for axis in axes.flat[len(watersheds) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
    )
    figure.suptitle(
        "Largest observed flood event in each watershed",
        x=0.045,
        y=0.992,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.045,
        0.963,
        "All models use common hourly timestamps and reference; D-LSTM is repeated within each calendar day",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.supylabel("Streamflow (cfs)", x=0.012, fontsize=10)
    figure.supxlabel("Date", y=0.012, fontsize=10)
    figure.tight_layout(rect=(0.025, 0.03, 0.995, 0.925), h_pad=1.6, w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def format_metric(value: float, decimals: int, grouping: bool = False) -> str:
    """Format a finite metric for a compact reconstruction-figure legend."""
    if not np.isfinite(value):
        return "N/A"
    grouping_code = "," if grouping else ""
    return f"{value:{grouping_code}.{decimals}f}"


def plot_basin_reconstruction_comparison(
    watershed: str,
    frame: pd.DataFrame,
    basin_metrics: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot the complete common test period for one watershed and four models."""
    figure, axis = plt.subplots(figsize=(15, 5.75))
    axis.scatter(
        frame["timestamp"],
        frame["obs_streamflow"],
        color="#303030",
        s=4,
        linewidths=0,
        alpha=0.24,
        label="Observed",
        rasterized=True,
        zorder=2,
    )

    metrics_by_model = basin_metrics.set_index("model")
    for model in MODEL_ORDER:
        color, _, linestyle = MODEL_STYLES[model]
        metrics = metrics_by_model.loc[model]
        label = (
            f"{model}   RMSE = {format_metric(metrics['RMSE'], 1, grouping=True)}"
            f"   |   NSE = {format_metric(metrics['NSE'], 3)}"
            f"   |   KGE = {format_metric(metrics['KGE'], 3)}"
        )
        axis.plot(
            frame["timestamp"],
            frame[PREDICTION_COLUMNS[model]],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            alpha=0.84,
            drawstyle="steps-post" if model == "D-LSTM" else "default",
            solid_capstyle="round",
            dash_capstyle="round",
            label=label,
            zorder=3,
        )

    coverage_start = frame["timestamp"].min()
    coverage_end = frame["timestamp"].max()
    axis.set_title(
        f"{basin_display_name(watershed)} Basin: Observed and Model-Predicted Hourly Streamflow",
        loc="center",
        fontsize=15,
        fontweight="semibold",
        pad=38,
    )
    axis.text(
        0.5,
        1.025,
        f"Common test period: {coverage_start:%b %d, %Y} to {coverage_end:%b %d, %Y}",
        transform=axis.transAxes,
        color="#555555",
        fontsize=10.5,
        ha="center",
        va="bottom",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Streamflow (cfs)")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.7, alpha=0.72)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.55, alpha=0.45)
    axis.margins(x=0)
    values = frame[
        ["obs_streamflow", *[PREDICTION_COLUMNS[model] for model in MODEL_ORDER]]
    ]
    if values.min().min() >= 0:
        axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.tick_params(axis="both", colors="#444444", labelsize=9)
    axis.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: f"{value:,.1f}" if 0 < abs(value) < 10 else f"{value:,.0f}"
        )
    )
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.92,
        fontsize=8.7,
        markerscale=2.8,
        ncol=1,
        title="Performance over displayed period",
        title_fontsize=9.2,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_all_reconstruction_comparisons(
    aligned_by_watershed: Mapping[str, pd.DataFrame],
    overall: pd.DataFrame,
    watersheds: Sequence[str],
    output_dir: Path,
    method: str,
    dpi: int,
) -> List[Path]:
    """Create one full-period comparison figure for every watershed."""
    reconstruction_dir = output_dir / "reconstructions"
    paths: List[Path] = []
    for watershed in watersheds:
        path = reconstruction_dir / f"{watershed}_{method}_comparison.png"
        plot_basin_reconstruction_comparison(
            watershed,
            aligned_by_watershed[watershed],
            overall.loc[overall["watershed"] == watershed],
            path,
            dpi,
        )
        paths.append(path)
    return paths


def write_markdown(
    path: Path,
    overall: pd.DataFrame,
    extreme: pd.DataFrame,
    audit: pd.DataFrame,
    watersheds: Sequence[str],
    args: argparse.Namespace,
) -> None:
    common_start = pd.to_datetime(audit["common_start"]).max()
    common_end = pd.to_datetime(audit["common_end"]).min()
    common_hours = int(audit["common_hourly_rows"].min())
    common_days = int(audit["common_complete_days"].min())
    text = f"""# Four-model test-period performance tables

## Methods

All reported values were recalculated on the exact intersection of D-LSTM, H-LSTM, MR-STF, and MR-PTF timestamps: **{common_start:%Y-%m-%d %H:%M} through {common_end:%Y-%m-%d %H:%M}**, comprising **{common_hours:,} hourly samples ({common_days:,} complete days) per watershed**. The shared reference is hourly `obs_streamflow` from `{EXPERIMENTS['H-LSTM']}`. MR-STF and MR-PTF use their leakage-controlled one-day-shift experiments. Native daily D-LSTM predictions from `{EXPERIMENTS['D-LSTM']}` are assigned unchanged to all 24 hours of the corresponding calendar day using `interpolate_daily_imv_to_hourly(..., day_shift=0)`.

Because the original pasted tables were not uniformly restricted to this four-model intersection, small differences from their previously reported H-LSTM/MR values are expected. Recalculating every model on the shared grid makes the rankings paired and directly comparable. Bold indicates the best model and underline the second best within a watershed/metric. RMSE and MAE are lower when better; NSE, KGE, correlation, and F1 are higher when better; KGE variability and bias ratios are ranked by closeness to 1.

## Table 1. Overall hourly performance

RMSE and MAE are in cfs. The average is the unweighted mean across {len(watersheds)} watersheds.

{markdown_overall_table(overall, watersheds)}

## Table 2. Extreme-flow summary across watersheds

Values are unweighted basin mean ± sample standard deviation. Conditional MAE is in cfs and flood-event F1 is in percent. Q10, Q90, and Q95 thresholds are calculated separately for each watershed from the shared hourly reference.

{markdown_extreme_summary(extreme)}

## Table 3. Basin-level extreme-flow performance

Events require at least {args.min_event_hours:g} exceedance-hours, bridge gaps up to {args.event_gap_hours:g} h, and are matched one-to-one by overlap or peak separation within ±{args.match_window_hours:g} h.

{markdown_extreme_basin_table(extreme, watersheds)}

## Table 4. KGE and components

`r` is Pearson correlation, `alpha` is the predicted-to-reference standard-deviation ratio, and `beta` is the predicted-to-reference mean ratio.

{markdown_kge_table(overall, watersheds)}

## Figure caption

**Representative flood events across all watersheds.** Reference hourly streamflow and predictions from D-LSTM, H-LSTM, MR-STF, and MR-PTF during the largest normalized observed flood event in each watershed. Events are defined from the watershed-specific observed Q95 threshold; shading marks the observed event and the horizontal dotted line marks Q95. H-LSTM, MR-STF, and MR-PTF are native-hourly predictions. D-LSTM is shown by assigning each native daily prediction to all 24 hours of the corresponding calendar day and therefore does not represent inferred intraday dynamics. All series use identical common test timestamps.

**Watershed reconstruction comparisons.** Each watershed figure shows the complete shared test-period reference and predictions in the order D-LSTM, H-LSTM, MR-STF, and MR-PTF. Legend labels report paired RMSE, NSE, and KGE over the displayed period. The stepped D-LSTM curve repeats each native daily prediction across its corresponding 24 hourly timestamps.

## Reproducibility

```bash
conda run --no-capture-output -n imerg_era5 python compare_four_model_test_performance.py
```

Exact unrounded values are saved alongside this file in `overall_metrics_by_basin.csv`, `extreme_metrics_by_basin.csv`, and `alignment_audit.csv`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows: List[Dict[str, object]] = []
    extreme_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    aligned_by_watershed: Dict[str, pd.DataFrame] = {}
    events_by_watershed: Dict[str, list] = {}
    thresholds_by_watershed: Dict[str, Dict[str, float]] = {}

    for watershed in args.watersheds:
        paths = {
            model: reconstruction_path(
                args.experiments_dir,
                experiment,
                args.results_subdir,
                watershed,
                args.split,
                args.method,
            )
            for model, experiment in EXPERIMENTS.items()
        }
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing source files for {watershed}: {', '.join(map(str, missing))}"
            )
        h_lstm = load_hourly_reconstruction(paths["H-LSTM"])
        mr_stf = load_hourly_reconstruction(paths["MR-STF"])
        mr_ptf = load_hourly_reconstruction(paths["MR-PTF"])
        d_lstm = load_daily_reconstruction(paths["D-LSTM"])
        aligned, audit = align_four_model_sources(
            h_lstm, mr_stf, mr_ptf, d_lstm, watershed
        )
        basin_overall, basin_extreme, events, thresholds = evaluate_watershed(
            aligned,
            watershed,
            args.low_quantile,
            args.high_quantile,
            args.flood_quantile,
            args.event_gap_hours,
            args.min_event_hours,
            args.match_window_hours,
        )
        overall_rows.extend(basin_overall)
        extreme_rows.extend(basin_extreme)
        audit_rows.append(audit)
        aligned_by_watershed[watershed] = aligned
        events_by_watershed[watershed] = events
        thresholds_by_watershed[watershed] = thresholds

    overall = pd.DataFrame(overall_rows)
    extreme = pd.DataFrame(extreme_rows)
    audit = pd.DataFrame(audit_rows)
    validate_outputs(overall, extreme, audit, args.watersheds)

    overall_path = output_dir / "overall_metrics_by_basin.csv"
    extreme_path = output_dir / "extreme_metrics_by_basin.csv"
    audit_path = output_dir / "alignment_audit.csv"
    markdown_path = output_dir / "four_model_test_tables.md"
    figure_path = output_dir / "model_representative_flood_events_all_basins.png"
    overall.to_csv(overall_path, index=False, float_format="%.10g")
    extreme.to_csv(extreme_path, index=False, float_format="%.10g")
    audit.to_csv(audit_path, index=False, float_format="%.10g")
    write_markdown(markdown_path, overall, extreme, audit, args.watersheds, args)
    plot_representative_flood_events_all_basins(
        aligned_by_watershed,
        events_by_watershed,
        thresholds_by_watershed,
        args.watersheds,
        args.event_padding_hours,
        figure_path,
        args.dpi,
    )
    reconstruction_paths = plot_all_reconstruction_comparisons(
        aligned_by_watershed,
        overall,
        args.watersheds,
        output_dir,
        args.method,
        args.dpi,
    )
    print(
        f"Compared {len(MODEL_ORDER)} models across {len(args.watersheds)} watersheds "
        f"and {len(overall)} basin/model rows."
    )
    print(f"Tables: {markdown_path}")
    print(f"Overall CSV: {overall_path}")
    print(f"Extreme CSV: {extreme_path}")
    print(f"Audit CSV: {audit_path}")
    print(f"Figure: {figure_path}")
    print(
        f"Reconstruction figures: {len(reconstruction_paths)} files in "
        f"{output_dir / 'reconstructions'}"
    )


if __name__ == "__main__":
    main()
