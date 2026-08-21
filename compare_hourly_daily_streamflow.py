#!/usr/bin/env python3
"""Compare hourly H-LSTM/MR-STF and daily D-LSTM streamflow predictions.

The comparison uses reconstructed test results from:

* ``hourly_global_streamflow_no_IMVs`` (H-LSTM), and
* ``hourly_global_streamflow_pred_streamflow_day_shift_1`` (MR-STF), and
* ``daily_global_streamflow`` (D-LSTM).

MR-STF is evaluated at its native hourly resolution. D-LSTM predictions are
expanded to the hourly grid with the project's existing
``interpolate_daily_imv_to_hourly`` function. That function performs a
calendar-day repeat (zero-order hold): one daily prediction is assigned to all
24 hours of its date. Both models are then evaluated against the *same* hourly
``obs_streamflow`` series from the H-LSTM reconstruction file and on the exact
same timestamps.

Because an hourly-grid comparison structurally tests subdaily behavior that a
daily model cannot express, the script also evaluates all three models after
calendar-day averaging. Additional diagnostics include Q95 high-flow MAE and
within-day anomaly RMSE/correlation.

Outputs
-------
``tables/hourly_daily_metrics_by_basin.csv``
    RMSE, NSE, KGE, supporting metrics, and Q95 MAE for all models at hourly
    and daily-mean evaluation scales.
``tables/hourly_daily_metric_summary.csv``
    Unweighted watershed mean and sample standard deviation for each metric.
``tables/hourly_daily_intraday_metrics.csv``
    Skill at reproducing within-day deviations from each day's mean.
``tables/hourly_daily_alignment_audit.csv``
    Timestamp coverage, complete-day counts, and daily-reference agreement.
``plots/hourly_daily_test_timeseries_all_basins.*``
    Full test-period hydrographs in a 4 x 3 layout. Each panel reports hourly
    RMSE, NSE, and KGE for all models.
``plots/hourly_daily_peak_event_zooms_all_basins.*``
    Basin-specific 14-day windows around the largest observed hourly peak.
``plots/hourly_daily_paired_performance.*``
    Paired basin comparisons of RMSE, NSE, and KGE on both evaluation grids.
``plots/hourly_daily_intraday_anomaly_rmse.*``
    Paired comparison of within-day anomaly RMSE.
``comparison_notes.md``
    Methods, interpretation guidance, and suggested paper captions.

Run from the repository root::

    conda run --no-capture-output -n imerg_era5 \
        python compare_hourly_daily_streamflow.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from combine_daily_imv_outputs_hourly_streamflow_hourly_eddy import (
    interpolate_daily_imv_to_hourly,
    load_imv_daily,
)
from hydrology_metrics import calculate_hydrology_metrics


DEFAULT_HOURLY_EXPERIMENT = "hourly_global_streamflow_no_IMVs"
DEFAULT_MR_STF_EXPERIMENT = "hourly_global_streamflow_pred_streamflow_day_shift_1"
DEFAULT_DAILY_EXPERIMENT = "daily_global_streamflow"
DEFAULT_OUTPUT_EXPERIMENT = "hourly_daily_streamflow_comparison"
DEFAULT_WATERSHEDS: Tuple[str, ...] = (
    "BlueEarth",
    "Cloquet",
    "KettleRiverModels",
    "LeSueur",
    "LittleFork",
    "SFCrow",
    "Sauk",
    "SnakeSE",
    "TwoRivers",
    "Watonwan",
    "WildRiceMarsh",
    "Zumbro",
)
MODEL_ORDER: Tuple[str, ...] = ("H-LSTM", "MR-STF", "D-LSTM")
MODEL_STYLES: Mapping[str, Tuple[str, str, object]] = {
    "H-LSTM": ("#0072B2", "o", "-"),
    "MR-STF": ("#D55E00", "s", (0, (4, 1.8))),
    "D-LSTM": ("#7A7F00", "D", (0, (6, 2.5))),
}
REFERENCE_COLOR = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hourly H-LSTM and MR-STF streamflow with daily D-LSTM "
            "after repeating each daily prediction over its 24 hourly timestamps."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing the two experiment folders.",
    )
    parser.add_argument(
        "--hourly-experiment",
        default=DEFAULT_HOURLY_EXPERIMENT,
        help=f"Hourly experiment folder (default: {DEFAULT_HOURLY_EXPERIMENT}).",
    )
    parser.add_argument(
        "--mr-stf-experiment",
        default=DEFAULT_MR_STF_EXPERIMENT,
        help=f"MR-STF experiment folder (default: {DEFAULT_MR_STF_EXPERIMENT}).",
    )
    parser.add_argument(
        "--daily-experiment",
        default=DEFAULT_DAILY_EXPERIMENT,
        help=f"Daily experiment folder (default: {DEFAULT_DAILY_EXPERIMENT}).",
    )
    parser.add_argument(
        "--results-subdir",
        default="test_results",
        help="Reconstruction subdirectory within each experiment.",
    )
    parser.add_argument("--split", default="test", help="Split to compare (default: test).")
    parser.add_argument(
        "--method",
        default="latest",
        help="Reconstruction method suffix (default: latest).",
    )
    parser.add_argument(
        "--watersheds",
        nargs="+",
        default=DEFAULT_WATERSHEDS,
        help="Watersheds to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to experiments/"
            f"{DEFAULT_OUTPUT_EXPERIMENT}/test_results."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Figure formats to write (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--peak-window-days",
        type=int,
        default=7,
        help="Days before and after the observed peak in zoom figures.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Continue with watersheds that have both source files.",
    )
    return parser.parse_args()


def basin_display_name(basin: str) -> str:
    """Convert an internal watershed identifier to a paper-facing label."""
    overrides = {
        "KettleRiverModels": "Kettle River",
        "SFCrow": "SF Crow",
        "SnakeSE": "Snake SE",
        "WildRiceMarsh": "Wild Rice Marsh",
    }
    if basin in overrides:
        return overrides[basin]
    name = basin.replace("_", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return name.removesuffix(" Models")


def reconstruction_path(
    experiments_dir: Path,
    experiment: str,
    results_subdir: str,
    watershed: str,
    split: str,
    method: str,
) -> Path:
    return (
        experiments_dir
        / experiment
        / results_subdir
        / f"{watershed}_{split}_reconstructed_{method}.csv"
    )


def load_hourly_reconstruction(path: Path) -> pd.DataFrame:
    """Load and validate an hourly streamflow reconstruction table."""
    frame = pd.read_csv(path)
    required = {"timestamp", "pred_streamflow", "obs_streamflow", "scenario"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    frame = frame.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path} contains invalid timestamps.")
    if frame.duplicated(["timestamp", "scenario"]).any():
        raise ValueError(f"{path} contains duplicate timestamp/scenario rows.")
    frame["pred_streamflow"] = pd.to_numeric(frame["pred_streamflow"], errors="coerce")
    frame["obs_streamflow"] = pd.to_numeric(frame["obs_streamflow"], errors="coerce")
    return frame.sort_values(["scenario", "timestamp"]).reset_index(drop=True)


def load_daily_reconstruction(path: Path) -> pd.DataFrame:
    """Load a daily reconstruction using the existing IMV pipeline helper."""
    frame = load_imv_daily(path)
    required = {"timestamp", "date", "pred_streamflow", "obs_streamflow", "scenario"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    if frame.duplicated(["date", "scenario"]).any():
        raise ValueError(f"{path} contains duplicate date/scenario rows.")
    frame["pred_streamflow"] = pd.to_numeric(frame["pred_streamflow"], errors="coerce")
    frame["obs_streamflow"] = pd.to_numeric(frame["obs_streamflow"], errors="coerce")
    return frame.sort_values(["scenario", "timestamp"]).reset_index(drop=True)


def expand_daily_prediction(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Reuse the project calendar-day-repeat method for every shared scenario."""
    shared_scenarios = sorted(
        set(hourly["scenario"].dropna()).intersection(daily["scenario"].dropna())
    )
    if not shared_scenarios:
        raise ValueError("The hourly and daily files do not share a scenario.")

    expanded_frames = []
    for scenario in shared_scenarios:
        hourly_scenario = hourly.loc[hourly["scenario"] == scenario]
        daily_scenario = daily.loc[daily["scenario"] == scenario]
        expanded_frames.append(
            interpolate_daily_imv_to_hourly(
                daily_imv=daily_scenario,
                hourly_timestamps=hourly_scenario["timestamp"],
                scenario=str(scenario),
                day_shift=0,
            )
        )
    expanded = pd.concat(expanded_frames, ignore_index=True).rename(
        columns={"Datetime": "timestamp", "pred_streamflow": "pred_d_lstm"}
    )
    if expanded.duplicated(["timestamp", "scenario"]).any():
        raise ValueError("Daily-to-hourly expansion produced duplicate timestamps.")
    return expanded.sort_values(["scenario", "timestamp"]).reset_index(drop=True)


def align_basin_sources(
    hourly: pd.DataFrame,
    mr_stf: pd.DataFrame,
    daily: pd.DataFrame,
    watershed: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Create common three-model grids and audit reference agreement."""
    expanded = expand_daily_prediction(daily, hourly)
    hourly_common = hourly.rename(
        columns={
            "pred_streamflow": "pred_h_lstm",
            "obs_streamflow": "obs_hourly",
        }
    ).merge(
        mr_stf.rename(
            columns={
                "pred_streamflow": "pred_mr_stf",
                "obs_streamflow": "obs_mr_stf",
            }
        )[["timestamp", "scenario", "pred_mr_stf", "obs_mr_stf"]],
        on=["timestamp", "scenario"],
        how="inner",
        validate="one_to_one",
    ).merge(
        expanded[["timestamp", "scenario", "pred_d_lstm"]],
        on=["timestamp", "scenario"],
        how="inner",
        validate="one_to_one",
    )
    hourly_common["date"] = hourly_common["timestamp"].dt.floor("D")
    finite = np.isfinite(
        hourly_common[
            ["obs_hourly", "obs_mr_stf", "pred_h_lstm", "pred_mr_stf", "pred_d_lstm"]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    hourly_common = hourly_common.loc[finite].copy()
    if hourly_common.empty:
        raise ValueError(f"No finite common hourly rows for {watershed}.")

    counts = hourly_common.groupby(["scenario", "date"]).size()
    incomplete_days = counts[counts != 24]
    if not incomplete_days.empty:
        raise ValueError(
            f"{watershed} has {len(incomplete_days)} common dates without exactly 24 hours."
        )

    daily_common = (
        hourly_common.groupby(["scenario", "date"], as_index=False)
        .agg(
            obs_streamflow=("obs_hourly", "mean"),
            pred_h_lstm=("pred_h_lstm", "mean"),
            pred_mr_stf=("pred_mr_stf", "mean"),
            pred_d_lstm=("pred_d_lstm", "mean"),
            n_hours=("timestamp", "size"),
        )
        .sort_values(["scenario", "date"])
        .reset_index(drop=True)
    )

    native_daily = daily[["scenario", "date", "obs_streamflow", "pred_streamflow"]]
    reference_check = daily_common.merge(
        native_daily,
        on=["scenario", "date"],
        how="inner",
        validate="one_to_one",
        suffixes=("_hourly_mean", "_daily_native"),
    )
    obs_difference = np.abs(
        reference_check["obs_streamflow_hourly_mean"]
        - reference_check["obs_streamflow_daily_native"]
    )
    pred_difference = np.abs(
        reference_check["pred_d_lstm"] - reference_check["pred_streamflow"]
    )
    mr_reference_difference = np.abs(
        hourly_common["obs_hourly"] - hourly_common["obs_mr_stf"]
    )
    audit = {
        "watershed": watershed,
        "watershed_name": basin_display_name(watershed),
        "hourly_source_rows": int(len(hourly)),
        "mr_stf_source_rows": int(len(mr_stf)),
        "daily_source_rows": int(len(daily)),
        "common_hourly_rows": int(len(hourly_common)),
        "common_complete_days": int(len(daily_common)),
        "common_start": hourly_common["timestamp"].min().isoformat(),
        "common_end": hourly_common["timestamp"].max().isoformat(),
        "n_scenarios": int(hourly_common["scenario"].nunique()),
        "daily_reference_mean_abs_difference": float(obs_difference.mean()),
        "daily_reference_max_abs_difference": float(obs_difference.max()),
        "daily_prediction_expansion_max_abs_difference": float(pred_difference.max()),
        "mr_stf_reference_mean_abs_difference": float(mr_reference_difference.mean()),
        "mr_stf_reference_max_abs_difference": float(mr_reference_difference.max()),
        "daily_to_hourly_method": "calendar-day repeat (zero-order hold)",
    }
    return hourly_common, daily_common, audit


def percent_bias(prediction: np.ndarray, observation: np.ndarray) -> float:
    denominator = float(np.sum(observation))
    if denominator == 0.0:
        return float("nan")
    return float(100.0 * np.sum(prediction - observation) / denominator)


def metric_record(
    observation: np.ndarray,
    prediction: np.ndarray,
    watershed: str,
    model: str,
    evaluation_scale: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    """Compute the requested metrics and supporting high-flow diagnostics."""
    observation = np.asarray(observation, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(observation) & np.isfinite(prediction)
    observation = observation[finite]
    prediction = prediction[finite]
    if observation.size < 2:
        raise ValueError(f"Insufficient finite values for {watershed}/{model}.")
    metrics = calculate_hydrology_metrics(prediction, observation)
    q95 = float(np.quantile(observation, 0.95))
    high_flow = observation >= q95
    row: Dict[str, object] = {
        "watershed": watershed,
        "watershed_name": basin_display_name(watershed),
        "evaluation_scale": evaluation_scale,
        "model": model,
        "n_samples": int(observation.size),
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        "observed_Q95": q95,
        "Q95_MAE": float(np.mean(np.abs(prediction[high_flow] - observation[high_flow]))),
        "PBIAS_pct": percent_bias(prediction, observation),
    }
    row.update(metrics)
    return row


def evaluate_basin(
    watershed: str,
    hourly_common: pd.DataFrame,
    daily_common: pd.DataFrame,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Evaluate all three models on common hourly and daily-mean grids."""
    metric_rows = []
    for model, prediction_column in (
        ("H-LSTM", "pred_h_lstm"),
        ("MR-STF", "pred_mr_stf"),
        ("D-LSTM", "pred_d_lstm"),
    ):
        metric_rows.append(
            metric_record(
                hourly_common["obs_hourly"].to_numpy(),
                hourly_common[prediction_column].to_numpy(),
                watershed,
                model,
                "hourly",
                hourly_common["timestamp"].min(),
                hourly_common["timestamp"].max(),
            )
        )
        metric_rows.append(
            metric_record(
                daily_common["obs_streamflow"].to_numpy(),
                daily_common[prediction_column].to_numpy(),
                watershed,
                model,
                "daily_mean",
                daily_common["date"].min(),
                daily_common["date"].max(),
            )
        )

    intraday = hourly_common.copy()
    group_keys = ["scenario", "date"]
    intraday["obs_anomaly"] = intraday["obs_hourly"] - intraday.groupby(group_keys)[
        "obs_hourly"
    ].transform("mean")
    intraday_rows = []
    for model, prediction_column in (
        ("H-LSTM", "pred_h_lstm"),
        ("MR-STF", "pred_mr_stf"),
        ("D-LSTM", "pred_d_lstm"),
    ):
        anomaly = intraday[prediction_column] - intraday.groupby(group_keys)[
            prediction_column
        ].transform("mean")
        observed_anomaly = intraday["obs_anomaly"].to_numpy(dtype=float)
        predicted_anomaly = anomaly.to_numpy(dtype=float)
        obs_std = float(np.std(observed_anomaly))
        pred_std = float(np.std(predicted_anomaly))
        # Calendar-day repetition should have exactly zero within-day variance.
        # Floating-point subtraction can leave numerical noise (~1e-13), which
        # must not be reported as a meaningful correlation.
        variability_tolerance = max(1.0, obs_std) * 1.0e-10
        correlation = (
            float(np.corrcoef(observed_anomaly, predicted_anomaly)[0, 1])
            if obs_std > variability_tolerance
            and pred_std > variability_tolerance
            else float("nan")
        )
        intraday_rows.append(
            {
                "watershed": watershed,
                "watershed_name": basin_display_name(watershed),
                "model": model,
                "n_hours": int(len(intraday)),
                "intraday_anomaly_RMSE": float(
                    np.sqrt(np.mean((predicted_anomaly - observed_anomaly) ** 2))
                ),
                "intraday_anomaly_correlation": correlation,
                "observed_intraday_std": obs_std,
                "predicted_intraday_std": pred_std,
            }
        )
    return metric_rows, intraday_rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize each performance metric across watersheds."""
    rows = []
    for (scale, model), group in metrics.groupby(
        ["evaluation_scale", "model"], sort=False
    ):
        for metric in ("RMSE", "NSE", "KGE", "MAE", "PBIAS_pct", "Q95_MAE"):
            values = group[metric].dropna()
            rows.append(
                {
                    "evaluation_scale": scale,
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "n_watersheds": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.7,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "path.simplify": True,
            "path.simplify_threshold": 0.15,
        }
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        output = base_path.with_suffix(f".{extension}")
        figure.savefig(output, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        outputs.append(output)
    return outputs


def format_rmse(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.0f}" if abs(value) >= 100.0 else f"{value:.1f}"


def metric_annotation(metric_lookup: pd.DataFrame, watershed: str) -> str:
    """Build the three-line RMSE/NSE/KGE label required for each basin panel."""
    subset = metric_lookup.loc[
        (metric_lookup["watershed"] == watershed)
        & (metric_lookup["evaluation_scale"] == "hourly")
    ].set_index("model")
    lines = []
    for model in MODEL_ORDER:
        row = subset.loc[model]
        lines.append(
            f"{model}: RMSE {format_rmse(row['RMSE'])} | "
            f"NSE {row['NSE']:.2f} | KGE {row['KGE']:.2f}"
        )
    return "\n".join(lines)


def plot_full_timeseries(
    aligned: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    watersheds: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot full common test-period hydrographs for all watersheds."""
    figure, axes = plt.subplots(4, 3, figsize=(18, 13.5), sharex=True)
    for axis, watershed in zip(axes.flat, watersheds):
        frame = aligned[watershed]
        axis.plot(
            frame["timestamp"],
            frame["obs_hourly"],
            color=REFERENCE_COLOR,
            linewidth=0.65,
            alpha=0.70,
            label="Reference",
            rasterized=True,
        )
        for model, column in (
            ("H-LSTM", "pred_h_lstm"),
            ("MR-STF", "pred_mr_stf"),
            ("D-LSTM", "pred_d_lstm"),
        ):
            color, _, linestyle = MODEL_STYLES[model]
            axis.plot(
                frame["timestamp"],
                frame[column],
                color=color,
                linewidth=0.70,
                linestyle=linestyle,
                alpha=0.76,
                label=model,
                rasterized=True,
            )
        axis.set_title(basin_display_name(watershed), loc="left", fontweight="bold")
        axis.text(
            0.012,
            0.97,
            metric_annotation(metrics, watershed),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.1,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(labelsize=8)

    for axis in axes.flat[len(watersheds) :]:
        axis.set_visible(False)

    for axis in axes[-1, :]:
        axis.set_xlabel("Date")
    figure.supylabel("Streamflow (native model-output units)", x=0.012, fontsize=10)
    handles = [plt.Line2D([], [], color=REFERENCE_COLOR, linewidth=1.5, label="Reference")]
    for model in MODEL_ORDER:
        color, _, linestyle = MODEL_STYLES[model]
        handles.append(
            plt.Line2D([], [], color=color, linewidth=1.5, linestyle=linestyle, label=model)
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Hourly and daily streamflow predictions during the test period",
        x=0.04,
        y=0.987,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.958,
        "D-LSTM is repeated over each calendar day's 24 hours; labels use the common hourly grid",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.025, 0.025, 0.995, 0.925), h_pad=1.3, w_pad=1.0)
    outputs = save_figure(
        figure,
        output_dir / "hourly_daily_test_timeseries_all_basins",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def plot_peak_event_zooms(
    aligned: Mapping[str, pd.DataFrame],
    watersheds: Sequence[str],
    window_days: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot short windows around each basin's largest observed hourly peak."""
    figure, axes = plt.subplots(4, 3, figsize=(18, 13.5))
    for axis, watershed in zip(axes.flat, watersheds):
        frame = aligned[watershed]
        peak_index = frame["obs_hourly"].idxmax()
        peak_time = pd.Timestamp(frame.loc[peak_index, "timestamp"])
        start = peak_time - pd.Timedelta(days=window_days)
        end = peak_time + pd.Timedelta(days=window_days)
        window = frame.loc[frame["timestamp"].between(start, end)]
        axis.plot(
            window["timestamp"],
            window["obs_hourly"],
            color=REFERENCE_COLOR,
            linewidth=1.2,
            alpha=0.85,
            label="Reference",
        )
        for model, column in (
            ("H-LSTM", "pred_h_lstm"),
            ("MR-STF", "pred_mr_stf"),
            ("D-LSTM", "pred_d_lstm"),
        ):
            color, _, linestyle = MODEL_STYLES[model]
            axis.plot(
                window["timestamp"],
                window[column],
                color=color,
                linewidth=1.15,
                linestyle=linestyle,
                alpha=0.85,
                label=model,
            )
        axis.axvline(peak_time, color="#777777", linewidth=0.7, linestyle=(0, (3, 3)))
        axis.set_title(
            f"{basin_display_name(watershed)} | peak {peak_time:%Y-%m-%d %H:%M}",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, window_days // 3)))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    for axis in axes.flat[len(watersheds) :]:
        axis.set_visible(False)
    figure.supylabel("Streamflow (native model-output units)", x=0.012, fontsize=10)
    figure.supxlabel("Date", y=0.012, fontsize=10)
    handles = [plt.Line2D([], [], color=REFERENCE_COLOR, linewidth=1.6, label="Reference")]
    for model in MODEL_ORDER:
        color, _, linestyle = MODEL_STYLES[model]
        handles.append(
            plt.Line2D([], [], color=color, linewidth=1.6, linestyle=linestyle, label=model)
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.946),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Hourly behavior around each watershed's largest observed test-period peak",
        x=0.04,
        y=0.987,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.958,
        f"Basin-specific ±{window_days}-day windows; D-LSTM steps reflect calendar-day repetition",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.025, 0.035, 0.995, 0.925), h_pad=1.6, w_pad=1.0)
    outputs = save_figure(
        figure,
        output_dir / "hourly_daily_peak_event_zooms_all_basins",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def metric_axis_limits(metrics: pd.DataFrame, metric: str) -> Tuple[float, float]:
    values = metrics[metric].dropna().to_numpy(dtype=float)
    if metric == "RMSE":
        return 0.0, float(values.max() * 1.08)
    lower = min(-0.05, float(values.min() - 0.06))
    return lower, 1.03


def plot_paired_performance(
    metrics: pd.DataFrame,
    watersheds: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot paired basin metrics at hourly and daily-mean evaluation scales."""
    scales = ("hourly", "daily_mean")
    metric_names = ("RMSE", "NSE", "KGE")
    figure, axes = plt.subplots(2, 3, figsize=(17, 11), sharey=True)

    for row_index, scale in enumerate(scales):
        scale_data = metrics.loc[metrics["evaluation_scale"] == scale]
        for column_index, metric in enumerate(metric_names):
            axis = axes[row_index, column_index]
            pivot = scale_data.pivot(index="watershed", columns="model", values=metric).reindex(
                watersheds
            )
            positions = np.arange(len(watersheds))
            for position, watershed in enumerate(watersheds):
                values = pivot.loc[watershed]
                if values.notna().all():
                    axis.plot(
                        [values[model] for model in MODEL_ORDER],
                        [position] * len(MODEL_ORDER),
                        color="#9A9A9A",
                        linewidth=0.75,
                        alpha=0.55,
                        zorder=1,
                    )
            for model in MODEL_ORDER:
                color, marker, _ = MODEL_STYLES[model]
                axis.scatter(
                    pivot[model],
                    positions,
                    color=color,
                    marker=marker,
                    s=38,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.5,
                    label=model,
                    zorder=3,
                )
            axis.set_xlim(*metric_axis_limits(metrics, metric))
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(
                f"{metric} ({'lower' if metric == 'RMSE' else 'higher'} is better)",
                loc="left",
                fontweight="bold",
            )
            if metric in {"NSE", "KGE"}:
                axis.axvline(0.0, color="#777777", linewidth=0.7, linestyle=(0, (3, 3)))
                axis.axvline(1.0, color="#222222", linewidth=0.8, linestyle=(0, (5, 3)))
            axis.set_yticks(positions)
            if column_index == 0:
                axis.set_yticklabels([basin_display_name(value) for value in watersheds])
            else:
                axis.tick_params(labelleft=False)
            axis.invert_yaxis()
            if row_index == 1:
                axis.set_xlabel(metric if metric != "RMSE" else "RMSE (native units)")
        axes[row_index, 0].set_ylabel(
            "Hourly evaluation" if scale == "hourly" else "Daily-mean evaluation"
        )

    handles = []
    for model in MODEL_ORDER:
        color, marker, _ = MODEL_STYLES[model]
        handles.append(
            plt.Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="none",
                markersize=7,
                label=model,
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Paired streamflow performance across temporal evaluation scales",
        x=0.04,
        y=0.987,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.958,
        "Each grey line connects H-LSTM, MR-STF, and D-LSTM for the same watershed",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.025, 0.025, 0.995, 0.925), h_pad=2.0, w_pad=1.3)
    outputs = save_figure(
        figure,
        output_dir / "hourly_daily_paired_performance",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def plot_intraday_anomaly_rmse(
    intraday: pd.DataFrame,
    watersheds: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Compare errors in hourly deviations from each model's daily mean."""
    pivot = intraday.pivot(
        index="watershed", columns="model", values="intraday_anomaly_RMSE"
    ).reindex(watersheds)
    positions = np.arange(len(watersheds))
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    for position, watershed in enumerate(watersheds):
        values = pivot.loc[watershed]
        axis.plot(
            [values[model] for model in MODEL_ORDER],
            [position] * len(MODEL_ORDER),
            color="#9A9A9A",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )
    for model in MODEL_ORDER:
        color, marker, _ = MODEL_STYLES[model]
        axis.scatter(
            pivot[model],
            positions,
            color=color,
            marker=marker,
            s=48,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=model,
            zorder=3,
        )
    axis.set_yticks(positions, [basin_display_name(value) for value in watersheds])
    axis.invert_yaxis()
    axis.set_xlim(left=0.0)
    axis.set_xlabel("Within-day anomaly RMSE (native streamflow units; lower is better)")
    axis.set_ylabel("Watershed")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, ncol=3)
    axis.set_title(
        "Subdaily streamflow-variation error",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.02,
        "Anomalies subtract each series' calendar-day mean; D-LSTM therefore represents a no-intraday-variation baseline",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
    )
    figure.tight_layout()
    outputs = save_figure(
        figure,
        output_dir / "hourly_daily_intraday_anomaly_rmse",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def summary_value(
    summary: pd.DataFrame,
    scale: str,
    model: str,
    metric: str,
) -> Tuple[float, float]:
    row = summary.loc[
        (summary["evaluation_scale"] == scale)
        & (summary["model"] == model)
        & (summary["metric"] == metric)
    ].iloc[0]
    return float(row["mean"]), float(row["std"])


def winner_count(metrics: pd.DataFrame, scale: str, metric: str, model: str) -> int:
    """Count watersheds where a model has the best score, retaining exact ties."""
    pivot = metrics.loc[metrics["evaluation_scale"] == scale].pivot(
        index="watershed", columns="model", values=metric
    )
    if metric in {"RMSE", "MAE", "Q95_MAE"}:
        best = pivot.min(axis=1)
    else:
        best = pivot.max(axis=1)
    return int(np.isclose(pivot[model], best, rtol=1e-10, atol=1e-12).sum())


def markdown_metric_table(
    metrics: pd.DataFrame,
    scale: str,
    watersheds: Sequence[str],
) -> str:
    header = ["Watershed"] + [
        f"{model} {metric}"
        for metric in ("RMSE", "NSE", "KGE")
        for model in MODEL_ORDER
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * (len(header) - 1),
    ]
    subset = metrics.loc[metrics["evaluation_scale"] == scale]
    for watershed in watersheds:
        basin = subset.loc[subset["watershed"] == watershed].set_index("model")
        if basin.empty:
            continue
        values = [basin_display_name(watershed)]
        for metric in ("RMSE", "NSE", "KGE"):
            precision = 2 if metric == "RMSE" else 3
            values.extend(
                f"{basin.loc[model, metric]:.{precision}f}" for model in MODEL_ORDER
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_notes(
    path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    intraday: pd.DataFrame,
    audit: pd.DataFrame,
    watersheds: Sequence[str],
    hourly_experiment: str,
    mr_stf_experiment: str,
    daily_experiment: str,
    split: str,
    method: str,
    peak_window_days: int,
) -> None:
    """Write methods, exact requested metrics, and paper-facing captions."""
    hourly_kge = {
        model: summary_value(summary, "hourly", model, "KGE")
        for model in MODEL_ORDER
    }
    daily_kge = {
        model: summary_value(summary, "daily_mean", model, "KGE")
        for model in MODEL_ORDER
    }
    intraday_pivot = intraday.pivot(
        index="watershed", columns="model", values="intraday_anomaly_RMSE"
    )
    intraday_best = intraday_pivot.min(axis=1)
    intraday_wins = {
        model: int(
            np.isclose(
                intraday_pivot[model], intraday_best, rtol=1e-10, atol=1e-12
            ).sum()
        )
        for model in MODEL_ORDER
    }
    max_reference_difference = float(audit["daily_reference_max_abs_difference"].max())
    max_mr_reference_difference = float(
        audit["mr_stf_reference_max_abs_difference"].max()
    )
    common_start = pd.to_datetime(audit["common_start"]).max()
    common_end = pd.to_datetime(audit["common_end"]).min()
    common_days = int(audit["common_complete_days"].min())
    hourly_kge_text = ", ".join(
        f"{model} {hourly_kge[model][0]:.3f} ± {hourly_kge[model][1]:.3f}"
        for model in MODEL_ORDER
    )
    daily_kge_text = ", ".join(
        f"{model} {daily_kge[model][0]:.3f} ± {daily_kge[model][1]:.3f}"
        for model in MODEL_ORDER
    )
    hourly_rmse_wins = ", ".join(
        f"{model} {winner_count(metrics, 'hourly', 'RMSE', model)}"
        for model in MODEL_ORDER
    )
    hourly_kge_wins = ", ".join(
        f"{model} {winner_count(metrics, 'hourly', 'KGE', model)}"
        for model in MODEL_ORDER
    )
    intraday_win_text = ", ".join(
        f"{model} {intraday_wins[model]}" for model in MODEL_ORDER
    )

    notes = f"""# H-LSTM, MR-STF, and D-LSTM streamflow comparison

## Technical summary

All three models were evaluated on identical timestamps and against the same hourly reference. Watershed wins for lowest hourly RMSE were: {hourly_rmse_wins}; wins for highest hourly KGE were: {hourly_kge_wins}. Across watersheds, hourly-grid KGE (mean ± sample standard deviation) was {hourly_kge_text}. After the predictions and common reference were averaged by calendar day, KGE was {daily_kge_text}. Wins for lowest within-day anomaly RMSE were: {intraday_win_text}. The last comparison tests whether native-hourly models add useful subdaily structure relative to D-LSTM's no-intraday-variation baseline.

## Sources and temporal alignment

- H-LSTM source: `experiments/{hourly_experiment}/test_results`.
- MR-STF source: `experiments/{mr_stf_experiment}/test_results`.
- D-LSTM source: `experiments/{daily_experiment}/test_results`.
- Split and reconstruction: `{split}`, `{method}`.
- Common evaluation window: `{common_start:%Y-%m-%d %H:%M}` through `{common_end:%Y-%m-%d %H:%M}` ({common_days:,} complete calendar days per watershed).
- D-LSTM conversion: the existing `interpolate_daily_imv_to_hourly(..., day_shift=0)` function assigns each daily prediction unchanged to all 24 timestamps of the same calendar day.
- Shared reference: hourly `obs_streamflow` from the H-LSTM reconstruction file.
- MR-STF's hourly reference agrees with the shared H-LSTM reference to a maximum absolute difference of {max_mr_reference_difference:.6f} native units on aligned timestamps.
- The daily `obs_streamflow` values agree with calendar-day means of the hourly reference to a maximum absolute difference of {max_reference_difference:.6f} native units.
- Metrics use only common, finite timestamps and complete 24-hour days.

## Why two evaluation scales are necessary

The hourly-grid comparison evaluates total error against hourly streamflow, including intraday behavior. It is operationally relevant when hourly predictions are required, but D-LSTM is structurally unable to vary within a day. The daily-mean comparison removes this resolution disadvantage and evaluates whether each model reproduces day-to-day streamflow. H-LSTM and MR-STF remain native-hourly models in both comparisons. Both views should be reported; neither alone is a complete cross-resolution comparison.

## Hourly-grid metrics

{markdown_metric_table(metrics, 'hourly', watersheds)}

## Daily-mean-grid metrics

{markdown_metric_table(metrics, 'daily_mean', watersheds)}

Exact values and supporting MAE, bias, KGE components, percent bias, and observed-Q95 MAE are saved in `tables/hourly_daily_metrics_by_basin.csv`.

## Additional comparison methods

1. **Within-day anomaly error.** Subtract each series' calendar-day mean and compare the residual hourly pattern. D-LSTM becomes a zero-anomaly baseline; H-LSTM and MR-STF only demonstrate useful subdaily skill when they improve on that baseline.
2. **High-flow error.** `Q95_MAE` evaluates all models during hours or days at or above the observed 95th percentile and is included in the detailed table.
3. **Peak-event zooms.** Basin-specific ±{peak_window_days}-day windows reveal timing, attenuation, and the step structure introduced by daily repetition.
4. **Paired watershed comparisons.** Connecting the three model scores within each watershed avoids misleading conclusions from pooled errors dominated by large-flow basins.
5. **For event-focused claims.** Add event detection F1, peak timing error, and peak magnitude bias; overall NSE or KGE cannot establish flood-event fidelity.

## Suggested figure captions

**Full test-period hydrographs.** Observed/reference hourly streamflow and predictions from the native-hourly H-LSTM and MR-STF models and daily D-LSTM for all {len(watersheds)} watersheds during the common test period. D-LSTM predictions were converted to hourly resolution by assigning each daily value to all 24 hours of the same calendar day. Panel labels report RMSE, Nash–Sutcliffe efficiency (NSE), and Kling–Gupta efficiency (KGE) calculated against the same hourly reference and common timestamps. Panel-specific y-axis scales preserve temporal detail.

**Peak-event zooms.** H-LSTM, MR-STF, and temporally expanded D-LSTM predictions in basin-specific ±{peak_window_days}-day windows around the largest observed hourly streamflow in the test period. Vertical lines identify the observed peak time. The daily model's stepwise trace reflects calendar-day repetition rather than inferred subdaily dynamics.

**Paired performance.** Basin-level RMSE, NSE, and KGE for H-LSTM, MR-STF, and D-LSTM on the common hourly grid and after calendar-day averaging. Grey lines connect scores for the same watershed. Lower RMSE and higher NSE/KGE indicate better performance. Reporting both grids separates day-to-day predictive skill from the ability to represent intraday variation.

**Intraday anomaly error.** RMSE of hourly deviations from each series' calendar-day mean. D-LSTM represents a no-intraday-variation baseline because its daily prediction is repeated over 24 hours. H-LSTM or MR-STF values below that baseline indicate useful reconstruction of subdaily variability; higher values indicate that modeled hourly fluctuations add error.

## Interpretation caveats

- The comparison is paired and descriptive; it does not isolate temporal resolution or MR-STF input design from differences in training targets, windowing, or optimization.
- Repeating daily values is the established pipeline behavior, not a learned temporal downscaling model. Do not interpret the D-LSTM hourly trace as an hourly forecast.
- RMSE is in native streamflow units and larger-flow watersheds contribute larger absolute errors. Use paired basin results rather than pooling all timestamps across basins.
- KGE and NSE summarize overall behavior but can obscure peak timing and low-flow errors.
- Confirm whether `obs_streamflow` is simulated HSPF output or field observation before using “observed” in the paper.

## Reproducibility

```bash
conda run --no-capture-output -n imerg_era5 python compare_hourly_daily_streamflow.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notes, encoding="utf-8")


def validate_results(
    metrics: pd.DataFrame,
    intraday: pd.DataFrame,
    audit: pd.DataFrame,
    watersheds: Sequence[str],
    allow_incomplete: bool,
) -> None:
    """Apply calculation, alignment, and completeness gates."""
    expected_metric_rows = len(watersheds) * 2 * len(MODEL_ORDER)
    expected_intraday_rows = len(watersheds) * len(MODEL_ORDER)
    if metrics.duplicated(["watershed", "evaluation_scale", "model"]).any():
        raise ValueError("Duplicate model metric rows were generated.")
    if intraday.duplicated(["watershed", "model"]).any():
        raise ValueError("Duplicate intraday rows were generated.")
    if not allow_incomplete and len(metrics) != expected_metric_rows:
        raise ValueError(f"Expected {expected_metric_rows} metric rows; got {len(metrics)}.")
    if not allow_incomplete and len(intraday) != expected_intraday_rows:
        raise ValueError(
            f"Expected {expected_intraday_rows} intraday rows; got {len(intraday)}."
        )
    if (audit["common_hourly_rows"] != audit["common_complete_days"] * 24).any():
        raise ValueError("At least one watershed does not have 24 rows per common day.")
    if (audit["daily_prediction_expansion_max_abs_difference"] > 1e-10).any():
        raise ValueError("Expanded D-LSTM predictions do not equal native daily values.")
    if (audit["daily_reference_max_abs_difference"] > 0.01).any():
        raise ValueError(
            "Daily reference values do not agree closely enough with hourly daily means."
        )
    # The independently written reconstruction CSVs differ by at most 0.01
    # native units because of output precision. This tolerance remains tiny
    # relative to the basin flows while still rejecting a material mismatch.
    if (audit["mr_stf_reference_max_abs_difference"] > 0.02).any():
        raise ValueError(
            "MR-STF reference values do not agree closely enough with the shared "
            "H-LSTM hourly reference."
        )
    requested = metrics[["RMSE", "NSE", "KGE"]].to_numpy(dtype=float)
    if not np.isfinite(requested).all():
        raise ValueError("RMSE, NSE, or KGE contains a non-finite value.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.experiments_dir / DEFAULT_OUTPUT_EXPERIMENT / "test_results"
    )
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    metric_rows: List[Dict[str, object]] = []
    intraday_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    aligned: Dict[str, pd.DataFrame] = {}
    processed_watersheds: List[str] = []

    for watershed in args.watersheds:
        hourly_path = reconstruction_path(
            args.experiments_dir,
            args.hourly_experiment,
            args.results_subdir,
            watershed,
            args.split,
            args.method,
        )
        daily_path = reconstruction_path(
            args.experiments_dir,
            args.daily_experiment,
            args.results_subdir,
            watershed,
            args.split,
            args.method,
        )
        mr_stf_path = reconstruction_path(
            args.experiments_dir,
            args.mr_stf_experiment,
            args.results_subdir,
            watershed,
            args.split,
            args.method,
        )
        missing_sources = [
            path
            for path in (hourly_path, mr_stf_path, daily_path)
            if not path.is_file()
        ]
        if missing_sources:
            if args.allow_incomplete:
                print(f"Skipping {watershed}: missing {', '.join(map(str, missing_sources))}")
                continue
            raise FileNotFoundError(
                f"Missing source for {watershed}: hourly={hourly_path.is_file()}, "
                f"mr_stf={mr_stf_path.is_file()}, "
                f"daily={daily_path.is_file()}."
            )

        hourly = load_hourly_reconstruction(hourly_path)
        mr_stf = load_hourly_reconstruction(mr_stf_path)
        daily = load_daily_reconstruction(daily_path)
        hourly_common, daily_common, audit = align_basin_sources(
            hourly, mr_stf, daily, watershed
        )
        basin_metrics, basin_intraday = evaluate_basin(
            watershed, hourly_common, daily_common
        )
        metric_rows.extend(basin_metrics)
        intraday_rows.extend(basin_intraday)
        audit_rows.append(audit)
        aligned[watershed] = hourly_common
        processed_watersheds.append(watershed)

    if not processed_watersheds:
        raise ValueError("No watersheds had both hourly and daily reconstruction files.")

    metrics = pd.DataFrame(metric_rows)
    intraday = pd.DataFrame(intraday_rows)
    audit = pd.DataFrame(audit_rows)
    summary = summarize_metrics(metrics)
    validate_results(
        metrics, intraday, audit, processed_watersheds, args.allow_incomplete
    )

    metrics_path = tables_dir / "hourly_daily_metrics_by_basin.csv"
    summary_path = tables_dir / "hourly_daily_metric_summary.csv"
    intraday_path = tables_dir / "hourly_daily_intraday_metrics.csv"
    audit_path = tables_dir / "hourly_daily_alignment_audit.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.8g")
    summary.to_csv(summary_path, index=False, float_format="%.8g")
    intraday.to_csv(intraday_path, index=False, float_format="%.8g")
    audit.to_csv(audit_path, index=False, float_format="%.8g")

    figure_outputs: List[Path] = []
    figure_outputs.extend(
        plot_full_timeseries(
            aligned,
            metrics,
            processed_watersheds,
            plots_dir,
            args.formats,
            args.dpi,
        )
    )
    figure_outputs.extend(
        plot_peak_event_zooms(
            aligned,
            processed_watersheds,
            args.peak_window_days,
            plots_dir,
            args.formats,
            args.dpi,
        )
    )
    figure_outputs.extend(
        plot_paired_performance(
            metrics,
            processed_watersheds,
            plots_dir,
            args.formats,
            args.dpi,
        )
    )
    figure_outputs.extend(
        plot_intraday_anomaly_rmse(
            intraday,
            processed_watersheds,
            plots_dir,
            args.formats,
            args.dpi,
        )
    )

    notes_path = output_dir / "comparison_notes.md"
    write_notes(
        notes_path,
        metrics,
        summary,
        intraday,
        audit,
        processed_watersheds,
        args.hourly_experiment,
        args.mr_stf_experiment,
        args.daily_experiment,
        args.split,
        args.method,
        args.peak_window_days,
    )

    print(
        f"Compared H-LSTM, MR-STF, and D-LSTM across {len(processed_watersheds)} watersheds "
        f"and {len(metrics):,} model/scale metric rows."
    )
    print(f"Basin metrics: {metrics_path}")
    print(f"Summary:       {summary_path}")
    print(f"Intraday:      {intraday_path}")
    print(f"Alignment:     {audit_path}")
    print(f"Notes:         {notes_path}")
    print(f"Figures:       {len(figure_outputs)} files under {plots_dir}")


if __name__ == "__main__":
    main()
