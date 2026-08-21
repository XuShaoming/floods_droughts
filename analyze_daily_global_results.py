#!/usr/bin/env python3
"""Analyze all daily ``daily_global_*`` reconstruction experiments.

The script reads the same reconstructed time series produced by
``inference_global.py``::

    experiments/daily_global_<target>/test_results/
        <watershed>_<split>_reconstructed_<method>.csv

It creates:

* a basin-level audit table containing KGE, NSE, RMSE, MAE, bias, and the KGE
  components for every target and train/validation/test split;
* a paper-facing table of the mean and sample standard deviation of KGE across
  watersheds, with each watershed weighted equally;
* a source-data audit table;
* one 4 x 3 observed-versus-predicted test-period figure per target;
* cross-target KGE summary and test-basin heatmap figures; and
* a Markdown methods/results report with reusable figure captions.

Kling-Gupta efficiency (KGE) is the primary metric because it is unitless and
combines temporal correlation, variability error, and mean bias. That makes it
more suitable than a unit-dependent error for comparing the differently scaled
IMVs and streamflow. MAPE is intentionally omitted because zeros and near-zero
reference values make it unstable for several targets.

Examples
--------
Run the complete analysis with the project environment::

    conda run --no-capture-output -n imerg_era5 \
        python analyze_daily_global_results.py

Generate only PNG figures in a custom location::

    conda run --no-capture-output -n imerg_era5 \
        python analyze_daily_global_results.py \
        --output-dir experiments/daily_global_comparison/analysis_results \
        --formats png
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from hydrology_metrics import calculate_hydrology_metrics


EXPECTED_FEATURES: Tuple[str, ...] = (
    "T2",
    "DEWPT",
    "PRECIP",
    "SWDNB",
    "WSPD10",
    "LH",
)
EXPECTED_WATERSHEDS: Tuple[str, ...] = (
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
TARGET_ORDER: Tuple[str, ...] = (
    "PET",
    "ET",
    "SUPY",
    "WYIE",
    "SNOW",
    "TWS",
    "LZS",
    "AGW",
    "streamflow",
)
SPLIT_LABELS: Mapping[str, str] = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
SUMMARY_COLUMN_PREFIXES: Mapping[str, str] = {
    "train": "train",
    "val": "valid",
    "test": "test",
}

OBSERVED_COLOR = "#222222"
PREDICTED_COLOR = "#D55E00"
SPLIT_STYLES: Mapping[str, Tuple[str, str]] = {
    "train": ("#0072B2", "o"),
    "val": ("#E69F00", "s"),
    "test": ("#D55E00", "D"),
}


@dataclass(frozen=True)
class ExperimentSpec:
    """Resolved metadata for one single-target daily experiment."""

    name: str
    directory: Path
    target: str
    feature_cols: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all daily_global_* IMV and streamflow reconstructions "
            "across watersheds and dataset splits."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing daily_global_* experiment folders.",
    )
    parser.add_argument(
        "--experiment-pattern",
        default="daily_global_*",
        help="Glob used to discover experiment directories.",
    )
    parser.add_argument(
        "--results-subdir",
        default="test_results",
        help="Reconstruction subdirectory within each experiment.",
    )
    parser.add_argument(
        "--method",
        default="latest",
        help="Reconstruction suffix in input filenames (default: latest).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to experiments/daily_global_comparison/"
            "analysis_results."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Figure formats to export (default: png pdf).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution (default: 300 dpi).",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Analyze available files instead of requiring all nine targets, "
            "12 watersheds, and three splits."
        ),
    )
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Skip the nine 4 x 3 test-period time-series figures.",
    )
    return parser.parse_args()


def basin_display_name(basin: str) -> str:
    """Return a readable watershed label from its internal identifier."""
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


def target_sort_key(target: str) -> Tuple[int, str]:
    """Keep the configured IMV order, followed by any future targets."""
    try:
        return TARGET_ORDER.index(target), target
    except ValueError:
        return len(TARGET_ORDER), target


def discover_experiments(
    experiments_dir: Path,
    pattern: str,
    results_subdir: str,
    allow_incomplete: bool,
) -> List[ExperimentSpec]:
    """Discover valid single-target experiments and verify their inputs."""
    specs: List[ExperimentSpec] = []
    targets_seen: Dict[str, str] = {}

    for directory in sorted(experiments_dir.glob(pattern)):
        model_config_path = directory / "model_config.json"
        results_dir = directory / results_subdir
        if not directory.is_dir() or not model_config_path.is_file() or not results_dir.is_dir():
            continue

        with model_config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        targets = model_config.get("target_cols", [])
        if len(targets) != 1:
            raise ValueError(
                f"{model_config_path} must define exactly one target; got {targets!r}."
            )
        target = str(targets[0])
        features = tuple(str(value) for value in model_config.get("feature_cols", []))
        if features != EXPECTED_FEATURES:
            raise ValueError(
                f"{directory.name} uses features {features!r}; expected "
                f"{EXPECTED_FEATURES!r}. Mixed feature sets are not comparable."
            )
        if target in targets_seen:
            raise ValueError(
                f"Targets must be unique, but {target!r} appears in both "
                f"{targets_seen[target]!r} and {directory.name!r}."
            )
        targets_seen[target] = directory.name
        specs.append(ExperimentSpec(directory.name, directory, target, features))

    if not specs:
        raise FileNotFoundError(
            f"No valid experiment folders matching {pattern!r} were found under "
            f"{experiments_dir}."
        )

    specs.sort(key=lambda item: target_sort_key(item.target))
    if not allow_incomplete:
        missing_targets = sorted(set(TARGET_ORDER).difference(targets_seen))
        if missing_targets:
            raise FileNotFoundError(
                "Required daily targets are missing: " + ", ".join(missing_targets)
            )
    return specs


def reconstructed_path(
    spec: ExperimentSpec,
    results_subdir: str,
    watershed: str,
    split: str,
    method: str,
) -> Path:
    return (
        spec.directory
        / results_subdir
        / f"{watershed}_{split}_reconstructed_{method}.csv"
    )


def discover_watersheds(
    specs: Sequence[ExperimentSpec],
    results_subdir: str,
    method: str,
    allow_incomplete: bool,
) -> List[str]:
    """Resolve and validate the watershed set across targets and splits."""
    discovered: Dict[Tuple[str, str], set[str]] = {}
    for spec in specs:
        results_dir = spec.directory / results_subdir
        for split in SPLIT_LABELS:
            suffix = f"_{split}_reconstructed_{method}.csv"
            watersheds = {
                path.name[: -len(suffix)]
                for path in results_dir.glob(f"*{suffix}")
                if path.name.endswith(suffix)
            }
            discovered[(spec.target, split)] = watersheds

    if allow_incomplete:
        union = set().union(*discovered.values())
        if not union:
            raise FileNotFoundError("No reconstructed CSV files were discovered.")
        return sorted(union, key=lambda value: (basin_display_name(value), value))

    expected = set(EXPECTED_WATERSHEDS)
    problems = []
    for (target, split), watersheds in discovered.items():
        if watersheds != expected:
            missing = sorted(expected.difference(watersheds))
            extra = sorted(watersheds.difference(expected))
            problems.append(
                f"{target}/{split}: missing={missing or 'none'}, extra={extra or 'none'}"
            )
    if problems:
        raise FileNotFoundError(
            "Reconstruction coverage is incomplete or inconsistent:\n"
            + "\n".join(problems)
        )
    return list(EXPECTED_WATERSHEDS)


def load_reconstruction(path: Path, target: str) -> pd.DataFrame:
    """Load, type-check, and sort one reconstructed daily series."""
    required = ["timestamp", f"pred_{target}", f"obs_{target}"]
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")

    keep = required + (["scenario"] if "scenario" in frame.columns else [])
    frame = frame.loc[:, keep].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path} contains invalid timestamps.")

    duplicate_key = ["timestamp"] + (["scenario"] if "scenario" in frame.columns else [])
    duplicate_count = int(frame.duplicated(duplicate_key).sum())
    if duplicate_count:
        raise ValueError(
            f"{path} contains {duplicate_count} duplicate timestamp/scenario rows."
        )

    frame[f"pred_{target}"] = pd.to_numeric(frame[f"pred_{target}"], errors="coerce")
    frame[f"obs_{target}"] = pd.to_numeric(frame[f"obs_{target}"], errors="coerce")
    return frame.sort_values(duplicate_key).reset_index(drop=True)


def percent_bias(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Return total-volume percent bias, or NaN for a zero denominator."""
    denominator = float(np.sum(observation))
    if denominator == 0.0:
        return float("nan")
    return float(100.0 * np.sum(prediction - observation) / denominator)


def evaluate_frame(
    frame: pd.DataFrame,
    spec: ExperimentSpec,
    split: str,
    watershed: str,
    source_path: Path,
) -> Dict[str, object]:
    """Calculate metrics and source checks for one target/basin/split."""
    pred_column = f"pred_{spec.target}"
    obs_column = f"obs_{spec.target}"
    pred_all = frame[pred_column].to_numpy(dtype=np.float64)
    obs_all = frame[obs_column].to_numpy(dtype=np.float64)
    paired = np.isfinite(pred_all) & np.isfinite(obs_all)
    pred = pred_all[paired]
    obs = obs_all[paired]
    if obs.size < 2:
        raise ValueError(f"{source_path} has fewer than two finite paired values.")

    metrics = calculate_hydrology_metrics(pred, obs)
    timestamps = frame.loc[paired, "timestamp"]
    scenarios = (
        ";".join(sorted(frame["scenario"].dropna().astype(str).unique()))
        if "scenario" in frame.columns
        else ""
    )
    row: Dict[str, object] = {
        "experiment": spec.name,
        "target": spec.target,
        "split": split,
        "watershed": watershed,
        "watershed_name": basin_display_name(watershed),
        "scenario": scenarios,
        "n_rows": int(len(frame)),
        "n_pairs": int(obs.size),
        "n_dropped_nonfinite": int(len(frame) - obs.size),
        "start_date": timestamps.min().date().isoformat(),
        "end_date": timestamps.max().date().isoformat(),
        "obs_mean": float(np.mean(obs)),
        "obs_std": float(np.std(obs)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "PBIAS_pct": percent_bias(pred, obs),
        "source_file": str(source_path),
    }
    row.update(metrics)
    return row


def analyze_reconstructions(
    specs: Sequence[ExperimentSpec],
    watersheds: Sequence[str],
    results_subdir: str,
    method: str,
    allow_incomplete: bool,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame], List[Path]]:
    """Compute all split metrics and retain test frames for plotting."""
    rows: List[Dict[str, object]] = []
    test_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    source_paths: List[Path] = []

    for spec in specs:
        for split in SPLIT_LABELS:
            for watershed in watersheds:
                path = reconstructed_path(spec, results_subdir, watershed, split, method)
                if not path.is_file():
                    if allow_incomplete:
                        continue
                    raise FileNotFoundError(path)
                frame = load_reconstruction(path, spec.target)
                rows.append(evaluate_frame(frame, spec, split, watershed, path))
                source_paths.append(path)
                if split == "test":
                    test_frames[(spec.target, watershed)] = frame

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No finite reconstruction pairs were available for analysis.")
    return metrics, test_frames, source_paths


def summarize_kge(
    metrics: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
) -> pd.DataFrame:
    """Return one KGE mean/std row per model target across watersheds."""
    rows: List[Dict[str, object]] = []
    for spec in specs:
        row: Dict[str, object] = {
            "experiment": spec.name,
            "target": spec.target,
            "metric": "KGE",
            "aggregation": "unweighted basin mean and sample SD",
        }
        for split, prefix in SUMMARY_COLUMN_PREFIXES.items():
            values = metrics.loc[
                (metrics["target"] == spec.target) & (metrics["split"] == split),
                "KGE",
            ].dropna()
            row[f"{prefix}_kge_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{prefix}_kge_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{prefix}_n_watersheds"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def build_data_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completeness and date coverage by target and split."""
    rows: List[Dict[str, object]] = []
    for (experiment, target, split), group in metrics.groupby(
        ["experiment", "target", "split"], sort=False
    ):
        rows.append(
            {
                "experiment": experiment,
                "target": target,
                "split": split,
                "n_watersheds": int(group["watershed"].nunique()),
                "n_source_files": int(len(group)),
                "total_rows": int(group["n_rows"].sum()),
                "total_finite_pairs": int(group["n_pairs"].sum()),
                "total_dropped_nonfinite": int(group["n_dropped_nonfinite"].sum()),
                "earliest_date": str(group["start_date"].min()),
                "latest_date": str(group["end_date"].max()),
                "scenarios": ";".join(sorted(group["scenario"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows)


def configure_style() -> None:
    """Apply a restrained, publication-friendly Matplotlib style."""
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
        }
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Save one figure in each requested format."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        output = base_path.with_suffix(f".{extension}")
        figure.savefig(output, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        outputs.append(output)
    return outputs


def plot_target_timeseries(
    spec: ExperimentSpec,
    watersheds: Sequence[str],
    test_frames: Mapping[Tuple[str, str], pd.DataFrame],
    test_metrics: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot all watershed test series for one target in a 4 x 3 layout."""
    figure, axes = plt.subplots(4, 3, figsize=(17.5, 13.0), sharex=True)
    metric_lookup = test_metrics.set_index("watershed")["KGE"].to_dict()

    for axis, watershed in zip(axes.flat, watersheds):
        frame = test_frames.get((spec.target, watershed))
        if frame is None:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.plot(
            frame["timestamp"],
            frame[f"obs_{spec.target}"],
            color=OBSERVED_COLOR,
            linewidth=0.85,
            alpha=0.78,
            label="Reference",
            rasterized=True,
        )
        axis.plot(
            frame["timestamp"],
            frame[f"pred_{spec.target}"],
            color=PREDICTED_COLOR,
            linewidth=0.80,
            linestyle=(0, (5, 2)),
            alpha=0.78,
            label="Prediction",
            rasterized=True,
        )
        kge = metric_lookup.get(watershed, np.nan)
        kge_label = f"KGE = {kge:.2f}" if np.isfinite(kge) else "KGE = N/A"
        axis.set_title(f"{basin_display_name(watershed)}  |  {kge_label}", loc="left")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        axis.tick_params(labelsize=8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Date")
    figure.supylabel(f"{spec.target} (native model-output units)", x=0.012, fontsize=10)
    handles = [
        plt.Line2D([], [], color=OBSERVED_COLOR, linewidth=1.5, label="Reference"),
        plt.Line2D(
            [],
            [],
            color=PREDICTED_COLOR,
            linewidth=1.5,
            linestyle=(0, (5, 2)),
            label="Prediction",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"Daily {spec.target}: reference and predicted test-period series",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.04,
        0.955,
        "All 12 watersheds; latest reconstruction; panel-specific y-axis scales",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0.025, 0.025, 0.995, 0.925), h_pad=1.3, w_pad=1.0)
    outputs = save_figure(
        figure,
        output_dir / f"daily_test_{spec.target}_timeseries",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def plot_kge_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot basin-mean KGE and between-basin standard deviation by split."""
    figure, axis = plt.subplots(figsize=(11.5, 7.5))
    targets = summary["target"].tolist()
    positions = np.arange(len(targets), dtype=float)
    offsets = {"train": -0.22, "val": 0.0, "test": 0.22}

    for split, label in SPLIT_LABELS.items():
        prefix = SUMMARY_COLUMN_PREFIXES[split]
        color, marker = SPLIT_STYLES[split]
        means = summary[f"{prefix}_kge_mean"].to_numpy(dtype=float)
        stds = summary[f"{prefix}_kge_std"].to_numpy(dtype=float)
        axis.errorbar(
            means,
            positions + offsets[split],
            xerr=stds,
            fmt=marker,
            color=color,
            markerfacecolor="white" if split == "val" else color,
            markeredgecolor=color,
            markeredgewidth=1.1,
            markersize=6.5,
            capsize=2.5,
            elinewidth=1.1,
            alpha=0.9,
            label=label,
        )

    axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle=(0, (3, 3)))
    axis.axvline(1.0, color="#222222", linewidth=0.9, linestyle=(0, (5, 3)))
    axis.text(1.0, -0.68, "ideal", ha="right", va="bottom", fontsize=8)
    axis.set_yticks(positions, targets)
    axis.invert_yaxis()
    axis.set_xlabel("Kling–Gupta efficiency (KGE; higher is better)")
    axis.set_ylabel("Prediction target")
    finite_lows = []
    for split in SPLIT_LABELS:
        prefix = SUMMARY_COLUMN_PREFIXES[split]
        finite_lows.extend(
            (
                summary[f"{prefix}_kge_mean"] - summary[f"{prefix}_kge_std"]
            ).dropna()
        )
    lower = min(-0.05, float(min(finite_lows)) - 0.03) if finite_lows else -0.05
    axis.set_xlim(lower, 1.035)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", frameon=False, ncol=3)
    axis.set_title(
        "Daily prediction skill across watersheds",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.02,
        "Points are unweighted basin means; error bars are sample SD (n = 12 watersheds)",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
    )
    figure.tight_layout()
    outputs = save_figure(figure, output_dir / "daily_kge_summary_by_split", formats, dpi)
    plt.close(figure)
    return outputs


def plot_test_kge_heatmap(
    metrics: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
    watersheds: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot test KGE for every target-watershed pair to expose heterogeneity."""
    targets = [spec.target for spec in specs]
    test = metrics.loc[metrics["split"] == "test"]
    pivot = test.pivot(index="target", columns="watershed", values="KGE")
    matrix = pivot.reindex(index=targets, columns=watersheds).to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(15.5, 7.0))
    image = axis.imshow(matrix, cmap="viridis", vmin=-0.5, vmax=1.0, aspect="auto")
    axis.set_xticks(
        np.arange(len(watersheds)),
        [basin_display_name(value) for value in watersheds],
        rotation=42,
        ha="right",
    )
    axis.set_yticks(np.arange(len(targets)), targets)
    axis.set_xlabel("Watershed")
    axis.set_ylabel("Prediction target")
    axis.set_title(
        "Test-period daily KGE by target and watershed",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.02,
        "Cell labels give exact KGE; the fixed color scale supports comparison across targets",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = f"{value:.2f}" if np.isfinite(value) else "N/A"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if not np.isfinite(value) or value < 0.60 else "#111111",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.018)
    colorbar.set_label("KGE (higher is better)")
    axis.set_xticks(np.arange(-0.5, len(watersheds), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.tight_layout()
    outputs = save_figure(
        figure,
        output_dir / "daily_test_kge_by_basin_heatmap",
        formats,
        dpi,
    )
    plt.close(figure)
    return outputs


def markdown_summary_table(summary: pd.DataFrame) -> str:
    """Format the main KGE table for the generated technical report."""
    lines = [
        "| Target | Train KGE | Validation KGE | Test KGE |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        values = []
        for prefix in ("train", "valid", "test"):
            mean = getattr(row, f"{prefix}_kge_mean")
            std = getattr(row, f"{prefix}_kge_std")
            values.append(
                f"{mean:.3f} ± {std:.3f}"
                if np.isfinite(mean) and np.isfinite(std)
                else "N/A"
            )
        lines.append(f"| {row.target} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
    watersheds: Sequence[str],
    source_paths: Sequence[Path],
    method: str,
) -> None:
    """Write a paper-oriented methods/results note and figure captions."""
    test_rank = summary.sort_values("test_kge_mean", ascending=False)
    best = test_rank.iloc[0]
    weakest = test_rank.iloc[-1]
    streamflow = summary.loc[summary["target"] == "streamflow"]
    streamflow_sentence = ""
    if not streamflow.empty:
        item = streamflow.iloc[0]
        streamflow_sentence = (
            f" Streamflow test KGE was {item['test_kge_mean']:.3f} ± "
            f"{item['test_kge_std']:.3f}."
        )

    source_modified = max(path.stat().st_mtime for path in source_paths)
    source_as_of = datetime.fromtimestamp(source_modified, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dates = (
        audit.groupby("split", sort=False)
        .agg(earliest=("earliest_date", "min"), latest=("latest_date", "max"))
        .to_dict("index")
    )
    split_ranges = "; ".join(
        f"{SPLIT_LABELS[split]} {dates[split]['earliest']} to {dates[split]['latest']}"
        for split in SPLIT_LABELS
        if split in dates
    )

    report = f"""# Daily IMV and streamflow prediction analysis

Generated: {generated}  
Newest reconstructed source file: {source_as_of}

## Technical summary

Across {len(watersheds)} watersheds, {best['target']} had the highest mean test-period KGE ({best['test_kge_mean']:.3f} ± {best['test_kge_std']:.3f}), while {weakest['target']} had the lowest ({weakest['test_kge_mean']:.3f} ± {weakest['test_kge_std']:.3f}).{streamflow_sentence} These values are descriptive model-evaluation results, not causal comparisons. Between-watershed standard deviation and the basin heatmap should be reported with the means because basin heterogeneity is substantial for some targets.

## Cross-watershed performance

Values are the unweighted mean ± sample standard deviation of watershed-level KGE. Each watershed therefore contributes equally, regardless of its record length or flow magnitude.

{markdown_summary_table(summary)}

The machine-readable version is `tables/daily_global_kge_summary.csv`. Exact basin-level KGE values and supporting metrics are in `tables/daily_global_metrics_by_basin.csv`.

## Scope and data

- Experiments: {', '.join(spec.name for spec in specs)}.
- Inputs in every model: {', '.join(EXPECTED_FEATURES)}.
- Targets: {', '.join(spec.target for spec in specs)}.
- Watersheds: {', '.join(basin_display_name(value) for value in watersheds)}.
- Evaluation records: `{method}` reconstructed daily prediction/reference pairs.
- Periods represented by the files: {split_ranges}.
- No prediction clipping or post-hoc physical constraints were applied.
- Rows with a non-finite prediction or reference value were omitted pairwise and counted in `tables/daily_global_data_audit.csv`.

## Metric definition and aggregation

Kling–Gupta efficiency is

`KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)`,

where `r` is Pearson correlation, `alpha` is predicted/reference standard-deviation ratio, and `beta` is predicted/reference mean ratio. KGE is unitless; 1 is ideal and larger values indicate better joint reproduction of timing, variability, and mean magnitude. It was selected as the primary paper table metric because the nine targets have different scales and native units.

KGE was first computed independently for every watershed, target, and split. The table then reports the arithmetic mean across watersheds and the sample standard deviation (`ddof = 1`). This is a macro-average: large watersheds and longer series do not receive extra weight. NSE, RMSE, MAE, signed bias, percent bias, and all KGE components are retained for diagnosis, but they are not mixed into the primary comparison table. MAPE is excluded because zeros and near-zero reference values make it unstable for several IMVs.

## Figures and suggested captions

### Daily KGE summary by split

**Caption.** Watershed-macro daily prediction skill for each intermediate model variable (IMV) and streamflow during the train, validation, and test periods. Points show the unweighted mean Kling–Gupta efficiency (KGE) across {len(watersheds)} watersheds, and error bars show the between-watershed sample standard deviation. KGE is unitless, 1 is ideal, and higher values indicate better joint reproduction of correlation, variability, and mean magnitude.

### Test KGE by basin

**Caption.** Test-period daily Kling–Gupta efficiency (KGE) for every target–watershed combination. Cell labels report exact basin-level KGE values and the fixed color scale supports comparison across targets. Higher values indicate better performance. The panel exposes geographic heterogeneity that is hidden by cross-watershed means.

### Target-specific test-period time series

**Caption template.** Daily reference and predicted **[TARGET]** during the test period for all {len(watersheds)} watersheds. Each panel uses its own y-axis scale to preserve temporal detail; the panel title reports the watershed-level KGE. Predictions are the `{method}` reconstruction and are shown without clipping or other post-processing.

## Limitations and interpretation

- The `obs_*` columns are treated as the reference series. Confirm whether these are simulated HSPF targets or field observations before choosing the word “observed” in the paper.
- KGE supports scale-independent comparison but does not communicate absolute error in native units. Use the detailed RMSE/MAE columns when the physical magnitude of error matters.
- KGE can be sensitive when a target mean or variance is close to zero. The saved KGE components, NSE, and time-series plots should be checked before interpreting an unusual value.
- Train-period performance is in-sample. Validation and test results are the appropriate evidence for temporal generalization.
- This analysis compares separately trained target models; it does not establish that differences arise from the target variable alone.

## Recommended robustness checks

1. Report the basin-level heatmap or a distribution of test KGE alongside the mean ± SD table.
2. Inspect `KGE_r`, `KGE_alpha`, and `KGE_beta` to distinguish timing, variability, and mean-bias failures.
3. For streamflow, supplement overall KGE with low-flow, high-flow, and flood-event diagnostics from the separate extremes experiment.
4. If the paper makes seasonal claims, compute season-specific errors in native units or robust normalized errors; seasonal KGE can become unstable when seasonal means approach zero.

## Reproducibility

Run from the repository root:

```bash
conda run --no-capture-output -n imerg_era5 python analyze_daily_global_results.py
```

Generate only PNG outputs:

```bash
conda run --no-capture-output -n imerg_era5 python analyze_daily_global_results.py --formats png
```
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def validate_outputs(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
    watersheds: Sequence[str],
    allow_incomplete: bool,
) -> None:
    """Apply completion and numerical-integrity checks before handoff."""
    key_columns = ["target", "split", "watershed"]
    if metrics.duplicated(key_columns).any():
        duplicates = metrics.loc[metrics.duplicated(key_columns, keep=False), key_columns]
        raise ValueError(f"Duplicate metric rows were generated:\n{duplicates}")
    if not np.isfinite(metrics["KGE"].dropna()).all():
        raise ValueError("The basin metric table contains non-finite non-null KGE values.")
    if (metrics["n_pairs"] <= 1).any():
        raise ValueError("Every metric row must contain at least two finite pairs.")
    if not allow_incomplete:
        expected_rows = len(specs) * len(SPLIT_LABELS) * len(watersheds)
        if len(metrics) != expected_rows:
            raise ValueError(f"Expected {expected_rows} metric rows; got {len(metrics)}.")
        expected_summary = len(specs)
        if len(summary) != expected_summary:
            raise ValueError(
                f"Expected {expected_summary} summary rows; got {len(summary)}."
            )
        expected_audit = len(specs) * len(SPLIT_LABELS)
        if len(audit) != expected_audit:
            raise ValueError(f"Expected {expected_audit} audit rows; got {len(audit)}.")
        n_columns = [column for column in summary if column.endswith("_n_watersheds")]
        if not (summary[n_columns] == len(watersheds)).all().all():
            raise ValueError("A summary split does not contain all watersheds.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.experiments_dir / "daily_global_comparison" / "analysis_results"
    )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    timeseries_dir = figures_dir / "timeseries"
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    specs = discover_experiments(
        args.experiments_dir,
        args.experiment_pattern,
        args.results_subdir,
        args.allow_incomplete,
    )
    watersheds = discover_watersheds(
        specs,
        args.results_subdir,
        args.method,
        args.allow_incomplete,
    )
    metrics, test_frames, source_paths = analyze_reconstructions(
        specs,
        watersheds,
        args.results_subdir,
        args.method,
        args.allow_incomplete,
    )
    summary = summarize_kge(metrics, specs)
    audit = build_data_audit(metrics)
    validate_outputs(metrics, summary, audit, specs, watersheds, args.allow_incomplete)

    metrics_path = tables_dir / "daily_global_metrics_by_basin.csv"
    summary_path = tables_dir / "daily_global_kge_summary.csv"
    audit_path = tables_dir / "daily_global_data_audit.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.8g")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    audit.to_csv(audit_path, index=False)

    figure_outputs: List[Path] = []
    if not args.skip_timeseries:
        for spec in specs:
            target_test_metrics = metrics.loc[
                (metrics["target"] == spec.target) & (metrics["split"] == "test")
            ]
            figure_outputs.extend(
                plot_target_timeseries(
                    spec,
                    watersheds,
                    test_frames,
                    target_test_metrics,
                    timeseries_dir,
                    args.formats,
                    args.dpi,
                )
            )
    figure_outputs.extend(
        plot_kge_summary(summary, figures_dir, args.formats, args.dpi)
    )
    figure_outputs.extend(
        plot_test_kge_heatmap(
            metrics,
            specs,
            watersheds,
            figures_dir,
            args.formats,
            args.dpi,
        )
    )

    report_path = output_dir / "daily_global_analysis_report.md"
    write_report(
        report_path,
        summary,
        audit,
        specs,
        watersheds,
        source_paths,
        args.method,
    )

    print(f"Analyzed {len(specs)} targets across {len(watersheds)} watersheds.")
    print(f"Basin metrics: {metrics_path}")
    print(f"KGE summary:   {summary_path}")
    print(f"Data audit:    {audit_path}")
    print(f"Report:        {report_path}")
    print(f"Figures:       {len(figure_outputs)} files under {figures_dir}")


if __name__ == "__main__":
    main()
