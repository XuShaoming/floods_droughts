#!/usr/bin/env python3
"""Plot reconstructed streamflow from multiple experiments on shared axes.

Edit ``EXPERIMENTS`` below to choose the experiment directories and the labels
shown in the legend. Reconstruction tables are read from

    experiments/<experiment>/test_results/<basin>_<split>_reconstructed_<method>.csv

By default, figures are written to a comparison experiment directory that
mirrors the analysis_global.py output layout.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from hydrology_metrics import calculate_hydrology_metrics


# (experiment directory name, label displayed in the plot legend)
EXPERIMENTS: List[Tuple[str, str]] = [
    ("hourly_global_streamflow_no_IMVs", "H-LSTM"),
    ("hourly_global_streamflow_pred_streamflow_day_shift_1", "MR-STF"),
    ("hourly_global_streamflow_day_shift_1", "MR-PTF"),
]

DEFAULT_OUTPUT_EXPERIMENT = "hourly_global_streamflow_comparison"

# Color-blind-friendly colors and complementary line styles make the model
# series distinguishable in color, grayscale, and projected presentations.
MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
MODEL_LINESTYLES = ("-", (0, (7, 2.5)), (0, (2, 1.5)), (0, (5, 2, 1, 2)))


def basin_display_name(basin: str) -> str:
    """Convert internal basin identifiers into presentation-ready names."""
    name = basin.replace("_", " ")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return name.removesuffix(" Models")


def format_metric(value: float, digits: int, use_grouping: bool = False) -> str:
    """Format a metric compactly while handling undefined values."""
    if not np.isfinite(value):
        return "N/A"
    grouping = "," if use_grouping else ""
    return f"{value:{grouping}.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot observed streamflow and reconstructed predictions from the "
            "experiments listed in EXPERIMENTS."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing the experiment folders (default: experiments).",
    )
    parser.add_argument(
        "--results-subdir",
        default="test_results",
        help="Inference-results directory within each experiment (default: test_results).",
    )
    parser.add_argument("--split", default="test", help="Dataset split to plot (default: test).")
    parser.add_argument(
        "--method",
        default="latest",
        help="Reconstruction method in the input filename (default: latest).",
    )
    parser.add_argument(
        "--target",
        default="streamflow",
        help="Target name used by pred_<target> and obs_<target> columns (default: streamflow).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional inclusive start timestamp, for example 2001-01-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional inclusive end timestamp, for example 2005-12-31 23:00:00.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, figures go under "
            "experiments/hourly_global_streamflow_comparison/test_results/"
            "analysis_results/<split>/plots/reconstructions."
        ),
    )
    parser.add_argument(
        "--observed-alpha",
        type=float,
        default=0.24,
        help="Opacity of observed dots from 0 to 1 (default: 0.24).",
    )
    parser.add_argument(
        "--model-alpha",
        type=float,
        default=0.84,
        help="Opacity of prediction lines from 0 to 1 (default: 0.84).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution (default: 300).")
    return parser.parse_args()


def validate_experiments(experiments: Sequence[Tuple[str, str]]) -> None:
    if not experiments:
        raise ValueError("EXPERIMENTS must contain at least one (experiment, label) tuple.")

    names = [name for name, _ in experiments]
    labels = [label for _, label in experiments]
    if any(not name or not label for name, label in experiments):
        raise ValueError("Experiment names and display labels cannot be empty.")
    if len(names) != len(set(names)):
        raise ValueError("Experiment names in EXPERIMENTS must be unique.")
    if len(labels) != len(set(labels)):
        raise ValueError("Display labels in EXPERIMENTS must be unique.")


def reconstruction_suffix(split: str, method: str) -> str:
    return f"_{split}_reconstructed_{method}.csv"


def find_basin_files(
    experiments_dir: Path,
    experiment: str,
    results_subdir: str,
    split: str,
    method: str,
) -> Dict[str, Path]:
    results_dir = experiments_dir / experiment / results_subdir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    suffix = reconstruction_suffix(split, method)
    files = {
        path.name[: -len(suffix)]: path
        for path in sorted(results_dir.glob(f"*{suffix}"))
    }
    if not files:
        raise FileNotFoundError(
            f"No '*{suffix}' reconstruction files found in {results_dir}"
        )
    return files


def common_basin_files(
    experiments_dir: Path,
    experiments: Sequence[Tuple[str, str]],
    results_subdir: str,
    split: str,
    method: str,
) -> Tuple[List[str], Dict[str, Dict[str, Path]]]:
    files_by_experiment = {
        experiment: find_basin_files(
            experiments_dir, experiment, results_subdir, split, method
        )
        for experiment, _ in experiments
    }

    basin_sets: List[Set[str]] = [set(files) for files in files_by_experiment.values()]
    common = set.intersection(*basin_sets)
    if not common:
        raise ValueError("The configured experiments do not share any reconstruction basins.")

    union = set.union(*basin_sets)
    for experiment, _ in experiments:
        missing = sorted(union - set(files_by_experiment[experiment]))
        if missing:
            print(
                f"Warning: {experiment} is missing {len(missing)} basin(s), which will not be "
                f"plotted: {', '.join(missing)}",
                file=sys.stderr,
            )

    return sorted(common), files_by_experiment


def load_reconstruction(
    path: Path,
    target: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    required = ["timestamp", f"obs_{target}", f"pred_{target}"]
    frame = pd.read_csv(path, usecols=lambda column: column in required)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps.")

    frame = frame.sort_values("timestamp")
    if start is not None:
        frame = frame[frame["timestamp"] >= start]
    if end is not None:
        frame = frame[frame["timestamp"] <= end]
    return frame.reset_index(drop=True)


def observations_agree(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    target: str,
) -> bool:
    obs_col = f"obs_{target}"
    overlap = reference[["timestamp", obs_col]].merge(
        candidate[["timestamp", obs_col]],
        on="timestamp",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    if overlap.empty:
        return True
    return bool(
        np.allclose(
            overlap[f"{obs_col}_reference"],
            overlap[f"{obs_col}_candidate"],
            rtol=1e-5,
            # Reconstruction CSVs are rounded independently when written.
            atol=5e-3,
            equal_nan=True,
        )
    )


def plot_basin(
    basin: str,
    frames: Sequence[Tuple[str, str, pd.DataFrame]],
    target: str,
    split: str,
    method: str,
    output_path: Path,
    dpi: int,
    observed_alpha: float,
    model_alpha: float,
) -> None:
    reference_experiment, _, reference = frames[0]
    if reference.empty:
        raise ValueError(f"No samples remain in the requested range for {basin}.")

    obs_col = f"obs_{target}"
    pred_col = f"pred_{target}"
    figure, axis = plt.subplots(figsize=(15, 5.75))

    # Keep dense observations subtle and behind the model lines. This preserves
    # their point-based appearance without hiding predictions where they agree.
    axis.scatter(
        reference["timestamp"],
        reference[obs_col],
        color="#303030",
        s=4,
        linewidths=0,
        alpha=observed_alpha,
        label="Observed",
        rasterized=True,
        zorder=2,
    )

    for index, (experiment, label, frame) in enumerate(frames):
        if frame.empty:
            print(
                f"Warning: no {basin} samples remain for {experiment} in the requested range.",
                file=sys.stderr,
            )
            continue
        if not observations_agree(reference, frame, target):
            print(
                f"Warning: observed {target} differs between {reference_experiment} and "
                f"{experiment} for {basin}; the observed points use {reference_experiment}.",
                file=sys.stderr,
            )
        metrics = calculate_hydrology_metrics(frame[pred_col], frame[obs_col])
        metric_label = (
            f"{label}   RMSE = {format_metric(metrics['RMSE'], 1, use_grouping=True)}"
            f"   |   NSE = {format_metric(metrics['NSE'], 3)}"
        )
        axis.plot(
            frame["timestamp"],
            frame[pred_col],
            color=MODEL_COLORS[index % len(MODEL_COLORS)],
            linestyle=MODEL_LINESTYLES[index % len(MODEL_LINESTYLES)],
            linewidth=1.35,
            alpha=model_alpha,
            solid_capstyle="round",
            dash_capstyle="round",
            label=metric_label,
            zorder=4,
        )

    coverage_start = reference["timestamp"].min()
    coverage_end = reference["timestamp"].max()
    period = f"{coverage_start:%b %d, %Y} to {coverage_end:%b %d, %Y}"
    target_label = target.replace("_", " ").title()
    axis.set_title(
        f"{basin_display_name(basin)} Basin: Observed and Model-Predicted Hourly {target_label}",
        loc="center",
        fontsize=15,
        fontweight="semibold",
        pad=38,
    )
    axis.text(
        0.5,
        1.025,
        f"{split.title()} period: {period}",
        transform=axis.transAxes,
        color="#555555",
        fontsize=10.5,
        ha="center",
        va="bottom",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel(target_label)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.7, alpha=0.72)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.55, alpha=0.45)
    axis.margins(x=0)
    if all(
        frame[column].dropna().ge(0).all()
        for _, _, frame in frames
        for column in (obs_col, pred_col)
        if not frame.empty
    ):
        axis.set_ylim(bottom=0)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.tick_params(axis="both", colors="#444444", labelsize=9)
    axis.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: f"{value:,.1f}" if 0 < abs(value) < 10 else f"{value:,.0f}"
        )
    )
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.92,
        fontsize=9.2,
        markerscale=2.8,
        ncol=1,
        title="Performance over displayed period",
        title_fontsize=9.5,
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_experiments(EXPERIMENTS)

    start = pd.to_datetime(args.start) if args.start else None
    end = pd.to_datetime(args.end) if args.end else None
    if start is not None and end is not None and start > end:
        raise ValueError(f"Start timestamp {start} is after end timestamp {end}.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not 0.0 <= args.observed_alpha <= 1.0:
        raise ValueError("--observed-alpha must be between 0 and 1.")
    if not 0.0 <= args.model_alpha <= 1.0:
        raise ValueError("--model-alpha must be between 0 and 1.")

    basins, files_by_experiment = common_basin_files(
        args.experiments_dir,
        EXPERIMENTS,
        args.results_subdir,
        args.split,
        args.method,
    )
    output_dir = args.output_dir or (
        args.experiments_dir
        / DEFAULT_OUTPUT_EXPERIMENT
        / args.results_subdir
        / "analysis_results"
        / args.split
        / "plots"
        / "reconstructions"
    )

    for basin in basins:
        frames = [
            (
                experiment,
                label,
                load_reconstruction(
                    files_by_experiment[experiment][basin],
                    args.target,
                    start,
                    end,
                ),
            )
            for experiment, label in EXPERIMENTS
        ]
        output_path = output_dir / f"{basin}_{args.method}_comparison.png"
        plot_basin(
            basin,
            frames,
            args.target,
            args.split,
            args.method,
            output_path,
            args.dpi,
            args.observed_alpha,
            args.model_alpha,
        )
        print(f"Saved {output_path}")

    print(f"Created {len(basins)} basin comparison plot(s) in {output_dir}")


if __name__ == "__main__":
    main()
