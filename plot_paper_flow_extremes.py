#!/usr/bin/env python3
"""Create publication-ready figures for the three-model extreme-flow experiment.

Run ``compare_flow_extremes.py`` first. This script reads its reviewed CSV
outputs and exports each figure as high-resolution PNG plus vector PDF and SVG.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from compare_flow_extremes import align_basin_frames
from plot_reconstruction_comparison import (
    EXPERIMENTS,
    basin_display_name,
    common_basin_files,
)


MODEL_ORDER = [label for _, label in EXPERIMENTS]
MODEL_COLORS = {
    "H-LSTM": "#0072B2",
    "MR-STF": "#D55E00",
    "MR-PTF": "#009E73",
}
MODEL_MARKERS = {"H-LSTM": "o", "MR-STF": "s", "MR-PTF": "D"}
MODEL_LINESTYLES = {"H-LSTM": "-", "MR-STF": "--", "MR-PTF": (0, (2, 1.5))}
BASIN_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C564B",
    "#9467BD",
    "#17BECF",
    "#6B7280",
    "#8A9A0A",
    "#E377C2",
)
INK = "#202124"
MID_GREY = "#6B7280"
LIGHT_GREY = "#D7DCE3"
VERY_LIGHT_GREY = "#EEF2F6"

DEFAULT_RESULTS_DIR = Path(
    "experiments/hourly_global_streamflow_comparison/test_results/"
    "extreme_flow_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready figures from the extreme-flow comparison."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-dir>/paper_figures",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument("--event-padding-hours", type=float, default=48.0)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def require_tables(results_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "regime_summary": "regime_model_summary.csv",
        "flood_summary": "flood_skill_model_summary.csv",
        "regime_basin": "regime_metrics_by_basin.csv",
        "event_basin": "flood_event_metrics_by_basin.csv",
        "event_matches": "flood_event_matches.csv",
    }
    missing = [str(results_dir / name) for name in files.values() if not (results_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment outputs. Run compare_flow_extremes.py first:\n  "
            + "\n  ".join(missing)
        )
    return {key: pd.read_csv(results_dir / name) for key, name in files.items()}


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.09,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
    )


def clean_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis=grid_axis)


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi if extension == "png" else None)
        saved.append(path)
    plt.close(figure)
    return saved


def plot_performance_summary(
    regime_summary: pd.DataFrame,
    flood_summary: pd.DataFrame,
) -> plt.Figure:
    """Four-panel dot-and-interval summary of the primary outcomes."""
    event_f1 = flood_summary[
        (flood_summary["level"] == "flood_events")
        & (flood_summary["metric"] == "event_F1")
    ].copy()
    specifications = [
        ("low", "Low flow (≤Q10)", "NRMSE (% of observed mean)", False),
        ("high", "High flow (≥Q90)", "NRMSE (% of observed mean)", False),
        ("flood_hours", "Flood hours (≥Q95)", "NRMSE (% of observed mean)", False),
        ("event_F1", "Flood-event detection", "Event F1 (%)", True),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    y_positions = np.arange(len(MODEL_ORDER))[::-1]

    for panel_index, (axis, (metric, title, xlabel, higher_better)) in enumerate(
        zip(axes.flat, specifications)
    ):
        if metric == "event_F1":
            subset = event_f1.set_index("model").loc[MODEL_ORDER]
            center = subset["macro_value"].to_numpy(dtype=float) * 100
            lower = subset["ci_low"].to_numpy(dtype=float) * 100
            upper = subset["ci_high"].to_numpy(dtype=float) * 100
        else:
            subset = (
                regime_summary[regime_summary["regime"] == metric]
                .set_index("model")
                .loc[MODEL_ORDER]
            )
            center = subset["median_NRMSE_mean"].to_numpy(dtype=float) * 100
            lower = subset["NRMSE_ci_low"].to_numpy(dtype=float) * 100
            upper = subset["NRMSE_ci_high"].to_numpy(dtype=float) * 100

        for y, model, value, low, high in zip(
            y_positions, MODEL_ORDER, center, lower, upper
        ):
            axis.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markerfacecolor="white" if model == "MR-PTF" else MODEL_COLORS[model],
                markeredgewidth=1.0,
                markersize=5.2,
                elinewidth=1.3,
                capsize=2.5,
                zorder=3,
            )
            axis.annotate(
                f"{value:.1f}",
                (value, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
            )

        axis.set_yticks(y_positions, MODEL_ORDER)
        axis.set_ylim(-0.55, len(MODEL_ORDER) - 0.25)
        axis.set_xlim(left=0)
        axis.set_title(title, loc="left", fontweight="semibold", color=INK)
        axis.set_xlabel(xlabel)
        axis.text(
            0.98 if higher_better else 0.02,
            0.04,
            "higher is better →" if higher_better else "← lower is better",
            transform=axis.transAxes,
            ha="right" if higher_better else "left",
            color=MID_GREY,
            fontsize=6.5,
        )
        clean_axis(axis, "x")
        panel_label(axis, chr(ord("a") + panel_index))

    figure.suptitle(
        "Extreme-flow performance across 12 watersheds",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.945,
        "Points are basin medians (NRMSE) or the macro mean (F1); bars show 95% watershed-bootstrap intervals",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MID_GREY,
    )
    figure.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.11, hspace=0.48, wspace=0.34)
    return figure


def plot_performance_by_basin(
    regime_basin: pd.DataFrame,
    event_basin: pd.DataFrame,
) -> plt.Figure:
    """Show every basin-model result using basin colors and model markers."""
    regime_watersheds = set(regime_basin["watershed"].unique())
    event_watersheds = set(event_basin["watershed"].unique())
    watersheds = sorted(regime_watersheds & event_watersheds)
    if not watersheds:
        raise ValueError("No watersheds are shared by the regime and event tables.")
    if len(watersheds) > len(BASIN_COLORS):
        raise ValueError(
            f"The basin palette supports {len(BASIN_COLORS)} watersheds, "
            f"but {len(watersheds)} were supplied."
        )
    basin_colors = dict(zip(watersheds, BASIN_COLORS))
    basin_offsets = dict(zip(watersheds, np.linspace(-0.22, 0.22, len(watersheds))))
    specifications = [
        ("low", "Low flow (≤Q10)", "MAE (streamflow units)", True),
        ("high", "High flow (≥Q90)", "MAE (streamflow units)", False),
        ("flood_hours", "Flood hours (≥Q95)", "MAE (streamflow units)", False),
        ("event_F1", "Flood-event detection", "Event F1 (%)", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 5.55))
    y_positions = dict(zip(MODEL_ORDER, np.arange(len(MODEL_ORDER))[::-1]))

    for panel_index, (axis, (metric, title, xlabel, log_scale)) in enumerate(
        zip(axes.flat, specifications)
    ):
        if metric == "event_F1":
            table = event_basin.pivot(
                index="watershed", columns="model", values="event_F1"
            ) * 100.0
        else:
            table = regime_basin[regime_basin["regime"] == metric].pivot(
                index="watershed", columns="model", values="MAE"
            )
        table = table.reindex(index=watersheds, columns=MODEL_ORDER)

        for watershed in watersheds:
            values = table.loc[watershed, MODEL_ORDER].to_numpy(dtype=float)
            y_values = np.array(
                [
                    y_positions[model] + basin_offsets[watershed]
                    for model in MODEL_ORDER
                ],
                dtype=float,
            )
            valid = np.isfinite(values)
            axis.plot(
                values[valid],
                y_values[valid],
                color=basin_colors[watershed],
                linewidth=0.75,
                alpha=0.28,
                zorder=1,
            )

        for model in MODEL_ORDER:
            for watershed in watersheds:
                value = table.at[watershed, model]
                if not np.isfinite(value):
                    continue
                axis.scatter(
                    value,
                    y_positions[model] + basin_offsets[watershed],
                    s=24,
                    marker=MODEL_MARKERS[model],
                    facecolor=basin_colors[watershed],
                    edgecolor=INK,
                    linewidth=0.35,
                    alpha=0.92,
                    zorder=3,
                )

        axis.set_yticks(
            [y_positions[model] for model in MODEL_ORDER], MODEL_ORDER
        )
        axis.set_ylim(-0.55, len(MODEL_ORDER) - 0.45)
        if log_scale:
            axis.set_xscale("log")
        else:
            axis.set_xlim(left=0)
        axis.set_title(title, loc="left", fontweight="semibold", color=INK)
        axis.set_xlabel(xlabel)
        clean_axis(axis, "x")
        panel_label(axis, chr(ord("a") + panel_index))

    model_handles = [
        Line2D(
            [0], [0], marker=MODEL_MARKERS[model], linestyle="none",
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.8,
            markersize=5.2, label=model,
        )
        for model in MODEL_ORDER
    ]
    basin_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=basin_colors[watershed], markeredgecolor=INK,
            markeredgewidth=0.35, markersize=5.2,
            label=basin_display_name(watershed),
        )
        for watershed in watersheds
    ]
    model_legend = figure.legend(
        handles=model_handles,
        title="Model marker",
        loc="upper left",
        bbox_to_anchor=(0.065, 0.91),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    figure.add_artist(model_legend)
    figure.legend(
        handles=basin_handles,
        title="Watershed color",
        loc="upper left",
        bbox_to_anchor=(0.785, 0.86),
        ncol=1,
        frameon=False,
        handletextpad=0.45,
        labelspacing=0.55,
    )
    figure.suptitle(
        "Basin-level extreme-flow performance",
        x=0.07,
        y=0.995,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.07,
        0.955,
        "Each line connects one watershed across models (n = 12); panel a uses a log scale to retain the full low-flow range",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MID_GREY,
    )
    figure.subplots_adjust(
        left=0.10, right=0.76, top=0.78, bottom=0.11, hspace=0.52, wspace=0.34
    )
    return figure


def paired_metric_frame(
    regime_basin: pd.DataFrame,
    event_basin: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    regime = regime_basin[regime_basin["regime"].isin(["high", "flood_hours"])].pivot(
        index="watershed", columns=["regime", "model"], values="NRMSE_mean"
    )
    events = event_basin.pivot(index="watershed", columns="model", values="event_F1")
    negative = regime_basin[regime_basin["regime"] == "low"].pivot(
        index="watershed", columns="model", values="negative_prediction_rate"
    )
    return {
        "high": pd.DataFrame(
            {
                model: 100 * (regime[("high", "H-LSTM")] - regime[("high", model)])
                for model in MODEL_ORDER[1:]
            }
        ),
        "flood": pd.DataFrame(
            {
                model: 100
                * (regime[("flood_hours", "H-LSTM")] - regime[("flood_hours", model)])
                for model in MODEL_ORDER[1:]
            }
        ),
        "event": pd.DataFrame(
            {model: 100 * (events[model] - events["H-LSTM"]) for model in MODEL_ORDER[1:]}
        ),
        "negative": 100 * negative[MODEL_ORDER],
    }


def plot_connected_points(
    axis: plt.Axes,
    values: pd.DataFrame,
    model_order: Sequence[str],
    ylabel: str,
    title: str,
    zero_line: bool,
) -> None:
    x = np.arange(len(model_order), dtype=float)
    for watershed, row in values[model_order].iterrows():
        axis.plot(x, row.to_numpy(dtype=float), color=LIGHT_GREY, linewidth=0.65, zorder=1)
    for position, model in enumerate(model_order):
        series = values[model].dropna()
        axis.scatter(
            np.full(len(series), position),
            series,
            s=17,
            marker=MODEL_MARKERS[model],
            facecolor="white" if model == "MR-PTF" else MODEL_COLORS[model],
            edgecolor=MODEL_COLORS[model],
            linewidth=0.8,
            zorder=3,
        )
        median = float(series.median())
        axis.plot(
            [position - 0.20, position + 0.20],
            [median, median],
            color=INK,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=4,
        )
    if zero_line:
        axis.axhline(0, color=MID_GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    axis.set_xticks(x, model_order)
    axis.set_xlim(-0.45, len(model_order) - 0.55)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="semibold", color=INK)
    clean_axis(axis, "y")


def plot_watershed_consistency(
    regime_basin: pd.DataFrame,
    event_basin: pd.DataFrame,
) -> plt.Figure:
    """Show paired watershed gains and physically invalid low-flow outputs."""
    metrics = paired_metric_frame(regime_basin, event_basin)
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    plot_connected_points(
        axes[0, 0], metrics["high"], MODEL_ORDER[1:],
        "NRMSE reduction vs H-LSTM (pp)", "High flow (≥Q90)", True,
    )
    plot_connected_points(
        axes[0, 1], metrics["flood"], MODEL_ORDER[1:],
        "NRMSE reduction vs H-LSTM (pp)", "Flood hours (≥Q95)", True,
    )
    shared_limit = max(45.0, float(max(metrics["high"].max().max(), metrics["flood"].max().max())) * 1.08)
    axes[0, 0].set_ylim(0, shared_limit)
    axes[0, 1].set_ylim(0, shared_limit)
    plot_connected_points(
        axes[1, 0], metrics["event"], MODEL_ORDER[1:],
        "Event-F1 gain vs H-LSTM (pp)", "Flood-event detection", True,
    )
    plot_connected_points(
        axes[1, 1], metrics["negative"], MODEL_ORDER,
        "Negative predictions in Q10 hours (%)", "Low-flow physical validity", False,
    )
    axes[1, 0].set_ylim(min(-15, metrics["event"].min().min() * 1.15), 70)
    axes[1, 1].set_ylim(0, 105)
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index))
    figure.suptitle(
        "Watershed-level gains and low-flow validity",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.945,
        "Thin lines pair watersheds; thick marks denote medians (n = 12); positive values in a–c indicate improvement",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MID_GREY,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.11, hspace=0.48, wspace=0.34)
    return figure


def choose_representative_events(matches: pd.DataFrame, count: int = 2) -> pd.DataFrame:
    observed = (
        matches[matches["model"] == "H-LSTM"]
        .drop_duplicates(["watershed", "obs_event_id"])
        .copy()
    )
    observed["peak_ratio"] = observed["obs_peak_flow"] / observed["flood_threshold"]
    observed = observed.sort_values("peak_ratio", ascending=False)
    selected = []
    used_basins = set()
    for _, row in observed.iterrows():
        if row["watershed"] in used_basins:
            continue
        selected.append(row)
        used_basins.add(row["watershed"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} distinct-basin observed events are available.")
    return pd.DataFrame(selected)


def plot_representative_hydrographs(
    matches: pd.DataFrame,
    experiments_dir: Path,
    padding_hours: float,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot two high-magnitude events from different basins, normalized by Q95."""
    selected = choose_representative_events(matches, 2)
    _, files_by_experiment = common_basin_files(
        experiments_dir, EXPERIMENTS, "test_results", "test", "latest"
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for index, (axis, (_, event)) in enumerate(zip(axes, selected.iterrows())):
        basin = str(event["watershed"])
        threshold = float(event["flood_threshold"])
        event_start = pd.Timestamp(event["obs_start"])
        event_end = pd.Timestamp(event["obs_end"])
        start = event_start - pd.Timedelta(hours=padding_hours)
        end = event_end + pd.Timedelta(hours=padding_hours)
        model_frames, _ = align_basin_frames(basin, files_by_experiment, "streamflow")
        reference = model_frames[EXPERIMENTS[0][0]]
        window = reference[(reference["timestamp"] >= start) & (reference["timestamp"] <= end)]
        axis.plot(
            window["timestamp"],
            window["obs_streamflow"] / threshold,
            color=INK,
            linewidth=1.8,
            label="Observed",
            zorder=4,
        )
        for experiment, model in EXPERIMENTS:
            frame = model_frames[experiment]
            window = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
            axis.plot(
                window["timestamp"],
                window["pred_streamflow"] / threshold,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=1.25,
                label=model,
                zorder=3,
            )
        axis.axhline(1, color=MID_GREY, linewidth=0.9, linestyle=":", label="Q95")
        axis.axvspan(event_start, event_end, color=VERY_LIGHT_GREY, zorder=0)
        axis.set_title(
            f"{basin_display_name(basin)} | observed peak = {event['peak_ratio']:.1f} × Q95",
            loc="left",
            fontweight="semibold",
            color=INK,
        )
        axis.set_ylabel("Flow / basin Q95")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(axis="x", rotation=0)
        clean_axis(axis, "y")
        panel_label(axis, chr(ord("a") + index))

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=5,
        frameon=False,
        handlelength=2.4,
    )
    figure.suptitle(
        "Representative observed flood hydrographs",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.91,
        f"Flows normalized by basin Q95; shading marks the observed event with ±{padding_hours:g} h context",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MID_GREY,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.70, bottom=0.18, wspace=0.25)
    return figure, selected


def performance_by_basin_markdown_table(
    regime_basin: pd.DataFrame,
    event_basin: pd.DataFrame,
) -> str:
    """Format the values plotted in Figure 1b as a long Markdown table."""
    regimes = regime_basin[regime_basin["regime"].isin(["low", "high", "flood_hours"])].pivot(
        index=["watershed", "model"], columns="regime", values="MAE"
    )
    events = event_basin.set_index(["watershed", "model"])["event_F1"] * 100.0
    watersheds = sorted(
        set(regime_basin["watershed"].unique())
        & set(event_basin["watershed"].unique())
    )
    lines = [
        "| Watershed | Model | Q10 low-flow MAE | Q90 high-flow MAE | Q95 flood-hour MAE | Flood-event F1 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for watershed in watersheds:
        for model in MODEL_ORDER:
            index = (watershed, model)
            lines.append(
                "| "
                + " | ".join(
                    [
                        basin_display_name(watershed),
                        model,
                        f"{regimes.at[index, 'low']:.2f}",
                        f"{regimes.at[index, 'high']:.2f}",
                        f"{regimes.at[index, 'flood_hours']:.2f}",
                        f"{events.at[index]:.1f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_captions(
    path: Path,
    selected_events: pd.DataFrame,
    padding_hours: float,
    regime_basin: pd.DataFrame,
    event_basin: pd.DataFrame,
) -> None:
    event_names = [basin_display_name(str(value)) for value in selected_events["watershed"]]
    figure_1b_table = performance_by_basin_markdown_table(regime_basin, event_basin)
    text = f"""# Suggested paper figures and captions

## Figure 1 — Main performance comparison

**Suggested use:** primary Results figure.

**Caption:** Extreme-flow performance of H-LSTM, MR-STF, and MR-PTF across 12 watersheds. Points show basin-median normalized root-mean-square error (NRMSE) for low-flow (≤Q10), high-flow (≥Q90), and flood-hour (≥Q95) regimes, and macro-averaged F1 for detected flood events. Error bars are 95% intervals from 5,000 watershed bootstrap resamples. Thresholds were calculated separately for each watershed from the shared observed test period. Lower NRMSE and higher F1 indicate better performance.

## Figure 1b — Basin-level performance

**Suggested use:** basin-level companion to Figure 1 or supplementary Results figure.

**Caption:** Basin-level extreme-flow performance of H-LSTM, MR-STF, and MR-PTF across 12 watersheds. Panels a–c show mean absolute error (MAE) during observed low-flow (≤Q10), high-flow (≥Q90), and flood-hour (≥Q95) conditions, respectively; panel d shows flood-event F1. Watershed-specific colors and connecting lines identify paired results from the same watershed, while circles, squares, and diamonds denote H-LSTM, MR-STF, and MR-PTF. The low-flow MAE axis uses a logarithmic scale to retain the full cross-basin range. Thresholds were calculated separately for each watershed from the shared observed test period. Lower MAE and higher F1 indicate better performance.

### Values plotted in Figure 1b

MAE values are in the streamflow units used by the reconstruction data and are rounded to two decimal places; event F1 is reported as a percentage rounded to one decimal place.

{figure_1b_table}

## Figure 2 — Watershed consistency and physical validity

**Suggested use:** secondary Results figure or main-text robustness panel.

**Caption:** Watershed-level changes in high-flow NRMSE, flood-hour NRMSE, and flood-event F1 relative to H-LSTM, together with the fraction of low-flow hours having negative predictions. Thin lines connect results from the same watershed and thick horizontal marks show medians (n = 12). Positive changes in panels a–c indicate improvement over H-LSTM. Panel d diagnoses physically invalid negative streamflow predictions during observed Q10 conditions.

## Figure 3 — Representative hydrographs

**Suggested use:** qualitative behavior figure or supplement.

**Caption:** Reconstructed hydrographs for high-magnitude observed flood events in {event_names[0]} and {event_names[1]}. Streamflow is normalized by the basin-specific Q95 threshold; shaded intervals identify the detected observed event and the displayed record includes {padding_hours:g} h of surrounding context on each side. Normalization permits comparison of event shape, timing, and peak attenuation across basins.

## Notes for the manuscript

- Define MR-STF and MR-PTF at first mention in the paper.
- State that Q95 denotes a relative test-period extreme, not a regulatory flood stage or return-period flood.
- Low-flow NRMSE can be unstable when the observed Q10 mean is close to zero; Figure 2d provides a complementary physical-validity diagnostic.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72.")
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "paper_figures").resolve()
    tables = require_tables(results_dir)
    configure_style()

    outputs: List[Path] = []
    outputs.extend(
        save_figure(
            plot_performance_summary(tables["regime_summary"], tables["flood_summary"]),
            output_dir,
            "paper_fig1_performance_summary",
            args.formats,
            args.dpi,
        )
    )
    outputs.extend(
        save_figure(
            plot_performance_by_basin(tables["regime_basin"], tables["event_basin"]),
            output_dir,
            "paper_fig1b_performance_by_basin",
            args.formats,
            args.dpi,
        )
    )
    outputs.extend(
        save_figure(
            plot_watershed_consistency(tables["regime_basin"], tables["event_basin"]),
            output_dir,
            "paper_fig2_watershed_consistency",
            args.formats,
            args.dpi,
        )
    )
    event_figure, selected_events = plot_representative_hydrographs(
        tables["event_matches"], args.experiments_dir, args.event_padding_hours
    )
    outputs.extend(
        save_figure(
            event_figure,
            output_dir,
            "paper_fig3_representative_hydrographs",
            args.formats,
            args.dpi,
        )
    )
    captions = output_dir / "captions.md"
    write_captions(
        captions,
        selected_events,
        args.event_padding_hours,
        tables["regime_basin"],
        tables["event_basin"],
    )
    outputs.append(captions)

    print(f"Saved {len(outputs)} paper-figure files to {output_dir}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
