#!/usr/bin/env python3
"""
Analysis script for per-watershed inference outputs.

Loads inference results saved by inference.py, computes metrics, and
generates diagnostic plots for windowed and reconstructed time series.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "") -> Dict:
    """Compute core and hydrological metrics on flattened arrays."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    bias = np.mean(y_pred_flat - y_true_flat)
    std_error = np.std(y_pred_flat - y_true_flat)

    nse = 1 - (np.sum((y_true_flat - y_pred_flat) ** 2) / np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))

    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else 0.0
    bias_ratio = np.mean(y_pred_flat) / (np.mean(y_true_flat) + 1e-8)
    variability_ratio = np.std(y_pred_flat) / (np.std(y_true_flat) + 1e-8)
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (bias_ratio - 1) ** 2 + (variability_ratio - 1) ** 2)

    metrics = {
        "dataset": dataset_name,
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "bias": float(bias),
        "std_error": float(std_error),
        "nse": float(nse),
        "kge": float(kge),
        "correlation": float(correlation),
        "n_samples": len(y_true_flat),
    }

    print(f"\n{dataset_name.upper()} SET METRICS:")
    print("=" * 50)
    print(f"MSE:         {mse:.2f}")
    print(f"RMSE:        {rmse:.2f}")
    print(f"MAE:         {mae:.2f}")
    print(f"R²:          {r2:.4f}")
    print(f"MAPE:        {mape:.2f}%")
    print(f"Bias:        {bias:.2f}")
    print(f"Std Error:   {std_error:.2f}")
    print(f"NSE:         {nse:.4f}")
    print(f"KGE:         {kge:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Samples:     {len(y_true_flat):,}")

    return metrics


def plot_windowed_time_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
    window_indices: Optional[List[int]] = None,
    max_windows: int = 5,
    save_dir: Optional[str] = None,
) -> None:
    """Plot a handful of windowed sequences."""
    if y_true.ndim != 3 or y_pred.ndim != 3:
        print(f"Warning: Expected 3D windowed data, got shapes {y_true.shape}, {y_pred.shape}")
        return

    n_windows, window_size, n_features = y_true.shape
    if window_indices is None:
        step = max(1, n_windows // max_windows)
        window_indices = list(range(0, n_windows, step))[:max_windows]
    else:
        window_indices = [idx for idx in window_indices if 0 <= idx < n_windows]
        if not window_indices:
            print("Warning: No valid window indices provided")
            return

    n_windows_to_plot = len(window_indices)
    fig, axes = plt.subplots(n_windows_to_plot, 1, figsize=(15, 4 * n_windows_to_plot))
    if n_windows_to_plot == 1:
        axes = [axes]

    fig.suptitle(f"{dataset_name.upper()} Set: Windowed Time Series (Selected Windows)", fontsize=16, fontweight="bold")

    for i, window_idx in enumerate(window_indices):
        for feature_idx in range(n_features):
            time_steps = np.arange(window_size)

            axes[i].plot(
                time_steps,
                y_true[window_idx, :, feature_idx],
                label=f"Ground Truth (Feature {feature_idx})",
                alpha=0.8,
                linewidth=2,
                linestyle="-",
            )
            axes[i].plot(
                time_steps,
                y_pred[window_idx, :, feature_idx],
                label=f"Predictions (Feature {feature_idx})",
                alpha=0.8,
                linewidth=2,
                linestyle="--",
            )

        axes[i].set_title(f"Window {window_idx} (of {n_windows})")
        axes[i].set_xlabel("Time Step in Window")
        axes[i].set_ylabel("Streamflow (cfs)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_windowed_timeseries.png"), dpi=300, bbox_inches="tight")
        print(f"Windowed time series plot saved: {save_dir}/{dataset_name}_windowed_timeseries.png")

    plt.show()


def plot_predictions_vs_truth(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, save_dir: Optional[str] = None) -> None:
    """Scatter vs. truth plot with residual histogram."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    residuals = y_pred_flat - y_true_flat

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{dataset_name.upper()} Set: Predictions Analysis", fontsize=16, fontweight="bold")

    axes[0].scatter(y_true_flat, y_pred_flat, alpha=0.6, s=1)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlabel("Ground Truth (cfs)")
    axes[0].set_ylabel("Predictions (cfs)")
    axes[0].set_title("Scatter Plot: Predictions vs Ground Truth")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", lw=2, label="Zero Error")
    axes[1].axvline(x=np.mean(residuals), color="g", linestyle="--", lw=2, label=f"Mean Error: {np.mean(residuals):.2f}")
    axes[1].set_xlabel("Residuals (Prediction - Ground Truth, cfs)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_predictions_analysis.png"), dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_dir}/{dataset_name}_predictions_analysis.png")

    plt.show()


def plot_time_series_reconstruction(
    pred_ts: pd.DataFrame,
    target_ts: pd.DataFrame,
    dataset_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Plot reconstructed time series comparing predictions vs ground truth."""
    combined_df = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, how="inner", suffixes=("_pred", "_target"))
    if len(combined_df) == 0:
        print(f"Warning: No overlapping data for {dataset_name} time series plot")
        return

    if start_date or end_date:
        original_length = len(combined_df)
        if start_date:
            combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date)]
        if end_date:
            combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date)]
        if len(combined_df) == 0:
            print(f"Warning: No data in specified date range for {dataset_name}")
            return
        print(f"Filtered data from {original_length} to {len(combined_df)} points for date range")

    pred_cols = [col for col in combined_df.columns if col.endswith("_pred")]
    target_cols = [col for col in combined_df.columns if col.endswith("_target")]

    base_cols = []
    for pred_col in pred_cols:
        base_name = pred_col.replace("_pred", "")
        target_col = base_name + "_target"
        if target_col in target_cols:
            base_cols.append(base_name)

    if len(base_cols) == 0:
        print(f"Warning: No matching columns for {dataset_name} time series plot")
        return

    n_vars = len(base_cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 5 * n_vars))
    if n_vars == 1:
        axes = [axes]

    title = f"{dataset_name.upper()} Set: Reconstructed Time Series"
    if start_date or end_date:
        date_range = f" ({start_date or 'start'} to {end_date or 'end'})"
        title += date_range
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, base_col in enumerate(base_cols):
        pred_col = base_col + "_pred"
        target_col = base_col + "_target"

        axes[i].plot(combined_df.index, combined_df[target_col], label="Ground Truth", alpha=0.8, linewidth=1.5, color="blue")
        axes[i].plot(combined_df.index, combined_df[pred_col], label="Predictions", alpha=0.8, linewidth=1.5, color="red")

        axes[i].set_title(f"{base_col} - Time Series Comparison")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Streamflow (cfs)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis="x", rotation=45)

        if len(combined_df) > 365:
            axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        elif len(combined_df) > 30:
            axes[i].xaxis.set_major_locator(mdates.WeekdayLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            axes[i].xaxis.set_major_locator(mdates.DayLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename_suffix = ""
        if start_date or end_date:
            filename_suffix = f"_{start_date or 'start'}_to_{end_date or 'end'}"
        plt.savefig(
            os.path.join(save_dir, f"{dataset_name}_time_series_reconstruction{filename_suffix}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Time series plot saved: {save_dir}/{dataset_name}_time_series_reconstruction{filename_suffix}.png")

    plt.show()


def calculate_timeseries_metrics(pred_ts: pd.DataFrame, target_ts: pd.DataFrame, dataset_name: str = "") -> Dict:
    """Calculate metrics on merged prediction/target time series."""
    combined_df = pd.merge(
        pred_ts,
        target_ts,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_pred", "_target"),
    )
    if len(combined_df) == 0:
        print(f"Warning: No overlapping data for {dataset_name}")
        return {}

    base_columns = []
    for col in pred_ts.columns:
        target_col = col
        pred_name = f"{col}_pred"
        target_name = f"{target_col}_target"
        if pred_name in combined_df.columns and target_name in combined_df.columns:
            base_columns.append((pred_name, target_name, col))

    if not base_columns:
        print(f"Warning: No matching columns for {dataset_name}")
        return {}

    metrics = {}
    for pred_col, target_col, base_name in base_columns:
        y_pred = combined_df[pred_col].values
        y_true = combined_df[target_col].values

        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        if len(y_pred_clean) == 0:
            continue

        var_metrics = calculate_comprehensive_metrics(
            y_true_clean.reshape(-1, 1),
            y_pred_clean.reshape(-1, 1),
            f"{dataset_name}_{base_name}",
        )

        metrics[base_name] = var_metrics

    return metrics


def print_analysis_summary(all_metrics: Dict, save_dir: Optional[str] = None) -> None:
    """Print and append a summary of metrics to a text file."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("ANALYSIS SUMMARY")
    summary_lines.append("=" * 60)

    for dataset_name, dataset_metrics in all_metrics.items():
        summary_lines.append(f"\n{dataset_name.upper()} Dataset:")

        if "windowed" in dataset_metrics:
            windowed = dataset_metrics["windowed"]
            summary_lines.append("  Windowed Data:")
            summary_lines.append(f"    R²:    {windowed.get('r2', float('nan')):.4f}")
            summary_lines.append(f"    RMSE:  {windowed.get('rmse', float('nan')):.2f}")
            summary_lines.append(f"    NSE:   {windowed.get('nse', float('nan')):.4f}")
            summary_lines.append(f"    KGE:   {windowed.get('kge', float('nan')):.4f}")

        if "timeseries" in dataset_metrics:
            ts_metrics = dataset_metrics["timeseries"]
            for method_name, method_metrics in ts_metrics.items():
                summary_lines.append(f"  Time Series ({method_name}):")
                if not method_metrics:
                    summary_lines.append("    No overlapping data available.")
                    continue
                for var_name, var_metrics in method_metrics.items():
                    summary_lines.append(f"    {var_name}:")
                    summary_lines.append(f"      R²:    {var_metrics.get('r2', float('nan')):.4f}")
                    summary_lines.append(f"      RMSE:  {var_metrics.get('rmse', float('nan')):.2f}")
                    summary_lines.append(f"      NSE:   {var_metrics.get('nse', float('nan')):.4f}")
                    summary_lines.append(f"      KGE:   {var_metrics.get('kge', float('nan')):.4f}")

    for line in summary_lines:
        print(line)

    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)

        summary_file = save_dir_path / "analysis_summary.txt"
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(summary_file, "a") as f:
            if summary_file.exists() and summary_file.stat().st_size > 0:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"NEW ANALYSIS RUN - {timestamp}\n")
                f.write("=" * 60 + "\n")
            else:
                f.write(f"ANALYSIS RUN - {timestamp}\n")

            for line in summary_lines:
                f.write(line + "\n")
            f.write("\n")

        print(f"\nAnalysis summary appended to: {summary_file}")


def analyze_inference_results(inference_results: Dict, save_dir: Optional[str] = None) -> Dict:
    """Compute metrics and produce plots for each dataset split."""
    all_metrics: Dict[str, Dict] = {}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        print(f"Analysis results will be saved to: {save_dir}")

    for dataset_name, results in inference_results.items():
        print(f"\n{'=' * 60}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'=' * 60}")

        pred_windowed = results["predictions_windowed"]
        target_windowed = results["targets_windowed"]

        reconstruction_series = results.get("reconstructed_timeseries")
        if not reconstruction_series:
            default_method = results.get("default_reconstruction_method", "default")
            reconstruction_series = {
                default_method: {
                    "predictions": results.get("predictions_timeseries", pd.DataFrame()),
                    "targets": results.get("targets_timeseries", pd.DataFrame()),
                }
            }

        windowed_metrics = calculate_comprehensive_metrics(target_windowed, pred_windowed, f"{dataset_name}_windowed")
        timeseries_metrics: Dict[str, Dict] = {}

        for method_name, series in reconstruction_series.items():
            pred_ts = series.get("predictions")
            target_ts = series.get("targets")
            if pred_ts is None or target_ts is None or pred_ts.empty or target_ts.empty:
                print(f"Skipping timeseries metrics for {dataset_name} method '{method_name}' due to empty data.")
                timeseries_metrics[method_name] = {}
                continue

            method_label = f"{dataset_name}_timeseries_{method_name}"
            timeseries_metrics[method_name] = calculate_timeseries_metrics(pred_ts, target_ts, method_label)

        all_metrics[dataset_name] = {"windowed": windowed_metrics, "timeseries": timeseries_metrics}

        print(f"\nCreating visualizations for {dataset_name} dataset...")
        plot_predictions_vs_truth(target_windowed, pred_windowed, f"{dataset_name}_analysis", save_dir)
        plot_windowed_time_series(target_windowed, pred_windowed, f"{dataset_name}_windowed", window_indices=None, max_windows=5, save_dir=save_dir)

        for method_name, series in reconstruction_series.items():
            pred_ts = series.get("predictions")
            target_ts = series.get("targets")
            if pred_ts is None or target_ts is None or pred_ts.empty or target_ts.empty:
                continue
            method_label = f"{dataset_name}_timeseries_{method_name}"
            plot_time_series_reconstruction(pred_ts, target_ts, method_label, start_date=None, end_date=None, save_dir=save_dir)

        if save_dir:
            for method_name, series in reconstruction_series.items():
                pred_ts = series.get("predictions")
                target_ts = series.get("targets")
                if pred_ts is None or target_ts is None or pred_ts.empty or target_ts.empty:
                    continue

                combined_ts = pd.merge(
                    pred_ts, target_ts, left_index=True, right_index=True, how="inner", suffixes=("_prediction", "_observation")
                )
                combined_ts.reset_index(inplace=True)

                if combined_ts.columns[0] != "date":
                    old_col_name = str(combined_ts.columns[0])
                    combined_ts.rename(columns={old_col_name: "date"}, inplace=True)

                column_mapping = {}
                for col in combined_ts.columns:
                    if col != "date":
                        if col.endswith("_prediction"):
                            base_name = col.replace("_prediction", "")
                            column_mapping[col] = f"{base_name}_prediction"
                        elif col.endswith("_observation"):
                            base_name = col.replace("_observation", "")
                            column_mapping[col] = f"{base_name}_observation"
                        elif col.endswith("_x"):
                            base_name = col.replace("_x", "")
                            column_mapping[col] = f"{base_name}_prediction"
                        elif col.endswith("_y"):
                            base_name = col.replace("_y", "")
                            column_mapping[col] = f"{base_name}_observation"

                if column_mapping:
                    combined_ts.rename(columns=column_mapping, inplace=True)

                method_safe = method_name.replace(" ", "_")
                csv_path = Path(save_dir) / f"{dataset_name}_reconstructed_timeseries_{method_safe}.csv"
                combined_ts.to_csv(csv_path, index=False)
                print(f"Time series saved: {csv_path}")

    if save_dir:
        metrics_path = Path(save_dir) / "all_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"All metrics saved: {save_dir}/all_metrics.json")

    print_analysis_summary(all_metrics, str(save_dir) if save_dir else None)
    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze inference results and generate plots.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the model directory containing results.")
    parser.add_argument("--model-trained", type=str, default="best_model.pth", help="Checkpoint name used during inference.")
    parser.add_argument("--dataset", type=str, choices=["train", "val", "test"], default=None, help="Dataset split to analyze.")
    parser.add_argument("--results-path", type=str, default=None, help="Optional direct path to inference_results_<split>.pkl.")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional directory to write analysis outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_label = args.dataset if args.dataset is not None else "None"
    results_path = (
        Path(args.results_path)
        if args.results_path
        else Path(args.model_dir) / f"{Path(args.model_trained).stem}_results" / f"inference_results_{dataset_label}.pkl"
    )

    if not results_path.exists():
        raise FileNotFoundError(f"Inference results not found: {results_path}")

    save_dir = Path(args.save_dir) if args.save_dir else results_path.parent

    with open(results_path, "rb") as f:
        inference_results = pickle.load(f)
    print(f"Loaded inference results from: {results_path}")

    analyze_inference_results(inference_results, str(save_dir))
    print(f"\nAnalysis complete. Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
