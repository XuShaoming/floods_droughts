#!/usr/bin/env python3
"""Hierarchical SCIF inference for multitask LSTM models."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from dataloader_hmtl import FloodDroughtDataLoader
from models.LSTM_HMTL import HierarchicalLSTMModel

warnings.filterwarnings("ignore")


def resolve_cmb_metadata(config: Dict, model_config: Dict) -> Dict:
    """Resolve conditional metadata describing CMB/SCIF state features."""

    metadata = config.get("cmb_metadata") or model_config.get("cmb_metadata")
    if metadata is not None:
        metadata = metadata.copy()
    else:
        state_order = list(config.get("intermediate_targets", [])) + list(config["target_cols"])
        metadata = {
            "state_targets_order": state_order,
            "state_feature_names": [f"CMB_{name}_state" for name in state_order],
            "cmb_feature_count": len(state_order),
            "window_size": config["window_size"],
            "stride": config["stride"],
            "many_to_many": True,
        }

    required_defaults = {
        "state_targets_order": list(config.get("intermediate_targets", [])) + list(config["target_cols"]),
        "cmb_feature_count": len(metadata.get("state_targets_order", [])),
        "stride": config["stride"],
        "window_size": config["window_size"],
    }

    for key, default_value in required_defaults.items():
        metadata.setdefault(key, default_value)

    metadata.setdefault(
        "state_feature_names",
        [f"CMB_{name}_state" for name in metadata.get("state_targets_order", [])],
    )
    metadata.setdefault("many_to_many", True)
    return metadata


def load_model_and_config(
    model_dir: str,
    model_trained: str,
    stride_override: Optional[int] = None,
) -> Tuple[HierarchicalLSTMModel, Dict, Dict, torch.device]:
    """Load trained model, configuration, and target device."""

    model_dir_path = Path(model_dir)
    config_path = model_dir_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must contain a mapping")

    if len(raw_config) == 1 and isinstance(next(iter(raw_config.values())), dict):
        experiment_name, config = next(iter(raw_config.items()))
    else:
        experiment_name = raw_config.get("experiment_name", "inference_experiment")
        config = raw_config

    config.setdefault("intermediate_targets", config.get("intermediate_targets", []))
    config.setdefault("target_cols", config.get("target_cols", ["streamflow"]))

    if stride_override is not None:
        print(f"Overriding stride to {stride_override}")
        config["stride"] = stride_override

    base_feature_cols = config.get("feature_cols")
    if not base_feature_cols:
        raise ValueError("Configuration must include 'feature_cols'.")

    model_path = model_dir_path / model_trained
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("config", {})

    metadata = resolve_cmb_metadata(config, model_config)
    metadata["stride"] = config["stride"]
    config["cmb_metadata"] = metadata

    input_size = len(base_feature_cols) + metadata.get("cmb_feature_count", 0)
    model = HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=config.get("intermediate_targets", []),
        final_targets=config["target_cols"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.2),
        batch_first=True,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Loaded experiment: {experiment_name}")
    print(f"Using device: {device}")

    return model, config, model_config, device


def _build_target_dict(array: Optional[np.ndarray], names: List[str]) -> Dict[str, np.ndarray]:
    """Create a dictionary mapping target names to windowed arrays."""

    if array is None or not names:
        return {}
    return {name: array[:, :, idx : idx + 1] for idx, name in enumerate(names)}


def _concatenate_hierarchical_outputs(
    intermediate_preds: Dict[str, np.ndarray],
    final_preds: Dict[str, np.ndarray],
    intermediate_targets: Dict[str, np.ndarray],
    final_targets: Dict[str, np.ndarray],
    intermediate_names: List[str],
    final_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack intermediate and final targets in canonical order."""

    pred_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []

    for name in intermediate_names:
        if name in intermediate_preds:
            pred_blocks.append(intermediate_preds[name])
        if name in intermediate_targets:
            target_blocks.append(intermediate_targets[name])

    for name in final_names:
        if name in final_preds:
            pred_blocks.append(final_preds[name])
        if name in final_targets:
            target_blocks.append(final_targets[name])

    if not pred_blocks or not target_blocks:
        raise RuntimeError("SCIF rollout produced empty predictions; check state configuration.")

    concatenated_predictions = np.concatenate(pred_blocks, axis=-1)
    concatenated_targets = np.concatenate(target_blocks, axis=-1)
    return concatenated_predictions, concatenated_targets


def _initial_state_value(
    target_name: str,
    window_idx: int,
    intermediate_targets: Optional[np.ndarray],
    final_targets: np.ndarray,
    intermediate_names: List[str],
    final_names: List[str],
) -> float:
    """Fallback initial state using observed data when predictions are unavailable."""

    if intermediate_targets is not None and target_name in intermediate_names:
        pos = intermediate_names.index(target_name)
        return float(intermediate_targets[window_idx, 0, pos])

    if target_name in final_names:
        pos = final_names.index(target_name)
        return float(final_targets[window_idx, 0, pos])

    return 0.0


def run_scif_rollout(
    model: HierarchicalLSTMModel,
    device: torch.device,
    base_features: np.ndarray,
    intermediate_targets: Optional[np.ndarray],
    final_targets: np.ndarray,
    state_order: List[str],
    intermediate_names: List[str],
    final_names: List[str],
    stride: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Sequentially roll out predictions using SCIF with dynamic state updates."""

    if base_features.ndim != 3:
        raise ValueError("Base features must be 3-dimensional (windows, timesteps, features).")

    num_windows, window_size, _ = base_features.shape
    stride_idx = min(stride, window_size - 1)

    predictions_intermediate: Dict[str, List[np.ndarray]] = {name: [] for name in intermediate_names}
    predictions_final: Dict[str, List[np.ndarray]] = {name: [] for name in final_names}
    state_values: Dict[str, float] = {}

    for window_idx in tqdm(range(num_windows), desc="SCIF rollout", leave=False):
        state_vector: List[float] = []
        for target_name in state_order:
            if target_name in state_values:
                state_val = state_values[target_name]
            else:
                state_val = _initial_state_value(
                    target_name,
                    window_idx,
                    intermediate_targets,
                    final_targets,
                    intermediate_names,
                    final_names,
                )
            state_vector.append(float(state_val))

        state_matrix = np.repeat(np.array(state_vector, dtype=np.float32)[np.newaxis, :], window_size, axis=0)
        augmented_window = np.concatenate([base_features[window_idx], state_matrix], axis=-1).astype(np.float32)

        window_tensor = torch.from_numpy(augmented_window).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(window_tensor)

        current_intermediate: Dict[str, np.ndarray] = {}
        for name in intermediate_names:
            tensor = outputs["intermediate"][name]
            pred_np = tensor.detach().cpu().numpy()[0]
            predictions_intermediate[name].append(pred_np)
            current_intermediate[name] = pred_np

        current_final: Dict[str, np.ndarray] = {}
        for name in final_names:
            tensor = outputs["final"][name]
            pred_np = tensor.detach().cpu().numpy()[0]
            predictions_final[name].append(pred_np)
            current_final[name] = pred_np

        if window_idx < num_windows - 1:
            for target_name in state_order:
                if target_name in current_intermediate:
                    seq = current_intermediate[target_name]
                else:
                    seq = current_final[target_name]
                state_values[target_name] = float(seq[stride_idx, 0])

    for name in intermediate_names:
        predictions_intermediate[name] = np.stack(predictions_intermediate[name], axis=0)
    for name in final_names:
        predictions_final[name] = np.stack(predictions_final[name], axis=0)

    return predictions_intermediate, predictions_final


def run_inference(
    model_dir: str,
    model_trained: str,
    dataset_names: Optional[List[str]] = None,
    stride_override: Optional[int] = None,
) -> Dict:
    """Run SCIF-based inference for the requested datasets."""

    model, config, model_config, device = load_model_and_config(
        model_dir, model_trained, stride_override=stride_override
    )
    cmb_metadata = config["cmb_metadata"]
    intermediate_names = list(config.get("intermediate_targets", []))
    final_names = list(config["target_cols"])

    effective_stride = config.get("stride")
    if effective_stride is None:
        raise ValueError("Configuration must define 'stride' either in config or via override.")
    cmb_metadata["stride"] = effective_stride

    data_loader = FloodDroughtDataLoader(
        csv_file=config["csv_file"],
        window_size=config["window_size"],
    stride=effective_stride,
        target_col=config["target_cols"],
        feature_cols=config["feature_cols"],
        intermediate_targets=config.get("intermediate_targets", []),
        train_years=tuple(config["train_years"]),
        val_years=tuple(config["val_years"]),
        test_years=tuple(config["test_years"]),
        batch_size=config.get("batch_size", 32),
        scale_features=config.get("scale_features", True),
        scale_targets=config.get("scale_targets", True),
        scale_intermediate_targets=config.get("scale_intermediate_targets", True),
        many_to_many=True,
        random_seed=config.get("seed", 42),
    )

    data_splits = data_loader.prepare_data()
    dataset_map = {
        "train": {
            "features": data_splits["x_train"],
            "final": data_splits["y_train"],
            "intermediate": data_splits.get("intermediate_train"),
            "dates": data_splits["date_train"],
        },
        "val": {
            "features": data_splits["x_val"],
            "final": data_splits["y_val"],
            "intermediate": data_splits.get("intermediate_val"),
            "dates": data_splits["date_val"],
        },
        "test": {
            "features": data_splits["x_test"],
            "final": data_splits["y_test"],
            "intermediate": data_splits.get("intermediate_test"),
            "dates": data_splits["date_test"],
        },
    }

    if dataset_names is None:
        dataset_names = ["train", "val", "test"]

    inference_results: Dict[str, Dict] = {}

    for dataset_name in dataset_names:
        if dataset_name not in dataset_map:
            print(f"Warning: dataset '{dataset_name}' not available in splits.")
            continue

        dataset_data = dataset_map[dataset_name]
        base_features = np.asarray(dataset_data["features"], dtype=np.float32)
        final_targets = np.asarray(dataset_data["final"], dtype=np.float32)
        intermediate_targets = (
            np.asarray(dataset_data["intermediate"], dtype=np.float32)
            if dataset_data["intermediate"] is not None
            else None
        )

        print(f"\n{'=' * 60}")
        print(f"RUNNING SCIF INFERENCE ON {dataset_name.upper()} DATASET")
        print(f"{'=' * 60}")
        print(f"Using stride: {effective_stride}{' (override)' if stride_override is not None else ''}")

        predictions_intermediate, predictions_final = run_scif_rollout(
            model=model,
            device=device,
            base_features=base_features,
            intermediate_targets=intermediate_targets,
            final_targets=final_targets,
            state_order=cmb_metadata["state_targets_order"],
            intermediate_names=intermediate_names,
            final_names=final_names,
            stride=cmb_metadata.get("stride", effective_stride),
        )

        intermediate_targets_dict = _build_target_dict(intermediate_targets, intermediate_names)
        final_targets_dict = _build_target_dict(final_targets, final_names)

        (
            denorm_intermediate_preds,
            denorm_final_preds,
            denorm_intermediate_targets,
            denorm_final_targets,
        ) = denormalize_hierarchical_data(
            predictions_intermediate,
            predictions_final,
            intermediate_targets_dict,
            final_targets_dict,
            data_loader.intermediate_target_scaler,
            data_loader.target_scaler,
        )

        concatenated_predictions, concatenated_targets = _concatenate_hierarchical_outputs(
            denorm_intermediate_preds,
            denorm_final_preds,
            denorm_intermediate_targets,
            denorm_final_targets,
            intermediate_names,
            final_names,
        )

        date_windows = dataset_data["dates"]
        all_target_names = intermediate_names + final_names

        pred_ts, target_ts = reconstruct_time_series_from_windows(
            concatenated_predictions,
            concatenated_targets,
            date_windows,
            {"target_cols": all_target_names},
            data_loader,
        )

        inference_results[dataset_name] = {
            "predictions_windowed": concatenated_predictions,
            "targets_windowed": concatenated_targets,
            "predictions_timeseries": pred_ts,
            "targets_timeseries": target_ts,
            "intermediate_predictions": denorm_intermediate_preds,
            "final_predictions": denorm_final_preds,
            "intermediate_targets": denorm_intermediate_targets,
            "final_targets": denorm_final_targets,
            "target_names_order": all_target_names,
            "date_windows": date_windows,
            "config": config,
            "model_config": model_config,
            "cmb_metadata": cmb_metadata,
        }

        print(f"Inference completed for {dataset_name}")
        print(f"  - Windows: {concatenated_predictions.shape[0]}")
        print(f"  - Window size: {concatenated_predictions.shape[1]}")
        print(f"  - Target order: {all_target_names}")

    return inference_results


# =============================================================================
# DENORMALIZATION AND RECONSTRUCTION UTILITIES
# =============================================================================

def denormalize_hierarchical_data(
    intermediate_predictions: Dict[str, np.ndarray],
    final_predictions: Dict[str, np.ndarray],
    intermediate_targets: Dict[str, np.ndarray],
    final_targets: Dict[str, np.ndarray],
    intermediate_scaler,
    final_scaler,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Denormalize hierarchical predictions and targets back to original scale."""

    denorm_intermediate_preds: Dict[str, np.ndarray] = {}
    denorm_final_preds: Dict[str, np.ndarray] = {}
    denorm_intermediate_targets: Dict[str, np.ndarray] = {}
    denorm_final_targets: Dict[str, np.ndarray] = {}

    if intermediate_scaler is not None and intermediate_predictions:
        intermediate_names = list(intermediate_predictions.keys())
        intermediate_pred_list = []
        intermediate_target_list = []

        for target_name in intermediate_names:
            pred_data = intermediate_predictions[target_name]
            target_data = intermediate_targets.get(target_name, pred_data)

            if pred_data.ndim == 3:
                pred_reshaped = pred_data.reshape(-1, 1)
                target_reshaped = target_data.reshape(-1, 1)
                intermediate_pred_list.append(pred_reshaped)
                intermediate_target_list.append(target_reshaped)
            else:
                intermediate_pred_list.append(pred_data)
                intermediate_target_list.append(target_data)

        if intermediate_pred_list:
            all_intermediate_preds = np.concatenate(intermediate_pred_list, axis=1)
            all_intermediate_targets = np.concatenate(intermediate_target_list, axis=1)

            denorm_all_preds = intermediate_scaler.inverse_transform(all_intermediate_preds)
            denorm_all_targets = intermediate_scaler.inverse_transform(all_intermediate_targets)

            offset = 0
            for target_name in intermediate_names:
                orig_shape = intermediate_predictions[target_name].shape
                denorm_intermediate_preds[target_name] = denorm_all_preds[:, offset : offset + 1].reshape(orig_shape)
                denorm_intermediate_targets[target_name] = denorm_all_targets[:, offset : offset + 1].reshape(orig_shape)
                offset += 1
    else:
        denorm_intermediate_preds = {k: v.copy() for k, v in intermediate_predictions.items()}
        denorm_intermediate_targets = {k: v.copy() for k, v in intermediate_targets.items()}

    if final_scaler is not None and final_predictions:
        final_names = list(final_predictions.keys())
        final_pred_list = []
        final_target_list = []

        for target_name in final_names:
            pred_data = final_predictions[target_name]
            target_data = final_targets.get(target_name, pred_data)

            if pred_data.ndim == 3:
                pred_reshaped = pred_data.reshape(-1, 1)
                target_reshaped = target_data.reshape(-1, 1)
                final_pred_list.append(pred_reshaped)
                final_target_list.append(target_reshaped)
            else:
                final_pred_list.append(pred_data)
                final_target_list.append(target_data)

        if final_pred_list:
            all_final_preds = np.concatenate(final_pred_list, axis=1)
            all_final_targets = np.concatenate(final_target_list, axis=1)

            denorm_all_preds = final_scaler.inverse_transform(all_final_preds)
            denorm_all_targets = final_scaler.inverse_transform(all_final_targets)

            offset = 0
            for target_name in final_names:
                orig_shape = final_predictions[target_name].shape
                denorm_final_preds[target_name] = denorm_all_preds[:, offset : offset + 1].reshape(orig_shape)
                denorm_final_targets[target_name] = denorm_all_targets[:, offset : offset + 1].reshape(orig_shape)
                offset += 1
    else:
        denorm_final_preds = {k: v.copy() for k, v in final_predictions.items()}
        denorm_final_targets = {k: v.copy() for k, v in final_targets.items()}

    return denorm_intermediate_preds, denorm_final_preds, denorm_intermediate_targets, denorm_final_targets


def reconstruct_time_series_from_windows(
    predictions: np.ndarray,
    targets: np.ndarray,
    date_windows: List[pd.DatetimeIndex],
    config: Dict,
    data_loader: FloodDroughtDataLoader,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct full time series from windowed predictions using dataloader's method."""

    target_names = config["target_cols"]

    pred_ts, _ = data_loader.reconstruct_time_series(
        predictions, date_windows, target_names, aggregation_method="mean"
    )
    target_ts, _ = data_loader.reconstruct_time_series(
        targets, date_windows, target_names, aggregation_method="mean"
    )

    if not isinstance(pred_ts, pd.DataFrame):
        pred_ts = pd.DataFrame()
    if not isinstance(target_ts, pd.DataFrame):
        target_ts = pd.DataFrame()

    return pred_ts, target_ts


# =============================================================================
# ANALYSIS FUNCTIONS (METRICS AND VISUALIZATION)
# =============================================================================

def calculate_comprehensive_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = ""
) -> Dict:
    """Calculate hydrologically relevant evaluation metrics."""

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
    """Plot selected windows comparing predictions and ground truth."""

    if y_true.ndim != 3 or y_pred.ndim != 3:
        return

    n_windows, window_size, n_features = y_true.shape

    if window_indices is None:
        step = max(1, n_windows // max_windows)
        window_indices = list(range(0, n_windows, step))[:max_windows]
    else:
        window_indices = [idx for idx in window_indices if 0 <= idx < n_windows]
        if not window_indices:
            return

    fig, axes = plt.subplots(len(window_indices), 1, figsize=(15, 4 * len(window_indices)))
    if len(window_indices) == 1:
        axes = [axes]

    fig.suptitle(f"{dataset_name.upper()} Set: Windowed Time Series", fontsize=16, fontweight="bold")

    for axis, window_idx in zip(axes, window_indices):
        for feature_idx in range(n_features):
            time_steps = np.arange(window_size)
            axis.plot(
                time_steps,
                y_true[window_idx, :, feature_idx],
                label=f"Truth (Feature {feature_idx})",
                alpha=0.8,
                linewidth=2,
            )
            axis.plot(
                time_steps,
                y_pred[window_idx, :, feature_idx],
                label=f"Pred (Feature {feature_idx})",
                alpha=0.8,
                linewidth=2,
                linestyle="--",
            )

        axis.set_title(f"Window {window_idx} of {n_windows}")
        axis.set_xlabel("Time Step in Window")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        axis.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_windowed_timeseries.png"), dpi=300, bbox_inches="tight")

    plt.show()


def plot_predictions_vs_truth(
    y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str, save_dir: Optional[str] = None
) -> None:
    """Scatter and residual plots comparing predictions and ground truth."""

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    residuals = y_pred_flat - y_true_flat

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{dataset_name.upper()} Set: Predictions Analysis", fontsize=16, fontweight="bold")

    axes[0].scatter(y_true_flat, y_pred_flat, alpha=0.6, s=1)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Ground Truth")
    axes[0].set_ylabel("Predictions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", lw=2, label="Zero Error")
    axes[1].axvline(x=np.mean(residuals), color="g", linestyle="--", lw=2, label=f"Mean Error: {np.mean(residuals):.2f}")
    axes[1].set_xlabel("Residuals (Pred - Truth)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_predictions_analysis.png"), dpi=300, bbox_inches="tight")

    plt.show()


def calculate_timeseries_metrics(
    pred_ts: pd.DataFrame, target_ts: pd.DataFrame, dataset_name: str = ""
) -> Dict:
    """Metric calculation on reconstructed time-series data."""

    combined_df = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, how="inner")
    if len(combined_df) == 0:
        return {}

    pred_cols = [col for col in combined_df.columns if col in pred_ts.columns]
    target_cols = [col for col in combined_df.columns if col in target_ts.columns]

    metrics = {}
    for pred_col, target_col in zip(pred_cols, target_cols):
        y_pred = combined_df[pred_col].values
        y_true = combined_df[target_col].values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        if len(y_pred_clean) == 0:
            continue
        metrics[target_col] = calculate_comprehensive_metrics(
            y_true_clean.reshape(-1, 1), y_pred_clean.reshape(-1, 1), f"{dataset_name}_{target_col}"
        )

    return metrics


def plot_time_series_reconstruction(
    pred_ts: pd.DataFrame,
    target_ts: pd.DataFrame,
    dataset_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Plot reconstructed predictions vs ground truth over time."""

    combined_df = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, how="inner", suffixes=("_pred", "_target"))
    if len(combined_df) == 0:
        return

    if start_date or end_date:
        if start_date:
            combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date)]
        if end_date:
            combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date)]
        if len(combined_df) == 0:
            return

    pred_cols = [col for col in combined_df.columns if col.endswith("_pred")]
    target_cols = [col for col in combined_df.columns if col.endswith("_target")]

    base_cols = []
    for pred_col in pred_cols:
        base_name = pred_col.replace("_pred", "")
        target_col = base_name + "_target"
        if target_col in target_cols:
            base_cols.append(base_name)

    if not base_cols:
        return

    fig, axes = plt.subplots(len(base_cols), 1, figsize=(16, 5 * len(base_cols)))
    if len(base_cols) == 1:
        axes = [axes]

    fig.suptitle(f"{dataset_name.upper()} Set: Reconstructed Time Series", fontsize=16, fontweight="bold")

    for axis, base_col in zip(axes, base_cols):
        pred_col = base_col + "_pred"
        target_col = base_col + "_target"
        axis.plot(combined_df.index, combined_df[target_col], label="Ground Truth", alpha=0.8, linewidth=1.5)
        axis.plot(combined_df.index, combined_df[pred_col], label="Predictions", alpha=0.8, linewidth=1.5)
        axis.set_title(base_col)
        axis.set_xlabel("Date")
        axis.set_ylabel("Value")
        axis.legend()
        axis.grid(True, alpha=0.3)

        if len(combined_df) > 365:
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        elif len(combined_df) > 30:
            axis.xaxis.set_major_locator(mdates.WeekdayLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        else:
            axis.xaxis.set_major_locator(mdates.DayLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        axis.tick_params(axis="x", rotation=45)

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

    plt.show()


def analyze_inference_results(
    inference_results: Dict, save_dir: Optional[str] = None, analyze_features: Optional[List[str]] = None
) -> Dict:
    """Compute metrics and generate visualizations for inference outputs."""

    all_metrics: Dict[str, Dict] = {}
    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(exist_ok=True)

    for dataset_name, results in inference_results.items():
        concatenated_preds = results["predictions_windowed"]
        concatenated_targets = results["targets_windowed"]
        pred_ts = results["predictions_timeseries"]
        target_ts = results["targets_timeseries"]
        target_names_order = results["target_names_order"]

        features_to_analyze = analyze_features if analyze_features is not None else target_names_order
        features_to_analyze = [f for f in features_to_analyze if f in target_names_order]
        if not features_to_analyze:
            continue

        all_metrics[dataset_name] = {}

        for feature_idx, feature_name in enumerate(target_names_order):
            if feature_name not in features_to_analyze:
                continue

            feature_pred_windowed = concatenated_preds[:, :, feature_idx : feature_idx + 1]
            feature_target_windowed = concatenated_targets[:, :, feature_idx : feature_idx + 1]

            feature_pred_ts = pred_ts[[feature_name]] if feature_name in pred_ts.columns else None
            feature_target_ts = target_ts[[feature_name]] if feature_name in target_ts.columns else None
            if feature_pred_ts is None or feature_target_ts is None:
                continue

            windowed_metrics = calculate_comprehensive_metrics(
                feature_target_windowed, feature_pred_windowed, f"{dataset_name}_{feature_name}_windowed"
            )
            ts_metrics = calculate_timeseries_metrics(
                feature_pred_ts, feature_target_ts, f"{dataset_name}_{feature_name}_timeseries"
            )

            all_metrics[dataset_name][feature_name] = {"windowed": windowed_metrics, "timeseries": ts_metrics}

            feature_save_dir = None
            if save_dir_path:
                feature_save_dir = save_dir_path / feature_name
                feature_save_dir.mkdir(exist_ok=True)

            plot_predictions_vs_truth(
                feature_target_windowed,
                feature_pred_windowed,
                f"{dataset_name}_{feature_name}",
                str(feature_save_dir) if feature_save_dir else None,
            )
            plot_windowed_time_series(
                feature_target_windowed,
                feature_pred_windowed,
                f"{dataset_name}_{feature_name}_windowed",
                window_indices=None,
                max_windows=5,
                save_dir=str(feature_save_dir) if feature_save_dir else None,
            )
            plot_time_series_reconstruction(
                feature_pred_ts,
                feature_target_ts,
                f"{dataset_name}_{feature_name}_timeseries",
                start_date=None,
                end_date=None,
                save_dir=str(feature_save_dir) if feature_save_dir else None,
            )

            if feature_save_dir:
                combined_feature_ts = pd.merge(
                    feature_pred_ts,
                    feature_target_ts,
                    left_index=True,
                    right_index=True,
                    how="inner",
                    suffixes=("_prediction", "_observation"),
                )

                column_mapping = {}
                for col in combined_feature_ts.columns:
                    if col.endswith("_prediction") or col.endswith("_observation"):
                        base_name = col.rsplit("_", 1)[0]
                        column_mapping[col] = col
                    elif col.endswith("_x"):
                        base_name = col[:-2]
                        column_mapping[col] = f"{base_name}_prediction"
                    elif col.endswith("_y"):
                        base_name = col[:-2]
                        column_mapping[col] = f"{base_name}_observation"
                if column_mapping:
                    combined_feature_ts.rename(columns=column_mapping, inplace=True)

                feature_csv_path = feature_save_dir / f"{dataset_name}_{feature_name}_reconstructed_timeseries.csv"
                combined_feature_ts.to_csv(feature_csv_path)

                feature_metrics_path = feature_save_dir / f"{dataset_name}_{feature_name}_metrics.json"
                with open(feature_metrics_path, "w") as f:
                    json.dump(all_metrics[dataset_name][feature_name], f, indent=2)

    if save_dir_path:
        all_metrics_path = save_dir_path / "all_metrics.json"
        with open(all_metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    print_analysis_summary(all_metrics, str(save_dir_path) if save_dir_path else None)
    return all_metrics


def print_analysis_summary(all_metrics: Dict, save_dir: Optional[str] = None) -> None:
    """Print and optionally persist a summary of analysis metrics."""

    summary_lines = ["=" * 60, "HIERARCHICAL ANALYSIS SUMMARY", "=" * 60]

    for dataset_name, dataset_metrics in all_metrics.items():
        summary_lines.append(f"\n{dataset_name.upper()} Dataset:")
        for feature_name, feature_metrics in dataset_metrics.items():
            summary_lines.append(f"\n  {feature_name}:")
            if "windowed" in feature_metrics:
                windowed = feature_metrics["windowed"]
                summary_lines.append("    Windowed Data:")
                summary_lines.append(f"      R²:    {windowed.get('r2', float('nan')):.4f}")
                summary_lines.append(f"      RMSE:  {windowed.get('rmse', float('nan')):.2f}")
                summary_lines.append(f"      NSE:   {windowed.get('nse', float('nan')):.4f}")
                summary_lines.append(f"      KGE:   {windowed.get('kge', float('nan')):.4f}")
            if "timeseries" in feature_metrics and feature_metrics["timeseries"]:
                ts_metrics = feature_metrics["timeseries"]
                for var_name, var_metrics in ts_metrics.items():
                    summary_lines.append(f"    Time Series ({var_name}):")
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


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical SCIF Inference for Flood/Drought Prediction")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument(
        "--model-trained",
        type=str,
        default="best_model.pth",
        help="Filename of the trained model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="Process a single dataset split (default: all splits)",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run comprehensive analysis with metrics and visualizations",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride used for inference windows",
    )
    parser.add_argument(
        "--analyze-features",
        type=str,
        default=None,
        help='Comma-separated list of specific targets to analyze (e.g., "PET,ET,streamflow")',
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    analyze_features = [f.strip() for f in args.analyze_features.split(",")] if args.analyze_features else None
    dataset_names = [args.dataset] if args.dataset else None

    inference_results = run_inference(
        args.model_dir,
        args.model_trained,
        dataset_names,
        stride_override=args.stride,
    )

    save_dir = Path(args.model_dir) / f"{Path(args.model_trained).stem}_results_{args.stride}"
    save_dir.mkdir(exist_ok=True)

    results_filename = f"inference_results_{args.dataset or 'all'}.pkl"
    with open(save_dir / results_filename, "wb") as f:
        pickle.dump(inference_results, f)

    print(f"Inference results saved to: {save_dir / results_filename}")

    if args.analysis:
        print(f"\n{'=' * 60}")
        print("RUNNING COMPREHENSIVE FEATURE-WISE ANALYSIS")
        print(f"{'=' * 60}")
        analyze_inference_results(inference_results, str(save_dir), analyze_features)

        print(f"\nAll analysis artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()


