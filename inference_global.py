#!/usr/bin/env python3
"""
Inference script for global multi-watershed LSTM models.

Loads checkpoints saved by train_global.py, rebuilds the data pipeline,
and runs evaluation on the requested dataset split. The script denormalizes
predictions, saves per-watershed windowed outputs, and reconstructs full
time series using configurable strategies.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from dataloader_global import GlobalFloodDroughtDataLoader
from models.CTLSTM import CTLSTM
from train import calculate_metrics


def get_experiment_config(config_path: str, experiment: Optional[str]) -> Tuple[Dict, str, Dict]:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        experiment: Name of the experiment to load. If None, uses default_experiment.
    
    Returns:
        Tuple of (experiment_config, experiment_name, full_config_dict).
    
    Raises:
        ValueError: If experiment not found or no experiments in config.
    """
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)

    if experiment:
        if experiment not in full_config:
            available = [
                k for k in full_config.keys() if not k.startswith("base_") and k != "default_experiment"
            ]
            raise ValueError(f"Experiment '{experiment}' not found. Available: {available}")
        return full_config[experiment], experiment, full_config

    default_exp = full_config.get("default_experiment")
    if default_exp and default_exp in full_config:
        return full_config[default_exp], default_exp, full_config

    for key, value in full_config.items():
        if key.startswith("base_") or key == "default_experiment":
            continue
        return value, key, full_config

    raise ValueError("No experiments found in configuration file.")


def load_model(model_dir: str, checkpoint_name: str) -> Tuple[CTLSTM, Dict, Dict, torch.device, str]:
    """
    Load a trained CTLSTM model from checkpoint with its configurations.
    
    Args:
        model_dir: Directory containing the saved model and config files.
        checkpoint_name: Name of the checkpoint file (e.g., 'best_model.pth').
    
    Returns:
        Tuple of (model, training_config, model_config, device, checkpoint_path).
    
    Raises:
        FileNotFoundError: If config or checkpoint files are missing.
    """
    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, "config.yaml")
    model_config_path = os.path.join(model_dir, "model_config.json")
    checkpoint_path = os.path.join(model_dir, checkpoint_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found at {model_config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    with open(config_path, "r") as f:
        training_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTLSTM(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        output_size=model_config["output_size"],
        dropout=model_config["dropout"],
        static_input_size=model_config.get("static_input_size", 0),
        static_embedding_dim=model_config.get("static_embedding_dim"),
        static_embedding_layers=model_config.get("static_embedding_layers"),
        static_dropout=model_config.get("static_dropout", 0.0),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path} on {device}")
    return model, training_config, model_config, device, checkpoint_path


def build_data_loader(
    config: Dict,
    seed: int,
    watersheds: Optional[List[str]],
    scenarios: Optional[List[str]],
) -> Tuple[GlobalFloodDroughtDataLoader, Dict[str, torch.utils.data.DataLoader]]:
    """
    Initialize global data loader and create train/val/test DataLoader objects.
    
    Args:
        config: Configuration dictionary with data loading parameters.
        seed: Random seed for reproducibility.
        watersheds: Optional list of watersheds to filter. If None, uses config.
        scenarios: Optional list of scenarios to filter. If None, uses config.
    
    Returns:
        Tuple of (GlobalFloodDroughtDataLoader, dict of split_name -> DataLoader).
    
    Raises:
        ValueError: If dataset_splits not defined in config.
    """
    dataset_splits = config.get("dataset_splits")
    if not dataset_splits:
        raise ValueError("dataset_splits must be defined in the experiment config.")

    loader = GlobalFloodDroughtDataLoader(
        data_dir=config["data_dir"],
        watersheds=watersheds or config.get("watersheds"),
        scenarios=scenarios or config.get("scenarios"),
        csv_pattern=config.get("csv_pattern", "{watershed}_{scenario}_combined.csv"),
        window_size=config["window_size"],
        stride=config["stride"],
        target_cols=config["target_cols"],
        feature_cols=config.get("feature_cols"),
        batch_size=config.get("batch_size", 64),
        scale_features=config.get("scale_features", True),
        scale_targets=config.get("scale_targets", True),
        many_to_many=config.get("many_to_many", True),
        random_seed=seed,
        use_static_attributes=config.get("use_static_attributes", True),
        static_attributes_file=config.get("static_attributes_file"),
        static_attribute_id_col=config.get("static_attribute_id_col", "characteristic_id"),
        static_attribute_value_col=config.get("static_attribute_value_col", "value"),
        static_attribute_model_col=config.get("static_attribute_model_col", "model"),
        scale_static_attributes=config.get("scale_static_attributes", True),
        dataset_splits=dataset_splits,
        scenario_date_ranges=config.get("scenario_date_ranges"),
    )
    loaders = loader.create_data_loaders(shuffle_train=False)
    return loader, loaders


def run_inference(
    model: CTLSTM,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a dataloader and collect predictions and targets.
    
    Args:
        model: Trained CTLSTM model in eval mode.
        dataloader: DataLoader providing batches of (dynamic_feats, static_feats, targets).
        device: Device to run inference on (CPU or CUDA).
    
    Returns:
        Tuple of (predictions, targets) as numpy arrays with shape (num_windows, seq_len, num_targets).
    """
    predictions = []
    targets = []

    with torch.no_grad():
        for dynamic_feats, static_feats, y_true in tqdm(dataloader, desc="Running inference"):
            dynamic_feats = dynamic_feats.to(device)
            static_feats = (
                static_feats.to(device)
                if model.static_input_size > 0 and static_feats is not None and static_feats.numel() > 0
                else None
            )
            outputs = model(dynamic_feats, static_inputs=static_feats)
            predictions.append(outputs.cpu().numpy())
            targets.append(y_true.numpy())

    predictions = np.concatenate(predictions, axis=0) if predictions else np.empty(0)
    targets = np.concatenate(targets, axis=0) if targets else np.empty(0)
    return predictions, targets


def ensure_dir(path: str):
    """
    Create directory if it doesn't exist, including parent directories.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def denormalize_sequences(
    predictions: np.ndarray,
    targets: np.ndarray,
    scaler,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denormalize predictions and targets using the specified scaler.
    
    Args:
        predictions: Normalized predictions array of shape (num_windows, seq_len, num_targets).
        targets: Normalized targets array of shape (num_windows, seq_len, num_targets).
        scaler: Fitted scaler object with inverse_transform method (e.g., StandardScaler).
        method: Denormalization method. Only 'target_scaler' is currently supported.
    
    Returns:
        Tuple of (denormalized_predictions, denormalized_targets).
    """
    if predictions.size == 0:
        return predictions, targets

    method = (method or "target_scaler").lower()
    if method != "target_scaler" or scaler is None:
        return predictions, targets

    orig_shape = predictions.shape
    preds_flat = predictions.reshape(-1, orig_shape[-1])
    targs_flat = targets.reshape(-1, orig_shape[-1])

    preds_denorm = scaler.inverse_transform(preds_flat).reshape(orig_shape)
    targs_denorm = scaler.inverse_transform(targs_flat).reshape(orig_shape)
    return preds_denorm, targs_denorm


def group_windows_by_watershed(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: Dict[str, List],
) -> Dict[str, List[Dict]]:
    """
    Group prediction windows by watershed for separate processing.
    
    Args:
        predictions: Predictions array of shape (num_windows, seq_len, num_targets).
        targets: Targets array of shape (num_windows, seq_len, num_targets).
        metadata: Dictionary with 'watershed', 'scenario', and 'dates' lists.
    
    Returns:
        Dictionary mapping watershed_name -> list of window dictionaries containing
        window_index, scenario, dates, pred, and obs.
    
    Raises:
        ValueError: If metadata length doesn't match number of windows.
    """
    watersheds = metadata.get("watershed", [])
    scenarios = metadata.get("scenario", [])
    date_windows = metadata.get("dates", [])

    if len(watersheds) != len(predictions):
        raise ValueError("Metadata length does not match number of prediction windows for this split.")

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for idx, watershed in enumerate(watersheds):
        entry = {
            "window_index": idx,
            "scenario": scenarios[idx] if scenarios else None,
            "dates": list(pd.to_datetime(date_windows[idx])) if date_windows else [],
            "pred": predictions[idx],
            "obs": targets[idx],
        }
        grouped[watershed].append(entry)
    return grouped


def build_window_dataframe(
    entries: List[Dict],
    target_names: Sequence[str],
    split_name: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with all windowed predictions in long format.
    
    Args:
        entries: List of window dictionaries with dates, pred, obs, scenario, window_index.
        target_names: List of target variable names (e.g., ['streamflow']).
        split_name: Name of the dataset split (e.g., 'train', 'val', 'test').
    
    Returns:
        DataFrame with columns: split, window_index, timestep, timestamp, scenario (optional),
        pred_{target}, obs_{target} for each target.
    """
    records = []
    for entry in entries:
        scenario = entry.get("scenario")
        dates = entry.get("dates", [])
        if not dates:
            continue
        for offset, timestamp in enumerate(dates):
            row = {
                "split": split_name,
                "window_index": entry["window_index"],
                "timestep": offset,
                "timestamp": pd.Timestamp(timestamp),
            }
            if scenario is not None:
                row["scenario"] = scenario
            for target_idx, target_name in enumerate(target_names):
                row[f"pred_{target_name}"] = entry["pred"][offset, target_idx]
                row[f"obs_{target_name}"] = entry["obs"][offset, target_idx]
            records.append(row)
    return pd.DataFrame(records)


def reconstruct_time_series(
    entries: List[Dict],
    target_names: Sequence[str],
    method: str,
    stride: Optional[int] = None,
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reconstruct full time series from overlapping prediction windows.
    
    Args:
        entries: List of window dictionaries with dates, pred, obs.
        target_names: List of target variable names.
        method: Reconstruction method - 'average' (average overlapping predictions),
                'latest' or 'tail' (prefer later windows for overlapping timestamps).
        stride: Number of timesteps between window starts. If provided, used for non-overlapping
                reconstruction in 'latest'/'tail' methods.
        window_size: Total number of timesteps in each window. Optional, for validation.
    
    Returns:
        DataFrame with columns: timestamp, pred_{target}, obs_{target} for each target,
        and optionally scenario if present in entries.
    
    Raises:
        ValueError: If method is not one of 'average', 'latest', or 'tail'.
    """
    method = method.lower()
    if method not in {"average", "latest", "tail"}:
        raise ValueError(f"Unknown reconstruction method '{method}'.")

    if method == "average":
        accumulator: Dict[pd.Timestamp, Dict[str, np.ndarray]] = {}
        for entry in entries:
            dates = entry.get("dates", [])
            for ts, pred_vec, obs_vec in zip(dates, entry["pred"], entry["obs"]):
                timestamp = pd.Timestamp(ts)
                if timestamp not in accumulator:
                    accumulator[timestamp] = {
                        "pred_sum": np.zeros(len(target_names), dtype=np.float64),
                        "obs_sum": np.zeros(len(target_names), dtype=np.float64),
                        "count": 0,
                    }
                accumulator[timestamp]["pred_sum"] += pred_vec
                accumulator[timestamp]["obs_sum"] += obs_vec
                accumulator[timestamp]["count"] += 1

        rows = []
        for timestamp in sorted(accumulator.keys()):
            aggregates = accumulator[timestamp]
            row = {"timestamp": timestamp}
            for idx, name in enumerate(target_names):
                row[f"pred_{name}"] = aggregates["pred_sum"][idx] / aggregates["count"]
                row[f"obs_{name}"] = aggregates["obs_sum"][idx] / aggregates["count"]
            rows.append(row)
        return pd.DataFrame(rows)

    # Non-overlapping reconstruction preferring later windows ("latest"/"tail")
    # Strategy: Keep all timesteps from first window, then only keep the last 'stride'
    # timesteps from subsequent windows to avoid overlap
    entries_sorted = sorted(
        entries,
        key=lambda e: pd.Timestamp(e["dates"][0]) if e.get("dates") else pd.Timestamp.min,
    )

    if stride is None:
        raise ValueError("stride parameter is required for 'latest'/'tail' reconstruction methods")

    timeline: Dict[pd.Timestamp, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_tracker: Dict[pd.Timestamp, Optional[str]] = {}

    for window_idx, entry in enumerate(entries_sorted):
        scenario = entry.get("scenario")
        dates = entry.get("dates", [])
        pred_array = entry["pred"]
        obs_array = entry["obs"]
        
        if window_idx == 0:
            # First window: keep all timesteps
            start_idx = 0
        else:
            # Subsequent windows: only keep the last 'stride' timesteps
            start_idx = len(dates) - stride
        
        for idx in range(start_idx, len(dates)):
            ts = dates[idx]
            timestamp = pd.Timestamp(ts)
            timeline[timestamp] = (pred_array[idx], obs_array[idx])
            scenario_tracker[timestamp] = scenario

    rows = []
    for timestamp in sorted(timeline.keys()):
        pred_vec, obs_vec = timeline[timestamp]
        row = {"timestamp": timestamp}
        for idx, name in enumerate(target_names):
            row[f"pred_{name}"] = pred_vec[idx]
            row[f"obs_{name}"] = obs_vec[idx]
        if scenario_tracker.get(timestamp) is not None:
            row["scenario"] = scenario_tracker[timestamp]
        rows.append(row)
    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: str, file_format: str = "csv"):
    """
    Save DataFrame to disk in the specified format.
    
    Args:
        df: DataFrame to save.
        path: Output file path.
        file_format: Output format - 'csv' or 'parquet'.
    
    Raises:
        ValueError: If file_format is not supported.
    """
    ensure_dir(os.path.dirname(path))
    if df.empty:
        print(f"Warning: No data to save for {path}")
        return
    if file_format == "csv":
        df.to_csv(path, index=False)
    elif file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format '{file_format}'")


def process_split_results(
    split_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: Dict[str, List],
    target_names: Sequence[str],
    reconstruction_methods: Sequence[str],
    output_dir: str,
    file_format: str,
    stride: Optional[int] = None,
    window_size: Optional[int] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Process and save inference results for a dataset split.
    
    Groups windows by watershed, saves windowed time series, and reconstructs
    full time series using specified methods.
    
    Args:
        split_name: Name of the split (e.g., 'train', 'val', 'test').
        predictions: Predictions array of shape (num_windows, seq_len, num_targets).
        targets: Targets array of shape (num_windows, seq_len, num_targets).
        metadata: Dictionary with 'watershed', 'scenario', and 'dates' lists.
        target_names: List of target variable names.
        reconstruction_methods: List of methods to use (e.g., ['average', 'latest']).
        output_dir: Directory to save output files.
        file_format: File format for output ('csv' or 'parquet').
        stride: Number of timesteps between window starts, used for reconstruction.
        window_size: Total number of timesteps in each window.
    
    Returns:
        Dictionary mapping watershed_name -> dict of file_type -> file_path.
    """
    grouped = group_windows_by_watershed(predictions, targets, metadata)
    saved_paths: Dict[str, Dict[str, str]] = {}

    for watershed, entries in grouped.items():
        watershed_key = watershed.replace(" ", "_")
        window_df = build_window_dataframe(entries, target_names, split_name)
        window_filename = f"{watershed_key}_{split_name}_windowed_timeseries.{file_format}"
        window_path = os.path.join(output_dir, window_filename)
        save_dataframe(window_df.sort_values(["window_index", "timestep"]), window_path, file_format)

        saved_paths.setdefault(watershed, {})
        saved_paths[watershed]["windowed"] = window_path

        for method in reconstruction_methods:
            recon_df = reconstruct_time_series(
                entries, target_names, method, stride=stride, window_size=window_size
            )
            recon_df = recon_df.sort_values("timestamp").reset_index(drop=True)
            recon_filename = f"{watershed_key}_{split_name}_reconstructed_{method}.{file_format}"
            recon_path = os.path.join(output_dir, recon_filename)
            save_dataframe(recon_df, recon_path, file_format)
            saved_paths[watershed][f"reconstructed_{method}"] = recon_path

    return saved_paths


def parse_args():
    """
    Parse command-line arguments for inference script.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Inference for global LSTM models.")
    parser.add_argument("--config", type=str, default="config_global.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="streamflow_global_exp1_inference",
        help="Experiment entry to use from the config file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic loaders.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["train", "val", "test", "all"],
        help="Dataset split(s) to run inference on.",
    )
    parser.add_argument("--watersheds", type=str, nargs="*", default=None, help="Optional subset of watersheds.")
    parser.add_argument("--scenarios", type=str, nargs="*", default=None, help="Optional subset of scenarios.")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Override checkpoint filename directly.")
    parser.add_argument(
        "--checkpoint-choice",
        type=str,
        default=None,
        help="If multiple checkpoints are defined in the config, pick which one to use.",
    )
    parser.add_argument(
        "--reconstruction-methods",
        type=str,
        nargs="+",
        default=None,
        help="Override reconstruction methods (e.g., average latest).",
    )
    parser.add_argument(
        "--source-experiment",
        type=str,
        default=None,
        help="Name of the trained experiment directory to load checkpoints from.",
    )
    return parser.parse_args()


def main():
    """
    Main inference pipeline for global multi-watershed LSTM models.
    
    Loads model checkpoint, rebuilds data pipeline, runs inference on specified
    dataset splits, denormalizes predictions, saves windowed and reconstructed
    time series by watershed, and computes evaluation metrics.
    """
    args = parse_args()
    config, exp_name, full_config = get_experiment_config(args.config, args.experiment)
    print(f"Using inference experiment: {exp_name}")

    source_experiment = args.source_experiment or config.get("source_experiment") or config.get("source_experiment_name")
    if not source_experiment:
        raise ValueError("source_experiment must be provided in config or via --source-experiment.")
    source_config = full_config.get(source_experiment)
    if source_config is None:
        raise ValueError(f"source_experiment '{source_experiment}' not found in {args.config}.")

    combined_config: Dict = {}
    combined_config.update(source_config)
    combined_config.update(config)

    checkpoint_spec = args.checkpoint_file or config.get("checkpoint_files") or config.get("checkpoint_file")
    checkpoint_choice = args.checkpoint_choice or config.get("checkpoint_choice")
    checkpoint_name = None
    results_subdir = config.get("results_subdir")

    if args.checkpoint_file:
        checkpoint_name = args.checkpoint_file
    elif isinstance(checkpoint_spec, dict):
        choice = checkpoint_choice or "best"
        checkpoint_name = checkpoint_spec.get(choice)
        if checkpoint_name is None:
            raise ValueError(f"Checkpoint choice '{choice}' not found in checkpoint_files.")
        if results_subdir is None:
            results_subdir = f"{choice}_results"
    elif isinstance(checkpoint_spec, str):
        checkpoint_name = checkpoint_spec
    else:
        checkpoint_name = "best_model.pth"

    if results_subdir is None:
        results_subdir = f"{Path(checkpoint_name).stem}_results"

    model_root = os.path.join(combined_config["save_dir"], source_experiment)
    output_dir = os.path.join(model_root, results_subdir)
    ensure_dir(output_dir)

    def normalize_splits(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    dataset_arg = args.dataset
    config_split = config.get("split")
    dataset_choice = dataset_arg if dataset_arg is not None else config_split
    valid_choices = {"train", "val", "test", "all"}

    if dataset_arg is not None and dataset_arg not in valid_choices:
        raise ValueError(f"Unsupported dataset choice '{dataset_arg}'. Valid options: {sorted(valid_choices)}")

    if dataset_arg == "all":
        splits_to_run = ["train", "val", "test"]
    elif dataset_arg is not None:
        splits_to_run = [dataset_arg]
    else:
        normalized = normalize_splits(config_split)
        if not normalized:
            splits_to_run = ["test"]
        else:
            splits_to_run = normalized

    aggregate_all = dataset_arg == "all"

    reconstruction_methods = args.reconstruction_methods or config.get("reconstruction_methods") or ["average", "latest"]
    if not reconstruction_methods:
        reconstruction_methods = ["average"]
    reconstruction_methods = [method.lower() for method in reconstruction_methods]
    reconstruction_methods = list(dict.fromkeys(reconstruction_methods))

    denorm_method = config.get("denormalization_method", "target_scaler")
    file_format = config.get("window_file_format", "csv")

    model, _, model_config, device, checkpoint_path = load_model(model_root, checkpoint_name)
    seed = args.seed if args.seed is not None else combined_config.get("seed", 42)

    data_loader, loaders = build_data_loader(combined_config, seed, args.watersheds, args.scenarios)
    target_names = combined_config["target_cols"]

    manifest = {
        "inference_experiment": exp_name,
        "source_experiment": source_experiment,
        "checkpoint_choice": checkpoint_choice,
        "checkpoint_path": checkpoint_path,
        "requested_splits": splits_to_run,
        "splits": [],
        "reconstruction_methods": reconstruction_methods,
        "window_file_format": file_format,
        "target_names": target_names,
        "dataset_choice": dataset_choice if dataset_choice is not None else "test",
        "watersheds_by_split": {},
        "results_subdir": results_subdir,
    }

    overall_predictions = []
    overall_targets = []
    metrics_collection = {}
    saved_files = {}

    for split_name in splits_to_run:
        loader = loaders.get(f"{split_name}_loader")
        if loader is None or len(loader.dataset) == 0:
            print(f"Skipping {split_name} split (no samples).")
            continue

        preds, targets = run_inference(model, loader, device)
        preds, targets = denormalize_sequences(preds, targets, data_loader.target_scaler, denorm_method)

        metadata = data_loader.metadata.get(split_name, {})
        split_paths = process_split_results(
            split_name,
            preds,
            targets,
            metadata,
            target_names,
            reconstruction_methods,
            output_dir,
            file_format,
            stride=combined_config.get("stride"),
            window_size=combined_config.get("window_size"),
        )

        saved_files[split_name] = split_paths
        manifest["watersheds_by_split"][split_name] = sorted(split_paths.keys())
        manifest["splits"].append(split_name)

        metrics = calculate_metrics(preds, targets, scaler=None)
        metrics_collection[split_name] = metrics
        with open(os.path.join(output_dir, f"{split_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        overall_predictions.append(preds)
        overall_targets.append(targets)

    if overall_predictions and aggregate_all:
        combined_preds = np.concatenate(overall_predictions, axis=0)
        combined_tgts = np.concatenate(overall_targets, axis=0)
        combined_metrics = calculate_metrics(combined_preds, combined_tgts, scaler=None)
        metrics_collection["all"] = combined_metrics
        with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
            json.dump(combined_metrics, f, indent=2)

    manifest["saved_files"] = saved_files
    with open(os.path.join(output_dir, "results_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\nInference complete. Artifacts stored in:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
