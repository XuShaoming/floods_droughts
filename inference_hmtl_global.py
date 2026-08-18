#!/usr/bin/env python3
"""
Inference script for global hierarchical LSTM models.

Loads checkpoints from train_hmtl_global.py, rebuilds the data pipeline, runs
predictions across requested splits, denormalizes both intermediate and final
targets, and saves windowed plus reconstructed time series per watershed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from dataloader_hmtl_global import GlobalHierarchicalDataLoader
from models.CTLSTM_Global import CTLSTMGlobal


def resolve_device(device_cfg):
    if device_cfg is None or device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device_cfg, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_cfg}")
        return torch.device("cpu")

    if isinstance(device_cfg, str):
        trimmed = device_cfg.strip().lower()
        if trimmed.isdigit():
            if torch.cuda.is_available():
                return torch.device(f"cuda:{trimmed}")
            return torch.device("cpu")
        if trimmed in {"cpu", "cuda"}:
            if trimmed == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(trimmed)
        if trimmed.startswith("cuda:") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_cfg)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_experiment_config(config_path: str, experiment: Optional[str]):
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)

    if experiment:
        if experiment not in full_config:
            available = [k for k in full_config.keys() if not k.startswith("base_") and k != "default_experiment"]
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def group_windows_by_watershed(predictions, targets, metadata):
    grouped = {}
    watersheds = metadata.get("watershed", [])
    scenarios = metadata.get("scenario", [])
    dates = metadata.get("dates", [])

    for idx in range(len(predictions)):
        watershed = watersheds[idx] if idx < len(watersheds) else f"watershed_{idx}"
        scenario = scenarios[idx] if idx < len(scenarios) else None
        grouped.setdefault(watershed, []).append(
            {
                "window_index": idx,
                "scenario": scenario,
                "pred": predictions[idx],
                "obs": targets[idx],
                "dates": dates[idx] if idx < len(dates) else [],
            }
        )
    return grouped


def build_window_dataframe(entries, target_names, split_name):
    """
    Vectorized construction of the windowed dataframe to avoid per-timestep Python loops.
    """
    frames: List[pd.DataFrame] = []

    for entry in entries:
        dates = entry.get("dates", [])
        if dates is None or len(dates) == 0:
            continue

        window_len = len(dates)
        pred_array = entry["pred"]
        obs_array = entry["obs"]
        if pred_array.shape[0] != window_len or obs_array.shape[0] != window_len:
            continue

        data_dict: Dict[str, Union[np.ndarray, List]] = {
            "split": np.full(window_len, split_name),
            "window_index": np.full(window_len, entry["window_index"]),
            "timestep": np.arange(window_len, dtype=np.int32),
            "timestamp": pd.to_datetime(dates),
        }

        scenario = entry.get("scenario")
        if scenario is not None:
            data_dict["scenario"] = np.full(window_len, scenario)

        for idx, name in enumerate(target_names):
            data_dict[f"pred_{name}"] = pred_array[:, idx]
            data_dict[f"obs_{name}"] = obs_array[:, idx]

        frames.append(pd.DataFrame(data_dict))

    if not frames:
        columns = ["split", "window_index", "timestep", "timestamp"] + [
            col for name in target_names for col in (f"pred_{name}", f"obs_{name}")
        ]
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def reconstruct_time_series(entries, target_names, method: str, stride: Optional[int], window_size: Optional[int]):
    stride = stride or 1
    timeline: Dict[pd.Timestamp, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_tracker: Dict[pd.Timestamp, str] = {}

    for window_idx, entry in enumerate(entries):
        scenario = entry.get("scenario")
        dates = entry.get("dates", [])
        pred_array = entry["pred"]
        obs_array = entry["obs"]

        if window_idx == 0 or method == "average":
            start_idx = 0
        else:
            start_idx = max(0, len(dates) - stride)

        for idx in range(start_idx, len(dates)):
            ts = pd.Timestamp(dates[idx])
            if method == "average" and ts in timeline:
                prev_pred, prev_obs = timeline[ts]
                timeline[ts] = ((prev_pred + pred_array[idx]) / 2.0, (prev_obs + obs_array[idx]) / 2.0)
            else:
                timeline[ts] = (pred_array[idx], obs_array[idx])
            if scenario is not None:
                scenario_tracker[ts] = scenario

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
    grouped = group_windows_by_watershed(predictions, targets, metadata)
    saved_paths: Dict[str, Dict[str, str]] = {}

    total_watersheds = len(grouped)
    for idx, (watershed, entries) in enumerate(grouped.items(), start=1):
        print(
            f"[{split_name}] Processing watershed {watershed} "
            f"({idx}/{total_watersheds}, {len(entries)} windows)"
        )
        watershed_key = watershed.replace(" ", "_")
        window_df = build_window_dataframe(entries, target_names, split_name)
        window_filename = f"{watershed_key}_{split_name}_windowed_timeseries.{file_format}"
        window_path = os.path.join(output_dir, window_filename)
        save_dataframe(window_df.sort_values(["window_index", "timestep"]), window_path, file_format)
        print(f"[{split_name}] Saved windowed timeseries for watershed: {watershed} at {window_path}")
        saved_paths.setdefault(watershed, {})
        saved_paths[watershed]["windowed"] = window_path

        for method in reconstruction_methods:
            print(f"[{split_name}] Reconstructing time series for watershed: {watershed} using method: {method}")
            recon_df = reconstruct_time_series(entries, target_names, method, stride=stride, window_size=window_size)
            recon_df = recon_df.sort_values("timestamp").reset_index(drop=True)
            recon_filename = f"{watershed_key}_{split_name}_reconstructed_{method}.{file_format}"
            recon_path = os.path.join(output_dir, recon_filename)
            save_dataframe(recon_df, recon_path, file_format)
            saved_paths[watershed][f"reconstructed_{method}"] = recon_path

    return saved_paths


def load_model(
    model_dir: str,
    checkpoint_name: str,
    device_spec: Optional[Union[str, int]] = None,
) -> Tuple[CTLSTMGlobal, Dict, Dict, torch.device, str]:
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

    device = resolve_device(device_spec)
    model = CTLSTMGlobal(
        input_size=model_config["input_size"],
        intermediate_targets=model_config.get("intermediate_targets", []),
        final_targets=model_config["target_cols"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        static_input_size=model_config.get("static_input_size", 0),
        static_embedding_layers=model_config.get("static_embedding_layers"),
        static_dropout=model_config.get("static_dropout", 0.0),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
) -> Tuple[GlobalHierarchicalDataLoader, Dict[str, torch.utils.data.DataLoader]]:
    dataset_splits = config.get("dataset_splits")
    if not dataset_splits:
        raise ValueError("dataset_splits must be defined in the experiment config.")

    loader = GlobalHierarchicalDataLoader(
        data_dir=config["data_dir"],
        watersheds=watersheds or config.get("watersheds"),
        scenarios=scenarios or config.get("scenarios"),
        csv_pattern=config.get("csv_pattern", "{watershed}_{scenario}_combined.csv"),
        window_size=config["window_size"],
        stride=config["stride"],
        target_cols=config["target_cols"],
        feature_cols=config.get("feature_cols"),
        intermediate_targets=config.get("intermediate_targets"),
        batch_size=config.get("batch_size", 64),
        scale_features=config.get("scale_features", True),
        scale_targets=config.get("scale_targets", True),
        scale_intermediate_targets=config.get("scale_intermediate_targets", True),
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
    model: CTLSTMGlobal,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    intermediate_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final_predictions = []
    final_targets = []
    intermediate_predictions = []
    intermediate_targets = []
    num_intermediate = len(intermediate_names)

    with torch.no_grad():
        for dynamic_feats, static_feats, final_y, intermediate_y, _ in tqdm(dataloader, desc="Running inference"):
            dynamic_feats = dynamic_feats.to(device)
            final_y = final_y.to(device)
            static_feats = static_feats.to(device) if model.static_input_size > 0 and static_feats.numel() > 0 else None
            intermediate_y = intermediate_y.to(device) if intermediate_y.numel() > 0 else torch.empty(0, device=device)

            outputs = model(dynamic_feats, static_inputs=static_feats)
            final_predictions.append(outputs["final"].cpu().numpy())
            final_targets.append(final_y.cpu().numpy())

            if num_intermediate > 0 and intermediate_y.numel() > 0:
                preds_stack = []
                target_stack = []
                for idx, name in enumerate(intermediate_names):
                    preds_stack.append(outputs["intermediate"][name].cpu().numpy())
                    target_stack.append(intermediate_y[:, :, idx : idx + 1].cpu().numpy())
                intermediate_predictions.append(np.concatenate(preds_stack, axis=-1))
                intermediate_targets.append(np.concatenate(target_stack, axis=-1))

    final_predictions = np.concatenate(final_predictions, axis=0) if final_predictions else np.empty((0,))
    final_targets = np.concatenate(final_targets, axis=0) if final_targets else np.empty((0,))

    if intermediate_predictions:
        intermediate_predictions = np.concatenate(intermediate_predictions, axis=0)
        intermediate_targets = np.concatenate(intermediate_targets, axis=0)
    else:
        intermediate_predictions = np.empty((final_predictions.shape[0], final_predictions.shape[1], 0))
        intermediate_targets = np.empty_like(intermediate_predictions)

    return final_predictions, final_targets, intermediate_predictions, intermediate_targets


def denormalize_array(array: np.ndarray, scaler) -> np.ndarray:
    if scaler is None or array.size == 0:
        return array
    shape = array.shape
    reshaped = array.reshape(-1, shape[-1])
    denorm = scaler.inverse_transform(reshaped)
    return denorm.reshape(shape)


def concatenate_targets(intermediate_array: np.ndarray, final_array: np.ndarray) -> np.ndarray:
    if intermediate_array.size == 0:
        return final_array
    if final_array.size == 0:
        return intermediate_array
    return np.concatenate([intermediate_array, final_array], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for hierarchical global LSTM models.")
    parser.add_argument("--config", type=str, default="config_global.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="streamflow_hmtl_global_inference",
        help="Experiment entry to use from the config file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic loaders.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: auto, cpu, 0, 1, ... or cuda:N. Overrides the config file.",
    )
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
        "--source-experiment",
        type=str,
        default=None,
        help="Name of the trained experiment directory to load checkpoints from.",
    )
    parser.add_argument(
        "--reconstruction-methods",
        type=str,
        nargs="+",
        default=None,
        help="Override reconstruction methods (e.g., average latest).",
    )
    return parser.parse_args()


def main():
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

    reconstruction_methods = args.reconstruction_methods or config.get("reconstruction_methods") or ["average", "latest"]
    if not reconstruction_methods:
        reconstruction_methods = ["average"]
    reconstruction_methods = [method.lower() for method in reconstruction_methods]
    reconstruction_methods = list(dict.fromkeys(reconstruction_methods))

    device_cfg = args.device if args.device is not None else combined_config.get("device", "auto")
    model, _, model_config, device, checkpoint_path = load_model(model_root, checkpoint_name, device_cfg)
    seed = args.seed if args.seed is not None else combined_config.get("seed", 42)

    data_loader, loaders = build_data_loader(combined_config, seed, args.watersheds, args.scenarios)
    final_names = combined_config["target_cols"]
    intermediate_names = combined_config.get("intermediate_targets", [])
    all_target_names = intermediate_names + final_names

    manifest = {
        "inference_experiment": exp_name,
        "source_experiment": source_experiment,
        "checkpoint_choice": checkpoint_choice,
        "checkpoint_path": checkpoint_path,
        "requested_splits": splits_to_run,
        "splits": [],
        "reconstruction_methods": reconstruction_methods,
        "target_names": all_target_names,
        "final_targets": final_names,
        "intermediate_targets": intermediate_names,
        "dataset_choice": dataset_choice if dataset_choice is not None else "test",
        "watersheds_by_split": {},
        "results_subdir": results_subdir,
    }

    saved_files = {}

    for split_name in splits_to_run:
        loader = loaders.get(f"{split_name}_loader")
        if loader is None or len(loader.dataset) == 0:
            print(f"Skipping {split_name} split (no samples).")
            continue

        print(f"[{split_name}] Running model forward pass...")
        final_preds, final_targets, intermediate_preds, intermediate_targets = run_inference(
            model, loader, device, intermediate_names
        )
        print(
            f"[{split_name}] Forward pass complete. Windows: {final_preds.shape[0]}, "
            f"Seq len: {final_preds.shape[1] if final_preds.ndim >= 2 else 0}"
        )

        print(f"[{split_name}] Denormalizing predictions/targets...")
        final_preds = denormalize_array(final_preds, data_loader.target_scaler)
        final_targets = denormalize_array(final_targets, data_loader.target_scaler)
        intermediate_preds = denormalize_array(intermediate_preds, data_loader.intermediate_scaler)
        intermediate_targets = denormalize_array(intermediate_targets, data_loader.intermediate_scaler)
        print(f"[{split_name}] Denormalization complete.")

        combined_preds = concatenate_targets(intermediate_preds, final_preds)
        combined_targets = concatenate_targets(intermediate_targets, final_targets)

        metadata = data_loader.metadata.get(split_name, {})
        print(f"[{split_name}] Saving windowed and reconstructed outputs...")
        split_paths = process_split_results(
            split_name,
            combined_preds,
            combined_targets,
            metadata,
            all_target_names,
            reconstruction_methods,
            output_dir,
            config.get("window_file_format", "csv"),
            stride=combined_config.get("stride"),
            window_size=combined_config.get("window_size"),
        )

        saved_files[split_name] = split_paths
        manifest["watersheds_by_split"][split_name] = sorted(split_paths.keys())
        manifest["splits"].append(split_name)
        print(f"Completed split {split_name}: saved outputs for {len(split_paths)} watersheds.")

    manifest_path = os.path.join(output_dir, "results_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nInference finished. Results stored in {output_dir}")


if __name__ == "__main__":
    main()
