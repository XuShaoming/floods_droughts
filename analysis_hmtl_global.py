#!/usr/bin/env python3
"""
Analysis script for hierarchical global inference outputs.

Extends analysis_global.py by allowing per-target-group (intermediate vs final)
metrics and plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hydrology_metrics import calculate_hydrology_metrics


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


def load_manifest(results_dir: str) -> Dict:
    manifest_path = os.path.join(results_dir, "results_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def discover_result_dirs(base_dir: str) -> Dict[str, str]:
    """
    Scan the base directory for subdirectories that contain a results_manifest.json.
    Attempts to map directory names to splits (train/val/test) based on their names.
    """
    discovered: Dict[str, str] = {}
    if not os.path.isdir(base_dir):
        return discovered

    known_splits = ["train", "val", "test"]
    entries = sorted(os.listdir(base_dir))
    for entry in entries:
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path):
            continue
        manifest_path = os.path.join(path, "results_manifest.json")
        if not os.path.exists(manifest_path):
            continue

        lowered = entry.lower()
        matched = False
        for split in known_splits:
            if split in lowered:
                discovered.setdefault(split, path)
                matched = True
                break

        if not matched:
            discovered.setdefault("__default__", path)

    return discovered


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_")


def load_dataframe(path: str, file_format: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return pd.DataFrame()
    if file_format == "csv":
        df = pd.read_csv(path)
    elif file_format == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{file_format}'")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def basic_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    return calculate_hydrology_metrics(pred, obs)


def compute_metrics(df: pd.DataFrame, target_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    flat_preds = []
    flat_obs = []
    for name in target_names:
        pred_col = f"pred_{name}"
        obs_col = f"obs_{name}"
        if pred_col not in df.columns or obs_col not in df.columns:
            continue
        preds = df[pred_col].to_numpy()
        obs = df[obs_col].to_numpy()
        metrics[name] = basic_metrics(preds, obs)
        flat_preds.append(preds)
        flat_obs.append(obs)

    if flat_preds:
        stacked_preds = np.concatenate(flat_preds)
        stacked_obs = np.concatenate(flat_obs)
        metrics["overall"] = basic_metrics(stacked_preds, stacked_obs)
    return metrics


def save_metrics(metrics: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_window_series(
    df: pd.DataFrame,
    watershed: str,
    window_index: int,
    target_names: Sequence[str],
    output_path: str,
):
    unique_windows = sorted(df["window_index"].unique())

    if window_index < 0 or window_index >= len(unique_windows):
        print(f"Warning: Window {window_index} out of range for {watershed} (has {len(unique_windows)} windows)")
        return

    actual_window_index = unique_windows[window_index]
    subset = df[df["window_index"] == actual_window_index].sort_values("timestep")

    if subset.empty:
        print(f"Warning: No samples for {watershed} window {window_index} (global index {actual_window_index})")
        return

    plt.figure(figsize=(10, 4))
    for name in target_names:
        pred_col = f"pred_{name}"
        obs_col = f"obs_{name}"
        if pred_col not in subset.columns or obs_col not in subset.columns:
            continue
        plt.plot(subset["timestamp"], subset[obs_col], label=f"Observed {name}", linestyle="--")
        plt.plot(subset["timestamp"], subset[pred_col], label=f"Predicted {name}")

    plt.title(f"{watershed} - Window {window_index} (global index: {actual_window_index})")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_window_series_single_target(
    df: pd.DataFrame,
    watershed: str,
    window_index: int,
    target_name: str,
    output_path: str,
):
    unique_windows = sorted(df["window_index"].unique())

    if window_index < 0 or window_index >= len(unique_windows):
        print(f"Warning: Window {window_index} out of range for {watershed} (has {len(unique_windows)} windows)")
        return

    actual_window_index = unique_windows[window_index]
    subset = df[df["window_index"] == actual_window_index].sort_values("timestep")

    if subset.empty:
        print(f"Warning: No samples for {watershed} window {window_index} (global index {actual_window_index})")
        return

    pred_col = f"pred_{target_name}"
    obs_col = f"obs_{target_name}"
    if pred_col not in subset.columns or obs_col not in subset.columns:
        print(f"Warning: Columns missing for target {target_name} in window plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(subset["timestamp"], subset[obs_col], label=f"Observed {target_name}", linestyle="--")
    plt.plot(subset["timestamp"], subset[pred_col], label=f"Predicted {target_name}")
    plt.title(f"{watershed} - Window {window_index} ({target_name})")
    plt.xlabel("Timestamp")
    plt.ylabel(target_name)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction_series(
    df: pd.DataFrame,
    watershed: str,
    method: str,
    target_names: Sequence[str],
    output_path: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
):
    subset = df.sort_values("timestamp")
    if start is not None:
        subset = subset[subset["timestamp"] >= start]
    if end is not None:
        subset = subset[subset["timestamp"] <= end]

    if subset.empty:
        print(f"Warning: No samples for {watershed} reconstruction ({method}) in requested range.")
        return

    plt.figure(figsize=(12, 4))
    for name in target_names:
        pred_col = f"pred_{name}"
        obs_col = f"obs_{name}"
        if pred_col not in subset.columns or obs_col not in subset.columns:
            continue
        plt.plot(subset["timestamp"], subset[obs_col], label=f"Observed {name}", linestyle="--")
        plt.plot(subset["timestamp"], subset[pred_col], label=f"Predicted {name}")

    plt.title(f"{watershed} - Reconstruction ({method})")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction_series_single_target(
    df: pd.DataFrame,
    watershed: str,
    method: str,
    target_name: str,
    output_path: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
):
    subset = df.sort_values("timestamp")
    if start is not None:
        subset = subset[subset["timestamp"] >= start]
    if end is not None:
        subset = subset[subset["timestamp"] <= end]

    if subset.empty:
        print(f"Warning: No samples for {watershed} reconstruction ({method}) for target {target_name}.")
        return

    pred_col = f"pred_{target_name}"
    obs_col = f"obs_{target_name}"
    if pred_col not in subset.columns or obs_col not in subset.columns:
        print(f"Warning: Columns missing for target {target_name} in reconstruction plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(subset["timestamp"], subset[obs_col], label=f"Observed {target_name}", linestyle="--")
    plt.plot(subset["timestamp"], subset[pred_col], label=f"Predicted {target_name}")
    plt.title(f"{watershed} - Reconstruction ({method}) ({target_name})")
    plt.xlabel("Timestamp")
    plt.ylabel(target_name)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_watershed_list(value):
    entries = normalize_list(value)
    return [str(item) for item in entries]


def get_window_indices_for_split(
    cfg: Union[Dict, Sequence, int, None], split_key: str, watershed: str
) -> List[int]:
    def resolve(node):
        if isinstance(node, dict):
            return normalize_list(node.get(watershed, node.get("__default__", [])))
        return normalize_list(node)

    if isinstance(cfg, dict):
        if split_key in cfg:
            return [int(idx) for idx in resolve(cfg[split_key])]
        if "__default__" in cfg:
            return [int(idx) for idx in resolve(cfg["__default__"])]
        if watershed in cfg:
            return [int(idx) for idx in resolve(cfg[watershed])]
        return []

    return [int(idx) for idx in normalize_list(cfg)]


def get_methods_for_split(
    cfg: Union[Dict, Sequence, str, None],
    split_key: str,
    default_methods: Sequence[str],
) -> List[str]:
    value = None
    if isinstance(cfg, dict):
        value = cfg.get(split_key) or cfg.get("__default__")
    else:
        value = cfg

    if value is None:
        return list(default_methods)

    return [str(method).lower() for method in normalize_list(value)]


def get_timeseries_requests_for_watershed(timeseries_cfg, split_key: str, watershed: str) -> List[Dict]:
    def interpret(node):
        if node is None:
            return []
        if isinstance(node, list):
            return node
        return [node]

    if not isinstance(timeseries_cfg, dict):
        return []

    split_cfg = timeseries_cfg.get(split_key)
    if isinstance(split_cfg, dict):
        requests = split_cfg.get(watershed, split_cfg.get("__default__", []))
        return interpret(requests)

    requests = timeseries_cfg.get(watershed, timeseries_cfg.get("__default__", []))
    return interpret(requests)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze hierarchical global inference outputs.")
    parser.add_argument("--config", type=str, default="config_global.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="streamflow_hmtl_global_inference",
        help="Inference experiment entry to use from the config file.",
    )
    parser.add_argument("--source-experiment", type=str, default=None, help="Override source experiment directory.")
    parser.add_argument("--results-subdir", type=str, default=None, help="Override results sub-directory.")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Checkpoint filename to infer results folder.")
    parser.add_argument(
        "--checkpoint-choice",
        type=str,
        default=None,
        help="If multiple checkpoints are defined, pick which one to analyze.",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to analyze (train/val/test). Use multiple to analyze multiple subdirectories.",
    )
    parser.add_argument(
        "--reconstruction-methods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of reconstruction methods to analyze.",
    )
    parser.add_argument(
        "--target-group",
        type=str,
        default="all",
        choices=["all", "final", "intermediate"],
        help="Filter targets for metrics/plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config, exp_name, full_config = get_experiment_config(args.config, args.experiment)
    print(f"Analyzing inference experiment: {exp_name}")

    source_experiment = args.source_experiment or config.get("source_experiment") or config.get("source_experiment_name")
    if not source_experiment:
        raise ValueError("source_experiment must be specified for analysis.")
    source_config = full_config.get(source_experiment)
    if source_config is None:
        raise ValueError(f"source_experiment '{source_experiment}' not found in {args.config}.")

    combined_config: Dict = {}
    combined_config.update(source_config)
    combined_config.update(config)

    checkpoint_spec = args.checkpoint_file or config.get("checkpoint_files") or config.get("checkpoint_file")
    checkpoint_choice = args.checkpoint_choice or config.get("checkpoint_choice")
    checkpoint_name = None

    if args.checkpoint_file:
        checkpoint_name = args.checkpoint_file
    elif isinstance(checkpoint_spec, dict):
        choice = checkpoint_choice or "best"
        checkpoint_name = checkpoint_spec.get(choice)
        if checkpoint_name is None:
            raise ValueError(f"Checkpoint choice '{choice}' not found in checkpoint_files.")
    elif isinstance(checkpoint_spec, str):
        checkpoint_name = checkpoint_spec
    else:
        checkpoint_name = "best_model.pth"

    derived_subdir = f"{Path(checkpoint_name).stem}_results"
    default_subdir = args.results_subdir or config.get("results_subdir") or derived_subdir
    base_results_dir = os.path.join(combined_config["save_dir"], source_experiment)

    results_subdirs_cfg = config.get("results_subdirs")
    results_dir_map: Dict[str, str] = {}
    canonical_splits = ["train", "val", "test"]
    if isinstance(results_subdirs_cfg, dict):
        for split_name, subdir in results_subdirs_cfg.items():
            results_dir_map[split_name.lower()] = os.path.join(base_results_dir, subdir)
    else:
        results_dir_map["__default__"] = os.path.join(base_results_dir, default_subdir)

    for split_name in canonical_splits:
        default_path = os.path.join(base_results_dir, f"{split_name}_results")
        results_dir_map.setdefault(split_name, default_path)

    existing_dirs = [path for path in results_dir_map.values() if os.path.exists(path)]
    if not existing_dirs:
        auto_discovered = discover_result_dirs(base_results_dir)
        results_dir_map.update(auto_discovered)
        existing_dirs = [path for path in results_dir_map.values() if os.path.exists(path)]

    if not existing_dirs:
        raise FileNotFoundError(
            f"No results directories found. Checked: {list(results_dir_map.values())}"
        )

    reference_manifest = load_manifest(existing_dirs[0])
    file_format_default = reference_manifest.get("window_file_format", config.get("window_file_format", "csv"))

    final_targets = reference_manifest.get("final_targets", combined_config["target_cols"])
    intermediate_targets = reference_manifest.get("intermediate_targets", combined_config.get("intermediate_targets", []))

    if args.target_group == "final":
        target_names = final_targets
    elif args.target_group == "intermediate":
        target_names = intermediate_targets
    else:
        target_names = reference_manifest.get("target_names", intermediate_targets + final_targets)

    available_splits = [split for split in results_dir_map.keys() if split != "__default__"]
    if not available_splits:
        manifest_splits = reference_manifest.get("requested_splits") or reference_manifest.get("splits")
        if manifest_splits:
            available_splits = [str(split).lower() for split in manifest_splits]

    preferred_order = ["train", "val", "test"]
    if available_splits:
        default_splits = [split for split in preferred_order if split in available_splits]
        remaining = [split for split in available_splits if split not in default_splits]
        default_splits.extend(remaining)
    else:
        default_splits = preferred_order.copy()

    def normalize_split_args(split_values: Optional[Sequence[str]]) -> List[str]:
        if not split_values:
            return []
        normalized: List[str] = []
        for token in split_values:
            cleaned = token.strip().strip("[]").strip().strip("\"'").strip(",").lower()
            if cleaned:
                normalized.append(cleaned)
        return normalized

    split_arg = normalize_split_args(args.split)
    if split_arg:
        if len(split_arg) == 1 and split_arg[0] == "all":
            splits_to_analyze = default_splits
        else:
            splits_to_analyze = split_arg
    else:
        splits_to_analyze = default_splits

    reconstruction_methods = (
        args.reconstruction_methods
        or config.get("reconstruction_methods")
        or reference_manifest.get("reconstruction_methods")
        or ["average", "latest"]
    )
    reconstruction_methods = [method.lower() for method in reconstruction_methods]
    reconstruction_methods = list(dict.fromkeys(reconstruction_methods))

    metrics_subdir = config.get("metrics_output", "analysis_results")

    window_plot_cfg = config.get("window_plots", {})
    timeseries_cfg = config.get("timeseries_plots", {})
    timeseries_methods_cfg = config.get("method")

    for split in splits_to_analyze:
        split_key = split.lower()
        split_results_dir = results_dir_map.get(split_key) or results_dir_map.get("__default__")
        if split_results_dir is None or not os.path.exists(split_results_dir):
            print(f"Skipping split '{split}' - results directory missing.")
            continue

        manifest = load_manifest(split_results_dir)
        file_format = manifest.get("window_file_format", file_format_default)
        split_reconstruction_methods = (
            args.reconstruction_methods
            or config.get("reconstruction_methods")
            or manifest.get("reconstruction_methods")
            or reconstruction_methods
        )
        split_reconstruction_methods = [method.lower() for method in split_reconstruction_methods]
        split_reconstruction_methods = list(dict.fromkeys(split_reconstruction_methods))

        analysis_dir = os.path.join(split_results_dir, metrics_subdir)
        ensure_dir(analysis_dir)

        split_summary = {"windowed": {}, "reconstructed": {}}

        split_analysis_dir = os.path.join(analysis_dir, split_key)
        split_plots_dir = os.path.join(split_analysis_dir, "plots")
        window_plot_dir = os.path.join(split_plots_dir, "windows")
        timeseries_plot_dir = os.path.join(split_plots_dir, "reconstructions")
        ensure_dir(split_analysis_dir)
        ensure_dir(split_plots_dir)

        manifest_watersheds = manifest.get("watersheds_by_split", {}).get(split_key, [])
        config_watersheds = combined_config.get("watersheds")
        split_config_watersheds = normalize_watershed_list(config.get(split_key))
        watersheds = manifest_watersheds or config_watersheds or []
        if split_config_watersheds:
            available = (
                set(watersheds)
                if watersheds
                else {
                    f.replace(f"_{split_key}_windowed_timeseries.{file_format}", "")
                    for f in os.listdir(split_results_dir)
                    if f.endswith(f"_{split_key}_windowed_timeseries.{file_format}")
                }
            )
            watersheds = [ws for ws in split_config_watersheds if ws in available]
        if not watersheds:
            suffix = f"_{split_key}_windowed_timeseries.{file_format}"
            watersheds = [
                f.replace(suffix, "")
                for f in os.listdir(split_results_dir)
                if f.endswith(suffix)
            ]

        if not watersheds:
            print(f"No watershed files found to analyze for split '{split}'. Skipping.")
            continue

        for watershed in watersheds:
            ws_key = sanitize_name(watershed)
            window_file = os.path.join(split_results_dir, f"{ws_key}_{split_key}_windowed_timeseries.{file_format}")
            window_df = load_dataframe(window_file, file_format)
            if window_df.empty:
                print(f"Skipping {watershed}: windowed data unavailable for split '{split}'.")
                continue

            window_metrics = compute_metrics(window_df, target_names)
            split_summary["windowed"][watershed] = window_metrics
            save_metrics(
                window_metrics,
                os.path.join(split_analysis_dir, f"{ws_key}_{split_key}_windowed_metrics.json"),
            )

            window_indices = get_window_indices_for_split(window_plot_cfg, split_key, watershed)
            for idx in window_indices:
                plot_window_series(
                    window_df,
                    watershed,
                    int(idx),
                    target_names,
                    os.path.join(window_plot_dir, f"{ws_key}_window_{idx}.png"),
                )
                for target_name in target_names:
                    plot_window_series_single_target(
                        window_df,
                        watershed,
                        int(idx),
                        target_name,
                        os.path.join(window_plot_dir, f"{ws_key}_window_{idx}_{target_name}.png"),
                    )

            recon_dfs: Dict[str, pd.DataFrame] = {}
            for method in split_reconstruction_methods:
                recon_file = os.path.join(split_results_dir, f"{ws_key}_{split_key}_reconstructed_{method}.{file_format}")
                recon_df = load_dataframe(recon_file, file_format)
                if recon_df.empty:
                    print(f"Skipping {watershed} reconstruction ({method}) for split '{split}': file missing or empty.")
                    continue

                recon_dfs[method] = recon_df
                recon_metrics = compute_metrics(recon_df, target_names)
                split_summary["reconstructed"].setdefault(method, {})[watershed] = recon_metrics
                save_metrics(
                    recon_metrics,
                    os.path.join(split_analysis_dir, f"{ws_key}_{split_key}_reconstructed_{method}_metrics.json"),
                )

            plot_requests = get_timeseries_requests_for_watershed(timeseries_cfg, split_key, watershed)
            if not plot_requests:
                requested_methods = get_methods_for_split(timeseries_methods_cfg, split_key, split_reconstruction_methods)
                plot_requests = [{"method": requested_methods}]

            for idx, request in enumerate(plot_requests):
                request_methods = request.get("method")
                if request_methods is None:
                    request_methods = list(recon_dfs.keys())
                elif isinstance(request_methods, str):
                    request_methods = [request_methods]
                request_methods = [str(method).lower() for method in request_methods]

                request_start = pd.Timestamp(request.get("start")) if request.get("start") else None
                request_end = pd.Timestamp(request.get("end")) if request.get("end") else None

                for method in request_methods:
                    recon_df = recon_dfs.get(method)
                    if recon_df is None:
                        continue
                    plot_reconstruction_series(
                        recon_df,
                        watershed,
                        method,
                        target_names,
                        os.path.join(timeseries_plot_dir, f"{ws_key}_{method}_plot_{idx}.png"),
                        start=request_start,
                        end=request_end,
                    )
                    for target_name in target_names:
                        plot_reconstruction_series_single_target(
                            recon_df,
                            watershed,
                            method,
                            target_name,
                            os.path.join(
                                timeseries_plot_dir,
                                f"{ws_key}_{method}_plot_{idx}_{target_name}.png",
                            ),
                            start=request_start,
                            end=request_end,
                        )

        split_summary_path = os.path.join(split_analysis_dir, "summary_metrics.json")
        with open(split_summary_path, "w") as f:
            json.dump(split_summary, f, indent=2)
        print(f"Saved analysis artifacts for split '{split}' to {split_analysis_dir}")


if __name__ == "__main__":
    main()
