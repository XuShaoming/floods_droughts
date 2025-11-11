#!/usr/bin/env python3
"""
Inference script for global multi-watershed LSTM models.

Loads checkpoints saved by train_global.py, rebuilds the data pipeline, and
runs evaluation on train/val/test splits or any subset of watersheds/periods.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import yaml

from dataloader_global import GlobalFloodDroughtDataLoader
from models.CTLSTM import CTLSTM
from train import calculate_metrics


def load_model(model_dir: str, model_name: str) -> Tuple[CTLSTM, Dict, Dict, torch.device]:
    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, "config.yaml")
    model_config_path = os.path.join(model_dir, "model_config.json")
    checkpoint_path = os.path.join(model_dir, model_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found at {model_config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
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
    return model, config, model_config, device


def build_data_loader(config: Dict, seed: int, watersheds: Optional[List[str]], scenarios: Optional[List[str]]):
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
    predictions = []
    targets = []

    with torch.no_grad():
        for dynamic_feats, static_feats, y_true in tqdm(dataloader, desc="Running inference"):
            dynamic_feats = dynamic_feats.to(device)
            static_feats = (
                static_feats.to(device)
                if model.static_input_size > 0 and static_feats.numel() > 0
                else None
            )
            outputs = model(dynamic_feats, static_inputs=static_feats)
            predictions.append(outputs.cpu().numpy())
            targets.append(y_true.numpy())

    predictions = np.concatenate(predictions, axis=0) if predictions else np.empty(0)
    targets = np.concatenate(targets, axis=0) if targets else np.empty(0)
    return predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Inference for global LSTM models.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained experiment directory.")
    parser.add_argument("--model-trained", type=str, default="best_model.pth", help="Checkpoint filename.")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split.")
    parser.add_argument("--watersheds", type=str, nargs="*", default=None, help="Optional subset of watersheds.")
    parser.add_argument("--scenarios", type=str, nargs="*", default=None, help="Optional subset of scenarios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic loaders.")
    args = parser.parse_args()

    model, config, _, device = load_model(args.model_dir, args.model_trained)
    data_loader, loaders = build_data_loader(config, args.seed, args.watersheds, args.scenarios)

    split_map = {
        "train": loaders["train_loader"],
        "val": loaders["val_loader"],
        "test": loaders["test_loader"],
    }

    metrics_collection = {}
    combined_predictions = []
    combined_targets = []

    splits_to_run = split_map.keys() if args.dataset == "all" else [args.dataset]

    for split_name in splits_to_run:
        loader = split_map[split_name]
        if len(loader.dataset) == 0:
            print(f"Skipping {split_name} split (no samples).")
            continue

        preds, targets = run_inference(model, loader, device)
        combined_predictions.append(preds)
        combined_targets.append(targets)

        metrics = calculate_metrics(preds, targets, data_loader.target_scaler)
        metrics_collection[split_name] = metrics

        metrics_path = os.path.join(args.model_dir, f"{split_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        np.save(os.path.join(args.model_dir, f"{split_name}_predictions.npy"), preds)
        np.save(os.path.join(args.model_dir, f"{split_name}_targets.npy"), targets)

        print(f"\n{split_name.capitalize()} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    if args.dataset == "all" and combined_predictions:
        preds = np.concatenate(combined_predictions, axis=0)
        targets = np.concatenate(combined_targets, axis=0)
        overall_metrics = calculate_metrics(preds, targets, data_loader.target_scaler)
        metrics_collection["all"] = overall_metrics
        with open(os.path.join(args.model_dir, "all_metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=2)
        print("\nOverall Metrics (all splits combined):")
        for key, value in overall_metrics.items():
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
