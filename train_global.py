#!/usr/bin/env python3
"""
Global LSTM Training Script for Multi-Watershed Streamflow Prediction.

This script mirrors train.py but loads multiple watershed/period CSVs and
embeds static watershed attributes. Usage:

    python train_global.py --config config_global.yaml --experiment streamflow_global_exp1
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import numpy as np

from dataloader_global import GlobalFloodDroughtDataLoader
from models.CTLSTM import CTLSTM
from train import (
    EarlyStopping,
    calculate_metrics,
    get_scheduler,
    plot_training_history,
    save_model,
    seed_everything,
)


def train_epoch(
    model: CTLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    target_names: List[str],
    grad_clip_norm: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    target_losses = {name: 0.0 for name in target_names}
    num_batches = 0

    for dynamic_feats, static_feats, targets in tqdm(dataloader, desc="Training", leave=False):
        dynamic_feats = dynamic_feats.to(device)
        targets = targets.to(device)
        static_feats = (
            static_feats.to(device)
            if model.static_input_size > 0 and static_feats.numel() > 0
            else None
        )

        optimizer.zero_grad()
        predictions = model(dynamic_feats, static_inputs=static_feats)
        loss = criterion(predictions, targets)
        loss.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if len(target_names) > 1:
            for i, name in enumerate(target_names):
                target_losses[name] += criterion(predictions[:, :, i], targets[:, :, i]).item()

    avg_loss = total_loss / max(num_batches, 1)
    avg_target_losses = {name: loss / max(num_batches, 1) for name, loss in target_losses.items()}
    return avg_loss, avg_target_losses


def validate_epoch(
    model: CTLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    target_losses = {name: 0.0 for name in target_names}
    num_batches = 0

    with torch.no_grad():
        for dynamic_feats, static_feats, targets in tqdm(dataloader, desc="Validation", leave=False):
            dynamic_feats = dynamic_feats.to(device)
            targets = targets.to(device)
            static_feats = (
                static_feats.to(device)
                if model.static_input_size > 0 and static_feats.numel() > 0
                else None
            )

            predictions = model(dynamic_feats, static_inputs=static_feats)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            num_batches += 1

            if len(target_names) > 1:
                for i, name in enumerate(target_names):
                    target_losses[name] += criterion(predictions[:, :, i], targets[:, :, i]).item()

    avg_loss = total_loss / max(num_batches, 1)
    avg_target_losses = {name: loss / max(num_batches, 1) for name, loss in target_losses.items()}
    return avg_loss, avg_target_losses


def get_experiment_config(config_path: str, experiment: Optional[str]) -> Tuple[Dict, str]:
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)

    if experiment:
        if experiment not in full_config:
            available = [k for k in full_config.keys() if not k.startswith("base_") and k != "default_experiment"]
            raise ValueError(f"Experiment '{experiment}' not found. Available: {available}")
        return full_config[experiment], experiment

    default_exp = full_config.get("default_experiment")
    if default_exp and default_exp in full_config:
        return full_config[default_exp], default_exp

    for key, value in full_config.items():
        if key.startswith("base_") or key == "default_experiment":
            continue
        return value, key

    raise ValueError("No experiments found in configuration file.")


def main():
    parser = argparse.ArgumentParser(description="Train global LSTM model across multiple watersheds.")
    parser.add_argument("--config", type=str, default="config_global.yaml", help="Path to YAML config file.")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name inside the config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found.")

    config, exp_name = get_experiment_config(args.config, args.experiment)
    print(f"Using experiment: {exp_name}")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)

    save_root = config["save_dir"]
    save_dir = os.path.join(save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"Using device: {device}")

    dataset_splits = config.get("dataset_splits")
    if not dataset_splits:
        raise ValueError("dataset_splits must be defined in the experiment config.")

    print("Initializing global data loader...")
    data_loader = GlobalFloodDroughtDataLoader(
        data_dir=config["data_dir"],
        watersheds=config.get("watersheds"),
        scenarios=config.get("scenarios"),
        csv_pattern=config.get("csv_pattern", "{watershed}_{scenario}_combined.csv"),
        window_size=config["window_size"],
        stride=config["stride"],
        target_cols=config["target_cols"],
        feature_cols=config.get("feature_cols"),
        batch_size=config["batch_size"],
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

    loaders = data_loader.create_data_loaders(shuffle_train=True)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    model = CTLSTM(
        input_size=data_loader.dynamic_input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=len(config["target_cols"]),
        dropout=config["dropout"],
        static_input_size=data_loader.static_input_size if config.get("use_static_attributes", True) else 0,
        static_embedding_dim=config.get("static_embedding_dim"),
        static_embedding_layers=config.get("static_embedding_layers"),
        static_dropout=config.get("static_dropout", 0.0),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler_config = {
        "type": config.get("scheduler_type", "ReduceLROnPlateau"),
        "patience": config.get("scheduler_patience", 5),
        "factor": config.get("scheduler_factor", 0.5),
        "min_lr": config.get("scheduler_min_lr", 1e-6),
        "step_size": config.get("scheduler_step_size", 20),
        "gamma": config.get("scheduler_gamma", 0.95),
        "T_0": config.get("scheduler_T_0", 10),
        "T_mult": config.get("scheduler_T_mult", 2),
        "T_max": config.get("scheduler_T_max", 50),
    }
    scheduler = get_scheduler(optimizer, scheduler_config)

    early_stopping = EarlyStopping(
        patience=int(config.get("early_stopping_patience", 10)),
        min_delta=float(config.get("early_stopping_min_delta", 1e-6)),
    )

    writer = SummaryWriter(os.path.join(save_dir, "logs")) if config.get("tensorboard_log", True) else None

    model_save_config = {
        "input_size": data_loader.dynamic_input_size,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "output_size": len(config["target_cols"]),
        "dropout": config["dropout"],
        "window_size": config["window_size"],
        "target_cols": config["target_cols"],
        "feature_cols": config.get("feature_cols"),
        "static_input_size": data_loader.static_input_size if config.get("use_static_attributes", True) else 0,
        "static_embedding_dim": config.get("static_embedding_dim"),
        "static_embedding_layers": config.get("static_embedding_layers"),
        "static_dropout": config.get("static_dropout", 0.0),
    }

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_save_config, f, indent=2)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train_loss, train_target_losses = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["target_cols"],
            grad_clip_norm=config.get("grad_clip_norm", 1.0),
        )

        val_loss, val_target_losses = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            config["target_cols"],
        )

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            for target_name in config["target_cols"]:
                if target_name in train_target_losses:
                    writer.add_scalar(f"Train_Loss/{target_name}", train_target_losses[target_name], epoch)
                    writer.add_scalar(f"Val_Loss/{target_name}", val_target_losses[target_name], epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        if len(config["target_cols"]) > 1:
            print("Per-target losses:")
            for name in config["target_cols"]:
                print(f"  {name}: Train={train_target_losses[name]:.6f}, Val={val_target_losses[name]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, model_save_config, os.path.join(save_dir, "best_model.pth"))

        if (epoch + 1) % config.get("save_every_n_epochs", 10) == 0:
            save_model(
                model,
                optimizer,
                epoch,
                val_loss,
                model_save_config,
                os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"),
            )

        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Final save
    save_model(model, optimizer, epoch, val_loss, model_save_config, os.path.join(save_dir, "final_model.pth"))
    plot_training_history(train_losses, val_losses, save_dir)

    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_predictions, test_targets = [], []

    with torch.no_grad():
        for dynamic_feats, static_feats, targets in tqdm(test_loader, desc="Testing"):
            dynamic_feats = dynamic_feats.to(device)
            static_feats = (
                static_feats.to(device)
                if model.static_input_size > 0 and static_feats.numel() > 0
                else None
            )
            preds = model(dynamic_feats, static_inputs=static_feats)
            test_predictions.append(preds.cpu().numpy())
            test_targets.append(targets.numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    metrics = calculate_metrics(test_predictions, test_targets, data_loader.target_scaler)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(save_dir, "test_predictions.npy"), test_predictions)
    np.save(os.path.join(save_dir, "test_targets.npy"), test_targets)

    if writer is not None:
        writer.close()
    print(f"\nTraining completed. Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()
