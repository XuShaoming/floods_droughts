#!/usr/bin/env python3
"""
Global Hierarchical Multi-Task Training Script.

Trains the CTLSTM_Global model that jointly learns intermediate and final targets
across multiple watersheds/scenarios, optionally leveraging static watershed attributes.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from dataloader_hmtl_global import GlobalHierarchicalDataLoader
from models.CTLSTM_Global import CTLSTMGlobal
from train import (
    EarlyStopping,
    calculate_metrics,
    get_scheduler,
    plot_training_history,
    save_model,
    seed_everything,
)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical global LSTM model.")
    parser.add_argument("--config", type=str, default="config_global.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="streamflow_hmtl_global",
        help="Experiment entry name inside the config file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    return parser.parse_args()


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


def normalize_intermediate_weights(config_value, target_names: List[str]) -> Dict[str, float]:
    if not target_names:
        return {}

    if config_value is None:
        return {name: 1.0 for name in target_names}

    if isinstance(config_value, dict):
        default = float(config_value.get("default", 1.0))
        return {name: float(config_value.get(name, default)) for name in target_names}

    if isinstance(config_value, (list, tuple)):
        if len(config_value) != len(target_names):
            raise ValueError(
                f"intermediate loss weights length ({len(config_value)}) "
                f"does not match number of targets ({len(target_names)})."
            )
        return {name: float(val) for name, val in zip(target_names, config_value)}

    value = float(config_value)
    return {name: value for name in target_names}


def build_loss_weights(config: Dict, intermediate_names: List[str]) -> Tuple[float, Dict[str, float]]:
    weights_cfg = config.get("loss_weights", {})
    final_weight = float(weights_cfg.get("final", 1.0)) if isinstance(weights_cfg, dict) else 1.0
    intermediate_cfg = weights_cfg.get("intermediate") if isinstance(weights_cfg, dict) else None
    intermediate_weights = normalize_intermediate_weights(intermediate_cfg, intermediate_names)
    if not intermediate_weights:
        intermediate_weights = {name: 1.0 for name in intermediate_names}
    return final_weight, intermediate_weights


def denormalize_single_target(array: np.ndarray, scaler, column_idx: int) -> np.ndarray:
    """
    Denormalize a single target column using the corresponding scaler statistics.

    Parameters:
        array: numpy array shaped (n_windows, seq_len, 1) or similar.
        scaler: Fitted StandardScaler with mean_/scale_ attributes.
        column_idx: Index of the column inside the scaler to apply.
    """
    if scaler is None or array.size == 0:
        return array

    if not hasattr(scaler, "scale_") or not hasattr(scaler, "mean_"):
        raise AttributeError("Scaler must provide 'scale_' and 'mean_' attributes for denormalization.")

    if column_idx >= len(scaler.scale_):
        raise IndexError(
            f"column_idx {column_idx} is out of bounds for scaler with {len(scaler.scale_)} features."
        )

    scale = scaler.scale_[column_idx]
    mean = scaler.mean_[column_idx]
    return array * scale + mean


def compute_intermediate_loss(
    criterion: nn.Module,
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: Dict[str, float],
    target_names: List[str],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = torch.tensor(0.0, device=targets.device)
    per_target_losses: Dict[str, float] = {}
    if targets.numel() == 0 or not target_names:
        return total_loss, per_target_losses

    for idx, name in enumerate(target_names):
        pred_tensor = predictions[name]
        target_tensor = targets[:, :, idx : idx + 1]
        loss_val = criterion(pred_tensor, target_tensor)
        weight = weights.get(name, 1.0)
        total_loss = total_loss + weight * loss_val
        per_target_losses[name] = loss_val.item()

    return total_loss, per_target_losses


def train_epoch(
    model: CTLSTMGlobal,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    intermediate_names: List[str],
    final_names: List[str],
    final_weight: float,
    intermediate_weights: Dict[str, float],
    grad_clip_norm: Optional[float] = 1.0,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.train()
    total_loss = 0.0
    intermediate_loss_accum = {name: 0.0 for name in intermediate_names}
    final_target_losses = {name: 0.0 for name in final_names}
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        dynamic_feats, static_feats, final_targets, intermediate_targets, _ = batch
        dynamic_feats = dynamic_feats.to(device)
        final_targets = final_targets.to(device)
        static_feats = static_feats.to(device) if model.static_input_size > 0 and static_feats.numel() > 0 else None
        intermediate_targets = (
            intermediate_targets.to(device) if intermediate_targets.numel() > 0 else torch.empty(0, device=device)
        )

        optimizer.zero_grad()
        outputs = model(dynamic_feats, static_inputs=static_feats)
        final_predictions = outputs["final"]
        final_loss_term = criterion(final_predictions, final_targets) * final_weight
        per_final_losses = {
            name: criterion(final_predictions[:, :, idx], final_targets[:, :, idx]).item() for idx, name in enumerate(final_names)
        }

        if intermediate_targets.numel() > 0:
            interm_loss, per_intermediate = compute_intermediate_loss(
                criterion,
                outputs["intermediate"],
                intermediate_targets,
                intermediate_weights,
                intermediate_names,
            )
        else:
            interm_loss = torch.tensor(0.0, device=device)
            per_intermediate = {}

        loss = final_loss_term + interm_loss
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        for name, value in per_final_losses.items():
            final_target_losses[name] += value
        for name, value in per_intermediate.items():
            intermediate_loss_accum[name] += value
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_final_losses = {name: value / max(num_batches, 1) for name, value in final_target_losses.items()}
    avg_intermediate_losses = {name: value / max(num_batches, 1) for name, value in intermediate_loss_accum.items()}
    return avg_loss, avg_final_losses, avg_intermediate_losses


def validate_epoch(
    model: CTLSTMGlobal,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    intermediate_names: List[str],
    final_names: List[str],
    final_weight: float,
    intermediate_weights: Dict[str, float],
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    intermediate_loss_accum = {name: 0.0 for name in intermediate_names}
    final_target_losses = {name: 0.0 for name in final_names}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            dynamic_feats, static_feats, final_targets, intermediate_targets, _ = batch
            dynamic_feats = dynamic_feats.to(device)
            final_targets = final_targets.to(device)
            static_feats = static_feats.to(device) if model.static_input_size > 0 and static_feats.numel() > 0 else None
            intermediate_targets = (
                intermediate_targets.to(device) if intermediate_targets.numel() > 0 else torch.empty(0, device=device)
            )

            outputs = model(dynamic_feats, static_inputs=static_feats)
            final_predictions = outputs["final"]
            final_loss_term = criterion(final_predictions, final_targets) * final_weight
            per_final_losses = {
                name: criterion(final_predictions[:, :, idx], final_targets[:, :, idx]).item() for idx, name in enumerate(final_names)
            }

            if intermediate_targets.numel() > 0:
                interm_loss, per_intermediate = compute_intermediate_loss(
                    criterion,
                    outputs["intermediate"],
                    intermediate_targets,
                    intermediate_weights,
                    intermediate_names,
                )
            else:
                interm_loss = torch.tensor(0.0, device=device)
                per_intermediate = {}

            loss = final_loss_term + interm_loss
            total_loss += loss.item()
            for name, value in per_final_losses.items():
                final_target_losses[name] += value
            for name, value in per_intermediate.items():
                intermediate_loss_accum[name] += value
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_final_losses = {name: value / max(num_batches, 1) for name, value in final_target_losses.items()}
    avg_intermediate_losses = {name: value / max(num_batches, 1) for name, value in intermediate_loss_accum.items()}
    return avg_loss, avg_final_losses, avg_intermediate_losses


def collect_predictions(
    model: CTLSTMGlobal,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    intermediate_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    final_preds, final_targets = [], []
    intermediate_preds = {name: [] for name in intermediate_names}
    intermediate_targets = {name: [] for name in intermediate_names}

    with torch.no_grad():
        for dynamic_feats, static_feats, final_t, intermediate_t, _ in tqdm(dataloader, desc="Collecting"):
            dynamic_feats = dynamic_feats.to(device)
            final_t = final_t.to(device)
            static_feats = static_feats.to(device) if model.static_input_size > 0 and static_feats.numel() > 0 else None
            intermediate_t = intermediate_t.to(device) if intermediate_t.numel() > 0 else torch.empty(0, device=device)

            outputs = model(dynamic_feats, static_inputs=static_feats)
            final_preds.append(outputs["final"].cpu().numpy())
            final_targets.append(final_t.cpu().numpy())

            if intermediate_t.numel() > 0:
                for idx, name in enumerate(intermediate_names):
                    intermediate_preds[name].append(outputs["intermediate"][name].cpu().numpy())
                    target_slice = intermediate_t[:, :, idx : idx + 1].cpu().numpy()
                    intermediate_targets[name].append(target_slice)

    final_pred_array = np.concatenate(final_preds, axis=0) if final_preds else np.empty((0,))
    final_target_array = np.concatenate(final_targets, axis=0) if final_targets else np.empty((0,))

    intermediate_pred_arrays = {}
    intermediate_target_arrays = {}
    for name in intermediate_names:
        if intermediate_preds[name]:
            intermediate_pred_arrays[name] = np.concatenate(intermediate_preds[name], axis=0)
            intermediate_target_arrays[name] = np.concatenate(intermediate_targets[name], axis=0)
        else:
            intermediate_pred_arrays[name] = np.empty((0,))
            intermediate_target_arrays[name] = np.empty((0,))

    return final_pred_array, final_target_array, intermediate_pred_arrays, intermediate_target_arrays


def save_scalers(data_loader: GlobalHierarchicalDataLoader, save_dir: str):
    scalers = {
        "feature_scaler": data_loader.feature_scaler,
        "target_scaler": data_loader.target_scaler,
        "intermediate_scaler": data_loader.intermediate_scaler,
        "static_scaler": data_loader.static_scaler,
    }
    with open(os.path.join(save_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)


def save_metadata(metadata: Dict[str, Dict[str, List]], save_dir: str):
    serializable = {}
    for split, values in metadata.items():
        serializable[split] = {
            "watershed": values.get("watershed", []),
            "scenario": values.get("scenario", []),
            "dates": [
                [ts.isoformat() for ts in window] if isinstance(window, (list, tuple, pd.Index)) else []
                for window in values.get("dates", [])
            ],
        }
    with open(os.path.join(save_dir, "loader_metadata.json"), "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    args = parse_args()
    config, exp_name = get_experiment_config(args.config, args.experiment)
    print(f"Using experiment: {exp_name}")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)

    save_root = config["save_dir"]
    save_dir = os.path.join(save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    device_cfg = config.get("device", "auto")
    device = resolve_device(device_cfg)
    print(f"Using device: {device}")

    dataset_splits = config.get("dataset_splits")
    if not dataset_splits:
        raise ValueError("dataset_splits must be defined in the experiment config.")

    data_loader = GlobalHierarchicalDataLoader(
        data_dir=config["data_dir"],
        watersheds=config.get("watersheds"),
        scenarios=config.get("scenarios"),
        csv_pattern=config.get("csv_pattern", "{watershed}_{scenario}_combined.csv"),
        window_size=config["window_size"],
        stride=config["stride"],
        target_cols=config["target_cols"],
        feature_cols=config.get("feature_cols"),
        intermediate_targets=config.get("intermediate_targets"),
        batch_size=config["batch_size"],
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

    loaders = data_loader.create_data_loaders(shuffle_train=True)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    model = CTLSTMGlobal(
        input_size=data_loader.dynamic_input_size,
        intermediate_targets=config.get("intermediate_targets", []),
        final_targets=config["target_cols"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        static_input_size=data_loader.static_input_size if config.get("use_static_attributes", True) else 0,
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
        "dropout": config["dropout"],
        "window_size": config["window_size"],
        "target_cols": config["target_cols"],
        "feature_cols": config.get("feature_cols"),
        "intermediate_targets": config.get("intermediate_targets", []),
        "static_input_size": data_loader.static_input_size if config.get("use_static_attributes", True) else 0,
        "static_embedding_layers": config.get("static_embedding_layers"),
        "static_dropout": config.get("static_dropout", 0.0),
    }

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_save_config, f, indent=2)

    final_weight, intermediate_weights = build_loss_weights(config, config.get("intermediate_targets", []))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train_loss, train_final_losses, train_inter_losses = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.get("intermediate_targets", []),
            config["target_cols"],
            final_weight,
            intermediate_weights,
            grad_clip_norm=config.get("grad_clip_norm", 1.0),
        )

        val_loss, val_final_losses, val_inter_losses = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            config.get("intermediate_targets", []),
            config["target_cols"],
            final_weight,
            intermediate_weights,
        )

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            for name, value in train_final_losses.items():
                writer.add_scalar(f"FinalLoss/Train/{name}", value, epoch)
            for name, value in val_final_losses.items():
                writer.add_scalar(f"FinalLoss/Val/{name}", value, epoch)
            for name, value in train_inter_losses.items():
                writer.add_scalar(f"IntermediateLoss/Train/{name}", value, epoch)
            for name, value in val_inter_losses.items():
                writer.add_scalar(f"IntermediateLoss/Val/{name}", value, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

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

    save_model(model, optimizer, epoch, val_loss, model_save_config, os.path.join(save_dir, "final_model.pth"))
    plot_training_history(train_losses, val_losses, save_dir)

    print("\nEvaluating on test set...")
    final_pred, final_target, intermediate_pred, intermediate_target = collect_predictions(
        model, test_loader, device, config.get("intermediate_targets", [])
    )
    metrics = {"final": {}, "intermediate": {}}
    if final_pred.size > 0:
        metrics["final"]["overall"] = calculate_metrics(final_pred, final_target, data_loader.target_scaler)

    intermediate_names = config.get("intermediate_targets", [])
    for idx, name in enumerate(intermediate_names):
        preds = intermediate_pred.get(name)
        targets = intermediate_target.get(name)
        if preds is None or preds.size == 0:
            continue

        denorm_preds = denormalize_single_target(preds, data_loader.intermediate_scaler, idx)
        denorm_targets = denormalize_single_target(targets, data_loader.intermediate_scaler, idx)
        metrics["intermediate"][name] = calculate_metrics(denorm_preds, denorm_targets, scaler=None)

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    save_scalers(data_loader, save_dir)
    save_metadata(data_loader.metadata, save_dir)

    if writer is not None:
        writer.close()
    print(f"\nTraining completed. Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()
