#!/usr/bin/env python3
"""
Uncertainty-Weighted Hierarchical Multi-Task LSTM Training

This training script extends the existing HMTL (joint) workflow by learning
per-target homoscedastic uncertainty weights (Kendall & Gal, 2018) that balance
intermediate and final task losses automatically. Targets with noisier labels
receive smaller effective weights, allowing the network to emphasise signals
that most improve streamflow prediction without manual tuning.

Usage example:
    python train_hmtl_uncertainty.py --experiment streamflow_hmtl_uncertainty

Configuration:
    Inherit from streamflow_hmtl in config.yaml and set:
        training_strategy: "hmtl_uncertainty"
    Optional extras:
        uncertainty_init_log_var: 0.0
        uncertainty_min_log_var: -6.0
        uncertainty_max_log_var: 6.0

Author: Floods & Droughts ML Team
"""

# Core libraries
import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
from dataloader_hmtl import FloodDroughtDataLoader
from models.LSTM_HMTL import HierarchicalLSTMModel

warnings.filterwarnings('ignore')


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducible runs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Seeds set to {seed}")


class HomoscedasticUncertainty(nn.Module):
    """Learnable per-task loss weighting following Kendall & Gal (2018)."""

    def __init__(self,
                 target_names: List[str],
                 init_log_var: float = 0.0,
                 min_log_var: float = -6.0,
                 max_log_var: float = 6.0) -> None:
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_var)))
            for name in target_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        contributions: Dict[str, float] = {}

        for name, loss_value in losses.items():
            if name not in self.log_vars:
                raise KeyError(f"Missing log variance for target '{name}'")

            raw_log_var = self.log_vars[name]
            log_var = torch.clamp(raw_log_var, self.min_log_var, self.max_log_var)
            weighted_loss = torch.exp(-log_var) * loss_value + log_var
            total_loss = total_loss + weighted_loss
            contributions[name] = float(weighted_loss.detach().cpu())

        return total_loss, contributions

    def export_sigma(self) -> Dict[str, float]:
        """Return sigma per task (σ = sqrt(exp(log_var)))."""
        stats: Dict[str, float] = {}
        for name, param in self.log_vars.items():
            log_var = float(param.detach().cpu().clamp(self.min_log_var, self.max_log_var))
            sigma = float(np.sqrt(np.exp(log_var)))
            stats[name] = sigma
        return stats


def calculate_metrics(predictions: np.ndarray,
                      targets: np.ndarray,
                      scaler=None) -> Dict[str, float]:
    """Regression metrics with optional denormalisation."""
    if scaler is not None:
        original_shape = predictions.shape
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        predictions = scaler.inverse_transform(predictions)
        targets = scaler.inverse_transform(targets)
        predictions = predictions.reshape(original_shape)
        targets = targets.reshape(original_shape)

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)

    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }


def create_data_loaders(config: Dict) -> Tuple[FloodDroughtDataLoader, Dict[str, torch.utils.data.DataLoader]]:
    loader = FloodDroughtDataLoader.from_config(config)
    return loader, loader.create_data_loaders(shuffle_train=True)


def _assemble_batch_targets(final_targets_tensor: torch.Tensor,
                            intermediate_targets_tensor: torch.Tensor,
                            has_intermediate: bool) -> torch.Tensor:
    if not has_intermediate or intermediate_targets_tensor is None:
        return final_targets_tensor
    return torch.cat([intermediate_targets_tensor, final_targets_tensor], dim=-1)


def train_epoch(model: HierarchicalLSTMModel,
                train_loader,
                criterion,
                device,
                intermediate_targets: List[str],
                final_targets: List[str],
                uncertainty: HomoscedasticUncertainty,
                optimizer: optim.Optimizer,
                grad_clip_norm: float) -> Dict[str, Dict[str, float]]:
    model.train()
    base_tracker = {name: 0.0 for name in intermediate_targets + final_targets}
    weighted_tracker = {name: 0.0 for name in intermediate_targets + final_targets}
    batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        if len(batch) == 4:
            inputs, final_targets_tensor, intermediate_targets_tensor, _ = batch
            has_intermediate = True
        else:
            inputs, final_targets_tensor, _ = batch
            intermediate_targets_tensor = None
            has_intermediate = False

        batch_targets = _assemble_batch_targets(final_targets_tensor, intermediate_targets_tensor, has_intermediate)

        inputs = inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        intermediate_outputs = outputs['intermediate']
        final_outputs = outputs['final']

        per_task_losses: Dict[str, torch.Tensor] = {}

        for idx, target_name in enumerate(intermediate_targets):
            if target_name not in intermediate_outputs:
                continue
            per_task_losses[target_name] = criterion(
                intermediate_outputs[target_name],
                batch_targets[:, :, idx:idx + 1]
            )

        for jdx, target_name in enumerate(final_targets):
            if target_name not in final_outputs:
                continue
            offset = len(intermediate_targets) + jdx
            per_task_losses[target_name] = criterion(
                final_outputs[target_name],
                batch_targets[:, :, offset:offset + 1]
            )

        total_loss, weighted = uncertainty(per_task_losses)
        total_loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        batches += 1

        for name, base_val in per_task_losses.items():
            base_tracker[name] += float(base_val.detach().cpu())
        for name, weighted_val in weighted.items():
            weighted_tracker[name] += weighted_val

    avg_base = {name: base_tracker[name] / max(1, batches) for name in base_tracker}
    avg_weighted = {name: weighted_tracker[name] / max(1, batches) for name in weighted_tracker}

    return {'base': avg_base, 'weighted': avg_weighted}


def validate_epoch(model: HierarchicalLSTMModel,
                   val_loader,
                   criterion,
                   device,
                   intermediate_targets: List[str],
                   final_targets: List[str],
                   uncertainty: HomoscedasticUncertainty) -> Dict[str, Dict[str, float]]:
    model.eval()
    base_tracker = {name: 0.0 for name in intermediate_targets + final_targets}
    weighted_tracker = {name: 0.0 for name in intermediate_targets + final_targets}
    batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if len(batch) == 4:
                inputs, final_targets_tensor, intermediate_targets_tensor, _ = batch
                has_intermediate = True
            else:
                inputs, final_targets_tensor, _ = batch
                intermediate_targets_tensor = None
                has_intermediate = False

            batch_targets = _assemble_batch_targets(final_targets_tensor, intermediate_targets_tensor, has_intermediate)

            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(inputs)
            intermediate_outputs = outputs['intermediate']
            final_outputs = outputs['final']

            per_task_losses: Dict[str, torch.Tensor] = {}

            for idx, target_name in enumerate(intermediate_targets):
                if target_name not in intermediate_outputs:
                    continue
                per_task_losses[target_name] = criterion(
                    intermediate_outputs[target_name],
                    batch_targets[:, :, idx:idx + 1]
                )

            for jdx, target_name in enumerate(final_targets):
                if target_name not in final_outputs:
                    continue
                offset = len(intermediate_targets) + jdx
                per_task_losses[target_name] = criterion(
                    final_outputs[target_name],
                    batch_targets[:, :, offset:offset + 1]
                )

            _, weighted = uncertainty(per_task_losses)
            batches += 1

            for name, base_val in per_task_losses.items():
                base_tracker[name] += float(base_val.detach().cpu())
            for name, weighted_val in weighted.items():
                weighted_tracker[name] += weighted_val

    avg_base = {name: base_tracker[name] / max(1, batches) for name in base_tracker}
    avg_weighted = {name: weighted_tracker[name] / max(1, batches) for name in weighted_tracker}

    return {'base': avg_base, 'weighted': avg_weighted}


def evaluate(model: HierarchicalLSTMModel,
             test_loader,
             device,
             final_targets: List[str],
             target_scaler,
             save_dir: str) -> Dict[str, float]:
    model.eval()
    preds = []
    targs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            if len(batch) == 4:
                inputs, final_targets_tensor, _, _ = batch
            else:
                inputs, final_targets_tensor, _ = batch

            inputs = inputs.to(device)
            outputs = model(inputs)
            final_outputs = outputs['final']

            target_preds = [final_outputs[name].detach().cpu().numpy()
                            for name in final_targets if name in final_outputs]
            if not target_preds:
                continue

            preds.append(np.concatenate(target_preds, axis=-1))
            targs.append(final_targets_tensor.numpy())

    predictions = np.concatenate(preds, axis=0)
    targets = np.concatenate(targs, axis=0)

    metrics = calculate_metrics(predictions, targets, target_scaler)

    np.save(os.path.join(save_dir, 'test_predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'test_targets.npy'), targets)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


def plot_history(train_curve: List[float],
                 val_curve: List[float],
                 save_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_curve, label='Train (weighted)')
    plt.plot(val_curve, label='Validation (weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Uncertainty-Weighted Loss History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train uncertainty-weighted hierarchical LSTM')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name inside config')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed override')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as fp:
        full_config = yaml.safe_load(fp)

    if args.experiment not in full_config:
        raise KeyError(f"Experiment '{args.experiment}' not found in {args.config}")

    config = full_config[args.experiment]

    if config.get('training_strategy') not in ('hmtl', 'hmtl_uncertainty'):
        raise ValueError("training_strategy must be 'hmtl_uncertainty' (can inherit from hmtl)")

    seed = args.seed if args.seed is not None else config.get('seed', 42)
    seed_everything(seed)

    device_choice = config.get('device', 'auto')
    if device_choice == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_choice)
    print(f"Using device: {device}")

    data_loader, data_splits = create_data_loaders(config)
    train_loader = data_splits['train_loader']
    val_loader = data_splits['val_loader']
    test_loader = data_splits['test_loader']

    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[-1]

    intermediate_targets = config.get('intermediate_targets', [])
    final_targets = config['target_cols']

    model = HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=intermediate_targets,
        final_targets=final_targets,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        batch_first=True
    ).to(device)

    init_log_var = float(config.get('uncertainty_init_log_var', 0.0))
    min_log_var = float(config.get('uncertainty_min_log_var', -6.0))
    max_log_var = float(config.get('uncertainty_max_log_var', 6.0))

    uncertainty = HomoscedasticUncertainty(
        intermediate_targets + final_targets,
        init_log_var=init_log_var,
        min_log_var=min_log_var,
        max_log_var=max_log_var
    ).to(device)

    criterion = nn.MSELoss()
    lr = float(config.get('learning_rate', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-5))

    optimizer = optim.AdamW(
        list(model.parameters()) + list(uncertainty.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(config.get('scheduler_factor', 0.5)),
        patience=int(config.get('scheduler_patience', 5)),
        min_lr=float(config.get('scheduler_min_lr', 1e-6)),
        verbose=True
    )

    patience = int(config.get('early_stopping_patience', 10))
    min_delta = float(config.get('early_stopping_min_delta', 0.0))
    best_val = float('inf')
    epochs_without_improve = 0

    save_dir = os.path.join(config.get('save_dir', 'experiments'), args.experiment)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp, indent=2)

    model_meta = {
        'input_size': input_size,
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config.get('dropout', 0.2),
        'intermediate_targets': intermediate_targets,
        'final_targets': final_targets
    }
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as fp:
        json.dump(model_meta, fp, indent=2)

    writer = SummaryWriter(os.path.join(save_dir, 'logs')) if config.get('tensorboard_log', True) else None

    epochs = int(config.get('epochs', 100))
    grad_clip_norm = float(config.get('grad_clip_norm', 1.0))

    train_curve: List[float] = []
    val_curve: List[float] = []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_stats = train_epoch(
            model, train_loader, criterion, device,
            intermediate_targets, final_targets, uncertainty, optimizer, grad_clip_norm
        )
        val_stats = validate_epoch(
            model, val_loader, criterion, device,
            intermediate_targets, final_targets, uncertainty
        )

        train_weighted = sum(train_stats['weighted'].values())
        val_weighted = sum(val_stats['weighted'].values())
        train_curve.append(train_weighted)
        val_curve.append(val_weighted)

        scheduler.step(val_weighted)

        if writer:
            writer.add_scalar('Loss/TrainWeighted', train_weighted, epoch)
            writer.add_scalar('Loss/ValWeighted', val_weighted, epoch)
            for name, base_loss in train_stats['base'].items():
                writer.add_scalar(f'BaseLoss/Train/{name}', base_loss, epoch)
            for name, base_loss in val_stats['base'].items():
                writer.add_scalar(f'BaseLoss/Val/{name}', base_loss, epoch)
            for name, sigma in uncertainty.export_sigma().items():
                writer.add_scalar(f'Uncertainty/Sigma/{name}', sigma, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if val_weighted + min_delta < best_val:
            best_val = val_weighted
            epochs_without_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'uncertainty_state_dict': uncertainty.state_dict(),
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ New best model (val={best_val:.6f})")
        else:
            epochs_without_improve += 1
            print(f"  ↳ No improvement for {epochs_without_improve} epoch(s)")

        if epochs_without_improve >= patience:
            print("Early stopping triggered")
            break

    duration_minutes = (time.time() - start_time) / 60.0
    print(f"\nTraining completed in {duration_minutes:.1f} minutes")

    plot_history(train_curve, val_curve, save_dir)

    if writer:
        writer.close()

    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    uncertainty.load_state_dict(checkpoint['uncertainty_state_dict'])

    metrics = evaluate(
        model,
        test_loader,
        device,
        final_targets,
        data_loader.target_scaler,
        save_dir
    )

    stats_payload = {
        'sigma': uncertainty.export_sigma(),
        'metrics': metrics
    }
    with open(os.path.join(save_dir, 'uncertainty_stats.json'), 'w') as fp:
        json.dump(stats_payload, fp, indent=2)

    print('\nTest metrics (final targets):')
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
