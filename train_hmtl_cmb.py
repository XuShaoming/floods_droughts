#!/usr/bin/env python3
"""
Hierarchical Multi-Task Learning (HMTL) Training Script for Flood/Drought Prediction

This script implements the joint training strategy for hierarchical LSTM where all tasks
(intermediate and final) are trained simultaneously. This approach leverages multi-task
learning to learn shared representations that benefit both intermediate and final predictions.

Key Features:
- Joint training of all hierarchical tasks
- Weighted multi-task loss function
- Comprehensive evaluation of all targets
- TensorBoard logging with task-specific metrics
- Model checkpointing and early stopping
- GPU acceleration support

Usage:
    python train_hmtl.py --experiment streamflow_hmtl

This differs from train_hstl.py in that it trains all tasks simultaneously rather than
in sequential phases, allowing for shared representation learning across all targets.

Author: GitHub Copilot
Date: 2024
"""

# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standard library
import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import yaml
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple

# Local imports
from dataloader_hmtl import FloodDroughtDataLoader, TimeSeriesDataset
from models.LSTM_HMTL import HierarchicalLSTMModel


def load_data(config):
    """Load data using FloodDroughtDataLoader from config"""
    return FloodDroughtDataLoader.from_config(config)


def _concat_targets(intermediate_windows: Optional[np.ndarray],
                    final_windows: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate intermediate and final target windows across the feature axis."""
    components = []
    if intermediate_windows is not None and intermediate_windows.size > 0:
        components.append(intermediate_windows)
    if final_windows is not None and final_windows.size > 0:
        components.append(final_windows)
    if not components:
        return None
    return np.concatenate(components, axis=-1)


def _repeat_initial_states(target_windows: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Repeat the initial timestep (conditional state) across the entire window."""
    if target_windows is None:
        return None
    if target_windows.ndim != 3:
        raise ValueError("Target windows must be 3-dimensional for CMB processing")
    initial = target_windows[:, 0, :]  # (n_windows, n_targets)
    repeated = np.repeat(initial[:, np.newaxis, :], target_windows.shape[1], axis=1)
    return repeated


def _augment_features_with_states(feature_windows: np.ndarray,
                                  state_windows: Optional[np.ndarray]) -> np.ndarray:
    """Append state feature windows to the base feature windows."""
    if state_windows is None:
        return feature_windows
    return np.concatenate([feature_windows, state_windows], axis=-1)


def create_data_loaders(data_loader: FloodDroughtDataLoader,
                        config: Dict) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Dict]:
    """Create DataLoaders augmented with CMB-style state features."""

    data_splits = data_loader.prepare_data()

    intermediate_targets = config.get('intermediate_targets', []) or []
    final_targets = config.get('target_cols', []) or []
    state_order = list(intermediate_targets) + list(final_targets)
    state_feature_names = [f"CMB_{target}_state" for target in state_order]

    train_target_stack = _concat_targets(
        data_splits.get('intermediate_train'),
        data_splits.get('y_train')
    )
    val_target_stack = _concat_targets(
        data_splits.get('intermediate_val'),
        data_splits.get('y_val')
    )
    test_target_stack = _concat_targets(
        data_splits.get('intermediate_test'),
        data_splits.get('y_test')
    )

    train_state_windows = _repeat_initial_states(train_target_stack)
    val_state_windows = _repeat_initial_states(val_target_stack)
    test_state_windows = _repeat_initial_states(test_target_stack)

    x_train = _augment_features_with_states(data_splits['x_train'], train_state_windows)
    x_val = _augment_features_with_states(data_splits['x_val'], val_state_windows)
    x_test = _augment_features_with_states(data_splits['x_test'], test_state_windows)

    train_dataset = TimeSeriesDataset(
        x_train,
        data_splits['y_train'],
        data_splits.get('intermediate_train')
    )
    val_dataset = TimeSeriesDataset(
        x_val,
        data_splits['y_val'],
        data_splits.get('intermediate_val')
    )
    test_dataset = TimeSeriesDataset(
        x_test,
        data_splits['y_test'],
        data_splits.get('intermediate_test')
    )

    batch_size = config.get('batch_size', data_loader.batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cmb_metadata = {
        'state_feature_names': state_feature_names,
        'state_targets_order': state_order,
        'cmb_feature_count': len(state_feature_names),
        'window_size': data_loader.window_size,
        'stride': data_loader.stride,
        'many_to_many': data_loader.many_to_many
    }

    return (train_loader, val_loader, test_loader), cmb_metadata


def seed_everything(seed=42):
    """
    Seed all random number generators for reproducible results.
    """
    import random
    import os
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (CPU)
    torch.manual_seed(seed)
    
    # PyTorch random (GPU) - handles both single and multi-GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Additional PyTorch settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ All random seeds set to {seed} for reproducible results")


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model state dict"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def calculate_metrics(predictions, targets, scaler=None):
    """Calculate evaluation metrics for predictions"""
    
    # Convert to numpy if tensor
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    # Denormalize if scaler provided
    if scaler is not None:
        # Reshape for scaler
        original_shape = predictions.shape
        predictions_flat = predictions.reshape(-1, predictions.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
        
        # Inverse transform
        predictions_denorm = scaler.inverse_transform(predictions_flat)
        targets_denorm = scaler.inverse_transform(targets_flat)
        
        # Reshape back
        predictions = predictions_denorm.reshape(original_shape)
        targets = targets_denorm.reshape(original_shape)
    
    # Flatten for metric calculation
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def save_model(model, optimizer, epoch, val_loss, config, filepath):
    """Save model checkpoint with all necessary information"""
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, filepath)


def train_epoch(model, train_loader, criterion, optimizer, device, 
                intermediate_targets, final_targets, 
                intermediate_weight=1.0, final_weight=1.0,
                grad_clip_norm=1.0):
    """
    Train for one epoch with joint multi-task learning
    
    Args:
        model: Hierarchical LSTM model
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device (cuda/cpu)
        intermediate_targets: List of intermediate target names
        final_targets: List of final target names
        intermediate_weight: Weight for intermediate losses
        final_weight: Weight for final losses
        grad_clip_norm: Gradient clipping norm
    """
    model.train()
    total_losses = {target: 0.0 for target in intermediate_targets + final_targets}
    total_loss = 0.0
    intermediate_losses = {target: 0.0 for target in intermediate_targets}
    final_losses = {target: 0.0 for target in final_targets}
    num_batches = 0
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        if len(batch) == 4:
            # Hierarchical format: (features, final_targets, intermediate_targets, index)
            inputs, final_batch_targets, intermediate_batch_targets, _ = batch
            
            # Convert to dictionaries for easier handling
            intermediate_targets_dict = {}
            for i, target_name in enumerate(intermediate_targets):
                if i < intermediate_batch_targets.shape[-1]:
                    intermediate_targets_dict[target_name] = intermediate_batch_targets[:, :, i:i+1]
            
            final_targets_dict = {}
            for i, target_name in enumerate(final_targets):
                if i < final_batch_targets.shape[-1]:
                    final_targets_dict[target_name] = final_batch_targets[:, :, i:i+1]
                    
        else:
            # Standard format: (features, targets, index)
            inputs, batch_targets, _ = batch
            
            if batch_count == 0:
                print(f"DEBUG: Standard format - inputs shape: {inputs.shape}")
                print(f"DEBUG: batch_targets shape: {batch_targets.shape}")
            
            # Split targets based on order (intermediate first, then final)
            intermediate_targets_dict = {}
            final_targets_dict = {}
            
            for i, target_name in enumerate(intermediate_targets + final_targets):
                if target_name in intermediate_targets:
                    intermediate_targets_dict[target_name] = batch_targets[:, :, i:i+1]
                else:
                    final_idx = i - len(intermediate_targets)
                    final_targets_dict[target_name] = batch_targets[:, :, final_idx:final_idx+1]
        
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Move targets to device
        for target_name in intermediate_targets:
            if target_name in intermediate_targets_dict:
                intermediate_targets_dict[target_name] = intermediate_targets_dict[target_name].to(device)
        
        for target_name in final_targets:
            if target_name in final_targets_dict:
                final_targets_dict[target_name] = final_targets_dict[target_name].to(device)
        
        # Forward pass
        outputs = model(inputs)
        intermediate_outputs = outputs['intermediate']
        final_outputs = outputs['final']
        
        # Calculate intermediate task losses
        total_intermediate_loss = None
        for target_name in intermediate_targets:
            if target_name in intermediate_outputs and target_name in intermediate_targets_dict:
                intermediate_loss = criterion(
                    intermediate_outputs[target_name], 
                    intermediate_targets_dict[target_name]
                )
                if total_intermediate_loss is None:
                    total_intermediate_loss = intermediate_loss
                else:
                    total_intermediate_loss = total_intermediate_loss + intermediate_loss
                intermediate_losses[target_name] += intermediate_loss.item()
        
        # Calculate final task losses
        total_final_loss = None
        for target_name in final_targets:
            if target_name in final_outputs and target_name in final_targets_dict:
                final_loss = criterion(
                    final_outputs[target_name], 
                    final_targets_dict[target_name]
                )
                if total_final_loss is None:
                    total_final_loss = final_loss
                else:
                    total_final_loss = total_final_loss + final_loss
                final_losses[target_name] += final_loss.item()
        
        # Combined loss with task weighting
        batch_loss = None
        
        if total_intermediate_loss is not None:
            if batch_loss is None:
                batch_loss = intermediate_weight * total_intermediate_loss
            else:
                batch_loss = batch_loss + intermediate_weight * total_intermediate_loss
        
        if total_final_loss is not None:
            if batch_loss is None:
                batch_loss = final_weight * total_final_loss
            else:
                batch_loss = batch_loss + final_weight * total_final_loss
        
        # Only perform backward pass if we have a valid loss tensor
        if batch_loss is not None:
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            total_loss += batch_loss.item()
        
        batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_intermediate_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in intermediate_losses.items()}
    avg_final_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in final_losses.items()}
    
    return avg_total_loss, avg_intermediate_losses, avg_final_losses


def validate_epoch(model, val_loader, criterion, device, 
                   intermediate_targets, final_targets,
                   intermediate_weight=1.0, final_weight=1.0):
    """
    Validate for one epoch with joint multi-task learning
    
    Args:
        model: Hierarchical LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        intermediate_targets: List of intermediate target names
        final_targets: List of final target names
        intermediate_weight: Weight for intermediate task losses
        final_weight: Weight for final task losses
    
    Returns:
        total_loss: Average total validation loss
        intermediate_losses: Dictionary of individual intermediate task losses
        final_losses: Dictionary of individual final task losses
    """
    
    model.eval()
    total_loss = 0.0
    intermediate_losses = {target: 0.0 for target in intermediate_targets}
    final_losses = {target: 0.0 for target in final_targets}
    batch_count = 0
    
    with torch.no_grad():
        # for batch in tqdm(val_loader, desc="Validating", leave=False):
        for batch in val_loader:
            if len(batch) == 4:
                # Hierarchical format: (features, final_targets, intermediate_targets, index)
                inputs, final_batch_targets, intermediate_batch_targets, _ = batch
                
                # Convert to dictionaries for easier handling
                intermediate_targets_dict = {}
                for i, target_name in enumerate(intermediate_targets):
                    if i < intermediate_batch_targets.shape[-1]:
                        intermediate_targets_dict[target_name] = intermediate_batch_targets[:, :, i:i+1]
                
                final_targets_dict = {}
                for i, target_name in enumerate(final_targets):
                    if i < final_batch_targets.shape[-1]:
                        final_targets_dict[target_name] = final_batch_targets[:, :, i:i+1]
                        
            else:
                # Standard format: (features, targets, index)
                inputs, batch_targets, _ = batch
                # Split targets based on order (intermediate first, then final)
                intermediate_targets_dict = {}
                final_targets_dict = {}
                
                for i, target_name in enumerate(intermediate_targets + final_targets):
                    if target_name in intermediate_targets:
                        intermediate_targets_dict[target_name] = batch_targets[:, :, i:i+1]
                    else:
                        final_idx = i - len(intermediate_targets)
                        final_targets_dict[target_name] = batch_targets[:, :, final_idx:final_idx+1]
            
            inputs = inputs.to(device)
            
            # Move targets to device
            for target_name in intermediate_targets:
                if target_name in intermediate_targets_dict:
                    intermediate_targets_dict[target_name] = intermediate_targets_dict[target_name].to(device)
            
            for target_name in final_targets:
                if target_name in final_targets_dict:
                    final_targets_dict[target_name] = final_targets_dict[target_name].to(device)
            
            # Forward pass
            outputs = model(inputs)
            intermediate_outputs = outputs['intermediate']
            final_outputs = outputs['final']
            
            # Calculate intermediate task losses
            total_intermediate_loss = torch.tensor(0.0, device=device)
            for target_name in intermediate_targets:
                if target_name in intermediate_outputs and target_name in intermediate_targets_dict:
                    intermediate_loss = criterion(
                        intermediate_outputs[target_name], 
                        intermediate_targets_dict[target_name]
                    )
                    total_intermediate_loss = total_intermediate_loss + intermediate_loss
                    intermediate_losses[target_name] += intermediate_loss.item()
            
            # Calculate final task losses
            total_final_loss = torch.tensor(0.0, device=device)
            for target_name in final_targets:
                if target_name in final_outputs and target_name in final_targets_dict:
                    final_loss = criterion(
                        final_outputs[target_name], 
                        final_targets_dict[target_name]
                    )
                    total_final_loss = total_final_loss + final_loss
                    final_losses[target_name] += final_loss.item()
            
            # Combined loss with task weighting
            batch_loss = (intermediate_weight * total_intermediate_loss + 
                         final_weight * total_final_loss)
            
            total_loss += batch_loss.item()
            batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count
    avg_intermediate_losses = {k: v / batch_count for k, v in intermediate_losses.items()}
    avg_final_losses = {k: v / batch_count for k, v in final_losses.items()}
    
    return avg_total_loss, avg_intermediate_losses, avg_final_losses


def main():
    """Main HMTL training function"""
    
    parser = argparse.ArgumentParser(description='Train Hierarchical Multi-Task Learning LSTM')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name from config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    if args.experiment not in full_config:
        raise ValueError(f"Experiment '{args.experiment}' not found in config file")
    
    config = full_config[args.experiment]
    print(f"Running HMTL experiment: {args.experiment}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set random seeds for reproducibility
    seed_everything(args.seed)
    
    # Start timing
    start_time = time.time()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    save_dir = os.path.join('experiments', args.experiment)
    os.makedirs(save_dir, exist_ok=True)
    
    config_path = os.path.join(save_dir, 'config.yaml')
    
    # Load data
    print("Loading data...")
    data_loader = load_data(config)
    
    # Create data loaders with CMB augmentation
    (train_loader, val_loader, test_loader), cmb_metadata = create_data_loaders(
        data_loader,
        config
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    cmb_state_features = cmb_metadata.get('state_feature_names', [])
    if cmb_state_features:
        print(f"CMB state features added: {cmb_state_features}")

    config['cmb_metadata'] = cmb_metadata
    if cmb_state_features:
        config['feature_cols_extended'] = config['feature_cols'] + cmb_state_features
    with open(config_path, 'w') as f:
        yaml.dump({args.experiment: config}, f, default_flow_style=False)
    
    # Extract target configurations
    intermediate_targets = config['intermediate_targets']
    final_targets = config['target_cols']
    
    print(f"Intermediate targets: {intermediate_targets}")
    print(f"Final targets: {final_targets}")
    
    # Model configuration
    input_size = len(config['feature_cols']) + cmb_metadata.get('cmb_feature_count', 0)
    
    # Create model
    model = HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=intermediate_targets,
        final_targets=final_targets,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        batch_first=True
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    weight_decay = float(config.get('weight_decay', 0)) if config.get('weight_decay') else 0

    # Use AdamW optimizer (generally better than Adam)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.get('scheduler_type'):
        scheduler_type = config['scheduler_type']
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('scheduler_step_size', 20),
                gamma=config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.get('scheduler_factor', 0.5),
                patience=config.get('scheduler_patience', 10),
                min_lr=float(config.get('scheduler_min_lr', 1e-6)),
                verbose=True
            )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 15),
        min_delta=float(config.get('early_stopping_min_delta', 1e-6))
    )
    
    # TensorBoard logging
    writer = None
    if config.get('use_tensorboard', True):
        log_dir = os.path.join(save_dir, 'tensorboard_logs')
        writer = SummaryWriter(log_dir)
    
    # Task weights
    intermediate_weight = config.get('intermediate_task_weight', 1.0)
    final_weight = config.get('final_task_weight', 1.0)
    
    print(f"\nTask weights - Intermediate: {intermediate_weight}, Final: {final_weight}")
    
    # Model save configuration
    model_save_config = {
        'input_size': input_size,
        'intermediate_targets': intermediate_targets,
        'final_targets': final_targets,
        'feature_cols': config['feature_cols'],
        'feature_cols_extended': config.get('feature_cols_extended', config['feature_cols']),
        'target_cols': config['target_cols'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config.get('dropout', 0.2),
        'cmb_metadata': cmb_metadata
    }
    
    # Training loop
    print(f"\nStarting HMTL training for {config['epochs']} epochs...")
    print("="*80)
    
    train_losses = []
    val_losses = []
    intermediate_train_losses = {target: [] for target in intermediate_targets}
    intermediate_val_losses = {target: [] for target in intermediate_targets}
    final_train_losses = {target: [] for target in final_targets}
    final_val_losses = {target: [] for target in final_targets}
    
    best_val_loss = float('inf')
    epoch = 0
    val_loss = 0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 30)
        
        # Training
        train_loss, train_intermediate_losses, train_final_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            intermediate_targets, final_targets,
            intermediate_weight, final_weight,
            grad_clip_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Validation
        val_loss, val_intermediate_losses, val_final_losses = validate_epoch(
            model, val_loader, criterion, device,
            intermediate_targets, final_targets,
            intermediate_weight, final_weight
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        for target in intermediate_targets:
            intermediate_train_losses[target].append(train_intermediate_losses.get(target, 0))
            intermediate_val_losses[target].append(val_intermediate_losses.get(target, 0))
        
        for target in final_targets:
            final_train_losses[target].append(train_final_losses.get(target, 0))
            final_val_losses[target].append(val_final_losses.get(target, 0))
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Total/Train_Loss', train_loss, epoch)
            writer.add_scalar('Total/Val_Loss', val_loss, epoch)
            writer.add_scalar('Total/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log intermediate target losses
            for target in intermediate_targets:
                if target in train_intermediate_losses:
                    writer.add_scalar(f'Intermediate_Train/{target}', train_intermediate_losses[target], epoch)
                    writer.add_scalar(f'Intermediate_Val/{target}', val_intermediate_losses[target], epoch)
            
            # Log final target losses
            for target in final_targets:
                if target in train_final_losses:
                    writer.add_scalar(f'Final_Train/{target}', train_final_losses[target], epoch)
                    writer.add_scalar(f'Final_Val/{target}', val_final_losses[target], epoch)
        
        # Print progress
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Print individual task losses
        if len(intermediate_targets) > 1:
            print("Intermediate task losses:")
            for target in intermediate_targets:
                if target in train_intermediate_losses:
                    print(f"  {target} - Train: {train_intermediate_losses[target]:.6f}, "
                          f"Val: {val_intermediate_losses[target]:.6f}")
        
        if len(final_targets) > 1:
            print("Final task losses:")
            for target in final_targets:
                if target in train_final_losses:
                    print(f"  {target} - Train: {train_final_losses[target]:.6f}, "
                          f"Val: {val_final_losses[target]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, optimizer, epoch, val_loss, model_save_config,
                os.path.join(save_dir, 'best_model.pth')
            )
        
        # Save model checkpoint
        save_every_n = config.get('save_every_n_epochs', 10)
        if (epoch + 1) % save_every_n == 0:
            save_model(
                model, optimizer, epoch, val_loss, model_save_config,
                os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
            )
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break
    
    # Final model save
    save_model(
        model, optimizer, epoch, val_loss, model_save_config,
        os.path.join(save_dir, 'final_model.pth')
    )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print(f"\nHMTL training completed! Results saved in: {save_dir}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    

if __name__ == "__main__":
    main()