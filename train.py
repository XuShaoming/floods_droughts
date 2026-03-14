#!/usr/bin/env python3
"""
LSTM Training Script for Flood and Drought Prediction with YAML Configuration

This script trains an LSTM model for many-to-many sequence prediction using
YAML configuration files for all hyperparameters and settings. It supports:
- Multi-task learning (multiple target variables)
- Many-to-many sequence modeling  
- Model checkpointing and early stopping
- Comprehensive logging and visualization
- GPU acceleration when available
- Fully YAML-based configuration with inheritance

Usage:
    python train.py                                    # Use default experiment
    python train.py --experiment hourly_short_term     # Specific experiment
    python train.py --config my_config.yaml            # Custom config file
    python train.py --seed 123                         # Custom random seed
    
Available Hourly Streamflow Experiments:
    - hourly_short_term:    Flash flood prediction (72h window, 1h stride)
    - hourly_daily_cycles:  Daily patterns/snowmelt (168h window, 6h stride)  
    - hourly_flood_events:  Multi-day floods (120h window, 12h stride)
    - hourly_baseflow:      Seasonal trends (360h window, 24h stride)
    - multitarget:          Joint streamflow+ET (30h window, 1h stride)

Configuration uses YAML inheritance for maintainable experiment management.
All hyperparameters are in YAML files for better reproducibility and version control.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import argparse
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataloader import FloodDroughtDataLoader
from models.LSTMModel import LSTMModel


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate required sections
    required_sections = ['data', 'model', 'training', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    return config


def get_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule
    config : dict
        Scheduler configuration
        
    Returns:
    --------
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'ReduceLROnPlateau')
    
    if scheduler_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=float(config.get('factor', 0.5)),
            patience=int(config.get('patience', 5)),
            min_lr=float(config.get('min_lr', 1e-6)),
            verbose=True
        )
    elif scheduler_type == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get('step_size', 20)),
            gamma=float(config.get('gamma', 0.5))
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config.get('T_0', 10)),
            T_mult=int(config.get('T_mult', 2)),
            eta_min=float(config.get('min_lr', 1e-6))
        )
    elif scheduler_type == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(config.get('gamma', 0.95))
        )
    elif scheduler_type == 'OneCycleLR':
        # This requires knowing the number of steps per epoch
        # Note: This would need to be set up differently with train_loader info
        raise NotImplementedError("OneCycleLR requires train_loader information. Use train_yaml.py for this scheduler.")
    elif scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get('T_max', 50)),
            eta_min=float(config.get('min_lr', 1e-6))
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
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
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()


def calculate_metrics(predictions, targets, scaler=None):
    """
    Calculate various evaluation metrics.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted values
    targets : np.ndarray
        True values
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler to inverse transform the data
        
    Returns:
    --------
    dict
        Dictionary containing various metrics
    """
    # Inverse transform if scaler provided
    if scaler is not None:
        print(f"Debug: Denormalizing data...")
        print(f"Before denorm - Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"Before denorm - Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
        
        if predictions.ndim == 3:
            # Reshape for inverse transform
            orig_shape = predictions.shape
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
            predictions = scaler.inverse_transform(predictions)
            targets = scaler.inverse_transform(targets)
            
            predictions = predictions.reshape(orig_shape)
            targets = targets.reshape(orig_shape)
        else:
            predictions = scaler.inverse_transform(predictions)
            targets = scaler.inverse_transform(targets)
            
        print(f"After denorm - Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"After denorm - Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
    else:
        print("Debug: No scaler provided, using raw values")
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate MAPE (avoid division by zero)
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape)
    }


def train_epoch(model, train_loader, criterion, optimizer, device, target_names, grad_clip_norm=1.0):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Track losses per target
    target_losses = {name: 0.0 for name in target_names}
    
    # for batch_idx, (features, targets, _) in enumerate(tqdm(train_loader, desc="Training")):
    for batch_idx, (features, targets, _) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Calculate individual target losses for monitoring
        if len(target_names) > 1:
            for i, name in enumerate(target_names):
                target_loss = criterion(predictions[:, :, i], targets[:, :, i])
                target_losses[name] += target_loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_target_losses = {name: loss / num_batches for name, loss in target_losses.items()}
    
    return avg_loss, avg_target_losses


def validate_epoch(model, val_loader, criterion, device, target_names):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Track losses per target
    target_losses = {name: 0.0 for name in target_names}
    
    with torch.no_grad():
        # for features, targets, _ in tqdm(val_loader, desc="Validation"):
        for features, targets, _ in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(features)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Calculate individual target losses for monitoring
            if len(target_names) > 1:
                for i, name in enumerate(target_names):
                    target_loss = criterion(predictions[:, :, i], targets[:, :, i])
                    target_losses[name] += target_loss.item()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_target_losses = {name: loss / num_batches for name, loss in target_losses.items()}
    
    return avg_loss, avg_target_losses


def save_model(model, optimizer, epoch, loss, model_config, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model_config
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def plot_training_history(train_losses, val_losses, save_dir, filename: str = "learning_curves.png"):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    # Plot overall losses
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {output_path}")


def seed_everything(seed=42):
    """
    Seed all random number generators for reproducible results.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy random number generator
    - PyTorch random number generators (CPU and GPU)
    - PyTorch backend settings for deterministic behavior
    
    Parameters:
    -----------
    seed : int, default=42
        Random seed value to use across all libraries
        
    Note:
    -----
    Setting deterministic behavior may impact performance but ensures
    complete reproducibility across runs and different hardware.
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


def main():
    """Main training function using YAML configuration."""
    parser = argparse.ArgumentParser(description='Train LSTM model for flood/drought prediction')
    
    # YAML config support
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name from config file (for multi-experiment configs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found. Please create a config.yaml file.")
    
    print(f"Loading configuration from {args.config}")
    
    # Load the full config file first
    with open(args.config, 'r') as file:
        full_config = yaml.safe_load(file)
    
    # Handle experiment selection
    if args.experiment:
        # Multi-experiment config file
        available_experiments = [key for key in full_config.keys() 
                               if key not in ['base_config', 'BASE_CONFIG', 'default_experiment'] 
                               and not key.startswith('&')]
        
        if args.experiment in available_experiments:
            config = full_config[args.experiment]
            print(f"Using experiment: {args.experiment}")
        else:
            print(f"Available experiments: {available_experiments}")
            raise ValueError(f"Experiment '{args.experiment}' not found in config file")
    else:
        # Check if this is a multi-experiment config or single config
        required_sections = ['data', 'model', 'training', 'output']
        is_single_config = all(section in full_config for section in required_sections)
        
        if is_single_config:
            # Single experiment config
            config = full_config
            print("Using single experiment configuration")
        else:
            # Multi-experiment config - use default
            default_exp = full_config.get('default_experiment')
            available_experiments = [key for key in full_config.keys() 
                                   if key not in ['base_config', 'BASE_CONFIG', 'default_experiment'] 
                                   and not key.startswith('&')]
            
            if default_exp and default_exp in available_experiments:
                config = full_config[default_exp]
                print(f"Using default experiment: {default_exp}")
            elif available_experiments:
                config = full_config[available_experiments[0]]
                print(f"Using first available experiment: {available_experiments[0]}")
                print(f"Available experiments: {available_experiments}")
            else:
                raise ValueError("No valid experiments found in config file")
    
    # Validate the selected config
    required_keys = ['csv_file', 'hidden_size', 'batch_size', 'learning_rate', 'epochs', 'save_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in experiment configuration")
    
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    # Extract configuration values (now flattened)
    # Use command line seed if provided, otherwise use config seed, otherwise default to 42
    if hasattr(args, 'seed') and args.seed is not None:
        seed = args.seed
        print(f"Using seed from command line: {seed}")
    else:
        seed = config.get('seed', 42)
        print(f"Using seed from config: {seed}")
    
    # Create save directory
    save_dir = config['save_dir']
    save_dir = os.path.join(save_dir, args.experiment)
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    seed_everything(seed)
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = FloodDroughtDataLoader(
        csv_file=config['csv_file'],
        window_size=config['window_size'],
        stride=config['stride'],
        target_col=config['target_cols'],
        feature_cols=config['feature_cols'],
        train_years=tuple(config['train_years']),
        val_years=tuple(config['val_years']),
        test_years=tuple(config['test_years']),
        batch_size=config['batch_size'],
        scale_features=config.get('scale_features', True),
        scale_targets=config.get('scale_targets', True),
        many_to_many=True,  # Many-to-many prediction
        random_seed=seed
    )
    
    # Create data loaders
    print("Creating data loaders...")
    loaders = data_loader.create_data_loaders(shuffle_train=True)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    
    # Get data dimensions
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[-1]  # Number of features
    output_size = sample_batch[1].shape[-1]  # Number of targets
    
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Window size: {config['window_size']}")
    
    # Initialize model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=output_size,
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    # Ensure weight_decay is properly converted to float
    weight_decay = config.get('weight_decay', 1e-5)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    learning_rate = config['learning_rate']
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    # Use AdamW optimizer (generally better than Adam)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler - build config dict from flattened parameters
    scheduler_config = {
        'type': config.get('scheduler_type', 'ReduceLROnPlateau'),
        'patience': config.get('scheduler_patience', 5),
        'factor': config.get('scheduler_factor', 0.5),
        'min_lr': config.get('scheduler_min_lr', 1e-6)
    }
    scheduler = get_scheduler(optimizer, scheduler_config)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=int(config.get('early_stopping_patience', 10)),
        min_delta=float(config.get('early_stopping_min_delta', 1e-6))
    )
    
    
    # TensorBoard logging
    if config.get('tensorboard_log', True):
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    else:
        writer = None
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Model configuration for saving
    model_save_config = {
        'input_size': input_size,
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'output_size': output_size,
        'dropout': config['dropout'],
        'window_size': config['window_size'],
        'target_cols': config['target_cols'],
        'feature_cols': data_loader.get_feature_names()
    }
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(model_save_config, f, indent=2)
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_target_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, config['target_cols'],
            grad_clip_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Validate
        val_loss, val_target_losses = validate_epoch(
            model, val_loader, criterion, device, config['target_cols']
        )
        
        # Update learning rate
        if hasattr(scheduler, 'step'):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log individual target losses
            for target_name in config['target_cols']:
                if target_name in train_target_losses:
                    writer.add_scalar(f'Train_Loss/{target_name}', train_target_losses[target_name], epoch)
                    writer.add_scalar(f'Val_Loss/{target_name}', val_target_losses[target_name], epoch)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        if len(config['target_cols']) > 1:
            print("Individual target losses:")
            for target_name in config['target_cols']:
                if target_name in train_target_losses:
                    print(f"  {target_name} - Train: {train_target_losses[target_name]:.6f}, "
                          f"Val: {val_target_losses[target_name]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, optimizer, epoch, val_loss, model_save_config,
                os.path.join(save_dir, 'best_model.pth')
            )
        
        # Save latest model
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
    
    # Plot training history
    plot_training_history(train_losses, val_losses, save_dir)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        # for features, targets, _ in tqdm(test_loader, desc="Testing"):
        for features, targets, _ in test_loader:
            features = features.to(device)
            predictions = model(features)
            
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(targets.numpy())
    
    # Concatenate predictions and targets
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    print(f"\nDebug info before denormalization:")
    print(f"Predictions shape: {test_predictions.shape}")
    print(f"Targets shape: {test_targets.shape}")
    print(f"Predictions range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    print(f"Targets range: [{test_targets.min():.6f}, {test_targets.max():.6f}]")
    print(f"Target scaler available: {data_loader.target_scaler is not None}")
    
    if data_loader.target_scaler is not None:
        print(f"Target scaler mean: {data_loader.target_scaler.mean_}")
        print(f"Target scaler scale: {data_loader.target_scaler.scale_}")
    
    # Calculate metrics
    metrics = calculate_metrics(test_predictions, test_targets, data_loader.target_scaler)
    
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Save metrics
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions for further analysis
    np.save(os.path.join(save_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(save_dir, 'test_targets.npy'), test_targets)
    
    if writer is not None:
        writer.close()
    print(f"\nTraining completed! Results saved to {save_dir}")


if __name__ == "__main__":
    main()
