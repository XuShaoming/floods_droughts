#!/usr/bin/env python3
"""
Hierarchical Sequential Task Learning (HSTL) Training Script for Flood/Drought Prediction

This script implements the sequential training strategy for hierarchical LSTM where 
intermediate and final tasks are trained in separate phases. This approach uses a 
two-phase training process:

Phase 1: Train all intermediate target models simultaneously (parallel training)
Phase 2: Load best intermediate models, freeze them, and train final streamflow model
Phase 3: Assemble and validate complete models

Key Features:
- Sequential two-phase training strategy
- Parallel training of intermediate targets in Phase 1
- Selective layer freezing/unfreezing
- Individual model component saving and loading
- Comprehensive model assembly and validation
- TensorBoard logging with phase-specific metrics
- Model checkpointing and early stopping
- GPU acceleration support

Usage:
    python train_hstl.py --experiment streamflow_hstl

This differs from train_hmtl.py in that it trains tasks sequentially rather than
jointly, allowing for specialized optimization of each task phase.

Author: GitHub Copilot
Date: 2024
"""

# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
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

# Local imports
from dataloader_hmtl import FloodDroughtDataLoader, TimeSeriesDataset
from models.LSTM_HMTL import HierarchicalLSTMModel


def load_data(config):
    """Load data using FloodDroughtDataLoader from config"""
    return FloodDroughtDataLoader.from_config(config)


def create_data_loaders(data_loader, config):
    """Create data loaders from FloodDroughtDataLoader instance"""
    loaders = data_loader.create_data_loaders(shuffle_train=True)
    return loaders['train_loader'], loaders['val_loader'], loaders['test_loader']


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


def save_intermediate_model(model, target_name, save_dir, model_type='best'):
    """
    Save weights for a specific intermediate target model.
    
    Args:
        model: HierarchicalLSTMModel instance
        target_name: Name of the intermediate target
        save_dir: Directory to save the model
        model_type: 'best' or 'final'
    """
    # Extract weights for specific intermediate target
    intermediate_weights = {
        f'intermediate_lstms.{target_name}.weight_ih_l0': model.intermediate_lstms[target_name].weight_ih_l0,
        f'intermediate_lstms.{target_name}.weight_hh_l0': model.intermediate_lstms[target_name].weight_hh_l0,
        f'intermediate_lstms.{target_name}.bias_ih_l0': model.intermediate_lstms[target_name].bias_ih_l0,
        f'intermediate_lstms.{target_name}.bias_hh_l0': model.intermediate_lstms[target_name].bias_hh_l0,
        f'intermediate_batch_norms.{target_name}.weight': model.intermediate_batch_norms[target_name].weight,
        f'intermediate_batch_norms.{target_name}.bias': model.intermediate_batch_norms[target_name].bias,
        f'intermediate_batch_norms.{target_name}.running_mean': model.intermediate_batch_norms[target_name].running_mean,
        f'intermediate_batch_norms.{target_name}.running_var': model.intermediate_batch_norms[target_name].running_var,
        f'intermediate_outputs.{target_name}.weight': model.intermediate_outputs[target_name].weight,
        f'intermediate_outputs.{target_name}.bias': model.intermediate_outputs[target_name].bias,
    }
    
    # Handle multi-layer LSTM
    if model.num_layers > 1:
        for layer in range(1, model.num_layers):
            intermediate_weights.update({
                f'intermediate_lstms.{target_name}.weight_ih_l{layer}': getattr(model.intermediate_lstms[target_name], f'weight_ih_l{layer}'),
                f'intermediate_lstms.{target_name}.weight_hh_l{layer}': getattr(model.intermediate_lstms[target_name], f'weight_hh_l{layer}'),
                f'intermediate_lstms.{target_name}.bias_ih_l{layer}': getattr(model.intermediate_lstms[target_name], f'bias_ih_l{layer}'),
                f'intermediate_lstms.{target_name}.bias_hh_l{layer}': getattr(model.intermediate_lstms[target_name], f'bias_hh_l{layer}'),
            })
    
    filepath = os.path.join(save_dir, f'{model_type}_model_{target_name}.pth')
    torch.save(intermediate_weights, filepath)
    print(f"Saved {model_type} model for {target_name} to {filepath}")


def load_intermediate_weights(model, target_name, save_dir, model_type='best'):
    """
    Load weights for a specific intermediate target model.
    
    Args:
        model: HierarchicalLSTMModel instance
        target_name: Name of the intermediate target
        save_dir: Directory containing the saved model
        model_type: 'best' or 'final'
    """
    filepath = os.path.join(save_dir, f'{model_type}_model_{target_name}.pth')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist. Skipping load for {target_name}")
        return
    
    weights = torch.load(filepath, map_location='cpu')
    
    # Load weights into the model
    model_dict = model.state_dict()
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {model_type} weights for {target_name} from {filepath}")


def save_final_model(model, save_dir, model_type='best'):
    """
    Save weights for the final streamflow model.
    
    Args:
        model: HierarchicalLSTMModel instance
        save_dir: Directory to save the model
        model_type: 'best' or 'final'
    """
    # Extract final layer weights
    final_weights = {
        'final_lstm.weight_ih_l0': model.final_lstm.weight_ih_l0,
        'final_lstm.weight_hh_l0': model.final_lstm.weight_hh_l0,
        'final_lstm.bias_ih_l0': model.final_lstm.bias_ih_l0,
        'final_lstm.bias_hh_l0': model.final_lstm.bias_hh_l0,
        'final_batch_norm.weight': model.final_batch_norm.weight,
        'final_batch_norm.bias': model.final_batch_norm.bias,
        'final_batch_norm.running_mean': model.final_batch_norm.running_mean,
        'final_batch_norm.running_var': model.final_batch_norm.running_var,
        'final_output.weight': model.final_output.weight,
        'final_output.bias': model.final_output.bias,
    }
    
    # Handle multi-layer LSTM
    if model.num_layers > 1:
        for layer in range(1, model.num_layers):
            final_weights.update({
                f'final_lstm.weight_ih_l{layer}': getattr(model.final_lstm, f'weight_ih_l{layer}'),
                f'final_lstm.weight_hh_l{layer}': getattr(model.final_lstm, f'weight_hh_l{layer}'),
                f'final_lstm.bias_ih_l{layer}': getattr(model.final_lstm, f'bias_ih_l{layer}'),
                f'final_lstm.bias_hh_l{layer}': getattr(model.final_lstm, f'bias_hh_l{layer}'),
            })
    
    filepath = os.path.join(save_dir, f'{model_type}_model_streamflow.pth')
    torch.save(final_weights, filepath)
    print(f"Saved {model_type} final model to {filepath}")


def load_final_weights(model, save_dir, model_type='best'):
    """
    Load weights for the final streamflow model.
    
    Args:
        model: HierarchicalLSTMModel instance
        save_dir: Directory containing the saved model
        model_type: 'best' or 'final'
    """
    filepath = os.path.join(save_dir, f'{model_type}_model_streamflow.pth')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist. Skipping final model load")
        return
    
    weights = torch.load(filepath, map_location='cpu')
    
    # Load weights into the model
    model_dict = model.state_dict()
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {model_type} final weights from {filepath}")


def train_intermediate_epoch(model, train_loader, criterion, optimizer, device, 
                           intermediate_targets, grad_clip_norm=1.0):
    """
    Train intermediate targets for one epoch (Phase 1)
    
    Args:
        model: Hierarchical LSTM model with final layer frozen
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device (cuda/cpu)
        intermediate_targets: List of intermediate target names
        grad_clip_norm: Gradient clipping norm
    
    Returns:
        avg_total_loss: Average total training loss
        avg_intermediate_losses: Dictionary of individual intermediate task losses
    """
    model.train()
    total_loss = 0.0
    intermediate_losses = {target: 0.0 for target in intermediate_targets}
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training Intermediate", leave=False):
        optimizer.zero_grad()
        
        if len(batch) == 4:
            # Hierarchical format: (features, final_targets, intermediate_targets, index)
            inputs, final_batch_targets, intermediate_batch_targets, _ = batch
            
            # Convert to dictionaries for easier handling
            intermediate_targets_dict = {}
            for i, target_name in enumerate(intermediate_targets):
                if i < intermediate_batch_targets.shape[-1]:
                    intermediate_targets_dict[target_name] = intermediate_batch_targets[:, :, i:i+1]
                    
        else:
            # Standard format: (features, targets, index)
            inputs, batch_targets, _ = batch
            
            # Split targets based on order (intermediate first, then final)
            intermediate_targets_dict = {}
            
            for i, target_name in enumerate(intermediate_targets):
                intermediate_targets_dict[target_name] = batch_targets[:, :, i:i+1]
        
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Move targets to device
        for target_name in intermediate_targets:
            if target_name in intermediate_targets_dict:
                intermediate_targets_dict[target_name] = intermediate_targets_dict[target_name].to(device)
        
        # Forward pass
        outputs = model(inputs)
        intermediate_outputs = outputs['intermediate']
        
        # Calculate intermediate task losses (parallel training)
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
        
        # Only perform backward pass if we have a valid loss tensor
        if total_intermediate_loss is not None:
            # Backward pass
            total_intermediate_loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            total_loss += total_intermediate_loss.item()
        
        batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_intermediate_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in intermediate_losses.items()}
    
    return avg_total_loss, avg_intermediate_losses


def validate_intermediate_epoch(model, val_loader, criterion, device, intermediate_targets):
    """
    Validate intermediate targets for one epoch (Phase 1)
    
    Args:
        model: Hierarchical LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        intermediate_targets: List of intermediate target names
    
    Returns:
        avg_total_loss: Average total validation loss
        avg_intermediate_losses: Dictionary of individual intermediate task losses
    """
    model.eval()
    total_loss = 0.0
    intermediate_losses = {target: 0.0 for target in intermediate_targets}
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                # Hierarchical format: (features, final_targets, intermediate_targets, index)
                inputs, final_batch_targets, intermediate_batch_targets, _ = batch
                
                # Convert to dictionaries for easier handling
                intermediate_targets_dict = {}
                for i, target_name in enumerate(intermediate_targets):
                    if i < intermediate_batch_targets.shape[-1]:
                        intermediate_targets_dict[target_name] = intermediate_batch_targets[:, :, i:i+1]
                        
            else:
                # Standard format: (features, targets, index)
                inputs, batch_targets, _ = batch
                
                # Split targets based on order (intermediate first, then final)
                intermediate_targets_dict = {}
                
                for i, target_name in enumerate(intermediate_targets):
                    intermediate_targets_dict[target_name] = batch_targets[:, :, i:i+1]
            
            inputs = inputs.to(device)
            
            # Move targets to device
            for target_name in intermediate_targets:
                if target_name in intermediate_targets_dict:
                    intermediate_targets_dict[target_name] = intermediate_targets_dict[target_name].to(device)
            
            # Forward pass
            outputs = model(inputs)
            intermediate_outputs = outputs['intermediate']
            
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
            
            total_loss += total_intermediate_loss.item()
            batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_intermediate_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in intermediate_losses.items()}
    
    return avg_total_loss, avg_intermediate_losses


def train_final_epoch(model, train_loader, criterion, optimizer, device, 
                     final_targets, grad_clip_norm=1.0):
    """
    Train final streamflow target for one epoch (Phase 2)
    
    Args:
        model: Hierarchical LSTM model with intermediate layers frozen
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device (cuda/cpu)
        final_targets: List of final target names
        grad_clip_norm: Gradient clipping norm
    
    Returns:
        avg_total_loss: Average total training loss
        avg_final_losses: Dictionary of individual final task losses
    """
    model.train()
    total_loss = 0.0
    final_losses = {target: 0.0 for target in final_targets}
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training Final", leave=False):
        optimizer.zero_grad()
        
        if len(batch) == 4:
            # Hierarchical format: (features, final_targets, intermediate_targets, index)
            inputs, final_batch_targets, intermediate_batch_targets, _ = batch
            
            # Convert to dictionaries for easier handling
            final_targets_dict = {}
            for i, target_name in enumerate(final_targets):
                if i < final_batch_targets.shape[-1]:
                    final_targets_dict[target_name] = final_batch_targets[:, :, i:i+1]
                    
        else:
            # Standard format: (features, targets, index)
            inputs, batch_targets, _ = batch
            
            # Split targets - final targets come after intermediate targets
            final_targets_dict = {}
            intermediate_targets_count = len(model.intermediate_targets)
            
            for i, target_name in enumerate(final_targets):
                final_idx = intermediate_targets_count + i
                final_targets_dict[target_name] = batch_targets[:, :, final_idx:final_idx+1]
        
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Move targets to device
        for target_name in final_targets:
            if target_name in final_targets_dict:
                final_targets_dict[target_name] = final_targets_dict[target_name].to(device)
        
        # Forward pass (intermediate predictions are generated automatically)
        outputs = model(inputs)
        final_outputs = outputs['final']
        
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
        
        # Only perform backward pass if we have a valid loss tensor
        if total_final_loss is not None:
            # Backward pass
            total_final_loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            total_loss += total_final_loss.item()
        
        batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_final_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in final_losses.items()}
    
    return avg_total_loss, avg_final_losses


def validate_final_epoch(model, val_loader, criterion, device, final_targets):
    """
    Validate final streamflow target for one epoch (Phase 2)
    
    Args:
        model: Hierarchical LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device
        final_targets: List of final target names
    
    Returns:
        avg_total_loss: Average total validation loss
        avg_final_losses: Dictionary of individual final task losses
    """
    model.eval()
    total_loss = 0.0
    final_losses = {target: 0.0 for target in final_targets}
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                # Hierarchical format: (features, final_targets, intermediate_targets, index)
                inputs, final_batch_targets, intermediate_batch_targets, _ = batch
                
                # Convert to dictionaries for easier handling
                final_targets_dict = {}
                for i, target_name in enumerate(final_targets):
                    if i < final_batch_targets.shape[-1]:
                        final_targets_dict[target_name] = final_batch_targets[:, :, i:i+1]
                        
            else:
                # Standard format: (features, targets, index)
                inputs, batch_targets, _ = batch
                
                # Split targets - final targets come after intermediate targets
                final_targets_dict = {}
                intermediate_targets_count = len(model.intermediate_targets)
                
                for i, target_name in enumerate(final_targets):
                    final_idx = intermediate_targets_count + i
                    final_targets_dict[target_name] = batch_targets[:, :, final_idx:final_idx+1]
            
            inputs = inputs.to(device)
            
            # Move targets to device
            for target_name in final_targets:
                if target_name in final_targets_dict:
                    final_targets_dict[target_name] = final_targets_dict[target_name].to(device)
            
            # Forward pass
            outputs = model(inputs)
            final_outputs = outputs['final']
            
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
            
            total_loss += total_final_loss.item()
            batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_final_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in final_losses.items()}
    
    return avg_total_loss, avg_final_losses


def validate_full_model(model, test_loader, criterion, device, intermediate_targets, final_targets):
    """
    Validate the full assembled model on test data (Phase 3)
    
    Args:
        model: Complete HierarchicalLSTMModel
        test_loader: Test data loader
        criterion: Loss function
        device: Computing device
        intermediate_targets: List of intermediate target names
        final_targets: List of final target names
    
    Returns:
        dict: Comprehensive metrics for all targets
    """
    model.eval()
    total_loss = 0.0
    intermediate_losses = {target: 0.0 for target in intermediate_targets}
    final_losses = {target: 0.0 for target in final_targets}
    batch_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
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
            
            # Combined loss
            batch_loss = total_intermediate_loss + total_final_loss
            total_loss += batch_loss.item()
            batch_count += 1
    
    # Average losses
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_intermediate_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in intermediate_losses.items()}
    avg_final_losses = {k: v / batch_count if batch_count > 0 else 0.0 for k, v in final_losses.items()}
    
    return {
        'total_loss': avg_total_loss,
        'intermediate_losses': avg_intermediate_losses,
        'final_losses': avg_final_losses
    }



def main():
    """Main HSTL training function"""
    
    parser = argparse.ArgumentParser(description='Train Hierarchical Sequential Task Learning LSTM')
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
    print(f"Running HSTL experiment: {args.experiment}")
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
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump({args.experiment: config}, f, default_flow_style=False)
    
    # Load data
    print("Loading data...")
    data_loader = load_data(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_loader, 
        config
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Extract target configurations
    intermediate_targets = config['intermediate_targets']
    final_targets = config['target_cols']
    
    print(f"Intermediate targets: {intermediate_targets}")
    print(f"Final targets: {final_targets}")
    
    # Model configuration
    input_size = len(config['feature_cols'])
    
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
    param_info = model.get_trainable_parameters()
    print(f"\nModel Summary:")
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Intermediate parameters: {param_info['intermediate']:,}")
    print(f"Final parameters: {param_info['final']:,}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Model save configuration
    model_save_config = {
        'input_size': input_size,
        'intermediate_targets': intermediate_targets,
        'final_targets': final_targets,
        'feature_cols': config['feature_cols'],
        'target_cols': config['target_cols'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config.get('dropout', 0.2)
    }
    
    # TensorBoard logging
    writer = None
    if config.get('tensorboard_log', True):
        log_dir = os.path.join(save_dir, 'tensorboard_logs')
        writer = SummaryWriter(log_dir)
    
    print(f"\nStarting HSTL training with {config['epochs']} epochs per phase...")
    print("="*80)
    
    # ==============================================
    # PHASE 1: Train Intermediate Targets
    # ==============================================
    print(f"\nPHASE 1: Training Intermediate Targets")
    print("="*60)
    
    # Freeze final layer for Phase 1
    model.freeze_final_layer()
    param_info = model.get_trainable_parameters()
    print(f"Phase 1 - Trainable parameters: {param_info['trainable']:,}")
    
    # Optimizer for Phase 1
    weight_decay = float(config.get('weight_decay', 0)) if config.get('weight_decay') else 0
    optimizer_phase1 = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler for Phase 1
    scheduler_phase1 = None
    if config.get('scheduler_type'):
        scheduler_type = config['scheduler_type']
        if scheduler_type == 'StepLR':
            scheduler_phase1 = optim.lr_scheduler.StepLR(
                optimizer_phase1,
                step_size=config.get('scheduler_step_size', 20),
                gamma=config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_phase1,
                mode='min',
                factor=config.get('scheduler_factor', 0.5),
                patience=config.get('scheduler_patience', 10),
                min_lr=float(config.get('scheduler_min_lr', 1e-6)),
                verbose=True
            )
    
    # Early stopping for individual intermediate targets
    early_stopping_intermediate = {
        target: EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=float(config.get('early_stopping_min_delta', 1e-6))
        ) for target in intermediate_targets
    }
    
    # Track best losses for each intermediate target
    best_intermediate_losses = {target: float('inf') for target in intermediate_targets}
    
    # Phase 1 training loop
    for epoch in range(config['epochs']):
        print(f"\nPhase 1 - Epoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        # Training
        train_loss, train_intermediate_losses = train_intermediate_epoch(
            model, train_loader, criterion, optimizer_phase1, device,
            intermediate_targets, grad_clip_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Validation
        val_loss, val_intermediate_losses = validate_intermediate_epoch(
            model, val_loader, criterion, device, intermediate_targets
        )
        
        # Update learning rate
        if scheduler_phase1 is not None:
            if isinstance(scheduler_phase1, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_phase1.step(val_loss)
            else:
                scheduler_phase1.step()
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Phase1/Total_Train_Loss', train_loss, epoch)
            writer.add_scalar('Phase1/Total_Val_Loss', val_loss, epoch)
            writer.add_scalar('Phase1/Learning_Rate', optimizer_phase1.param_groups[0]['lr'], epoch)
            
            # Log individual intermediate target losses
            for target in intermediate_targets:
                if target in train_intermediate_losses:
                    writer.add_scalar(f'Phase1_Intermediate_Train/{target}', train_intermediate_losses[target], epoch)
                    writer.add_scalar(f'Phase1_Intermediate_Val/{target}', val_intermediate_losses[target], epoch)
        
        # Print progress
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer_phase1.param_groups[0]['lr']:.8f}")
        
        # Print individual intermediate task losses
        print("Intermediate task losses:")
        for target in intermediate_targets:
            if target in train_intermediate_losses:
                print(f"  {target} - Train: {train_intermediate_losses[target]:.6f}, "
                      f"Val: {val_intermediate_losses[target]:.6f}")
        
        # Save best models for each intermediate target
        for target in intermediate_targets:
            if target in val_intermediate_losses:
                target_val_loss = val_intermediate_losses[target]
                if target_val_loss < best_intermediate_losses[target]:
                    best_intermediate_losses[target] = target_val_loss
                    save_intermediate_model(model, target, save_dir, 'best')
        
        # Save model checkpoint
        save_every_n = config.get('save_every_n_epochs', 10)
        if (epoch + 1) % save_every_n == 0:
            save_model(
                model, optimizer_phase1, epoch, val_loss, model_save_config,
                os.path.join(save_dir, f'phase1_model_epoch_{epoch + 1}.pth')
            )
    
    # Save final models for all intermediate targets
    for target in intermediate_targets:
        save_intermediate_model(model, target, save_dir, 'final')
    
    print(f"\nPhase 1 completed! Best intermediate losses:")
    for target, loss in best_intermediate_losses.items():
        print(f"  {target}: {loss:.6f}")
    
    # ==============================================
    # PHASE 2: Train Final Streamflow Target
    # ==============================================
    print(f"\nPHASE 2: Training Final Streamflow Target")
    print("="*60)
    
    # Load best intermediate models and freeze them
    print("Loading best intermediate models...")
    for target in intermediate_targets:
        load_intermediate_weights(model, target, save_dir, 'best')
    
    # Freeze intermediate layers and unfreeze final layer
    model.freeze_intermediate_layers()
    model.unfreeze_final_layer()
    param_info = model.get_trainable_parameters()
    print(f"Phase 2 - Trainable parameters: {param_info['trainable']:,}")
    
    # Optimizer for Phase 2
    optimizer_phase2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler for Phase 2
    scheduler_phase2 = None
    if config.get('scheduler_type'):
        scheduler_type = config['scheduler_type']
        if scheduler_type == 'StepLR':
            scheduler_phase2 = optim.lr_scheduler.StepLR(
                optimizer_phase2,
                step_size=config.get('scheduler_step_size', 20),
                gamma=config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_phase2,
                mode='min',
                factor=config.get('scheduler_factor', 0.5),
                patience=config.get('scheduler_patience', 10),
                min_lr=float(config.get('scheduler_min_lr', 1e-6)),
                verbose=True
            )
    
    # Early stopping for final target
    early_stopping_final = EarlyStopping(
        patience=config.get('early_stopping_patience', 15),
        min_delta=float(config.get('early_stopping_min_delta', 1e-6))
    )
    
    # Track best loss for final target
    best_final_loss = float('inf')
    
    # Initialize variables for final model assembly
    final_epoch = 0
    final_val_loss = 0.0
    
    # Phase 2 training loop
    for epoch in range(config['epochs']):
        final_epoch = epoch  # Track final epoch
        print(f"\nPhase 2 - Epoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        # Training
        train_loss, train_final_losses = train_final_epoch(
            model, train_loader, criterion, optimizer_phase2, device,
            final_targets, grad_clip_norm=config.get('grad_clip_norm', 1.0)
        )
        
        # Validation
        val_loss, val_final_losses = validate_final_epoch(
            model, val_loader, criterion, device, final_targets
        )
        
        final_val_loss = val_loss  # Track final validation loss
        
        # Update learning rate
        if scheduler_phase2 is not None:
            if isinstance(scheduler_phase2, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_phase2.step(val_loss)
            else:
                scheduler_phase2.step()
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Phase2/Total_Train_Loss', train_loss, epoch)
            writer.add_scalar('Phase2/Total_Val_Loss', val_loss, epoch)
            writer.add_scalar('Phase2/Learning_Rate', optimizer_phase2.param_groups[0]['lr'], epoch)
            
            # Log final target losses
            for target in final_targets:
                if target in train_final_losses:
                    writer.add_scalar(f'Phase2_Final_Train/{target}', train_final_losses[target], epoch)
                    writer.add_scalar(f'Phase2_Final_Val/{target}', val_final_losses[target], epoch)
        
        # Print progress
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {optimizer_phase2.param_groups[0]['lr']:.8f}")
        
        # Print final task losses
        print("Final task losses:")
        for target in final_targets:
            if target in train_final_losses:
                print(f"  {target} - Train: {train_final_losses[target]:.6f}, "
                      f"Val: {val_final_losses[target]:.6f}")
        
        # Save best final model
        if val_loss < best_final_loss:
            best_final_loss = val_loss
            save_final_model(model, save_dir, 'best')
        
        # Save model checkpoint
        save_every_n = config.get('save_every_n_epochs', 10)
        if (epoch + 1) % save_every_n == 0:
            save_model(
                model, optimizer_phase2, epoch, val_loss, model_save_config,
                os.path.join(save_dir, f'phase2_model_epoch_{epoch + 1}.pth')
            )
        
        # Early stopping
        if early_stopping_final(val_loss, model):
            print(f"\nPhase 2 early stopping triggered after epoch {epoch + 1}")
            break
    
    # Save final streamflow model
    save_final_model(model, save_dir, 'final')
    
    print(f"\nPhase 2 completed! Best final loss: {best_final_loss:.6f}")
    
    # ==============================================
    # PHASE 3: Model Assembly and Validation
    # ==============================================
    print(f"\nPHASE 3: Model Assembly and Validation")
    print("="*60)
    
    # Assemble BEST model
    print("Assembling BEST model...")
    best_model = HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=intermediate_targets,
        final_targets=final_targets,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        batch_first=True
    ).to(device)
    
    # Load best weights
    for target in intermediate_targets:
        load_intermediate_weights(best_model, target, save_dir, 'best')
    load_final_weights(best_model, save_dir, 'best')
    
    # Save assembled best model
    save_model(
        best_model, optimizer_phase2, final_epoch, best_final_loss, model_save_config,
        os.path.join(save_dir, 'best_model.pth')
    )
    
    # Assemble FINAL model
    print("Assembling FINAL model...")
    final_model = HierarchicalLSTMModel(
        input_size=input_size,
        intermediate_targets=intermediate_targets,
        final_targets=final_targets,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        batch_first=True
    ).to(device)
    
    # Load final weights
    for target in intermediate_targets:
        load_intermediate_weights(final_model, target, save_dir, 'final')
    load_final_weights(final_model, save_dir, 'final')
    
    # Save assembled final model
    save_model(
        final_model, optimizer_phase2, final_epoch, final_val_loss, model_save_config,
        os.path.join(save_dir, 'final_model.pth')
    )
    
    # Validate assembled models
    print("Validating assembled models on test data...")
    
    best_metrics = validate_full_model(
        best_model, test_loader, criterion, device, intermediate_targets, final_targets
    )
    final_metrics = validate_full_model(
        final_model, test_loader, criterion, device, intermediate_targets, final_targets
    )
    
    print(f"\nTest Results:")
    print(f"Best model test loss: {best_metrics['total_loss']:.6f}")
    print(f"Final model test loss: {final_metrics['total_loss']:.6f}")
    
    print(f"\nBest model - Intermediate test losses:")
    for target, loss in best_metrics['intermediate_losses'].items():
        print(f"  {target}: {loss:.6f}")
    
    print(f"\nBest model - Final test losses:")
    for target, loss in best_metrics['final_losses'].items():
        print(f"  {target}: {loss:.6f}")
    
    # Log final results to TensorBoard
    if writer is not None:
        writer.add_scalar('Phase3/Best_Model_Test_Loss', best_metrics['total_loss'], 0)
        writer.add_scalar('Phase3/Final_Model_Test_Loss', final_metrics['total_loss'], 0)
        
        for target, loss in best_metrics['intermediate_losses'].items():
            writer.add_scalar(f'Phase3_Best_Intermediate/{target}', loss, 0)
        
        for target, loss in best_metrics['final_losses'].items():
            writer.add_scalar(f'Phase3_Best_Final/{target}', loss, 0)
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    # Save final metrics
    metrics_summary = {
        'phase1_best_intermediate_losses': best_intermediate_losses,
        'phase2_best_final_loss': best_final_loss,
        'phase3_best_model_test_metrics': best_metrics,
        'phase3_final_model_test_metrics': final_metrics,
        'training_time_seconds': time.time() - start_time
    }
    
    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\nHSTL training completed! Results saved in: {save_dir}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print("\nModel files created:")
    print(f"  - Individual intermediate models: best_model_{{target}}.pth, final_model_{{target}}.pth")
    print(f"  - Individual final model: best_model_streamflow.pth, final_model_streamflow.pth")
    print(f"  - Assembled models: best_model.pth, final_model.pth")
    print(f"  - Training metrics: training_metrics.json")
    

if __name__ == "__main__":
    main()