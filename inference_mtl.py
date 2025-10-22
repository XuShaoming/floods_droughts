#!/usr/bin/env python3
"""
Hierarchical LSTM Model Inference Script for Flood/Drought Prediction

This script loads a trained HierarchicalLSTMModel and performs comprehensive evaluation using
the same YAML configuration system as train_hmtl.py. It provides:

1. Model and configuration loading from hierarchical experiment directories
2. Predictions on train/validation/test sets with proper hierarchical data loading
3. Denormalization back to original scale using separate scalers for intermediate and final targets
4. Comprehensive metrics calculation for each target feature (MSE, RMSE, MAE, R², NSE, KGE, etc.)
5. Rich visualizations per feature (scatter plots, time series, residuals, error distributions)
6. Time series reconstruction from windowed predictions for all targets
7. Feature-specific analysis with individual folders and results
8. Results saving and logging with hierarchical structure

Key Features:
- Support for HierarchicalLSTMModel with intermediate and final targets
- Feature selection for analysis (analyze specific targets only)
- Individual folders for each analyzed feature
- Concatenated predictions/targets in configurable order
- Separate handling of intermediate and final target normalization

Usage:
    # Analyze all features
    python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset test --analysis

    # Analyze specific features only
    python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset test --analysis --analyze-features PET,ET,streamflow

    # Run inference only (no analysis)
    python inference_mtl.py --model-dir experiments/streamflow_hmtl --model-trained best_model.pth --dataset test

The script automatically detects the hierarchical experiment configuration and uses the same
data loading, normalization, and model architecture as during training.
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Import our modules
from dataloader_hmtl import FloodDroughtDataLoader
from models.LSTM_HMTL import HierarchicalLSTMModel


def load_config(config_path):
    """
    Load configuration from YAML file (same function as train.py).
    
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
    
    # Validate required keys for flattened config
    required_keys = ['csv_file', 'hidden_size', 'batch_size', 'learning_rate', 'epochs', 'save_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config file")
    
    return config


def calculate_metrics(predictions, targets, scaler=None):
    """
    Calculate various evaluation metrics (same function as train.py).
    
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


def load_model_and_config(model_dir: str, model_trained: str) -> Tuple[HierarchicalLSTMModel, Dict, Dict]:
    """
    Load trained HierarchicalLSTMModel and its configuration from experiment directory.
    
    Args:
        model_dir: Path to the model directory (e.g., experiments/streamflow_hmtl)
        model_trained: Name of the trained model file
        
    Returns:
        Tuple of (model, config, model_config)
    """
    model_dir_path = Path(model_dir)
    
    # Load configuration (same as saved during training)
    config_path = model_dir_path / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the full config and extract the experiment
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Get the first experiment config (assuming single experiment in file)
    experiment_name = list(full_config.keys())[0]
    config = full_config[experiment_name]
    
    print(f"Loaded experiment: {experiment_name}")
    
    # Initialize model with hierarchical configuration
    model = HierarchicalLSTMModel(
        input_size=len(config['feature_cols']),
        intermediate_targets=config['intermediate_targets'],
        final_targets=config['target_cols'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        batch_first=True
    )
    
    # Load trained weights
    model_path = model_dir_path / model_trained
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model_config = checkpoint.get('config', {})
    else:
        model.load_state_dict(checkpoint)
        model_config = {}
    
    model.to(device)
    model.eval()
    
    print(f"HierarchicalLSTMModel loaded from {model_dir}")
    print(f"Device: {device}")
    print(f"Intermediate targets: {config['intermediate_targets']}")
    print(f"Final targets: {config['target_cols']}")
    
    return model, config, model_config


def make_predictions(model: HierarchicalLSTMModel, data_loader: DataLoader, device: torch.device, 
                    dataset_name: str = "") -> Tuple[Dict, Dict, Dict, Dict, List]:
    """
    Make predictions on a dataset using HierarchicalLSTMModel.
    
    Args:
        model: Trained HierarchicalLSTMModel
        data_loader: DataLoader for the dataset
        device: torch device
        dataset_name: Name of dataset for progress bar
        
    Returns:
        Tuple of (intermediate_predictions, final_predictions, intermediate_targets, final_targets, dates)
    """
    model.eval()
    all_intermediate_predictions = {}
    all_final_predictions = {}
    all_intermediate_targets = {}
    all_final_targets = {}
    all_dates = []
    
    # Get target names from first batch to initialize storage
    intermediate_target_names = []
    final_target_names = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Predicting {dataset_name}")):
            if len(batch_data) == 4:
                # Hierarchical format: (features, final_targets, intermediate_targets, dates)
                batch_x, batch_final_y, batch_intermediate_y, batch_dates = batch_data
            else:
                # Standard format: (features, all_targets, dates)
                batch_x, batch_all_y, batch_dates = batch_data
                # Split targets later based on target order
                batch_final_y = batch_all_y
                batch_intermediate_y = batch_all_y
            
            batch_x = batch_x.to(device)
            
            # Make predictions using hierarchical model
            model_outputs = model(batch_x)
            intermediate_outputs = model_outputs['intermediate']
            final_outputs = model_outputs['final']
            
            # Initialize target names on first batch
            if batch_idx == 0:
                intermediate_target_names = list(intermediate_outputs.keys())
                final_target_names = list(final_outputs.keys())
            
            # Store intermediate predictions and targets
            for target_name, predictions in intermediate_outputs.items():
                if target_name not in all_intermediate_predictions:
                    all_intermediate_predictions[target_name] = []
                all_intermediate_predictions[target_name].append(predictions.cpu().numpy())
            
            # Store final predictions and targets
            for target_name, predictions in final_outputs.items():
                if target_name not in all_final_predictions:
                    all_final_predictions[target_name] = []
                all_final_predictions[target_name].append(predictions.cpu().numpy())
            
            # Store targets (will be processed later for proper separation)
            if len(batch_data) == 4:
                # Store intermediate targets
                batch_intermediate_y = batch_intermediate_y.to(device)
                if not all_intermediate_targets:
                    # Initialize intermediate targets storage
                    for i, target_name in enumerate(intermediate_target_names):
                        all_intermediate_targets[target_name] = []
                
                for i, target_name in enumerate(intermediate_target_names):
                    if i < batch_intermediate_y.shape[-1]:
                        all_intermediate_targets[target_name].append(
                            batch_intermediate_y[:, :, i:i+1].cpu().numpy()
                        )
                
                # Store final targets
                batch_final_y = batch_final_y.to(device)
                if not all_final_targets:
                    # Initialize final targets storage
                    for i, target_name in enumerate(final_target_names):
                        all_final_targets[target_name] = []
                
                for i, target_name in enumerate(final_target_names):
                    if i < batch_final_y.shape[-1]:
                        all_final_targets[target_name].append(
                            batch_final_y[:, :, i:i+1].cpu().numpy()
                        )
            
            all_dates.append(batch_dates)
    
    # Concatenate all batches for each target
    for target_name in all_intermediate_predictions:
        all_intermediate_predictions[target_name] = np.concatenate(
            all_intermediate_predictions[target_name], axis=0
        )
    
    for target_name in all_final_predictions:
        all_final_predictions[target_name] = np.concatenate(
            all_final_predictions[target_name], axis=0
        )
    
    for target_name in all_intermediate_targets:
        all_intermediate_targets[target_name] = np.concatenate(
            all_intermediate_targets[target_name], axis=0
        )
    
    for target_name in all_final_targets:
        all_final_targets[target_name] = np.concatenate(
            all_final_targets[target_name], axis=0
        )
    
    print(f"{dataset_name} predictions completed:")
    print(f"  Intermediate predictions: {list(all_intermediate_predictions.keys())}")
    print(f"  Final predictions: {list(all_final_predictions.keys())}")
    for target_name, preds in all_intermediate_predictions.items():
        print(f"    {target_name} shape: {preds.shape}")
    for target_name, preds in all_final_predictions.items():
        print(f"    {target_name} shape: {preds.shape}")
    
    return (all_intermediate_predictions, all_final_predictions, 
            all_intermediate_targets, all_final_targets, all_dates)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  dataset_name: str = "") -> Dict:
    """
    Calculate comprehensive evaluation metrics for hydrological modeling.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    bias = np.mean(y_pred_flat - y_true_flat)
    std_error = np.std(y_pred_flat - y_true_flat)
    
    # Nash-Sutcliffe Efficiency (hydrological standard)
    nse = 1 - (np.sum((y_true_flat - y_pred_flat) ** 2) / 
               np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    
    # Kling-Gupta Efficiency (hydrological standard)
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else 0.0
    bias_ratio = np.mean(y_pred_flat) / (np.mean(y_true_flat) + 1e-8)
    variability_ratio = np.std(y_pred_flat) / (np.std(y_true_flat) + 1e-8)
    kge = 1 - np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (variability_ratio - 1)**2)
    
    metrics = {
        'dataset': dataset_name,
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'bias': float(bias),
        'std_error': float(std_error),
        'nse': float(nse),
        'kge': float(kge),
        'correlation': float(correlation),
        'n_samples': len(y_true_flat)
    }
    
    print(f"\n{dataset_name.upper()} SET METRICS:")
    print("="*50)
    print(f"MSE:         {mse:.2f}")
    print(f"RMSE:        {rmse:.2f}")
    print(f"MAE:         {mae:.2f}")
    print(f"R²:          {r2:.4f}")
    print(f"MAPE:        {mape:.2f}%")
    print(f"Bias:        {bias:.2f}")
    print(f"Std Error:   {std_error:.2f}")
    print(f"NSE:         {nse:.4f}")
    print(f"KGE:         {kge:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Samples:     {len(y_true_flat):,}")
    
    return metrics


def plot_windowed_time_series(y_true: np.ndarray, y_pred: np.ndarray, 
                             dataset_name: str, window_indices: Optional[List[int]] = None,
                             max_windows: int = 5, save_dir: Optional[str] = None) -> None:
    """
    Plot specific windows from windowed time series data.
    
    Args:
        y_true: Ground truth windowed data (n_windows, window_size, n_features)
        y_pred: Predicted windowed data (n_windows, window_size, n_features)
        dataset_name: Name of the dataset
        window_indices: Specific window indices to plot. If None, will select evenly spaced windows
        max_windows: Maximum number of windows to plot if window_indices not specified
        save_dir: Directory to save plots (optional)
    """
    if y_true.ndim != 3 or y_pred.ndim != 3:
        print(f"Warning: Expected 3D windowed data, got shapes {y_true.shape}, {y_pred.shape}")
        return
    
    n_windows, window_size, n_features = y_true.shape
    
    # Select windows to plot
    if window_indices is None:
        # Select evenly spaced windows
        step = max(1, n_windows // max_windows)
        window_indices = list(range(0, n_windows, step))[:max_windows]
    else:
        # Validate provided indices
        window_indices = [idx for idx in window_indices if 0 <= idx < n_windows]
        if not window_indices:
            print(f"Warning: No valid window indices provided")
            return
    
    # Create figure
    n_windows_to_plot = len(window_indices)
    fig, axes = plt.subplots(n_windows_to_plot, 1, figsize=(15, 4*n_windows_to_plot))
    if n_windows_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(f'{dataset_name.upper()} Set: Windowed Time Series (Selected Windows)', 
                fontsize=16, fontweight='bold')
    
    for i, window_idx in enumerate(window_indices):
        # For each feature, plot the time series
        for feature_idx in range(n_features):
            time_steps = np.arange(window_size)
            
            axes[i].plot(time_steps, y_true[window_idx, :, feature_idx], 
                        label=f'Ground Truth (Feature {feature_idx})', 
                        alpha=0.8, linewidth=2, linestyle='-')
            axes[i].plot(time_steps, y_pred[window_idx, :, feature_idx], 
                        label=f'Predictions (Feature {feature_idx})', 
                        alpha=0.8, linewidth=2, linestyle='--')
        
        axes[i].set_title(f'Window {window_idx} (of {n_windows})')
        axes[i].set_xlabel('Time Step in Window')
        axes[i].set_ylabel('Streamflow (cfs)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_windowed_timeseries.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Windowed time series plot saved: {save_dir}/{dataset_name}_windowed_timeseries.png")
    
    plt.show()

def plot_predictions_vs_truth(y_true: np.ndarray, y_pred: np.ndarray, 
                             dataset_name: str, save_dir: Optional[str] = None) -> None:
    """
    Create visualization plots combining scatter plot and error distribution.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        dataset_name: Name of the dataset
        save_dir: Directory to save plots (optional)
    """
    # Flatten for plotting
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    residuals = y_pred_flat - y_true_flat
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{dataset_name.upper()} Set: Predictions Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    axes[0].scatter(y_true_flat, y_pred_flat, alpha=0.6, s=1)
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Ground Truth (cfs)')
    axes[0].set_ylabel('Predictions (cfs)')
    axes[0].set_title('Scatter Plot: Predictions vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    axes[1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1].axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2, 
                   label=f'Mean Error: {np.mean(residuals):.2f}')
    axes[1].set_xlabel('Residuals (Prediction - Ground Truth, cfs)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_predictions_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_dir}/{dataset_name}_predictions_analysis.png")
    
    plt.show()



def run_inference(
    model_dir: str,
    model_trained: str,
    dataset_names: Optional[List[str]] = None,
    stride_override: Optional[int] = None,
) -> Dict:
    """
    Pure inference function - load hierarchical model and generate predictions without analysis.
    
    Args:
        model_dir: Path to the model directory
        model_trained: Name of the trained model file
        dataset_names: List of datasets to process ['train', 'val', 'test']
        
    Returns:
        Dictionary containing raw inference results for each dataset
    """
    # Load model and configuration
    model, config, model_config = load_model_and_config(model_dir, model_trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine which stride to use
    config_stride = config.get('stride')
    effective_stride = stride_override if stride_override is not None else config_stride
    if effective_stride is None:
        raise ValueError("Stride must be defined in config or provided via --stride")
    if stride_override is not None:
        print(f"Overriding stride: using {effective_stride} instead of config value {config_stride}")
    config['stride'] = effective_stride

    # Initialize hierarchical data loader with same configuration as training
    data_loader = FloodDroughtDataLoader(
        csv_file=config['csv_file'],
        window_size=config['window_size'],
        stride=effective_stride,
        target_col=config['target_cols'],
        feature_cols=config['feature_cols'],
        intermediate_targets=config.get('intermediate_targets', []),
        train_years=tuple(config['train_years']),
        val_years=tuple(config['val_years']),
        test_years=tuple(config['test_years']),
        batch_size=32,  # Use smaller batch size for inference
        scale_features=config.get('scale_features', True),
        scale_targets=config.get('scale_targets', True),
        scale_intermediate_targets=config.get('scale_intermediate_targets', True),
        many_to_many=True,
        random_seed=42
    )
    
    # Create data loaders
    loaders = data_loader.create_data_loaders(shuffle_train=False)
    
    # Determine which datasets to process
    if dataset_names is None:
        dataset_names = ['train', 'val', 'test']
    
    inference_results = {}
    
    # Process each dataset
    for dataset_name in dataset_names:
        loader_key = f'{dataset_name}_loader'
        if loader_key not in loaders:
            print(f"Warning: {loader_key} not found in loaders")
            print(f"Available loader keys: {list(loaders.keys())}")
            continue
            
        print(f"\n{'='*60}")
        print(f"RUNNING INFERENCE ON {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        print(f"Using stride: {effective_stride}{' (override)' if stride_override is not None else ''}")
        
        # Step 1: Make hierarchical predictions (normalized)
        (intermediate_preds, final_preds, 
         intermediate_targets, final_targets, dates) = make_predictions(
            model, loaders[loader_key], device, dataset_name
        )

        # Step 2: Denormalize predictions and targets using separate scalers
        (denorm_intermediate_preds, denorm_final_preds,
         denorm_intermediate_targets, denorm_final_targets) = denormalize_hierarchical_data(
            intermediate_preds, final_preds, intermediate_targets, final_targets,
            data_loader.intermediate_target_scaler, data_loader.target_scaler
        )

        # Step 3: Concatenate intermediate and final predictions/targets in order
        # Order: intermediate targets first, then final targets
        all_target_names = config.get('intermediate_targets', []) + config['target_cols']
        
        # Concatenate predictions
        concat_predictions_list = []
        concat_targets_list = []
        
        # Add intermediate predictions and targets
        for target_name in config.get('intermediate_targets', []):
            if target_name in denorm_intermediate_preds:
                concat_predictions_list.append(denorm_intermediate_preds[target_name])
                concat_targets_list.append(denorm_intermediate_targets[target_name])
        
        # Add final predictions and targets
        for target_name in config['target_cols']:
            if target_name in denorm_final_preds:
                concat_predictions_list.append(denorm_final_preds[target_name])
                concat_targets_list.append(denorm_final_targets[target_name])
        
        # Concatenate all predictions and targets
        if concat_predictions_list:
            concatenated_predictions = np.concatenate(concat_predictions_list, axis=-1)
            concatenated_targets = np.concatenate(concat_targets_list, axis=-1)
        else:
            print(f"Warning: No predictions to concatenate for {dataset_name}")
            continue

        # Step 4: Get the correct date windows for this dataset
        date_key = f'date_{dataset_name}'
        date_windows = loaders[date_key]

        # Step 5: Reconstruct time series from windowed data
        pred_ts, target_ts = reconstruct_time_series_from_windows(
            concatenated_predictions, concatenated_targets, date_windows, 
            {'target_cols': all_target_names}, data_loader
        )
        
        # Store results
        inference_results[dataset_name] = {
            'predictions_windowed': concatenated_predictions,    # Shape: (n_windows, window_size, n_all_features)
            'targets_windowed': concatenated_targets,           # Shape: (n_windows, window_size, n_all_features)
            'predictions_timeseries': pred_ts,                  # DataFrame with datetime index
            'targets_timeseries': target_ts,                    # DataFrame with datetime index
            'intermediate_predictions': denorm_intermediate_preds,  # Dict by target name
            'final_predictions': denorm_final_preds,            # Dict by target name
            'intermediate_targets': denorm_intermediate_targets, # Dict by target name
            'final_targets': denorm_final_targets,              # Dict by target name
            'target_names_order': all_target_names,             # List of target names in concatenation order
            'date_windows': date_windows,                       # Original date windows
            'config': config,                                   # Configuration used
            'model_config': model_config                        # Model architecture info
        }
        
        print(f"Inference completed for {dataset_name}")
        print(f"  - Concatenated predictions shape: {concatenated_predictions.shape}")
        print(f"  - Time series length: {len(pred_ts)}")
        print(f"  - Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
        print(f"  - Target order: {all_target_names}")
    
    return inference_results


def denormalize_hierarchical_data(intermediate_predictions: Dict, final_predictions: Dict,
                                 intermediate_targets: Dict, final_targets: Dict,
                                 intermediate_scaler, final_scaler) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Denormalize hierarchical predictions and targets back to original scale.
    
    Args:
        intermediate_predictions: Dict of intermediate predictions by target name
        final_predictions: Dict of final predictions by target name
        intermediate_targets: Dict of intermediate targets by target name  
        final_targets: Dict of final targets by target name
        intermediate_scaler: Scaler used for intermediate targets
        final_scaler: Scaler used for final targets
        
    Returns:
        Tuple of (denorm_intermediate_preds, denorm_final_preds, denorm_intermediate_targets, denorm_final_targets)
    """
    denorm_intermediate_preds = {}
    denorm_final_preds = {}
    denorm_intermediate_targets = {}
    denorm_final_targets = {}
    
    # Denormalize intermediate predictions and targets
    if intermediate_scaler is not None and intermediate_predictions:
        print("Denormalizing intermediate predictions and targets...")
        
        # Concatenate all intermediate predictions for denormalization
        intermediate_pred_list = []
        intermediate_target_list = []
        intermediate_names = list(intermediate_predictions.keys())
        
        for target_name in intermediate_names:
            if target_name in intermediate_predictions:
                pred_data = intermediate_predictions[target_name]
                target_data = intermediate_targets.get(target_name, pred_data)
                
                print(f"  {target_name} - Pred range: [{pred_data.min():.6f}, {pred_data.max():.6f}]")
                
                # Handle 3D data by reshaping
                if pred_data.ndim == 3:
                    orig_shape = pred_data.shape
                    pred_reshaped = pred_data.reshape(-1, 1)
                    target_reshaped = target_data.reshape(-1, 1)
                    
                    intermediate_pred_list.append(pred_reshaped)
                    intermediate_target_list.append(target_reshaped)
                else:
                    intermediate_pred_list.append(pred_data)
                    intermediate_target_list.append(target_data)
        
        if intermediate_pred_list:
            # Concatenate all intermediate data
            all_intermediate_preds = np.concatenate(intermediate_pred_list, axis=1)
            all_intermediate_targets = np.concatenate(intermediate_target_list, axis=1)
            
            # Denormalize
            denorm_all_intermediate_preds = intermediate_scaler.inverse_transform(all_intermediate_preds)
            denorm_all_intermediate_targets = intermediate_scaler.inverse_transform(all_intermediate_targets)
            
            # Split back to individual targets
            for i, target_name in enumerate(intermediate_names):
                if target_name in intermediate_predictions:
                    orig_shape = intermediate_predictions[target_name].shape
                    denorm_intermediate_preds[target_name] = denorm_all_intermediate_preds[:, i:i+1].reshape(orig_shape)
                    denorm_intermediate_targets[target_name] = denorm_all_intermediate_targets[:, i:i+1].reshape(orig_shape)
                    
                    print(f"  {target_name} - Denorm range: [{denorm_intermediate_preds[target_name].min():.6f}, {denorm_intermediate_preds[target_name].max():.6f}]")
    else:
        print("Warning: No intermediate scaler found, using raw intermediate values")
        denorm_intermediate_preds = intermediate_predictions.copy()
        denorm_intermediate_targets = intermediate_targets.copy()
    
    # Denormalize final predictions and targets
    if final_scaler is not None and final_predictions:
        print("Denormalizing final predictions and targets...")
        
        # Concatenate all final predictions for denormalization
        final_pred_list = []
        final_target_list = []
        final_names = list(final_predictions.keys())
        
        for target_name in final_names:
            if target_name in final_predictions:
                pred_data = final_predictions[target_name]
                target_data = final_targets.get(target_name, pred_data)
                
                print(f"  {target_name} - Pred range: [{pred_data.min():.6f}, {pred_data.max():.6f}]")
                
                # Handle 3D data by reshaping
                if pred_data.ndim == 3:
                    orig_shape = pred_data.shape
                    pred_reshaped = pred_data.reshape(-1, 1)
                    target_reshaped = target_data.reshape(-1, 1)
                    
                    final_pred_list.append(pred_reshaped)
                    final_target_list.append(target_reshaped)
                else:
                    final_pred_list.append(pred_data)
                    final_target_list.append(target_data)
        
        if final_pred_list:
            # Concatenate all final data
            all_final_preds = np.concatenate(final_pred_list, axis=1)
            all_final_targets = np.concatenate(final_target_list, axis=1)
            
            # Denormalize
            denorm_all_final_preds = final_scaler.inverse_transform(all_final_preds)
            denorm_all_final_targets = final_scaler.inverse_transform(all_final_targets)
            
            # Split back to individual targets
            for i, target_name in enumerate(final_names):
                if target_name in final_predictions:
                    orig_shape = final_predictions[target_name].shape
                    denorm_final_preds[target_name] = denorm_all_final_preds[:, i:i+1].reshape(orig_shape)
                    denorm_final_targets[target_name] = denorm_all_final_targets[:, i:i+1].reshape(orig_shape)
                    
                    print(f"  {target_name} - Denorm range: [{denorm_final_preds[target_name].min():.6f}, {denorm_final_preds[target_name].max():.6f}]")
    else:
        print("Warning: No final scaler found, using raw final values")
        denorm_final_preds = final_predictions.copy()
        denorm_final_targets = final_targets.copy()
        
    return denorm_intermediate_preds, denorm_final_preds, denorm_intermediate_targets, denorm_final_targets


def reconstruct_time_series_from_windows(predictions: np.ndarray, targets: np.ndarray, 
                                        date_windows: List, config: Dict, 
                                        data_loader: FloodDroughtDataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct full time series from windowed predictions using dataloader's method.
    
    Args:
        predictions: Windowed predictions (batch, window, features)
        targets: Windowed targets (batch, window, features)
        date_windows: List of date windows for each window
        config: Configuration dictionary with target_cols
        data_loader: FloodDroughtDataLoader instance with reconstruction method
        
    Returns:
        Tuple of (reconstructed_predictions, reconstructed_targets)
    """
    print("Reconstructing time series from windowed predictions...")
    
    target_names = config['target_cols']
    
    print(f"Total windows: {len(predictions)}")
    print(f"Total date windows: {len(date_windows)}")
    
    # Use dataloader's reconstruct_time_series method
    pred_ts, pred_counts = data_loader.reconstruct_time_series(
        predictions, date_windows, target_names, aggregation_method='mean'
    )
    
    target_ts, target_counts = data_loader.reconstruct_time_series(
        targets, date_windows, target_names, aggregation_method='mean'
    )
    
    # Ensure we return DataFrames
    if not isinstance(pred_ts, pd.DataFrame):
        print("Warning: Prediction reconstruction did not return DataFrame")
        pred_ts = pd.DataFrame()
    if not isinstance(target_ts, pd.DataFrame):
        print("Warning: Target reconstruction did not return DataFrame")
        target_ts = pd.DataFrame()
    
    print(f"Reconstructed time series length: {len(pred_ts)}")
    if len(pred_ts) > 0:
        print(f"Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
    
    return pred_ts, target_ts


# =============================================================================
# ANALYSIS FUNCTIONS (METRICS AND VISUALIZATION)
# =============================================================================

def analyze_inference_results(inference_results: Dict, save_dir: Optional[str] = None, 
                             analyze_features: Optional[List[str]] = None) -> Dict:
    """
    Analyze hierarchical inference results by computing metrics and creating visualizations.
    
    Args:
        inference_results: Results from run_inference function
        save_dir: Directory to save plots and metrics (optional)
        analyze_features: List of specific features to analyze (optional, analyzes all if None)
        
    Returns:
        Dictionary of all computed metrics
    """
    all_metrics = {}
    
    # Create save directory for analysis results
    save_dir_path = None
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)
        print(f"Analysis results will be saved to: {save_dir_path}")
    
    # Analyze each dataset
    for dataset_name, results in inference_results.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Extract data
        concatenated_preds = results['predictions_windowed']
        concatenated_targets = results['targets_windowed']
        pred_ts = results['predictions_timeseries']
        target_ts = results['targets_timeseries']
        target_names_order = results['target_names_order']
        
        # Determine which features to analyze
        features_to_analyze = analyze_features if analyze_features is not None else target_names_order
        features_to_analyze = [f for f in features_to_analyze if f in target_names_order]
        
        if not features_to_analyze:
            print(f"Warning: No valid features to analyze for {dataset_name}")
            continue
        
        print(f"Analyzing features: {features_to_analyze}")
        
        all_metrics[dataset_name] = {}
        
        # Analyze each feature individually
        for feature_idx, feature_name in enumerate(target_names_order):
            if feature_name not in features_to_analyze:
                continue
                
            print(f"\n{'-'*40}")
            print(f"ANALYZING FEATURE: {feature_name}")
            print(f"{'-'*40}")
            
            # Extract feature-specific data from concatenated arrays
            feature_pred_windowed = concatenated_preds[:, :, feature_idx:feature_idx+1]
            feature_target_windowed = concatenated_targets[:, :, feature_idx:feature_idx+1]
            
            # Extract feature-specific time series
            feature_pred_ts = pred_ts[[feature_name]] if feature_name in pred_ts.columns else None
            feature_target_ts = target_ts[[feature_name]] if feature_name in target_ts.columns else None
            
            if feature_pred_ts is None or feature_target_ts is None:
                print(f"Warning: Feature {feature_name} not found in time series data")
                continue
            
            # Calculate comprehensive metrics on windowed data
            windowed_metrics = calculate_comprehensive_metrics(
                feature_target_windowed, feature_pred_windowed, f"{dataset_name}_{feature_name}_windowed"
            )
            
            # Calculate metrics on time series data
            ts_metrics = calculate_timeseries_metrics(
                feature_pred_ts, feature_target_ts, f"{dataset_name}_{feature_name}_timeseries"
            )
            
            # Store metrics for this feature
            all_metrics[dataset_name][feature_name] = {
                'windowed': windowed_metrics,
                'timeseries': ts_metrics
            }
            
            # Create feature-specific save directory
            feature_save_dir = None
            if save_dir_path:
                feature_save_dir = save_dir_path / feature_name
                feature_save_dir.mkdir(exist_ok=True)
            
            # Create visualizations for this feature
            print(f"\nCreating visualizations for {feature_name}...")
            
            # Plot analysis (scatter + error distribution)
            plot_predictions_vs_truth(
                feature_target_windowed, feature_pred_windowed, 
                f"{dataset_name}_{feature_name}", str(feature_save_dir) if feature_save_dir else None
            )
            
            # Plot windowed time series (selected windows)
            plot_windowed_time_series(
                feature_target_windowed, feature_pred_windowed, 
                f"{dataset_name}_{feature_name}_windowed", 
                window_indices=None, max_windows=5, 
                save_dir=str(feature_save_dir) if feature_save_dir else None
            )
            
            # Plot reconstructed time series (full range by default)
            plot_time_series_reconstruction(
                feature_pred_ts, feature_target_ts, 
                f"{dataset_name}_{feature_name}_timeseries", 
                start_date=None, end_date=None, 
                save_dir=str(feature_save_dir) if feature_save_dir else None
            )
            
            # Save feature-specific time series data
            if feature_save_dir:
                combined_feature_ts = pd.merge(
                    feature_pred_ts, feature_target_ts, 
                    left_index=True, right_index=True, how='inner', 
                    suffixes=('_prediction', '_observation')
                )
                
                # Fix column names to ensure clear distinction between predictions and observations
                # Handle cases where pandas defaults to _x, _y suffixes
                column_mapping = {}
                for col in combined_feature_ts.columns:
                    if col.endswith('_prediction'):
                        # Already has correct suffix
                        base_name = col.replace('_prediction', '')
                        column_mapping[col] = f'{base_name}_prediction'
                    elif col.endswith('_observation'):
                        # Already has correct suffix
                        base_name = col.replace('_observation', '')
                        column_mapping[col] = f'{base_name}_observation'
                    elif col.endswith('_x'):
                        # pandas default suffix for left dataframe (predictions)
                        base_name = col.replace('_x', '')
                        column_mapping[col] = f'{base_name}_prediction'
                    elif col.endswith('_y'):
                        # pandas default suffix for right dataframe (observations)
                        base_name = col.replace('_y', '')
                        column_mapping[col] = f'{base_name}_observation'
                
                if column_mapping:
                    combined_feature_ts.rename(columns=column_mapping, inplace=True)
                
                feature_csv_path = feature_save_dir / f'{dataset_name}_{feature_name}_reconstructed_timeseries.csv'
                combined_feature_ts.to_csv(feature_csv_path)
                print(f"Feature time series saved: {feature_csv_path}")
                
                # Save feature-specific metrics
                feature_metrics_path = feature_save_dir / f'{dataset_name}_{feature_name}_metrics.json'
                with open(feature_metrics_path, 'w') as f:
                    json.dump(all_metrics[dataset_name][feature_name], f, indent=2)
                print(f"Feature metrics saved: {feature_metrics_path}")
    
    # Save all metrics
    if save_dir_path:
        all_metrics_path = save_dir_path / 'all_metrics.json'
        with open(all_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"All metrics saved: {all_metrics_path}")
    
    # Print summary
    print_analysis_summary(all_metrics, str(save_dir_path) if save_dir_path else None)
    
    return all_metrics


def plot_time_series_reconstruction(pred_ts: pd.DataFrame, target_ts: pd.DataFrame,
                                   dataset_name: str, start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None, save_dir: Optional[str] = None) -> None:
    """
    Plot reconstructed time series comparing predictions vs ground truth with datetime x-axis.
    
    Args:
        pred_ts: Predictions time series DataFrame with datetime index
        target_ts: Targets time series DataFrame with datetime index
        dataset_name: Name of the dataset
        start_date: Start date for plotting (format: 'YYYY-MM-DD'). If None, uses full range
        end_date: End date for plotting (format: 'YYYY-MM-DD'). If None, uses full range
        save_dir: Directory to save plots (optional)
    """
    # Merge on datetime index
    combined_df = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, 
                          how='inner', suffixes=('_pred', '_target'))
    
    if len(combined_df) == 0:
        print(f"Warning: No overlapping data for {dataset_name} time series plot")
        return
    
    # Filter by date range if specified
    if start_date or end_date:
        original_length = len(combined_df)
        if start_date:
            combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date)]
        if end_date:
            combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date)]
        
        if len(combined_df) == 0:
            print(f"Warning: No data in specified date range for {dataset_name}")
            return
        
        print(f"Filtered data from {original_length} to {len(combined_df)} points for date range")
    
    # Get column names - after merge, predictions have '_pred' suffix and targets have '_target' suffix
    pred_cols = [col for col in combined_df.columns if col.endswith('_pred')]
    target_cols = [col for col in combined_df.columns if col.endswith('_target')]
    
    # Ensure we have matching pairs
    base_cols = []
    for pred_col in pred_cols:
        base_name = pred_col.replace('_pred', '')
        target_col = base_name + '_target'
        if target_col in target_cols:
            base_cols.append(base_name)
    
    if len(base_cols) == 0:
        print(f"Warning: No matching columns for {dataset_name} time series plot")
        return
    
    # Create figure - each variable gets its own subplot
    n_vars = len(base_cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 5*n_vars))
    if n_vars == 1:
        axes = [axes]
    
    # Create title with date range info
    title = f'{dataset_name.upper()} Set: Reconstructed Time Series'
    if start_date or end_date:
        date_range = f" ({start_date or 'start'} to {end_date or 'end'})"
        title += date_range
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, base_col in enumerate(base_cols):
        pred_col = base_col + '_pred'
        target_col = base_col + '_target'
        
        # Plot time series with datetime x-axis
        axes[i].plot(combined_df.index, combined_df[target_col], 
                    label='Ground Truth', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(combined_df.index, combined_df[pred_col], 
                    label='Predictions', alpha=0.8, linewidth=1.5, color='red')
        
        axes[i].set_title(f'{base_col} - Time Series Comparison')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Streamflow (cfs)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Format datetime x-axis
        axes[i].tick_params(axis='x', rotation=45)
        
        # Auto-format dates based on time range
        if len(combined_df) > 365:  # More than a year of data
            axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif len(combined_df) > 30:  # More than a month
            axes[i].xaxis.set_major_locator(mdates.WeekdayLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        else:  # Less than a month
            axes[i].xaxis.set_major_locator(mdates.DayLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename_suffix = ""
        if start_date or end_date:
            filename_suffix = f"_{start_date or 'start'}_to_{end_date or 'end'}"
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_time_series_reconstruction{filename_suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Time series plot saved: {save_dir}/{dataset_name}_time_series_reconstruction{filename_suffix}.png")
    
    plt.show()


def calculate_timeseries_metrics(pred_ts: pd.DataFrame, target_ts: pd.DataFrame, 
                               dataset_name: str = "") -> Dict:
    """
    Calculate metrics specifically for time series data.
    
    Args:
        pred_ts: Predictions time series DataFrame
        target_ts: Targets time series DataFrame
        dataset_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    # Merge on datetime index
    combined_df = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, how='inner')
    
    if len(combined_df) == 0:
        print(f"Warning: No overlapping data for {dataset_name}")
        return {}
    
    # Get column names
    pred_cols = [col for col in combined_df.columns if col in pred_ts.columns]
    target_cols = [col for col in combined_df.columns if col in target_ts.columns]
    
    if len(pred_cols) == 0 or len(target_cols) == 0:
        print(f"Warning: No matching columns for {dataset_name}")
        return {}
    
    # Calculate metrics for each target variable
    metrics = {}
    for pred_col, target_col in zip(pred_cols, target_cols):
        y_pred = combined_df[pred_col].values
        y_true = combined_df[target_col].values
        
        # Remove any NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        
        if len(y_pred_clean) == 0:
            continue
            
        var_metrics = calculate_comprehensive_metrics(
            y_true_clean.reshape(-1, 1), 
            y_pred_clean.reshape(-1, 1), 
            f"{dataset_name}_{target_col}"
        )
        
        metrics[target_col] = var_metrics
    
    return metrics


def print_analysis_summary(all_metrics: Dict, save_dir: Optional[str] = None) -> None:
    """Print and save a summary of all hierarchical analysis results."""
    
    # Create the summary text
    summary_lines = []
    summary_lines.append(f"{'='*60}")
    summary_lines.append("HIERARCHICAL ANALYSIS SUMMARY")
    summary_lines.append(f"{'='*60}")
    
    for dataset_name, dataset_metrics in all_metrics.items():
        summary_lines.append(f"\n{dataset_name.upper()} Dataset:")
        
        # Print metrics for each feature
        for feature_name, feature_metrics in dataset_metrics.items():
            summary_lines.append(f"\n  {feature_name}:")
            
            # Print windowed metrics if available
            if 'windowed' in feature_metrics:
                windowed = feature_metrics['windowed']
                summary_lines.append(f"    Windowed Data:")
                summary_lines.append(f"      R²:    {windowed.get('r2', 'N/A'):.4f}")
                summary_lines.append(f"      RMSE:  {windowed.get('rmse', 'N/A'):.2f}")
                summary_lines.append(f"      NSE:   {windowed.get('nse', 'N/A'):.4f}")
                summary_lines.append(f"      KGE:   {windowed.get('kge', 'N/A'):.4f}")
            
            # Print time series metrics if available
            if 'timeseries' in feature_metrics and feature_metrics['timeseries']:
                ts_metrics = feature_metrics['timeseries']
                for var_name, var_metrics in ts_metrics.items():
                    summary_lines.append(f"    Time Series ({var_name}):")
                    summary_lines.append(f"      R²:    {var_metrics.get('r2', 'N/A'):.4f}")
                    summary_lines.append(f"      RMSE:  {var_metrics.get('rmse', 'N/A'):.2f}")
                    summary_lines.append(f"      NSE:   {var_metrics.get('nse', 'N/A'):.4f}")
                    summary_lines.append(f"      KGE:   {var_metrics.get('kge', 'N/A'):.4f}")
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Save to file if save_dir provided
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True)
        
        # Use fixed filename and append mode
        summary_file = save_dir_path / 'analysis_summary.txt'
        
        # Add timestamp header for this run
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(summary_file, 'a') as f:
            # Add separator and timestamp if file already exists
            if summary_file.exists() and summary_file.stat().st_size > 0:
                f.write('\n' + '='*60 + '\n')
                f.write(f'NEW ANALYSIS RUN - {timestamp}\n')
                f.write('='*60 + '\n')
            else:
                f.write(f'ANALYSIS RUN - {timestamp}\n')
            
            for line in summary_lines:
                f.write(line + '\n')
            f.write('\n')  # Add extra newline at the end
        
        print(f"\nAnalysis summary appended to: {summary_file}")

def main():
    """
    Main inference function - separates pure inference from analysis.
    """
    parser = argparse.ArgumentParser(description='Hierarchical LSTM Model Inference for Flood/Drought Prediction')
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory (e.g., experiments/streamflow_hmtl)')
    parser.add_argument('--model-trained', type=str, default='best_model.pth', 
                       help='Name of the trained model file (default: best_model.pth)')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], default=None,
                       help='Evaluate only specific dataset (default: all datasets)')
    parser.add_argument('--analysis', action='store_true',
                       help='Run comprehensive analysis (creates feature-specific folders and metrics)')
    parser.add_argument('--analyze-features', type=str, default=None,
                       help='Comma-separated list of features to analyze (e.g., "PET,ET,streamflow"). If not specified, analyzes all features.')
    parser.add_argument('--stride', type=int, default=None,
                        help='Override stride to use for inference (defaults to value from config)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    print("="*60)
    print("HIERARCHICAL LSTM MODEL INFERENCE AND ANALYSIS")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Dataset filter: {args.dataset or 'all'}")
    
    # Parse analyze_features if provided
    analyze_features = None
    if args.analyze_features:
        analyze_features = [f.strip() for f in args.analyze_features.split(',')]
        print(f"Features to analyze: {analyze_features}")
    
    # Determine which datasets to process
    dataset_names = ['train', 'val', 'test']
    if args.dataset:
        dataset_names = [args.dataset]
    
    # Set up save directory
    save_dir = Path(args.model_dir) / f'{Path(args.model_trained).stem}_results_{args.stride}'
    
    print(f"\n{'='*60}")
    print("STEP 1: RUNNING HIERARCHICAL INFERENCE")
    print(f"{'='*60}")
    
    inference_results = run_inference(
        args.model_dir,
        args.model_trained,
        dataset_names,
        stride_override=args.stride,
    )
    
    # Save inference results
    save_dir.mkdir(exist_ok=True)
    
    # Save inference results as pickle for later analysis
    import pickle
    results_filename = f'inference_results_{args.dataset or "all"}.pkl'
    with open(save_dir / results_filename, 'wb') as f:
        pickle.dump(inference_results, f)
    print(f"Inference results saved: {save_dir}/{results_filename}")
    
    # Step 2: Run analysis if requested
    if args.analysis:
        print(f"\n{'='*60}")
        print("STEP 2: RUNNING COMPREHENSIVE FEATURE-WISE ANALYSIS")
        print(f"{'='*60}")
        
        all_metrics = analyze_inference_results(
            inference_results, str(save_dir), analyze_features
        )
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        print(f"All results saved to: {save_dir}")
        if analyze_features:
            print(f"Feature-specific analysis completed for: {analyze_features}")
        else:
            print("Analysis completed for all available features")

if __name__ == "__main__":
    main()


