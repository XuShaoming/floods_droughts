#!/usr/bin/env python3
"""
LSTM Model Inference Script for Flood/Drought Prediction

This script loads a trained LSTM model and performs comprehensive evaluation using
the same YAML configuration system as train.py. It provides:

1. Model and configuration loading from experiment directories
2. Predictions on train/validation/test sets with proper data loading
3. Denormalization back to original scale using saved scalers
4. Comprehensive metrics calculation (MSE, RMSE, MAE, R², NSE, KGE, etc.)
5. Rich visualizations (scatter plots, time series, residuals, error distributions)
6. Time series reconstruction from windowed predictions
7. Results saving and logging

Usage:
    python inference.py --model-dir experiments/hourly_flood_events --model-trained best_model.pth --dataset test --analysis

The script automatically detects the experiment configuration and uses the same
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
from dataloader import FloodDroughtDataLoader
from models.LSTMModel import LSTMModel


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


def load_model_and_config(model_dir: str, model_trained: str) -> Tuple[LSTMModel, Dict, Dict]:
    """
    Load trained model and its configuration from experiment directory.
    
    Args:
        model_dir: Path to the model directory (e.g., experiments/hourly_flood_events)
        
    Returns:
        Tuple of (model, config, model_config)
    """
    model_dir = Path(model_dir)
    
    # Load configuration (same as saved during training)
    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Load model configuration
    model_config_path = model_dir / 'model_config.json'
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Initialize model with exact same architecture
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=model_config['output_size'],
        dropout=model_config['dropout']
    )
    
    # Load trained weights
    # model_path = model_dir / 'best_model.pth'
    model_path = model_dir / model_trained
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_dir}")
    print(f"Device: {device}")
    print(f"Model architecture: {model_config}")
    
    return model, config, model_config


def make_predictions(model: LSTMModel, data_loader: DataLoader, device: torch.device, 
                    dataset_name: str = "") -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Make predictions on a dataset.
    
    Args:
        model: Trained LSTM model
        data_loader: DataLoader for the dataset
        device: torch device
        dataset_name: Name of dataset for progress bar
        
    Returns:
        Tuple of (predictions, targets, dates)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_dates = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_dates in tqdm(data_loader, desc=f"Predicting {dataset_name}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Make predictions
            predictions = model(batch_x)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_dates.append(batch_dates)  # Keep as list of date windows
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"{dataset_name} predictions shape: {predictions.shape}")
    print(f"{dataset_name} targets shape: {targets.shape}")
    
    return predictions, targets, all_dates


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


def reconstruct_time_series(predictions: np.ndarray, targets: np.ndarray, 
                           dates: List, config: Dict, data_loader: FloodDroughtDataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct full time series from windowed predictions using dataloader's method.
    
    Args:
        predictions: Windowed predictions (batch, window, features)
        targets: Windowed targets (batch, window, features)
        dates: Date information for each window (list of date windows)
        config: Configuration dictionary
        data_loader: FloodDroughtDataLoader instance with reconstruction method
        
    Returns:
        Tuple of (reconstructed_predictions, reconstructed_targets)
    """
    print("Reconstructing time series from windowed predictions using dataloader method...")
    
    target_names = config['target_cols']
    
    # Flatten the dates list (each element is a batch of date windows)
    all_date_windows = []
    for batch_dates in dates:
        all_date_windows.extend(batch_dates)
    
    print(f"Total windows: {len(predictions)}")
    print(f"Total date windows: {len(all_date_windows)}")
    
    # Use dataloader's reconstruct_time_series method
    pred_ts, pred_counts = data_loader.reconstruct_time_series(
        predictions, all_date_windows, target_names, aggregation_method='mean'
    )
    
    target_ts, target_counts = data_loader.reconstruct_time_series(
        targets, all_date_windows, target_names, aggregation_method='mean'
    )
    
    print(f"Reconstructed time series length: {len(pred_ts)}")
    print(f"Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
    
    return pred_ts, target_ts


def run_inference(model_dir: str, model_trained: str, dataset_names: List[str] = None) -> Dict:
    """
    Pure inference function - load model and generate predictions without analysis.
    
    Args:
        model_dir: Path to the model directory
        dataset_names: List of datasets to process ['train', 'val', 'test']
        
    Returns:
        Dictionary containing raw inference results for each dataset
    """
    # Load model and configuration
    model, config, model_config = load_model_and_config(model_dir, model_trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader with same configuration as training
    data_loader = FloodDroughtDataLoader(
        csv_file=config['csv_file'],
        window_size=config['window_size'],
        stride=config['stride'],
        target_col=config['target_cols'],
        feature_cols=config['feature_cols'],
        train_years=tuple(config['train_years']),
        val_years=tuple(config['val_years']),
        test_years=tuple(config['test_years']),
        batch_size=32,  # Use smaller batch size for inference
        scale_features=config.get('scale_features', True),
        scale_targets=config.get('scale_targets', True),
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
         # Step 1: Make predictions (normalized)
        predictions, targets, _ = make_predictions(
            model, loaders[loader_key], device, dataset_name
        )

        # Step 2: Denormalize predictions and targets
        pred_denorm, target_denorm = denormalize_data(
            predictions, targets, data_loader.target_scaler
        )

        # Step 3: Get the correct date windows for this dataset
        date_key = f'date_{dataset_name}'
        date_windows = loaders[date_key]

        # Step 4: Reconstruct time series from windowed data
        pred_ts, target_ts = reconstruct_time_series_from_windows(
            pred_denorm, target_denorm, date_windows, config, data_loader
        )
        
        # Store results
        inference_results[dataset_name] = {
            'predictions_windowed': pred_denorm,  # Shape: (n_windows, window_size, n_features)
            'targets_windowed': target_denorm,    # Shape: (n_windows, window_size, n_features)
            'predictions_timeseries': pred_ts,    # DataFrame with datetime index
            'targets_timeseries': target_ts,      # DataFrame with datetime index
            'date_windows': date_windows,          # Original date windows
            'config': config,                     # Configuration used
            'model_config': model_config          # Model architecture info
        }
        
        print(f"Inference completed for {dataset_name}")
        print(f"  - Windowed predictions shape: {pred_denorm.shape}")
        print(f"  - Time series length: {len(pred_ts)}")
        print(f"  - Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
    
    return inference_results


def denormalize_data(predictions: np.ndarray, targets: np.ndarray, 
                    target_scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    Denormalize predictions and targets back to original scale.
    
    Args:
        predictions: Normalized predictions
        targets: Normalized targets  
        target_scaler: Scaler used for targets
        
    Returns:
        Tuple of (denormalized_predictions, denormalized_targets)
    """
    if target_scaler is not None:
        print("Denormalizing predictions and targets...")
        print(f"Before denorm - Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"Before denorm - Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
        
        # Handle 3D data (batch, window, features) by reshaping
        if predictions.ndim == 3:
            orig_shape = predictions.shape
            pred_reshaped = predictions.reshape(-1, predictions.shape[-1])
            target_reshaped = targets.reshape(-1, targets.shape[-1])
            
            pred_denorm = target_scaler.inverse_transform(pred_reshaped)
            target_denorm = target_scaler.inverse_transform(target_reshaped)
            
            pred_denorm = pred_denorm.reshape(orig_shape)
            target_denorm = target_denorm.reshape(orig_shape)
        else:
            pred_denorm = target_scaler.inverse_transform(predictions)
            target_denorm = target_scaler.inverse_transform(targets)
            
        print(f"After denorm - Predictions range: [{pred_denorm.min():.6f}, {pred_denorm.max():.6f}]")
        print(f"After denorm - Targets range: [{target_denorm.min():.6f}, {target_denorm.max():.6f}]")
    else:
        print("Warning: No target scaler found, using raw values")
        pred_denorm = predictions
        target_denorm = targets
        
    return pred_denorm, target_denorm


def reconstruct_time_series_from_windows(predictions: np.ndarray, targets: np.ndarray, 
                                        date_windows: List, config: Dict, 
                                        data_loader: FloodDroughtDataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct full time series from windowed predictions using dataloader's method.
    
    Args:
        predictions: Windowed predictions (batch, window, features)
        targets: Windowed targets (batch, window, features)
        date_windows: List of date windows for each window
        config: Configuration dictionary
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
    
    print(f"Reconstructed time series length: {len(pred_ts)}")
    print(f"Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
    
    return pred_ts, target_ts


# =============================================================================
# ANALYSIS FUNCTIONS (METRICS AND VISUALIZATION)
# =============================================================================

def analyze_inference_results(inference_results: Dict, save_dir: Optional[str] = None) -> Dict:
    """
    Analyze inference results by computing metrics and creating visualizations.
    
    Args:
        inference_results: Results from run_inference function
        save_dir: Directory to save plots and metrics (optional)
        
    Returns:
        Dictionary of all computed metrics
    """
    all_metrics = {}
    
    # Create save directory for analysis results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        print(f"Analysis results will be saved to: {save_dir}")
    
    # Analyze each dataset
    for dataset_name, results in inference_results.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Extract data
        pred_windowed = results['predictions_windowed']
        target_windowed = results['targets_windowed']
        pred_ts = results['predictions_timeseries']
        target_ts = results['targets_timeseries']
        
        # Calculate comprehensive metrics on windowed data
        windowed_metrics = calculate_comprehensive_metrics(
            target_windowed, pred_windowed, f"{dataset_name}_windowed"
        )
        
        # Calculate metrics on time series data
        ts_metrics = calculate_timeseries_metrics(
            pred_ts, target_ts, f"{dataset_name}_timeseries"
        )
        
        # Store metrics
        all_metrics[dataset_name] = {
            'windowed': windowed_metrics,
            'timeseries': ts_metrics
        }
        
        # Create visualizations
        print(f"\nCreating visualizations for {dataset_name} dataset...")
        
        # Plot analysis (scatter + error distribution)
        plot_predictions_vs_truth(
            target_windowed, pred_windowed, f"{dataset_name}_analysis", save_dir
        )
        
        # Plot windowed time series (selected windows)
        plot_windowed_time_series(
            target_windowed, pred_windowed, f"{dataset_name}_windowed", 
            window_indices=None, max_windows=5, save_dir=save_dir
        )
        
        # Plot reconstructed time series (full range by default)
        plot_time_series_reconstruction(
            pred_ts, target_ts, f"{dataset_name}_timeseries", 
            start_date=None, end_date=None, save_dir=save_dir
        )
        
        # Save time series data
        if save_dir:
            # Merge with explicit suffixes to distinguish prediction vs observation
            combined_ts = pd.merge(pred_ts, target_ts, left_index=True, right_index=True, how='inner', suffixes=('_prediction', '_observation'))
            
            # Reset index to make date a column named 'date'
            combined_ts.reset_index(inplace=True)
            
            # Rename the index column to 'date' if it's not already named that
            if combined_ts.columns[0] != 'date':
                old_col_name = str(combined_ts.columns[0])
                combined_ts.rename(columns={old_col_name: 'date'}, inplace=True)
            
            # Fix column names to ensure clear distinction between predictions and observations
            # Handle cases where pandas defaults to _x, _y suffixes
            column_mapping = {}
            for col in combined_ts.columns:
                if col != 'date':
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
                combined_ts.rename(columns=column_mapping, inplace=True)
            
            csv_path = Path(save_dir) / f'{dataset_name}_reconstructed_timeseries.csv'
            combined_ts.to_csv(csv_path, index=False)
            print(f"Time series saved: {save_dir}/{dataset_name}_reconstructed_timeseries.csv")
    
    # Save all metrics
    if save_dir:
        metrics_path = Path(save_dir) / 'all_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"All metrics saved: {save_dir}/all_metrics.json")
    
    # Print summary
    print_analysis_summary(all_metrics, str(save_dir) if save_dir else None)
    
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
    """Print and save a summary of all analysis results."""
    
    # Create the summary text
    summary_lines = []
    summary_lines.append(f"{'='*60}")
    summary_lines.append("ANALYSIS SUMMARY")
    summary_lines.append(f"{'='*60}")
    
    for dataset_name, dataset_metrics in all_metrics.items():
        summary_lines.append(f"\n{dataset_name.upper()} Dataset:")
        
        # Print windowed metrics if available
        if 'windowed' in dataset_metrics:
            windowed = dataset_metrics['windowed']
            summary_lines.append(f"  Windowed Data:")
            summary_lines.append(f"    R²:    {windowed.get('r2', 'N/A'):.4f}")
            summary_lines.append(f"    RMSE:  {windowed.get('rmse', 'N/A'):.2f}")
            summary_lines.append(f"    NSE:   {windowed.get('nse', 'N/A'):.4f}")
            summary_lines.append(f"    KGE:   {windowed.get('kge', 'N/A'):.4f}")
        
        # Print time series metrics if available
        if 'timeseries' in dataset_metrics:
            ts_metrics = dataset_metrics['timeseries']
            for var_name, var_metrics in ts_metrics.items():
                summary_lines.append(f"  Time Series ({var_name}):")
                summary_lines.append(f"    R²:    {var_metrics.get('r2', 'N/A'):.4f}")
                summary_lines.append(f"    RMSE:  {var_metrics.get('rmse', 'N/A'):.2f}")
                summary_lines.append(f"    NSE:   {var_metrics.get('nse', 'N/A'):.4f}")
                summary_lines.append(f"    KGE:   {var_metrics.get('kge', 'N/A'):.4f}")
    
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
    parser = argparse.ArgumentParser(description='LSTM Model Inference for Flood/Drought Prediction')
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory (e.g., experiments/hourly_flood_events)')
    parser.add_argument('--model-trained', type=str, default='best_model.pth', 
                       help='Name of the trained model file (default: best_model.pth)')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], default=None,
                       help='Evaluate only specific dataset (default: all datasets)')
    parser.add_argument('--analysis', action='store_true',
                       help='Run only analysis (requires existing inference results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    print("="*60)
    print("LSTM MODEL INFERENCE AND ANALYSIS")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Dataset filter: {args.dataset or 'all'}")
    
    # Determine which datasets to process
    dataset_names = ['train', 'val', 'test']
    if args.dataset:
        dataset_names = [args.dataset]
    
    # Set up save directory if needed
    save_dir = Path(args.model_dir) / f'{Path(args.model_trained).stem}_results'
    
    print(f"\n{'='*60}")
    print("STEP 1: RUNNING PURE INFERENCE")
    print(f"{'='*60}")
    
    inference_results = run_inference(args.model_dir, args.model_trained, dataset_names)
    
    # Save inference results if requested
    if save_dir:
        save_dir.mkdir(exist_ok=True)
        
        # Save inference results as pickle for later analysis
        import pickle
        with open(save_dir / f'inference_results_{args.dataset}.pkl', 'wb') as f:
            pickle.dump(inference_results, f)
        print(f"Inference results saved: {save_dir}/inference_results_{args.dataset}.pkl")
    
    # Step 2: Run analysis (unless inference-only mode)
    if args.analysis:
        print(f"\n{'='*60}")
        print("STEP 2: RUNNING COMPREHENSIVE ANALYSIS")
        print(f"{'='*60}")

        # Load existing inference results
        inference_results_path = Path(args.model_dir) / f'{Path(args.model_trained).stem}_results' / f'inference_results_{args.dataset}.pkl'
        if not inference_results_path.exists():
            raise FileNotFoundError(f"Inference results not found: {inference_results_path}")
        
        import pickle
        with open(inference_results_path, 'rb') as f:
            inference_results = pickle.load(f)
        print(f"Loaded existing inference results from: {inference_results_path}")

        
        all_metrics = analyze_inference_results(inference_results, str(save_dir))
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        if save_dir:
            print(f"All results saved to: {save_dir}")

if __name__ == "__main__":
    main()


