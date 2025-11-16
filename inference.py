#!/usr/bin/env python3
"""
LSTM Model Inference Script for Flood/Drought Prediction

This script loads a trained LSTM model and performs comprehensive evaluation using
the same YAML configuration system as train.py. It provides:

1. Model and configuration loading from experiment directories
2. Predictions on train/validation/test sets with proper data loading
3. Denormalization back to original scale using saved scalers
4. Time series reconstruction from windowed predictions
5. Results saving for downstream analysis (see analysis.py for metrics/plots)

Usage:
    python inference.py --model-dir experiments/hourly_flood_events --model-trained best_model.pth --dataset test --analysis

The script automatically detects the experiment configuration and uses the same
data loading, normalization, and model architecture as during training.
"""

import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
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


def run_inference(model_dir: str,
                  model_trained: str,
                  dataset_names: Optional[List[str]] = None,
                  reconstruction_methods: Optional[List[str]] = None) -> Dict:
    """
    Pure inference function - load model and generate predictions without analysis.
    
    Args:
        model_dir: Path to the model directory
        dataset_names: List of datasets to process ['train', 'val', 'test']
        reconstruction_methods: Ordered list of reconstruction strategies (e.g., ['average', 'latest'])
        
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
    if not reconstruction_methods:
        reconstruction_methods = ['average', 'latest']
    
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

        # Step 4: Reconstruct time series from windowed data using requested methods
        reconstruction_series = reconstruct_time_series_from_windows(
            pred_denorm,
            target_denorm,
            date_windows,
            config,
            data_loader,
            reconstruction_methods,
        )

        default_method = next(iter(reconstruction_series.keys()))
        default_recon = reconstruction_series[default_method]
        
        # Store results
        inference_results[dataset_name] = {
            'predictions_windowed': pred_denorm,
            'targets_windowed': target_denorm,
            'predictions_timeseries': default_recon['predictions'],
            'targets_timeseries': default_recon['targets'],
            'reconstructed_timeseries': reconstruction_series,
            'default_reconstruction_method': default_method,
            'date_windows': date_windows,
            'config': config,
            'model_config': model_config,
        }
        
        print(f"Inference completed for {dataset_name}")
        print(f"  - Windowed predictions shape: {pred_denorm.shape}")
        for method_name, series in reconstruction_series.items():
            pred_ts = series['predictions']
            print(f"  - {method_name} reconstruction length: {len(pred_ts)}")
            if len(pred_ts) > 0:
                print(f"    Date range: {pred_ts.index.min()} to {pred_ts.index.max()}")
    
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


def reconstruct_time_series_from_windows(
    predictions: np.ndarray,
    targets: np.ndarray,
    date_windows: List,
    config: Dict,
    data_loader: FloodDroughtDataLoader,
    reconstruction_methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Reconstruct full time series for multiple aggregation strategies.

    Args:
        predictions: Windowed predictions (batch, window, features)
        targets: Windowed targets (batch, window, features)
        date_windows: List of date windows for each window
        config: Configuration dictionary
        data_loader: FloodDroughtDataLoader instance with reconstruction helper
        reconstruction_methods: Methods such as ['average', 'latest']

    Returns:
        Dictionary mapping method name -> dict containing prediction/target DataFrames.
    """
    print("Reconstructing time series from windowed predictions...")

    target_names = config['target_cols']

    print(f"Total windows: {len(predictions)}")
    print(f"Total date windows: {len(date_windows)}")

    if not reconstruction_methods:
        reconstruction_methods = ['average']

    alias_map = {
        'average': ('average', 'mean'),
        'mean': ('average', 'mean'),
        'latest': ('latest', 'last'),
        'last': ('latest', 'last'),
        'first': ('first', 'first'),
        'median': ('median', 'median'),
    }

    reconstruction_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for method in reconstruction_methods:
        method_key = method.lower()
        if method_key not in alias_map:
            raise ValueError(f"Unsupported reconstruction method '{method}'.")

        label, aggregation = alias_map[method_key]
        if label in reconstruction_results:
            print(f"Skipping duplicate reconstruction method '{method}' (alias for '{label}').")
            continue

        print(f"  - Applying '{label}' reconstruction (aggregation='{aggregation}')")
        pred_ts, pred_counts = data_loader.reconstruct_time_series(
            predictions, date_windows, target_names, aggregation_method=aggregation
        )
        target_ts, target_counts = data_loader.reconstruct_time_series(
            targets, date_windows, target_names, aggregation_method=aggregation
        )

        reconstruction_results[label] = {
            'predictions': pred_ts,
            'targets': target_ts,
            'prediction_counts': pred_counts,
            'target_counts': target_counts,
        }

    return reconstruction_results



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
                       help='After inference, run analysis using the saved results')
    parser.add_argument('--reconstruction-methods', type=str, nargs='+', default=None,
                       help='Reconstruction methods to save (e.g., average latest)')
    
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
    dataset_label = args.dataset if args.dataset is not None else "None"
    
    # Set up save directory if needed
    save_dir = Path(args.model_dir) / f'{Path(args.model_trained).stem}_results'
    
    print(f"\n{'='*60}")
    print("STEP 1: RUNNING PURE INFERENCE")
    print(f"{'='*60}")
    
    inference_results = run_inference(
        args.model_dir,
        args.model_trained,
        dataset_names,
        reconstruction_methods=args.reconstruction_methods,
    )
    
    # Save inference results if requested
    if save_dir:
        save_dir.mkdir(exist_ok=True)
        
        # Save inference results as pickle for later analysis
        import pickle
        results_filename = f'inference_results_{dataset_label}.pkl'
        with open(save_dir / results_filename, 'wb') as f:
            pickle.dump(inference_results, f)
        print(f"Inference results saved: {save_dir}/{results_filename}")
    
    # Step 2: Run analysis (unless inference-only mode)
    if args.analysis:
        print(f"\n{'='*60}")
        print("STEP 2: RUNNING COMPREHENSIVE ANALYSIS")
        print(f"{'='*60}")

        # Load existing inference results
        from analysis import analyze_inference_results

        inference_results_path = Path(args.model_dir) / f'{Path(args.model_trained).stem}_results' / f'inference_results_{dataset_label}.pkl'
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
