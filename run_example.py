#!/usr/bin/env python3
"""
Example Usage Script for LSTM Flood/Drought Prediction

This script demonstrates how to use the train.py and inference.py scripts
for flood and drought prediction using the FloodDroughtDataLoader.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print("ERROR!")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        return False
    
    return True


def main():
    """Main function to demonstrate training and inference."""
    
    print("LSTM Flood/Drought Prediction - Example Usage")
    print("=" * 60)
    
    # Check if data file exists
    data_file = "processed/KettleRiverModels_hist_scaled_combined.csv"
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please ensure the data file exists before running this example.")
        return
    
    # Training configuration
    train_config = {
        'window_size': 14,
        'stride': 7,
        'batch_size': 16,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 5,  # Small number for quick demo
        'patience': 3,
        'target_cols': ['streamflow'],  # Single target for simplicity
        'train_years': '1980 2000',
        'val_years': '1975 1979',
        'test_years': '2001 2005',
        'save_dir': 'demo_models',
        'seed': 42
    }
    
    # Build training command
    train_cmd = f"""python train.py \\
        --window_size {train_config['window_size']} \\
        --stride {train_config['stride']} \\
        --batch_size {train_config['batch_size']} \\
        --hidden_size {train_config['hidden_size']} \\
        --num_layers {train_config['num_layers']} \\
        --dropout {train_config['dropout']} \\
        --learning_rate {train_config['learning_rate']} \\
        --epochs {train_config['epochs']} \\
        --patience {train_config['patience']} \\
        --target_cols {' '.join(train_config['target_cols'])} \\
        --train_years {train_config['train_years']} \\
        --val_years {train_config['val_years']} \\
        --test_years {train_config['test_years']} \\
        --save_dir {train_config['save_dir']} \\
        --seed {train_config['seed']}"""
    
    # Run training
    print(f"1. TRAINING PHASE")
    if not run_command(train_cmd, "Training LSTM model"):
        print("Training failed! Exiting.")
        return
    
    # Check if model was saved
    model_path = f"{train_config['save_dir']}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found after training!")
        return
    
    # Inference configuration
    inference_config = {
        'model_path': model_path,
        'window_size': train_config['window_size'],
        'stride': train_config['stride'],
        'batch_size': train_config['batch_size'],
        'target_cols': train_config['target_cols'],
        'test_years': train_config['test_years'],
        'save_dir': 'demo_inference_results',
        'seed': train_config['seed']
    }
    
    # Build inference command
    inference_cmd = f"""python inference.py \\
        --model_path {inference_config['model_path']} \\
        --window_size {inference_config['window_size']} \\
        --stride {inference_config['stride']} \\
        --batch_size {inference_config['batch_size']} \\
        --target_cols {' '.join(inference_config['target_cols'])} \\
        --test_years {inference_config['test_years']} \\
        --save_dir {inference_config['save_dir']} \\
        --save_windowed \\
        --seed {inference_config['seed']}"""
    
    # Run inference
    print(f"\\n2. INFERENCE PHASE")
    if not run_command(inference_cmd, "Running inference on test data"):
        print("Inference failed!")
        return
    
    print(f"\\n{'='*60}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Training results saved to: {train_config['save_dir']}")
    print(f"Inference results saved to: {inference_config['save_dir']}")
    print(f"\\nKey files created:")
    print(f"  - {train_config['save_dir']}/best_model.pth (trained model)")
    print(f"  - {train_config['save_dir']}/test_metrics.json (training test metrics)")
    print(f"  - {inference_config['save_dir']}/predictions_reconstructed.csv (reconstructed predictions)")
    print(f"  - {inference_config['save_dir']}/inference_metrics.json (inference metrics)")
    print(f"  - {inference_config['save_dir']}/prediction_plots.png (visualization)")


if __name__ == "__main__":
    main()
