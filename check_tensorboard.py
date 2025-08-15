#!/usr/bin/env python3
"""
TensorBoard Results Checker for Flood/Drought Prediction Experiments

This script provides comprehensive analysis and visualization of training results
from TensorBoard logs for hierarchical LSTM experiments.

Features:
- Load and parse TensorBoard event files
- Generate comprehensive training/validation loss plots
- Task-specific performance analysis
- Comparative analysis between experiments
- Export results to various formats (PNG, PDF, CSV)
- Summary statistics and best model identification

Usage:
    python check_tensorboard.py --experiment streamflow_hmtl
    python check_tensorboard.py --experiment streamflow_hstl --compare streamflow_hmtl
    python check_tensorboard.py --list-experiments

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not available. Install with: pip install pyyaml")
    yaml = None


class TensorBoardAnalyzer:
    """Analyze TensorBoard logs and generate comprehensive visualizations"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.experiments = {}
        self.data = {}
        
    def list_experiments(self):
        """List all available experiments with TensorBoard logs"""
        experiments = []
        if not self.base_dir.exists():
            print(f"Experiments directory '{self.base_dir}' not found!")
            return experiments
            
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                tb_dir = exp_dir / "tensorboard_logs"
                if tb_dir.exists() and any(tb_dir.glob("events.out.tfevents.*")):
                    experiments.append(exp_dir.name)
                    
        return sorted(experiments)
    
    def load_experiment_data(self, experiment_name):
        """Load TensorBoard data for a specific experiment"""
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is required. Install with: pip install tensorboard")
            
        exp_path = self.base_dir / experiment_name
        tb_path = exp_path / "tensorboard_logs"
        
        if not tb_path.exists():
            raise FileNotFoundError(f"TensorBoard logs not found for experiment: {experiment_name}")
        
        # Find the event file
        event_files = list(tb_path.glob("events.out.tfevents.*"))
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found in: {tb_path}")
        
        # Load the event accumulator
        ea = EventAccumulator(str(tb_path))
        ea.Reload()
        
        # Extract scalar data
        data = {}
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = pd.DataFrame({'step': steps, 'value': values})
        
        # Load experiment config if available
        config_path = exp_path / "config.yaml"
        config = None
        if config_path.exists() and yaml is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        self.experiments[experiment_name] = {
            'data': data,
            'config': config,
            'path': exp_path
        }
        
        return data
    
    def plot_training_curves(self, experiment_name, save_path=None, show_individual_tasks=True):
        """Generate comprehensive training curve plots"""
        if experiment_name not in self.experiments:
            self.load_experiment_data(experiment_name)
        
        data = self.experiments[experiment_name]['data']
        config = self.experiments[experiment_name].get('config', {})
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Determine the number of subplots needed
        n_plots = 1  # Always have total loss
        intermediate_tasks = [tag for tag in data.keys() if tag.startswith('Intermediate_')]
        final_tasks = [tag for tag in data.keys() if tag.startswith('Final_')]
        
        if show_individual_tasks:
            if intermediate_tasks:
                n_plots += 1
            if final_tasks:
                n_plots += 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Total Loss
        ax = axes[plot_idx]
        if 'Total/Train_Loss' in data and 'Total/Val_Loss' in data:
            train_data = data['Total/Train_Loss']
            val_data = data['Total/Val_Loss']
            
            ax.plot(train_data['step'], train_data['value'], 
                   label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(val_data['step'], val_data['value'], 
                   label='Validation Loss', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{experiment_name} - Total Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Find best validation loss
            best_val_idx = val_data['value'].idxmin()
            best_val_loss = val_data.loc[best_val_idx, 'value']
            best_epoch = val_data.loc[best_val_idx, 'step']
            
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                      label=f'Best Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch}')
            ax.legend()
        
        plot_idx += 1
        
        # Plot 2: Intermediate Task Losses
        if show_individual_tasks and intermediate_tasks:
            ax = axes[plot_idx]
            
            # Group by train/val
            intermediate_train = [tag for tag in intermediate_tasks if '_Train/' in tag]
            intermediate_val = [tag for tag in intermediate_tasks if '_Val/' in tag]
            
            # Plot training losses
            for tag in intermediate_train:
                task_name = tag.split('/')[-1]
                task_data = data[tag]
                ax.plot(task_data['step'], task_data['value'], 
                       label=f'{task_name} (Train)', linestyle='-', alpha=0.7)
            
            # Plot validation losses
            for tag in intermediate_val:
                task_name = tag.split('/')[-1]
                task_data = data[tag]
                ax.plot(task_data['step'], task_data['value'], 
                       label=f'{task_name} (Val)', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{experiment_name} - Intermediate Task Losses')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot 3: Final Task Losses
        if show_individual_tasks and final_tasks:
            ax = axes[plot_idx]
            
            # Group by train/val
            final_train = [tag for tag in final_tasks if '_Train/' in tag]
            final_val = [tag for tag in final_tasks if '_Val/' in tag]
            
            # Plot training losses
            for tag in final_train:
                task_name = tag.split('/')[-1]
                task_data = data[tag]
                ax.plot(task_data['step'], task_data['value'], 
                       label=f'{task_name} (Train)', linestyle='-', alpha=0.7)
            
            # Plot validation losses
            for tag in final_val:
                task_name = tag.split('/')[-1]
                task_data = data[tag]
                ax.plot(task_data['step'], task_data['value'], 
                       label=f'{task_name} (Val)', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{experiment_name} - Final Task Losses')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_summary_stats(self, experiment_name):
        """Generate summary statistics for an experiment"""
        if experiment_name not in self.experiments:
            self.load_experiment_data(experiment_name)
        
        data = self.experiments[experiment_name]['data']
        config = self.experiments[experiment_name].get('config', {})
        
        stats = {
            'experiment': experiment_name,
            'total_epochs': 0,
            'best_val_loss': None,
            'best_epoch': None,
            'final_train_loss': None,
            'final_val_loss': None,
            'learning_rate_final': None,
            'intermediate_tasks': {},
            'final_tasks': {}
        }
        
        # Total loss statistics
        if 'Total/Train_Loss' in data and 'Total/Val_Loss' in data:
            train_data = data['Total/Train_Loss']
            val_data = data['Total/Val_Loss']
            
            stats['total_epochs'] = len(val_data)
            stats['final_train_loss'] = train_data['value'].iloc[-1]
            stats['final_val_loss'] = val_data['value'].iloc[-1]
            
            # Best validation loss
            best_idx = val_data['value'].idxmin()
            stats['best_val_loss'] = val_data.loc[best_idx, 'value']
            stats['best_epoch'] = val_data.loc[best_idx, 'step']
        
        # Learning rate
        if 'Total/Learning_Rate' in data:
            lr_data = data['Total/Learning_Rate']
            stats['learning_rate_final'] = lr_data['value'].iloc[-1]
        
        # Intermediate task statistics
        intermediate_tasks = [tag for tag in data.keys() if tag.startswith('Intermediate_')]
        for tag in intermediate_tasks:
            task_type = 'train' if '_Train/' in tag else 'val'
            task_name = tag.split('/')[-1]
            
            if task_name not in stats['intermediate_tasks']:
                stats['intermediate_tasks'][task_name] = {}
            
            task_data = data[tag]
            stats['intermediate_tasks'][task_name][f'{task_type}_loss_final'] = task_data['value'].iloc[-1]
            if task_type == 'val':
                best_idx = task_data['value'].idxmin()
                stats['intermediate_tasks'][task_name]['best_val_loss'] = task_data.loc[best_idx, 'value']
                stats['intermediate_tasks'][task_name]['best_epoch'] = task_data.loc[best_idx, 'step']
        
        # Final task statistics
        final_tasks = [tag for tag in data.keys() if tag.startswith('Final_')]
        for tag in final_tasks:
            task_type = 'train' if '_Train/' in tag else 'val'
            task_name = tag.split('/')[-1]
            
            if task_name not in stats['final_tasks']:
                stats['final_tasks'][task_name] = {}
            
            task_data = data[tag]
            stats['final_tasks'][task_name][f'{task_type}_loss_final'] = task_data['value'].iloc[-1]
            if task_type == 'val':
                best_idx = task_data['value'].idxmin()
                stats['final_tasks'][task_name]['best_val_loss'] = task_data.loc[best_idx, 'value']
                stats['final_tasks'][task_name]['best_epoch'] = task_data.loc[best_idx, 'step']
        
        return stats
    
    def compare_experiments(self, experiment_names, save_path=None):
        """Compare multiple experiments side by side"""
        # Load all experiments
        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                self.load_experiment_data(exp_name)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Training Loss Comparison
        ax = axes[0]
        for exp_name in experiment_names:
            data = self.experiments[exp_name]['data']
            if 'Total/Train_Loss' in data:
                train_data = data['Total/Train_Loss']
                ax.plot(train_data['step'], train_data['value'], 
                       label=exp_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss Comparison
        ax = axes[1]
        for exp_name in experiment_names:
            data = self.experiments[exp_name]['data']
            if 'Total/Val_Loss' in data:
                val_data = data['Total/Val_Loss']
                ax.plot(val_data['step'], val_data['value'], 
                       label=exp_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Comparison
        ax = axes[2]
        for exp_name in experiment_names:
            data = self.experiments[exp_name]['data']
            if 'Total/Learning_Rate' in data:
                lr_data = data['Total/Learning_Rate']
                ax.plot(lr_data['step'], lr_data['value'], 
                       label=exp_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Comparison')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary Statistics
        ax = axes[3]
        stats_data = []
        for exp_name in experiment_names:
            stats = self.generate_summary_stats(exp_name)
            stats_data.append([
                exp_name,
                stats.get('best_val_loss', 0),
                stats.get('best_epoch', 0),
                stats.get('total_epochs', 0)
            ])
        
        stats_df = pd.DataFrame(stats_data, 
                               columns=['Experiment', 'Best Val Loss', 'Best Epoch', 'Total Epochs'])
        
        # Create a simple bar plot for best validation loss
        ax.bar(stats_df['Experiment'], stats_df['Best Val Loss'], alpha=0.7)
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Best Validation Loss Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
        
        return fig, stats_df
    
    def export_data(self, experiment_name, output_dir=None):
        """Export experiment data to CSV files"""
        if experiment_name not in self.experiments:
            self.load_experiment_data(experiment_name)
        
        if output_dir is None:
            output_dir = self.base_dir / experiment_name / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        data = self.experiments[experiment_name]['data']
        
        # Export each metric to CSV
        for tag, df in data.items():
            safe_filename = tag.replace('/', '_').replace('\\', '_')
            csv_path = output_dir / f"{safe_filename}.csv"
            df.to_csv(csv_path, index=False)
        
        # Export summary statistics
        stats = self.generate_summary_stats(experiment_name)
        stats_path = output_dir / "summary_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Data exported to: {output_dir}")
        
        return output_dir


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='TensorBoard Results Checker')
    parser.add_argument('--experiment', type=str, 
                       help='Experiment name to analyze')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Additional experiments to compare with')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--base-dir', type=str, default='experiments',
                       help='Base directory containing experiments')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--export-data', action='store_true',
                       help='Export data to CSV files')
    parser.add_argument('--no-individual-tasks', action='store_true',
                       help='Don\'t show individual task plots')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TensorBoardAnalyzer(args.base_dir)
    
    # List experiments if requested
    if args.list_experiments:
        experiments = analyzer.list_experiments()
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")
        else:
            print("No experiments with TensorBoard logs found.")
        return
    
    # Check if experiment specified
    if not args.experiment:
        print("Please specify an experiment name with --experiment or use --list-experiments")
        return
    
    # Check if experiment exists
    available_experiments = analyzer.list_experiments()
    if args.experiment not in available_experiments:
        print(f"Experiment '{args.experiment}' not found.")
        print("Available experiments:", available_experiments)
        return
    
    try:
        # Single experiment analysis
        print(f"Analyzing experiment: {args.experiment}")
        print("="*50)
        
        # Generate plots
        save_path = None
        if args.save_plots:
            save_path = str(Path(args.base_dir) / args.experiment / f"{args.experiment}_training_curves.png")
        
        analyzer.plot_training_curves(
            args.experiment, 
            save_path=save_path,
            show_individual_tasks=not args.no_individual_tasks
        )
        
        # Generate summary statistics
        stats = analyzer.generate_summary_stats(args.experiment)
        print("\nSummary Statistics:")
        print(f"Total Epochs: {stats['total_epochs']}")
        print(f"Best Validation Loss: {stats['best_val_loss']:.6f} @ Epoch {stats['best_epoch']}")
        print(f"Final Train Loss: {stats['final_train_loss']:.6f}")
        print(f"Final Val Loss: {stats['final_val_loss']:.6f}")
        
        if stats['intermediate_tasks']:
            print("\nIntermediate Tasks:")
            for task, task_stats in stats['intermediate_tasks'].items():
                if 'best_val_loss' in task_stats:
                    print(f"  {task}: Best Val Loss = {task_stats['best_val_loss']:.6f}")
        
        if stats['final_tasks']:
            print("\nFinal Tasks:")
            for task, task_stats in stats['final_tasks'].items():
                if 'best_val_loss' in task_stats:
                    print(f"  {task}: Best Val Loss = {task_stats['best_val_loss']:.6f}")
        
        # Comparison analysis
        if args.compare:
            all_experiments = [args.experiment] + args.compare
            print(f"\nComparing experiments: {all_experiments}")
            
            compare_save_path = None
            if args.save_plots:
                compare_save_path = str(Path(args.base_dir) / f"{'_vs_'.join(all_experiments)}_comparison.png")
            
            fig, comparison_df = analyzer.compare_experiments(all_experiments, compare_save_path)
            print("\nComparison Summary:")
            print(comparison_df.to_string(index=False))
        
        # Export data if requested
        if args.export_data:
            analyzer.export_data(args.experiment)
    
    except Exception as e:
        print(f"Error analyzing experiment: {e}")
        return


if __name__ == "__main__":
    main()
