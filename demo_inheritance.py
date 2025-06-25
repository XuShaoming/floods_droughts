#!/usr/bin/env python3
"""
Demonstration of YAML inheritance structure for experiment management.
This script shows how to use YAML anchors and references for clean experiment configuration.
"""

import yaml
import os


def demonstrate_yaml_inheritance():
    """Show how YAML inheritance works in our configuration."""
    
    print("YAML Inheritance Demo - LSTM Experiment Configuration")
    print("=" * 60)
    
    # Load the config file
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Show available experiments
    experiments = [key for key in config.keys() 
                  if key not in ['base_config', 'BASE_CONFIG', 'default_experiment'] 
                  and not key.startswith('&')]
    
    print(f"Available experiments: {experiments}")
    print("\n" + "=" * 60)
    
    # Show how inheritance works
    for exp_name in experiments[:3]:  # Show first 3 experiments
        exp_config = config[exp_name]
        
        print(f"\nEXPERIMENT: {exp_name}")
        print("-" * 30)
        
        # Show key differences from base
        print("Key Configuration:")
        print(f"  Window Size: {exp_config['data']['window_size']}")
        print(f"  Hidden Size: {exp_config['model']['hidden_size']}")
        print(f"  Num Layers: {exp_config['model']['num_layers']}")
        print(f"  Batch Size: {exp_config['training']['batch_size']}")
        print(f"  Learning Rate: {exp_config['training']['learning_rate']}")
        print(f"  Epochs: {exp_config['training']['epochs']}")
        print(f"  Scheduler: {exp_config['training']['scheduler']['type']}")
        print(f"  Save Dir: {exp_config['output']['save_dir']}")
        
        if exp_name == 'multitarget':
            print(f"  Targets: {exp_config['data']['target_cols']}")
            if exp_config['data']['feature_cols']:
                print(f"  Features: {exp_config['data']['feature_cols'][:3]}...")


def show_usage_examples():
    """Show usage examples for the inheritance-based config."""
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Default experiment (baseline)", "python train.py"),
        ("Large model experiment", "python train.py --experiment large_model"),
        ("Fast baseline experiment", "python train.py --experiment fast_baseline"),
        ("Multi-target experiment", "python train.py --experiment multitarget"),
        ("Deep model experiment", "python train.py --experiment deep_model"),
        ("Long sequence experiment", "python train.py --experiment long_sequence"),
        ("Override parameters", "python train.py --experiment baseline --override training.epochs=200"),
        ("Custom learning rate", "python train.py --experiment large_model --override training.learning_rate=0.0001"),
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"{i}. {desc}:")
        print(f"   {cmd}")
        print()


def compare_with_base():
    """Compare experiment configs with base to show inheritance."""
    
    print("\n" + "=" * 60)
    print("INHERITANCE ANALYSIS")
    print("=" * 60)
    
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_config = config.get('BASE_CONFIG', {})
    baseline = config.get('baseline', {})
    large_model = config.get('large_model', {})
    
    print("BASE CONFIG vs LARGE MODEL EXPERIMENT:")
    print("-" * 40)
    
    changes = [
        ("Window Size", "data.window_size", 30, 60),
        ("Hidden Size", "model.hidden_size", 64, 128),
        ("Num Layers", "model.num_layers", 2, 3),
        ("Dropout", "model.dropout", 0.2, 0.3),
        ("Batch Size", "training.batch_size", 32, 16),
        ("Learning Rate", "training.learning_rate", 0.001, 0.0005),
        ("Epochs", "training.epochs", 100, 150),
        ("Scheduler", "training.scheduler.type", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts"),
    ]
    
    for param, path, base_val, exp_val in changes:
        print(f"{param:15} | Base: {base_val:20} | Large Model: {exp_val}")
    
    print("\n✅ Benefits of YAML Inheritance:")
    print("  - DRY (Don't Repeat Yourself) principle")
    print("  - Easy to see what changed between experiments")  
    print("  - Consistent base configuration across experiments")
    print("  - Easy to add new experiments")
    print("  - Maintainable and version-control friendly")


if __name__ == "__main__":
    demonstrate_yaml_inheritance()
    show_usage_examples()
    compare_with_base()
