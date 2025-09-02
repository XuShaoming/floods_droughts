#!/usr/bin/env python3
"""
Simple experiment comparison script
Compares streamflow_exp1 vs streamflow_exp2 with scatter plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading data...")
exp1 = pd.read_csv("experiments/streamflow_exp1/best_model_results/val_reconstructed_timeseries.csv")
exp2 = pd.read_csv("experiments/streamflow_exp2/best_model_results/val_reconstructed_timeseries.csv")

print(f"Exp1: {len(exp1)} points")
print(f"Exp2: {len(exp2)} points")

# Get data
obs = exp1['streamflow_observation']  # Should be same for both
pred1 = exp1['streamflow_prediction']
pred2 = exp2['streamflow_prediction']

# Calculate NSE for labels
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_exp1 = r2_score(obs, pred1)
r2_exp2 = r2_score(obs, pred2)

print(f"Exp1 NSE: {r2_exp1:.4f}")
print(f"Exp2 NSE: {r2_exp2:.4f}")

# Create scatter plot
plt.figure(figsize=(10, 8))

# Sample data if too many points
n = len(obs)
obs_plot = obs
pred1_plot = pred1
pred2_plot = pred2

# Plot scatter points
plt.scatter(obs_plot, pred1_plot, alpha=0.6, s=1, color='blue', 
           label=f'Antecedent conditions, None, (NSE = {r2_exp1:.3f})')
plt.scatter(obs_plot, pred2_plot, alpha=0.6, s=1, color='red', 
           label=f'Antecedent conditions, Included, (NSE = {r2_exp2:.3f})')

# Add 1:1 line
min_val = min(obs_plot.min(), pred1_plot.min(), pred2_plot.min())
max_val = max(obs_plot.max(), pred1_plot.max(), pred2_plot.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')

# Formatting
plt.xlabel('Observed Streamflow (cfs)', fontsize=12)
plt.ylabel('Predicted Streamflow (cfs)', fontsize=12)
plt.title('Streamflow Prediction Comparison\nValidation Dataset', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
# plt.axis('equal')

# Save and show
plt.tight_layout()
plt.savefig('experiments/streamflow_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved: experiments/streamflow_comparison.png")
plt.show()
