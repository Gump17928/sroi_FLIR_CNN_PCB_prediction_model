"""
Quick test script to verify the new regression feature plot visualization.

This script loads existing ML training data and generates the new
regression feature plot showing FLIR delta T vs Sand delta T relationship.

Usage:
    python test_regression_plot.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from viz_phase8_ml_results import create_regression_feature_plot

# Find most recent ML training data
outputs_dir = Path("outputs")
training_data_files = list(outputs_dir.rglob("*ml_training_data.csv"))

if not training_data_files:
    print("Error: No ML training data files found in outputs directory")
    print("Please run the ML training workflow first (Phase 8)")
    exit(1)

# Sort by modification time (most recent first)
training_data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
most_recent = training_data_files[0]

print(f"Loading training data from: {most_recent}")

# Load training data
df = pd.read_csv(most_recent)

print(f"Loaded {len(df)} components")
print(f"Component types: {df['component_type'].unique()}")

# Extract data
flir_delta_t = df['delta_t_flir_air'].values
y_true = df['delta_t_sand'].values
y_pred = df['predicted_sand_delta_t'].values
component_types = df['component_type'].values

# Extract PCB name from file
pcb_name = most_recent.stem.replace('_ml_training_data', '')
output_dir = most_recent.parent

print(f"\nGenerating regression feature plot for {pcb_name}...")

# Create plot
plot_path = create_regression_feature_plot(
    flir_delta_t=flir_delta_t,
    y_true=y_true,
    y_pred=y_pred,
    component_types=component_types,
    output_dir=str(output_dir),
    pcb_name=pcb_name
)

print(f"\nSuccess! Plot saved to: {plot_path}")
print("\nThis plot shows:")
print("  - Circles: Actual measured sand temperatures")
print("  - Lines: Model's predicted relationship for each component type")
print("  - How the model uses FLIR air delta T to predict sand delta T")
