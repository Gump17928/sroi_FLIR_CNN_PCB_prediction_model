#!/usr/bin/env python3
"""
Quick diagnostic: Verify denormalization is working correctly.

Checks:
1. Normalization stats from HDF5
2. Sample prediction denormalization
3. Visual inspection of output ranges
"""

import h5py
import numpy as np
from pathlib import Path

# Paths
dataset_path = Path(__file__).parent / "datasets" / "HBridge_cnn_dataset.h5"
results_path = Path(__file__).parent / "results" / "analysis_20260127_154615" / "evaluation_metrics.txt"

print("="*80)
print("DENORMALIZATION VERIFICATION")
print("="*80)

# 1. Check HDF5 normalization stats
print("\n1. Checking HDF5 normalization metadata:")
with h5py.File(dataset_path, 'r') as f:
    if 'metadata/sand_min' in f:
        sand_min = float(f['metadata/sand_min'][()])
        sand_max = float(f['metadata/sand_max'][()])
        flir_min = float(f['metadata/flir_min'][()])
        flir_max = float(f['metadata/flir_max'][()])
        time_min = float(f['metadata/time_min'][()])
        time_max = float(f['metadata/time_max'][()])
        
        print(f"  ✓ Cached stats found in HDF5:")
        print(f"    FLIR: [{flir_min:.2f}, {flir_max:.2f}]°C")
        print(f"    Sand: [{sand_min:.2f}, {sand_max:.2f}]°C")
        print(f"    Time: [{time_min:.2f}, {time_max:.2f}]s")
    else:
        print("  ⚠️  No cached stats found - will compute from data")
        sand_temps = f['sand_temps'][:]
        sand_min = float(np.min(sand_temps))
        sand_max = float(np.max(sand_temps))
        print(f"    Sand: [{sand_min:.2f}, {sand_max:.2f}]°C (computed)")

# 2. Check denormalization formula
print("\n2. Testing denormalization formula:")
print(f"   Formula: pred_temp = pred_norm × (sand_max - sand_min) + sand_min")
print(f"   Formula: pred_temp = pred_norm × ({sand_max:.2f} - {sand_min:.2f}) + {sand_min:.2f}")
print(f"   Formula: pred_temp = pred_norm × {sand_max - sand_min:.2f} + {sand_min:.2f}")

# Test cases
test_normalized = [0.0, 0.5, 1.0]
print("\n   Test cases:")
for norm in test_normalized:
    denorm = norm * (sand_max - sand_min) + sand_min
    print(f"     Normalized {norm:.1f} → Denormalized {denorm:.2f}°C")

# 3. Check evaluation metrics from file
print("\n3. Checking evaluation metrics:")
if results_path.exists():
    with open(results_path, 'r') as f:
        content = f.read()
        print(content)
else:
    print("  ⚠️  Metrics file not found")

# 4. Expected temperature range check
print("\n4. Expected temperature ranges:")
print(f"   Training data: [{sand_min:.2f}, {sand_max:.2f}]°C")
print(f"   Normalized range: [0.0, 1.0]")
print(f"   Denormalized range: [{sand_min:.2f}, {sand_max:.2f}]°C")

# 5. Load actual sand temps and check validation components
print("\n5. Checking validation component temperatures:")
with h5py.File(dataset_path, 'r') as f:
    if 'metadata/val_component_indices' in f:
        val_comp_indices = f['metadata/val_component_indices'][:]
        component_names = [name.decode('utf-8') for name in f['metadata/component_names'][:]]
        sand_temps = f['sand_temps'][:]
        
        print(f"   Validation components: {len(val_comp_indices)}")
        for comp_idx in val_comp_indices:
            comp_name = component_names[comp_idx]
            comp_temps = sand_temps[:, comp_idx]
            comp_min = np.min(comp_temps)
            comp_max = np.max(comp_temps)
            comp_mean = np.mean(comp_temps)
            print(f"     {comp_name} (idx={comp_idx}): [{comp_min:.2f}, {comp_max:.2f}]°C, mean={comp_mean:.2f}°C")
        
        # Overall validation stats
        val_temps = sand_temps[:, val_comp_indices]
        print(f"\n   Overall validation component temps:")
        print(f"     Min: {np.min(val_temps):.2f}°C")
        print(f"     Max: {np.max(val_temps):.2f}°C")
        print(f"     Mean: {np.mean(val_temps):.2f}°C")
        print(f"     Std: {np.std(val_temps):.2f}°C")
        
        # Baseline prediction (mean of validation temps)
        baseline_pred = np.mean(val_temps)
        baseline_mse = np.mean((val_temps - baseline_pred) ** 2)
        baseline_rmse = np.sqrt(baseline_mse)
        print(f"\n   Baseline (predict mean={baseline_pred:.2f}°C):")
        print(f"     RMSE: {baseline_rmse:.2f}°C")
        print(f"     R²: 0.00 (by definition)")
        
        # Check if model is worse than baseline
        from_metrics = 9.18  # RMSE from evaluation
        if from_metrics > baseline_rmse:
            print(f"\n   ⚠️  Model RMSE ({from_metrics:.2f}°C) > Baseline RMSE ({baseline_rmse:.2f}°C)")
            print(f"       This explains negative R² score!")
        else:
            print(f"\n   ✓ Model RMSE ({from_metrics:.2f}°C) < Baseline RMSE ({baseline_rmse:.2f}°C)")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
