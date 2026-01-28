"""
Diagnose why CNN training performance is poor (R² = -0.67).

Checks:
1. Training history analysis
2. Dataset quality metrics
3. Comparison with expected performance
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# Load training history
history_file = Path("ml_model/cnn_thermal_modeling/results/analysis_20260116_182621/training_history.npz")
history = np.load(history_file)

print("="*80)
print("TRAINING HISTORY ANALYSIS")
print("="*80)

for key in history.keys():
    data = history[key]
    print(f"\n{key}:")
    print(f"  Epochs: {len(data)}")
    print(f"  Initial (epoch 1): {data[0]:.4f}")
    print(f"  Final (epoch {len(data)}): {data[-1]:.4f}")
    print(f"  Best: {data.min():.4f} at epoch {data.argmin()+1}")
    print(f"  Improvement: {((data[0] - data[-1])/data[0]*100):.1f}%")

# Check dataset
dataset_file = Path("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")

print("\n" + "="*80)
print("DATASET QUALITY CHECK")
print("="*80)

with h5py.File(dataset_file, 'r') as f:
    flir = f['flir_frames'][:]
    temps = f['sand_temps'][:]
    masks = f['roi_masks'][:]
    timestamps = f['timestamps'][:]
    
    print(f"\nDataset dimensions:")
    print(f"  FLIR: {flir.shape}")
    print(f"  Temps: {temps.shape}")
    print(f"  Masks: {masks.shape}")
    
    print(f"\nTemperature ranges:")
    print(f"  FLIR: {flir.min():.1f}°C to {flir.max():.1f}°C (mean: {flir.mean():.1f}°C)")
    print(f"  Thermistor: {temps.min():.1f}°C to {temps.max():.1f}°C (mean: {temps.mean():.1f}°C)")
    
    print(f"\nROI coverage:")
    total_pixels = masks.shape[1] * masks.shape[2]
    for i in range(min(10, masks.shape[0])):
        roi_pixels = masks[i].sum()
        coverage = roi_pixels / total_pixels * 100
        print(f"  Component {i}: {roi_pixels:.0f} pixels ({coverage:.4f}%)")
    
    # Check correlation per component
    print(f"\nFLIR-Thermistor Correlation:")
    correlations = []
    for i in range(temps.shape[1]):
        # Extract FLIR temps at ROI
        roi_mask = masks[i] > 0
        flir_temps = []
        for frame_idx in range(flir.shape[0]):
            roi_values = flir[frame_idx][roi_mask]
            if len(roi_values) > 0:
                flir_temps.append(roi_values.mean())
            else:
                flir_temps.append(np.nan)
        
        flir_temps = np.array(flir_temps)
        therm_temps = temps[:, i]
        
        # Calculate correlation
        valid = ~np.isnan(flir_temps) & ~np.isnan(therm_temps)
        if valid.sum() > 10:
            corr = np.corrcoef(flir_temps[valid], therm_temps[valid])[0, 1]
            correlations.append(corr)
            if i < 5:
                print(f"  Component {i}: {corr:.3f}")
    
    avg_corr = np.mean(correlations)
    print(f"\nAverage correlation: {avg_corr:.3f}")
    print(f"Components with corr < 0.5: {sum(c < 0.5 for c in correlations)}/{len(correlations)}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Read evaluation metrics
metrics_file = Path("ml_model/cnn_thermal_modeling/results/analysis_20260116_182621/evaluation_metrics.txt")
with open(metrics_file) as f:
    metrics = f.read()
    
print("\nModel Performance:")
for line in metrics.split('\n'):
    if any(x in line for x in ['R²', 'RMSE', 'MAE']):
        print(f"  {line}")

print("\nPossible Issues:")
if avg_corr < 0.6:
    print("  ⚠️  Low correlation suggests:")
    print("      - Temporal alignment might be off")
    print("      - Surface temps (FLIR) differ significantly from embedded temps (thermistor)")
    print("      - ROI positions might be incorrect")

val_mae = history['val_mae'][-1]
if val_mae > 5.0:
    print(f"  ⚠️  High validation MAE ({val_mae:.1f}°C) suggests:")
    print("      - Model not learning meaningful patterns")
    print("      - Dataset quality insufficient")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Verify ROI pixel positions match actual component locations on thermal image")
print("2. Try larger ROI sizes (5x5 or 7x7) to capture more thermal context")
print("3. Re-run temporal alignment with finer time offset search")
print("4. Check if filtered FLIR frames have artifacts")
