"""
Quick verification script for HDF5 training dataset.
Displays dataset structure and summary statistics.
"""

import h5py
import numpy as np
from pathlib import Path

# Path to HDF5 dataset
h5_file = Path(__file__).parent.parent.parent / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"

print("="*80)
print("HDF5 DATASET VERIFICATION")
print("="*80)
print(f"\nFile: {h5_file}")
print(f"Size: {h5_file.stat().st_size / 1024 / 1024:.2f} MB\n")

with h5py.File(h5_file, 'r') as f:
    print("Dataset Structure:")
    print("-" * 40)
    
    # FLIR frames
    flir_frames = f['flir_frames'][:]
    print(f"✓ flir_frames: {flir_frames.shape}")
    print(f"  - Temperature range: {flir_frames.min():.2f}°C to {flir_frames.max():.2f}°C")
    print(f"  - Mean: {flir_frames.mean():.2f}°C, Std: {flir_frames.std():.2f}°C")
    
    # Sand temps (thermistor ground truth)
    sand_temps = f['sand_temps'][:]
    print(f"\n✓ sand_temps: {sand_temps.shape}")
    print(f"  - Temperature range: {sand_temps.min():.2f}°C to {sand_temps.max():.2f}°C")
    print(f"  - Mean: {sand_temps.mean():.2f}°C, Std: {sand_temps.std():.2f}°C")
    
    # ROI masks
    roi_masks = f['roi_masks'][:]
    print(f"\n✓ roi_masks: {roi_masks.shape}")
    print(f"  - Total ROI pixels: {roi_masks.sum():.0f}")
    print(f"  - Coverage: {roi_masks.sum() / (roi_masks.shape[1] * roi_masks.shape[2]) * 100:.2f}%")
    
    # Timestamps
    timestamps = f['timestamps'][:]
    print(f"\n✓ timestamps: {timestamps.shape}")
    print(f"  - Time range: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
    print(f"  - Duration: {timestamps[-1] - timestamps[0]:.1f}s")
    print(f"  - Avg interval: {np.mean(np.diff(timestamps)):.2f}s")
    
    # Metadata
    print("\n✓ Metadata:")
    for key, val in f['metadata'].attrs.items():
        print(f"  - {key}: {val}")

print("\n" + "="*80)
print("✓ DATASET VERIFICATION COMPLETE")
print("="*80)
