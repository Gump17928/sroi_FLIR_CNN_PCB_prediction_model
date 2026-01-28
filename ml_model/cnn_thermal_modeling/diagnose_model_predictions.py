#!/usr/bin/env python3
"""
Diagnose what the CNN model actually learned.

Compares:
- Predictions vs FLIR input (are we copying input?)
- Predictions vs ground truth (are we learning the target?)
- FLIR vs ground truth (what's the actual transformation?)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from scipy.stats import pearsonr

def diagnose_predictions(npz_path, dataset_path, output_dir):
    """
    Analyze model predictions to determine what was learned.
    """
    print("="*80)
    print("DIAGNOSTIC ANALYSIS: What did the model learn?")
    print("="*80)
    
    # Load predictions
    print(f"\nLoading predictions from: {npz_path}")
    data = np.load(npz_path)
    predictions = data['predictions']
    timestamps = data['timestamps']
    timestamp_indices = data['timestamp_indices']
    frame_indices_map = data['frame_indices_map']
    
    n_samples = len(predictions)
    print(f"  Loaded {n_samples} predictions")
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    with h5py.File(dataset_path, 'r') as f:
        # Load data for these timestamps
        sand_temps = f['sand_temps'][:][timestamp_indices]
        roi_masks = f['roi_masks'][:]
        flir_frames_all = f['flir_frames'][:]
        
        # Load FLIR frames for each timestamp
        flir_frames = np.array([flir_frames_all[frame_indices_map[i]] for i in range(n_samples)])
        
        # Get component split
        if 'metadata/train_component_indices' in f:
            train_comp_indices = f['metadata/train_component_indices'][:]
            all_comp_indices = np.arange(sand_temps.shape[1])
            val_comp_indices = np.array([i for i in all_comp_indices if i not in train_comp_indices])
        else:
            n_components = sand_temps.shape[1]
            split_idx = int(n_components * 0.8)
            train_comp_indices = np.arange(0, split_idx)
            val_comp_indices = np.arange(split_idx, n_components)
    
    print(f"  Training components: {len(train_comp_indices)}")
    print(f"  Validation components: {len(val_comp_indices)}")
    
    # Extract values at ROI locations for all components
    print("\nExtracting temperatures at ROI locations...")
    
    results = []
    
    for i in range(n_samples):
        timestamp = timestamps[i]
        
        for comp_idx in range(len(roi_masks)):
            mask = roi_masks[comp_idx]
            if np.sum(mask) == 0:
                continue
            
            # Extract average temperatures at this ROI
            flir_temp = flir_frames[i][mask > 0].mean()
            pred_temp = predictions[i][mask > 0].mean()
            actual_temp = sand_temps[i, comp_idx]
            
            is_validation = comp_idx in val_comp_indices
            
            results.append({
                'timestamp': timestamp,
                'component_id': comp_idx,
                'component_type': 'validation' if is_validation else 'training',
                'flir_surface_temp': flir_temp,
                'predicted_temp': pred_temp,
                'actual_embedded_temp': actual_temp,
                'pred_vs_flir_diff': pred_temp - flir_temp,
                'pred_vs_actual_diff': pred_temp - actual_temp,
                'flir_vs_actual_diff': flir_temp - actual_temp
            })
    
    df = pd.DataFrame(results)
    
    # Save detailed results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "diagnostic_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved detailed results: {csv_file}")
    
    # Analysis by component type
    print("\n" + "="*80)
    print("ANALYSIS: Validation Components Only")
    print("="*80)
    
    df_val = df[df['component_type'] == 'validation']
    
    # Key metrics
    flir_temps = df_val['flir_surface_temp'].values
    pred_temps = df_val['predicted_temp'].values
    actual_temps = df_val['actual_embedded_temp'].values
    
    # Correlations
    corr_pred_flir, p_pred_flir = pearsonr(pred_temps, flir_temps)
    corr_pred_actual, p_pred_actual = pearsonr(pred_temps, actual_temps)
    corr_flir_actual, p_flir_actual = pearsonr(flir_temps, actual_temps)
    
    print(f"\nCorrelation Analysis:")
    print(f"  Predictions vs FLIR input:     r = {corr_pred_flir:.4f} (p={p_pred_flir:.2e})")
    print(f"  Predictions vs Ground truth:   r = {corr_pred_actual:.4f} (p={p_pred_actual:.2e})")
    print(f"  FLIR vs Ground truth:          r = {corr_flir_actual:.4f} (p={p_flir_actual:.2e})")
    
    print(f"\nInterpretation:")
    if corr_pred_flir > 0.9:
        print(f"  ⚠️  Predictions HIGHLY correlated with FLIR input (r={corr_pred_flir:.3f})")
        print(f"      → Model is mostly COPYING the input, not transforming it")
    elif corr_pred_flir > 0.7:
        print(f"  ⚠️  Predictions moderately correlated with FLIR input (r={corr_pred_flir:.3f})")
        print(f"      → Model is partially copying input")
    else:
        print(f"  ✓  Predictions weakly correlated with FLIR input (r={corr_pred_flir:.3f})")
    
    if corr_pred_actual > 0.7:
        print(f"  ✓  Predictions correlated with ground truth (r={corr_pred_actual:.3f})")
        print(f"      → Model IS learning the target transformation")
    else:
        print(f"  ✗  Predictions poorly correlated with ground truth (r={corr_pred_actual:.3f})")
        print(f"      → Model NOT learning the target transformation")
    
    # Temperature offset analysis
    print(f"\nTemperature Offset Analysis:")
    print(f"  Mean FLIR surface temp:     {flir_temps.mean():.2f}°C ± {flir_temps.std():.2f}°C")
    print(f"  Mean predicted temp:        {pred_temps.mean():.2f}°C ± {pred_temps.std():.2f}°C")
    print(f"  Mean actual embedded temp:  {actual_temps.mean():.2f}°C ± {actual_temps.std():.2f}°C")
    
    print(f"\nMean Differences:")
    print(f"  Prediction - FLIR:          {df_val['pred_vs_flir_diff'].mean():.2f}°C ± {df_val['pred_vs_flir_diff'].std():.2f}°C")
    print(f"  Prediction - Actual:        {df_val['pred_vs_actual_diff'].mean():.2f}°C ± {df_val['pred_vs_actual_diff'].std():.2f}°C")
    print(f"  FLIR - Actual (true offset):{df_val['flir_vs_actual_diff'].mean():.2f}°C ± {df_val['flir_vs_actual_diff'].std():.2f}°C")
    
    # Per-component analysis
    print(f"\nPer-Component Analysis (Validation):")
    for comp_id in val_comp_indices:
        df_comp = df_val[df_val['component_id'] == comp_id]
        if len(df_comp) == 0:
            continue
        
        flir = df_comp['flir_surface_temp'].mean()
        pred = df_comp['predicted_temp'].mean()
        actual = df_comp['actual_embedded_temp'].mean()
        
        print(f"  Component {comp_id}:")
        print(f"    FLIR: {flir:.2f}°C | Predicted: {pred:.2f}°C | Actual: {actual:.2f}°C")
        print(f"    Pred error: {pred - actual:.2f}°C | FLIR would give: {flir - actual:.2f}°C")
    
    # Create diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diagnostic Analysis: What Did the Model Learn?', fontsize=16, fontweight='bold')
    
    # Plot 1: Predictions vs FLIR
    axes[0, 0].scatter(flir_temps, pred_temps, alpha=0.5, s=20)
    axes[0, 0].plot([flir_temps.min(), flir_temps.max()], 
                    [flir_temps.min(), flir_temps.max()], 'r--', label='y=x (perfect copy)')
    axes[0, 0].set_xlabel('FLIR Surface Temp (°C)')
    axes[0, 0].set_ylabel('Predicted Temp (°C)')
    axes[0, 0].set_title(f'Predictions vs FLIR Input\nr = {corr_pred_flir:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    axes[0, 1].scatter(actual_temps, pred_temps, alpha=0.5, s=20)
    axes[0, 1].plot([actual_temps.min(), actual_temps.max()], 
                    [actual_temps.min(), actual_temps.max()], 'g--', label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Embedded Temp (°C)')
    axes[0, 1].set_ylabel('Predicted Temp (°C)')
    axes[0, 1].set_title(f'Predictions vs Ground Truth\nr = {corr_pred_actual:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: FLIR vs Actual (shows true transformation)
    axes[1, 0].scatter(flir_temps, actual_temps, alpha=0.5, s=20)
    axes[1, 0].plot([flir_temps.min(), flir_temps.max()], 
                    [flir_temps.min(), flir_temps.max()], 'k--', alpha=0.3, label='y=x')
    axes[1, 0].set_xlabel('FLIR Surface Temp (°C)')
    axes[1, 0].set_ylabel('Actual Embedded Temp (°C)')
    axes[1, 0].set_title(f'FLIR vs Embedded (True Transform)\nr = {corr_flir_actual:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    axes[1, 1].hist(df_val['pred_vs_actual_diff'], bins=30, alpha=0.7, label='Prediction Error', edgecolor='black')
    axes[1, 1].axvline(0, color='g', linestyle='--', linewidth=2, label='Zero error')
    axes[1, 1].axvline(df_val['pred_vs_actual_diff'].mean(), color='r', linestyle='-', linewidth=2, 
                      label=f'Mean: {df_val["pred_vs_actual_diff"].mean():.2f}°C')
    axes[1, 1].set_xlabel('Prediction Error (°C)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_path / "diagnostic_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plots: {plot_file}")
    plt.close()
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if corr_pred_flir > 0.9 and corr_pred_actual < 0.5:
        print("\n❌ DIAGNOSIS: Model is COPYING FLIR input, NOT learning embedded temps")
        print("\nEvidence:")
        print(f"  - Predictions highly correlated with FLIR (r={corr_pred_flir:.3f})")
        print(f"  - Predictions poorly correlated with actual (r={corr_pred_actual:.3f})")
        print(f"  - Model outputting surface temps, not embedded temps")
        print("\nROOT CAUSE: Masked loss + component holdout")
        print("  - Model only trained on 18 component ROIs (~1800 pixels)")
        print("  - Validation ROIs (4 components) never received gradients")
        print("  - Model defaults to FLIR-like output at unseen locations")
        print("\nRECOMMENDED FIX:")
        print("  1. Use ALL 22 components for training (no component holdout)")
        print("  2. Validate on TEMPORAL split (future time periods)")
        print("  3. Tests: 'Can model predict at new thermal conditions?'")
    elif corr_pred_actual > 0.7:
        print("\n✓ DIAGNOSIS: Model IS learning embedded temp transformation")
        print(f"\n  - Predictions correlate with ground truth (r={corr_pred_actual:.3f})")
        print(f"  - Predictions differ from FLIR input (r={corr_pred_flir:.3f})")
        print("\nModel is working as intended!")
    else:
        print("\n⚠️  DIAGNOSIS: Model learning something, but not clearly either input or target")
        print(f"\n  - Moderate correlation with FLIR (r={corr_pred_flir:.3f})")
        print(f"  - Moderate correlation with actual (r={corr_pred_actual:.3f})")
        print("\nFurther investigation needed - check diagnostic plots")
    
    print(f"\nAll diagnostic files saved to: {output_path}")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose CNN model predictions')
    parser.add_argument('--npz', type=str, default='results/latest/predictions_thermal_maps.npz',
                       help='Path to predictions NPZ file')
    parser.add_argument('--dataset', type=str, default='datasets/HBridge_cnn_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='results/latest',
                       help='Output directory for diagnostic files')
    
    args = parser.parse_args()
    
    diagnose_predictions(args.npz, args.dataset, args.output)
