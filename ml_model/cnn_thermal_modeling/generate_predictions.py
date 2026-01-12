"""
Generate CNN predictions and thermal field visualizations.

This script:
1. Loads trained U-Net model
2. Generates predictions on test frames
3. Creates thermal field visualizations (FLIR | Predicted | Actual | Error)
4. Exports component-level predictions to CSV
5. Calculates performance metrics (R², RMSE, MAE)

Author: CNN Pipeline
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import TensorFlow only when needed
import tensorflow as tf
from viz_phase8c_spatial import Phase8cVisualizer


def load_trained_model(model_dir):
    """Find and load the most recent trained model."""
    model_path = Path(model_dir)
    
    # Find all .keras model files
    model_files = list(model_path.glob("unet_*.keras"))
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")
    
    # Get most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading model: {latest_model.name}")
    model = tf.keras.models.load_model(latest_model, compile=False)
    
    return model, latest_model


def generate_predictions(model, dataset_path, n_frames=None):
    """
    Generate predictions on dataset.
    
    Args:
        model: Trained Keras model
        dataset_path: Path to HDF5 dataset
        n_frames: Number of frames to predict (None = all)
    
    Returns:
        Dictionary with predictions and ground truth
    """
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    with h5py.File(dataset_path, 'r') as f:
        flir_frames = f['flir_frames'][:]
        sand_temps = f['sand_temps'][:]
        roi_masks = f['roi_masks'][:]
        timestamps = f['timestamps'][:]
        component_names = [name.decode('utf-8') for name in f['metadata']['component_names'][:]]
    
    if n_frames:
        flir_frames = flir_frames[:n_frames]
        sand_temps = sand_temps[:n_frames]
        timestamps = timestamps[:n_frames]
    
    # Generate predictions
    print(f"\nPredicting {len(flir_frames)} frames...")
    predictions = model.predict(flir_frames[..., np.newaxis], verbose=1)
    predictions = predictions.squeeze()
    
    print(f"✓ Predictions complete: {predictions.shape}")
    
    return {
        'flir_frames': flir_frames,
        'predictions': predictions,
        'sand_temps': sand_temps,
        'roi_masks': roi_masks,
        'timestamps': timestamps,
        'component_names': component_names
    }


def extract_component_predictions(results):
    """
    Extract component-level predictions from spatial predictions.
    
    Args:
        results: Dictionary from generate_predictions()
    
    Returns:
        DataFrame with component predictions
    """
    print("\n" + "="*80)
    print("EXTRACTING COMPONENT PREDICTIONS")
    print("="*80)
    
    predictions = results['predictions']
    sand_temps = results['sand_temps']
    roi_masks = results['roi_masks']
    component_names = results['component_names']
    timestamps = results['timestamps']
    
    n_frames = len(predictions)
    n_components = len(component_names)
    
    records = []
    
    for frame_idx in range(n_frames):
        for comp_idx, comp_name in enumerate(component_names):
            if comp_idx >= sand_temps.shape[1]:
                # Component in ROI mask but not in thermistor data
                continue
                
            mask = roi_masks[comp_idx]
            
            if np.sum(mask) > 0:
                # Extract predicted temperature at ROI pixels (mean)
                pred_temp = predictions[frame_idx][mask > 0].mean()
                actual_temp = sand_temps[frame_idx, comp_idx]
                
                records.append({
                    'timestamp': timestamps[frame_idx],
                    'component': comp_name,
                    'actual_temp': actual_temp,
                    'predicted_temp': pred_temp,
                    'error': pred_temp - actual_temp,
                    'abs_error': abs(pred_temp - actual_temp)
                })
    
    df = pd.DataFrame(records)
    
    print(f"✓ Extracted predictions for {len(records)} component-frame pairs")
    print(f"  Components: {n_components}")
    print(f"  Frames: {n_frames}")
    
    return df


def calculate_metrics(df):
    """Calculate overall and per-component metrics."""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    # Overall metrics
    r2 = r2_score(df['actual_temp'], df['predicted_temp'])
    rmse = np.sqrt(mean_squared_error(df['actual_temp'], df['predicted_temp']))
    mae = mean_absolute_error(df['actual_temp'], df['predicted_temp'])
    
    print(f"\nOverall Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  Sample count: {len(df)}")
    
    # Per-component metrics
    print(f"\nPer-Component Performance:")
    component_metrics = []
    
    for comp in df['component'].unique():
        comp_df = df[df['component'] == comp]
        comp_r2 = r2_score(comp_df['actual_temp'], comp_df['predicted_temp'])
        comp_rmse = np.sqrt(mean_squared_error(comp_df['actual_temp'], comp_df['predicted_temp']))
        comp_mae = mean_absolute_error(comp_df['actual_temp'], comp_df['predicted_temp'])
        
        component_metrics.append({
            'component': comp,
            'r2': comp_r2,
            'rmse': comp_rmse,
            'mae': comp_mae,
            'n_samples': len(comp_df)
        })
    
    comp_metrics_df = pd.DataFrame(component_metrics).sort_values('rmse', ascending=False)
    print(comp_metrics_df.to_string(index=False))
    
    return {
        'overall': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'per_component': comp_metrics_df
    }


def create_visualizations(results, output_dir, n_examples=5):
    """Create thermal field visualizations using viz_phase8c_spatial."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = Phase8cVisualizer()
    visualizer.output_dir = output_path  # Set output directory
    
    # Select representative frames (spread across time)
    n_frames = len(results['flir_frames'])
    frame_indices = np.linspace(0, n_frames-1, n_examples, dtype=int)
    
    # Create combined ROI mask (all components)
    combined_mask = np.max(results['roi_masks'], axis=0)
    
    for idx in frame_indices:
        flir = results['flir_frames'][idx]
        pred = results['predictions'][idx]
        actual = results['sand_temps'][idx]
        masks = results['roi_masks']
        timestamp = results['timestamps'][idx]
        
        # Build actual thermal map from component temps
        H, W = flir.shape
        actual_map = np.zeros((H, W))
        for comp_idx in range(len(actual)):
            if comp_idx < masks.shape[0]:
                mask = masks[comp_idx]
                actual_map[mask > 0] = actual[comp_idx]
        
        # Create comparison plot
        save_name = f"thermal_comparison_t{timestamp:.1f}s.png"
        visualizer.plot_thermal_field_comparison(
            flir_frame=flir,
            predicted_frame=pred,
            ground_truth_frame=actual_map,
            roi_mask=combined_mask,
            frame_index=idx,
            save_name=save_name
        )
    
    print(f"\n✓ Created {n_examples} thermal field visualizations")


def create_scatter_plot(df, output_dir):
    """Create actual vs predicted scatter plot."""
    output_path = Path(output_dir)
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(df['actual_temp'], df['predicted_temp'], 
                alpha=0.3, s=30, edgecolors='k', linewidth=0.5, label='Predictions')
    
    # Perfect prediction line
    min_val = min(df['actual_temp'].min(), df['predicted_temp'].min())
    max_val = max(df['actual_temp'].max(), df['predicted_temp'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Metrics
    r2 = r2_score(df['actual_temp'], df['predicted_temp'])
    rmse = np.sqrt(mean_squared_error(df['actual_temp'], df['predicted_temp']))
    mae = mean_absolute_error(df['actual_temp'], df['predicted_temp'])
    
    plt.xlabel('Actual Temperature (°C)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Temperature (°C)', fontsize=14, fontweight='bold')
    plt.title(f'Phase 8c U-Net: Actual vs Predicted\nR² = {r2:.4f}, RMSE = {rmse:.2f}°C, MAE = {mae:.2f}°C',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save
    output_file = output_path / f"scatter_actual_vs_predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Scatter plot saved: {output_file.name}")
    plt.close()


def export_results(df, metrics, output_dir, model_file):
    """Export predictions and metrics to CSV and text files."""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export predictions CSV
    pred_file = output_path / f"predictions_{timestamp}.csv"
    df.to_csv(pred_file, index=False)
    print(f"\n✓ Predictions exported: {pred_file.name}")
    
    # Export metrics
    metrics_file = output_path / f"metrics_{timestamp}.txt"
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Phase 8c U-Net CNN - Performance Metrics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_file.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Overall Performance:\n")
        f.write("-"*40 + "\n")
        f.write(f"R² Score: {metrics['overall']['r2']:.4f}\n")
        f.write(f"RMSE: {metrics['overall']['rmse']:.2f}°C\n")
        f.write(f"MAE: {metrics['overall']['mae']:.2f}°C\n")
        f.write(f"Total predictions: {len(df)}\n\n")
        
        f.write("Per-Component Performance:\n")
        f.write("-"*40 + "\n")
        f.write(metrics['per_component'].to_string(index=False))
    
    print(f"✓ Metrics exported: {metrics_file.name}")


def main():
    """Main prediction pipeline."""
    # Paths
    results_dir = parent_dir / "ml_model" / "cnn_thermal_modeling" / "results"
    dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
    
    # Verify dataset exists
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    # Load trained model
    model, model_file = load_trained_model(results_dir)
    
    # Generate predictions
    results = generate_predictions(model, dataset_file, n_frames=None)
    
    # Extract component-level predictions
    df = extract_component_predictions(results)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Create visualizations
    create_visualizations(results, results_dir, n_examples=10)
    create_scatter_plot(df, results_dir)
    
    # Export results
    export_results(df, metrics, results_dir, model_file)
    
    print("\n" + "="*80)
    print("PREDICTION GENERATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nFinal Performance:")
    print(f"  R² Score: {metrics['overall']['r2']:.4f}")
    print(f"  RMSE: {metrics['overall']['rmse']:.2f}°C")
    print(f"  MAE: {metrics['overall']['mae']:.2f}°C")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
