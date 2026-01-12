"""
Train U-Net model for HBridge thermal field reconstruction.

This script:
1. Loads HDF5 training dataset
2. Initializes U-Net model with custom masked MSE loss
3. Trains with validation split
4. Saves trained model and training history
5. Generates performance metrics and visualizations

Author: CNN Pipeline
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from phase8c_spatial_cnn import SpatialCNNTrainer


def load_dataset_info(h5_file):
    """Load and display dataset information."""
    print("="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        print(f"\nDataset: {h5_file}")
        print(f"\nShapes:")
        print(f"  FLIR frames: {f['flir_frames'].shape}")
        print(f"  Sand temps: {f['sand_temps'].shape}")
        print(f"  ROI masks: {f['roi_masks'].shape}")
        print(f"  Timestamps: {f['timestamps'].shape}")
        
        print(f"\nMetadata:")
        for key, val in f['metadata'].attrs.items():
            print(f"  {key}: {val}")
        
        # Get component names
        component_names = [name.decode('utf-8') for name in f['metadata']['component_names'][:]]
        print(f"\n  Components ({len(component_names)}): {', '.join(component_names[:10])}{'...' if len(component_names) > 10 else ''}")
        
        # Temperature statistics
        temps = f['sand_temps'][:]
        print(f"\nTemperature Statistics:")
        print(f"  Min: {np.min(temps):.2f}°C")
        print(f"  Max: {np.max(temps):.2f}°C")
        print(f"  Mean: {np.mean(temps):.2f}°C")
        print(f"  Std: {np.std(temps):.2f}°C")
        
        # FLIR statistics
        frames = f['flir_frames'][:]
        print(f"\nFLIR Frame Statistics:")
        print(f"  Min: {np.min(frames):.2f}°C")
        print(f"  Max: {np.max(frames):.2f}°C")
        print(f"  Mean: {np.mean(frames):.2f}°C")
        print(f"  Std: {np.std(frames):.2f}°C")
    
    print()


def train_model(dataset_path, output_dir, epochs=100, batch_size=8, validation_split=0.2):
    """
    Train U-Net model on HBridge dataset.
    
    Args:
        dataset_path: Path to HDF5 dataset
        output_dir: Directory to save model and results
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
    """
    print("\n" + "="*80)
    print("TRAINING U-NET MODEL")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SpatialCNNTrainer(verbose=True)
    
    # Load dataset
    trainer.load_dataset(str(dataset_path))
    
    # Prepare training data
    X_train, y_train, X_val, y_val = trainer.prepare_training_data(
        val_split=validation_split,
        temporal_split=True
    )
    
    # Build model
    trainer.build_model(learning_rate=0.001)
    
    # Train model
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split:.1%}")
    print(f"  Output directory: {output_path}")
    print()
    
    trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=15
    )
    
    # Save model
    model_file = output_path / f"unet_hbridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    trainer.model.save(model_file)
    print(f"\n✓ Model saved: {model_file}")
    
    history = trainer.history
    
    # Save training history
    history_file = output_path / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(history_file, **history.history)
    print(f"\n✓ Training history saved: {history_file}")
    
    # Plot training curves
    plot_training_curves(history, output_path)
    
    return trainer, history


def plot_training_curves(history, output_dir):
    """Plot and save training/validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Masked MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE curve
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE (°C)', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {output_file}")
    plt.close()


def evaluate_model(trainer, dataset_path, output_dir):
    """
    Evaluate trained model and generate metrics.
    
    Args:
        trainer: Trained SpatialCNNTrainer instance
        dataset_path: Path to HDF5 dataset
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load test data (use a subset for quick evaluation)
    with h5py.File(dataset_path, 'r') as f:
        flir_frames = f['flir_frames'][:50]  # First 50 frames
        sand_temps = f['sand_temps'][:50]
        roi_masks = f['roi_masks'][:]
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = trainer.model.predict(flir_frames[..., np.newaxis], verbose=0)
    predictions = predictions.squeeze()
    
    # Calculate metrics at ROI locations
    print("\nCalculating metrics at ROI locations...")
    
    all_actual = []
    all_predicted = []
    
    for frame_idx in range(len(flir_frames)):
        for comp_idx in range(roi_masks.shape[0]):
            mask = roi_masks[comp_idx]
            if np.sum(mask) > 0:
                # Extract values at ROI pixels
                actual_temp = sand_temps[frame_idx, comp_idx]
                predicted_temp = predictions[frame_idx][mask > 0].mean()
                
                all_actual.append(actual_temp)
                all_predicted.append(predicted_temp)
    
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    
    # Calculate R², RMSE, MAE
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(all_actual, all_predicted)
    rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
    mae = mean_absolute_error(all_actual, all_predicted)
    
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  Sample count: {len(all_actual)}")
    
    # Save metrics
    metrics_file = Path(output_dir) / f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(metrics_file, 'w') as f:
        f.write("HBridge U-Net Model Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f}°C\n")
        f.write(f"MAE: {mae:.2f}°C\n")
        f.write(f"Sample count: {len(all_actual)}\n")
        f.write(f"\nTemperature Range:\n")
        f.write(f"  Actual: {all_actual.min():.2f}°C - {all_actual.max():.2f}°C\n")
        f.write(f"  Predicted: {all_predicted.min():.2f}°C - {all_predicted.max():.2f}°C\n")
    
    print(f"\n✓ Metrics saved: {metrics_file}")
    
    # Create scatter plot
    plot_scatter(all_actual, all_predicted, r2, rmse, output_dir)
    
    return r2, rmse, mae


def plot_scatter(actual, predicted, r2, rmse, output_dir):
    """Plot actual vs predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    
    plt.scatter(actual, predicted, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title(f'U-Net Predictions vs Actual Thermistor Readings\nR² = {r2:.4f}, RMSE = {rmse:.2f}°C', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save figure
    output_file = Path(output_dir) / f"scatter_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Scatter plot saved: {output_file}")
    plt.close()


def main():
    """Main training pipeline."""
    # Paths
    dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
    output_dir = parent_dir / "ml_model" / "cnn_thermal_modeling" / "results"
    
    # Verify dataset exists
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    # Load and display dataset info
    load_dataset_info(dataset_file)
    
    # Train model
    trainer, history = train_model(
        dataset_path=dataset_file,
        output_dir=output_dir,
        epochs=100,
        batch_size=8,
        validation_split=0.2
    )
    
    # Evaluate model
    r2, rmse, mae = evaluate_model(trainer, dataset_file, output_dir)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"\nModel saved to: {output_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
