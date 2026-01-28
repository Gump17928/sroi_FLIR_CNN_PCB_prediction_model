"""
===============================================================================
VISUALIZATION - Phase 8c Spatial CNN Results
===============================================================================
Visualization tools for U-Net thermal field predictions.

Purpose:
    - Visualize FLIR input vs CNN predictions vs ground truth
    - Generate thermal field heatmaps
    - Plot component-level scatter (actual vs predicted)
    - Show spatial error distributions
    - Compare Phase 8 (linear) vs Phase 8c (CNN) performance
    - Create training curves and diagnostic plots

Visualization Types:
    1. Side-by-side thermal maps: FLIR | Predicted | Error
    2. Component scatter: Actual vs Predicted temperatures
    3. Spatial error heatmap: |Predicted - Actual| across image
    4. Training curves: Loss and metrics vs epochs
    5. Temporal evolution: Animation or multi-frame comparison
    6. Phase 8 vs 8c comparison: R² improvement demonstration

Created: January 12, 2026
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import h5py
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras


class Phase8cVisualizer:
    """
    Visualizer for Phase 8c spatial CNN results.
    
    Generates publication-quality plots for CNN thermal predictions.
    """
    
    def __init__(self, output_dir: str = "ml_model/cnn_thermal_modeling/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for output plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_thermal_field_comparison(self, flir_frame, predicted_frame, 
                                     ground_truth_frame, roi_mask,
                                     frame_index: int, save_name: str):
        """
        Plot side-by-side thermal field comparison.
        
        Args:
            flir_frame: FLIR input [H, W]
            predicted_frame: CNN prediction [H, W]
            ground_truth_frame: Sparse ground truth [H, W]
            roi_mask: Combined ROI mask [H, W]
            frame_index: Frame index for title
            save_name: Filename for saving
        """
        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)
        
        # Common temperature range for color scaling
        vmin = min(np.min(flir_frame), np.min(predicted_frame[roi_mask == 1]))
        vmax = max(np.max(flir_frame), np.max(predicted_frame[roi_mask == 1]))
        
        # FLIR Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(flir_frame, cmap='hot', vmin=vmin, vmax=vmax)
        ax1.set_title(f'FLIR Input\n(Frame {frame_index})', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, label='Temperature (°C)')
        
        # CNN Prediction
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(predicted_frame, cmap='hot', vmin=vmin, vmax=vmax)
        ax2.set_title('CNN Predicted\nSand Temperature', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, label='Temperature (°C)')
        
        # Ground Truth (sparse)
        ax3 = fig.add_subplot(gs[0, 2])
        gt_display = np.copy(ground_truth_frame)
        gt_display[gt_display == 0] = np.nan  # Hide non-ROI pixels
        im3 = ax3.imshow(gt_display, cmap='hot', vmin=vmin, vmax=vmax)
        ax3.set_title('Ground Truth\n(Thermistor ROIs)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='Temperature (°C)')
        
        # Error Map
        ax4 = fig.add_subplot(gs[0, 3])
        error = np.abs(predicted_frame - ground_truth_frame)
        error[roi_mask == 0] = np.nan  # Show error only at ROI locations
        im4 = ax4.imshow(error, cmap='coolwarm', vmin=0, vmax=5)
        ax4.set_title('Absolute Error\n(at ROI locations)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, label='|Error| (°C)')
        
        plt.suptitle(f'Phase 8c: Spatial CNN Thermal Reconstruction', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def plot_component_scatter(self, y_true_components, y_pred_components, 
                              component_names, save_name: str):
        """
        Plot component-level scatter: Actual vs Predicted.
        
        Args:
            y_true_components: Actual temperatures [n_samples, n_components]
            y_pred_components: Predicted temperatures [n_samples, n_components]
            component_names: List of component names
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Flatten all component predictions
        y_true_flat = y_true_components.flatten()
        y_pred_flat = y_pred_components.flatten()
        
        # Calculate R² and RMSE
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # Scatter plot
        ax.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        # 1:1 reference line
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (1:1)')
        
        # Labels and title
        ax.set_xlabel('Actual Sand Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Sand Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title(f'Phase 8c CNN: Actual vs Predicted\nR² = {r2:.4f}, RMSE = {rmse:.2f}°C', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = f'R² Score: {r2:.4f}\nRMSE: {rmse:.2f}°C\nSamples: {len(y_true_flat)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def plot_training_curves(self, history_path: str, save_name: str):
        """
        Plot training and validation loss curves.
        
        Args:
            history_path: Path to training history JSON
            save_name: Filename for saving
        """
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Masked MSE Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE curve
        axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Absolute Error (°C)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Phase 8c CNN Training Curves', fontsize=16, fontweight='bold', y=1.02)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def plot_phase8_vs_phase8c_comparison(self, phase8_r2: float, phase8_rmse: float,
                                         phase8c_r2: float, phase8c_rmse: float,
                                         save_name: str):
        """
        Compare Phase 8 (linear) vs Phase 8c (CNN) performance.
        
        Args:
            phase8_r2: Phase 8 R² score
            phase8_rmse: Phase 8 RMSE
            phase8c_r2: Phase 8c R² score
            phase8c_rmse: Phase 8c RMSE
            save_name: Filename for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        models = ['Phase 8\n(Linear Regression)', 'Phase 8c\n(Spatial CNN)']
        r2_scores = [phase8_r2, phase8c_r2]
        rmse_scores = [phase8_rmse, phase8c_rmse]
        
        colors = ['#e74c3c', '#27ae60']
        
        # R² comparison
        bars1 = axes[0].bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance: R² Score', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Target (0.90)')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # RMSE comparison
        bars2 = axes[1].bar(models, rmse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('RMSE (°C)', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Performance: RMSE', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.2f}°C', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle('Phase 8 vs Phase 8c: Performance Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
    
    def plot_spatial_error_distribution(self, y_true_frame, y_pred_frame, 
                                       roi_mask, save_name: str):
        """
        Plot spatial error distribution across image.
        
        Args:
            y_true_frame: Ground truth [H, W]
            y_pred_frame: Prediction [H, W]
            roi_mask: ROI mask [H, W]
            save_name: Filename for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute error map
        error = np.abs(y_pred_frame - y_true_frame)
        error_display = np.copy(error)
        error_display[roi_mask == 0] = np.nan
        
        im1 = axes[0].imshow(error_display, cmap='YlOrRd', vmin=0)
        axes[0].set_title('Spatial Error Distribution\n(Absolute Error)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, label='|Error| (°C)')
        
        # Error histogram (at ROI locations)
        roi_errors = error[roi_mask == 1]
        roi_errors = roi_errors[~np.isnan(roi_errors)]
        
        axes[1].hist(roi_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Absolute Error (°C)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Error Distribution at ROI Locations', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_error = np.mean(roi_errors)
        std_error = np.std(roi_errors)
        textstr = f'Mean: {mean_error:.2f}°C\nStd: {std_error:.2f}°C'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[1].text(0.95, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.suptitle('Spatial Error Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")


def generate_all_visualizations(dataset_h5: str, model_path: str, 
                               history_path: str, output_dir: str,
                               phase8_r2: Optional[float] = None,
                               phase8_rmse: Optional[float] = None):
    """
    Generate complete set of visualizations for Phase 8c.
    
    Args:
        dataset_h5: Path to HDF5 dataset
        model_path: Path to trained model
        history_path: Path to training history JSON
        output_dir: Output directory for plots
        phase8_r2: Phase 8 R² for comparison (optional)
        phase8_rmse: Phase 8 RMSE for comparison (optional)
    """
    print("\n" + "="*80)
    print("  GENERATING PHASE 8C VISUALIZATIONS")
    print("="*80 + "\n")
    
    viz = Phase8cVisualizer(output_dir=output_dir)
    
    # Load dataset
    print("Loading dataset...")
    with h5py.File(dataset_h5, 'r') as f:
        X_val = f['flir_frames'][-20:, ..., np.newaxis]  # Last 20 frames
        y_val = np.zeros((20, f['flir_frames'].shape[1], f['flir_frames'].shape[2]))
        
        # Build sparse ground truth
        for comp_idx in range(f['metadata'].attrs['n_components']):
            roi_mask = f['roi_masks'][comp_idx]
            for frame_idx in range(20):
                comp_temp = f['sand_temps'][-20 + frame_idx, comp_idx]
                y_val[frame_idx][roi_mask == 1] = comp_temp
        
        roi_masks_combined = np.any(f['roi_masks'][:], axis=0)
    
    # Load model and predict
    print("Loading model and generating predictions...")
    model = keras.models.load_model(model_path, custom_objects={'MaskedMSELoss': tf.keras.losses.MeanSquaredError()})
    y_pred = model.predict(X_val, verbose=0)
    
    # 1. Thermal field comparison (multiple frames)
    print("\nGenerating thermal field comparisons...")
    for idx in [0, 5, 10, 15, 19]:
        viz.plot_thermal_field_comparison(
            flir_frame=X_val[idx, :, :, 0],
            predicted_frame=y_pred[idx, :, :, 0],
            ground_truth_frame=y_val[idx],
            roi_mask=roi_masks_combined,
            frame_index=idx,
            save_name=f'thermal_field_comparison_frame_{idx}.png'
        )
    
    # 2. Component scatter
    print("\nGenerating component scatter plot...")
    y_true_roi = y_val[roi_masks_combined == 1]
    y_pred_roi = y_pred[:, roi_masks_combined == 1, 0]
    viz.plot_component_scatter(
        y_true_components=y_true_roi.reshape(-1, 1),
        y_pred_components=y_pred_roi.reshape(-1, 1),
        component_names=['All Components'],
        save_name='component_scatter_actual_vs_predicted.png'
    )
    
    # 3. Training curves
    print("\nGenerating training curves...")
    viz.plot_training_curves(history_path, 'training_curves.png')
    
    # 4. Phase 8 vs 8c comparison
    if phase8_r2 is not None and phase8_rmse is not None:
        print("\nGenerating Phase 8 vs 8c comparison...")
        from sklearn.metrics import r2_score, mean_squared_error
        
        y_true_flat = y_true_roi.flatten()
        y_pred_flat = y_pred_roi.flatten()
        phase8c_r2 = r2_score(y_true_flat, y_pred_flat)
        phase8c_rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        viz.plot_phase8_vs_phase8c_comparison(
            phase8_r2=phase8_r2,
            phase8_rmse=phase8_rmse,
            phase8c_r2=phase8c_r2,
            phase8c_rmse=phase8c_rmse,
            save_name='phase8_vs_phase8c_comparison.png'
        )
    
    # 5. Spatial error distribution
    print("\nGenerating spatial error distribution...")
    viz.plot_spatial_error_distribution(
        y_true_frame=y_val[10],
        y_pred_frame=y_pred[10, :, :, 0],
        roi_mask=roi_masks_combined,
        save_name='spatial_error_distribution.png'
    )
    
    print("\n" + "="*80)
    print("  ALL VISUALIZATIONS GENERATED ✓")
    print("="*80 + "\n")


def main():
    """
    Example usage: Generate visualizations for HBridge model.
    """
    generate_all_visualizations(
        dataset_h5="ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5",
        model_path="ml_model/cnn_thermal_modeling/models/unet_thermal_hbridge.keras",
        history_path="ml_model/cnn_thermal_modeling/models/training_history_hbridge.json",
        output_dir="ml_model/cnn_thermal_modeling/visualizations/hbridge",
        phase8_r2=0.72,  # From Phase 8 linear regression
        phase8_rmse=3.5
    )


if __name__ == "__main__":
    main()
