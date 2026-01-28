"""
3D Thermal Field Visualization

Creates 3D visualizations of thermal predictions:
1. 3D surface plots (height = temperature)
2. 3D scatter plots with color-coded temperatures
3. Interactive rotatable 3D views
4. Time-series 3D animations

Author: CNN Pipeline
Date: 2026-01-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import h5py


class Thermal3DVisualizer:
    """
    Create 3D visualizations of thermal fields.
    
    Provides multiple 3D rendering options for spatial thermal data.
    """
    
    def __init__(self, output_dir="results"):
        """
        Initialize 3D visualizer.
        
        Args:
            output_dir: Directory for output plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_3d_surface(self, thermal_frame, title="Thermal Field", save_name=None, 
                       elev=30, azim=45, stride=10):
        """
        Create 3D surface plot where height represents temperature.
        
        Args:
            thermal_frame: 2D array [H, W] of temperatures
            title: Plot title
            save_name: Filename to save (None = display only)
            elev: Elevation angle for viewing
            azim: Azimuth angle for viewing
            stride: Stride for mesh (higher = faster but coarser)
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        H, W = thermal_frame.shape
        
        # Create coordinate meshes
        X = np.arange(0, W, stride)
        Y = np.arange(0, H, stride)
        X, Y = np.meshgrid(X, Y)
        
        # Downsample thermal data to match mesh
        Z = thermal_frame[::stride, ::stride]
        
        # Normalize Z for better visualization (0-1 range)
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min()) if Z.max() > Z.min() else Z
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cm.hot,
                              linewidth=0, antialiased=True,
                              alpha=0.9, vmin=Z.min(), vmax=Z.max())
        
        # Add contour lines on the bottom
        ax.contour(X, Y, Z, zdir='z', offset=Z.min()-2, cmap=cm.hot, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('X Pixel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Pixel', fontsize=12, fontweight='bold')
        ax.set_zlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Temperature (°C)', fontsize=11, fontweight='bold')
        
        # Add temperature statistics as text
        stats_text = f"Min: {Z.min():.1f}°C\\nMax: {Z.max():.1f}°C\\nMean: {Z.mean():.1f}°C"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved 3D surface: {save_name}")
            plt.close()
        else:
            plt.show()
    
    def plot_3d_comparison(self, flir_frame, predicted_frame, ground_truth_frame, 
                          roi_mask, save_name=None, stride=10):
        """
        Create side-by-side 3D surface comparison.
        
        Args:
            flir_frame: FLIR input [H, W]
            predicted_frame: CNN prediction [H, W]
            ground_truth_frame: Ground truth (sparse) [H, W]
            roi_mask: ROI mask [H, W]
            save_name: Filename to save
            stride: Mesh stride
        """
        fig = plt.figure(figsize=(20, 6))
        
        # Common temperature range for z-axis
        valid_temps = predicted_frame[roi_mask > 0]
        z_min = min(flir_frame.min(), valid_temps.min())
        z_max = max(flir_frame.max(), valid_temps.max())
        
        H, W = flir_frame.shape
        X = np.arange(0, W, stride)
        Y = np.arange(0, H, stride)
        X, Y = np.meshgrid(X, Y)
        
        # Panel 1: FLIR Input
        ax1 = fig.add_subplot(131, projection='3d')
        Z1 = flir_frame[::stride, ::stride]
        surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.hot, linewidth=0,
                                antialiased=True, alpha=0.9,
                                vmin=z_min, vmax=z_max)
        ax1.set_title('FLIR Input\\n(Full Thermal Image)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        ax1.set_zlabel('Temp (°C)', fontsize=10)
        ax1.view_init(elev=30, azim=45)
        
        # Panel 2: CNN Prediction
        ax2 = fig.add_subplot(132, projection='3d')
        Z2 = predicted_frame[::stride, ::stride]
        surf2 = ax2.plot_surface(X, Y, Z2, cmap=cm.hot, linewidth=0,
                                antialiased=True, alpha=0.9,
                                vmin=z_min, vmax=z_max)
        ax2.set_title('CNN Predicted\\n(Sand Temperature)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_zlabel('Temp (°C)', fontsize=10)
        ax2.view_init(elev=30, azim=45)
        
        # Panel 3: Ground Truth (only show ROI regions)
        ax3 = fig.add_subplot(133, projection='3d')
        Z3 = ground_truth_frame[::stride, ::stride].copy()
        Z3[Z3 == 0] = np.nan  # Hide non-ROI regions
        surf3 = ax3.plot_surface(X, Y, Z3, cmap=cm.hot, linewidth=0,
                                antialiased=True, alpha=0.9,
                                vmin=z_min, vmax=z_max)
        ax3.set_title('Ground Truth\\n(Thermistor ROIs)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X', fontsize=10)
        ax3.set_ylabel('Y', fontsize=10)
        ax3.set_zlabel('Temp (°C)', fontsize=10)
        ax3.view_init(elev=30, azim=45)
        
        # Add single colorbar for all plots
        fig.colorbar(surf2, ax=[ax1, ax2, ax3], shrink=0.6, aspect=15, pad=0.1,
                    label='Temperature (°C)')
        
        plt.suptitle('3D Thermal Field Comparison', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved 3D comparison: {save_name}")
            plt.close()
        else:
            plt.show()
    
    def plot_3d_multiview(self, thermal_frame, title="Thermal Field", save_name=None, stride=10):
        """
        Create multiple viewing angles of same thermal field.
        
        Args:
            thermal_frame: 2D array [H, W] of temperatures
            title: Plot title
            save_name: Filename to save
            stride: Mesh stride
        """
        fig = plt.figure(figsize=(16, 12))
        
        H, W = thermal_frame.shape
        X = np.arange(0, W, stride)
        Y = np.arange(0, H, stride)
        X, Y = np.meshgrid(X, Y)
        Z = thermal_frame[::stride, ::stride]
        
        # Define 4 different viewing angles
        views = [
            (30, 45, "Isometric View"),
            (10, 90, "Front View"),
            (10, 0, "Side View"),
            (90, 0, "Top View")
        ]
        
        for idx, (elev, azim, view_title) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection='3d')
            
            surf = ax.plot_surface(X, Y, Z, cmap=cm.hot, linewidth=0,
                                  antialiased=True, alpha=0.9)
            
            ax.set_xlabel('X Pixel', fontsize=10)
            ax.set_ylabel('Y Pixel', fontsize=10)
            ax.set_zlabel('Temp (°C)', fontsize=10)
            ax.set_title(view_title, fontsize=12, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            
            # Add contour projection
            if idx != 4:  # Skip for top view
                ax.contour(X, Y, Z, zdir='z', offset=Z.min()-1, cmap=cm.hot, alpha=0.3)
        
        plt.suptitle(f'{title}\\nMultiple Viewing Angles', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved 3D multiview: {save_name}")
            plt.close()
        else:
            plt.show()
    
    def plot_3d_scatter_components(self, predictions, ground_truth, roi_masks, 
                                  component_names, frame_idx=0, save_name=None):
        """
        3D scatter plot showing component locations and temperatures.
        
        Args:
            predictions: Predicted temperatures [H, W]
            ground_truth: Actual temperatures [n_components]
            roi_masks: ROI masks [n_components, H, W]
            component_names: List of component names
            frame_idx: Frame index for title
            save_name: Filename to save
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract centroid and temperature for each component
        pred_points = []
        actual_points = []
        
        for comp_idx, comp_name in enumerate(component_names):
            if comp_idx >= len(roi_masks):
                continue
                
            mask = roi_masks[comp_idx]
            if mask.sum() == 0:
                continue
            
            # Get centroid
            Y_coords, X_coords = np.where(mask > 0)
            if len(Y_coords) == 0:
                continue
                
            centroid_x = X_coords.mean()
            centroid_y = Y_coords.mean()
            
            # Get predicted temperature at centroid
            pred_temp = predictions[int(centroid_y), int(centroid_x)]
            
            # Get actual temperature
            if comp_idx < len(ground_truth):
                actual_temp = ground_truth[comp_idx]
            else:
                continue
            
            pred_points.append([centroid_x, centroid_y, pred_temp, comp_name])
            actual_points.append([centroid_x, centroid_y, actual_temp, comp_name])
        
        if not pred_points:
            print("⚠️  No valid component points to plot")
            plt.close()
            return
        
        pred_arr = np.array([[p[0], p[1], p[2]] for p in pred_points])
        actual_arr = np.array([[p[0], p[1], p[2]] for p in actual_points])
        
        # Plot predicted (hollow circles)
        scatter_pred = ax.scatter(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2],
                                 c=pred_arr[:, 2], cmap='Reds', s=150, 
                                 alpha=0.7, marker='o', edgecolors='red',
                                 linewidth=2, label='CNN Predicted')
        
        # Plot actual (filled circles)
        scatter_actual = ax.scatter(actual_arr[:, 0], actual_arr[:, 1], actual_arr[:, 2],
                                   c=actual_arr[:, 2], cmap='Blues', s=100,
                                   alpha=0.9, marker='s', label='Thermistor Actual')
        
        # Connect predicted and actual with lines
        for i in range(len(pred_arr)):
            ax.plot([pred_arr[i, 0], actual_arr[i, 0]],
                   [pred_arr[i, 1], actual_arr[i, 1]],
                   [pred_arr[i, 2], actual_arr[i, 2]],
                   'k--', alpha=0.3, linewidth=1)
        
        # Labels and formatting
        ax.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title(f'Component-Level Predictions (Frame {frame_idx})\\nRed=Predicted, Blue=Actual, Dashed=Error',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left')
        ax.view_init(elev=25, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved 3D component scatter: {save_name}")
            plt.close()
        else:
            plt.show()


def demo_3d_visualizations(dataset_path, predictions_frame, output_dir):
    """
    Generate all 3D visualization types.
    
    Args:
        dataset_path: Path to HDF5 dataset
        predictions_frame: Predicted thermal frame [H, W]
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("CREATING 3D VISUALIZATIONS")
    print("="*80)
    
    viz = Thermal3DVisualizer(output_dir)
    
    # Load dataset
    with h5py.File(dataset_path, 'r') as f:
        flir_frames = f['flir_frames'][:]
        sand_temps = f['sand_temps'][:]
        roi_masks = f['roi_masks'][:]
        component_names = [name.decode('utf-8') for name in f['metadata']['component_names'][:]]
    
    # Select representative frame (middle of dataset)
    frame_idx = len(flir_frames) // 2
    flir = flir_frames[frame_idx]
    actual = sand_temps[frame_idx]
    
    # Build ground truth map
    H, W = flir.shape
    gt_map = np.zeros((H, W))
    for comp_idx in range(len(actual)):
        if comp_idx < len(roi_masks):
            mask = roi_masks[comp_idx]
            gt_map[mask > 0] = actual[comp_idx]
    
    # Create combined ROI mask
    combined_mask = np.max(roi_masks, axis=0)
    
    print(f"\nGenerating 3D visualizations for frame {frame_idx}...")
    
    # 1. Single 3D surface of prediction
    viz.plot_3d_surface(predictions_frame, 
                       title=f"CNN Predicted Thermal Field (Frame {frame_idx})",
                       save_name=f"3d_surface_predicted_frame{frame_idx}.png",
                       stride=5)
    
    # 2. 3D comparison (FLIR vs Predicted vs Actual)
    viz.plot_3d_comparison(flir, predictions_frame, gt_map, combined_mask,
                          save_name=f"3d_comparison_frame{frame_idx}.png",
                          stride=8)
    
    # 3. Multi-view of prediction
    viz.plot_3d_multiview(predictions_frame,
                         title=f"CNN Predicted Field (Frame {frame_idx})",
                         save_name=f"3d_multiview_frame{frame_idx}.png",
                         stride=8)
    
    # 4. Component scatter in 3D
    viz.plot_3d_scatter_components(predictions_frame, actual, roi_masks, 
                                  component_names, frame_idx,
                                  save_name=f"3d_scatter_components_frame{frame_idx}.png")
    
    print(f"\n✓ Created 4 types of 3D visualizations")


if __name__ == "__main__":
    print("3D Thermal Visualization Module")
    print("Import this module to use 3D plotting functions")
