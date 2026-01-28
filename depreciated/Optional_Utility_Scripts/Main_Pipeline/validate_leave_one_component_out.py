#!/usr/bin/env python3
"""
===============================================================================
LEAVE-ONE-COMPONENT-OUT (LOCO) VALIDATION
===============================================================================
Tests whether the CNN learns genuine thermal physics or just copies thermistor
inputs by masking out one component's thermistor channel at a time and predicting
its temperature from FLIR + other components' context.

If LOCO performance >> naive baselines, the model understands thermal coupling.
===============================================================================
"""

import numpy as np
import h5py
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Import existing infrastructure
from phase8c_spatial_cnn import SpatialCNNTrainer


class LOCOValidator:
    """Leave-One-Component-Out validation for thermal CNN."""
    
    def __init__(self, model_path, dataset_path, output_dir=None, verbose=True):
        """
        Initialize validator.
        
        Args:
            model_path: Path to trained .keras model
            dataset_path: Path to HDF5 dataset
            output_dir: Where to save results (auto-generated if None)
            verbose: Print progress messages
        """
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        
        # Auto-generate output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"ml_model/cnn_thermal_modeling/results/loco_validation_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if self.verbose:
            print(f"\n{'='*80}")
            print("LEAVE-ONE-COMPONENT-OUT VALIDATION")
            print(f"{'='*80}\n")
            print(f"Model: {self.model_path.name}")
            print(f"Dataset: {self.dataset_path.name}")
            print(f"Output: {self.output_dir}\n")
        
        self.trainer = SpatialCNNTrainer(verbose=False)
        self.trainer.load_model(str(self.model_path))
        
        # Load dataset metadata
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load component names and dataset structure."""
        with h5py.File(self.dataset_path, 'r') as f:
            self.component_names = [name.decode() if isinstance(name, bytes) else name 
                                   for name in f['metadata']['component_names'][:]]
            self.n_components = len(self.component_names)
            self.roi_masks = f['roi_masks'][:]
            
            # Get normalization stats
            if hasattr(self.trainer, 'normalization_stats') and self.trainer.normalization_stats:
                stats = self.trainer.normalization_stats
                self.flir_min = stats['flir_min']
                self.flir_max = stats['flir_max']
                self.sand_min = stats['sand_min']
                self.sand_max = stats['sand_max']
            else:
                # Compute from dataset
                flir_data = f['flir_frames'][:]
                sand_data = f['sand_temps'][:]
                self.flir_min = float(np.min(flir_data))
                self.flir_max = float(np.max(flir_data))
                self.sand_min = float(np.min(sand_data))
                self.sand_max = float(np.max(sand_data))
        
        if self.verbose:
            print(f"Components: {self.n_components}")
            print(f"  {', '.join(self.component_names)}\n")
    
    def _normalize(self, data, min_val, max_val):
        """Min-max normalize to [0, 1]."""
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    def _denormalize(self, data, min_val, max_val):
        """Denormalize from [0, 1] back to original scale."""
        return data * (max_val - min_val) + min_val
    
    def _build_input_masked(self, flir_frames, sand_temps, mask_component_idx=None):
        """
        Build 2-channel input with optional component masking.
        
        Args:
            flir_frames: Raw FLIR data [n_frames, H, W]
            sand_temps: Thermistor temps [n_frames, n_components]
            mask_component_idx: Component index to zero out (None = no masking)
        
        Returns:
            X: Input array [n_frames, H, W, 2]
        """
        n_frames, H, W = flir_frames.shape
        X = np.zeros((n_frames, H, W, 2), dtype=np.float32)
        
        # Channel 0: FLIR (normalized)
        for i in range(n_frames):
            X[i, :, :, 0] = self._normalize(flir_frames[i], self.flir_min, self.flir_max)
        
        # Channel 1: Thermistor temps embedded at ROI locations (normalized)
        for i in range(n_frames):
            for comp_idx in range(self.n_components):
                # Skip masked component
                if comp_idx == mask_component_idx:
                    continue
                
                roi_mask = self.roi_masks[comp_idx]
                comp_temp = sand_temps[i, comp_idx]
                comp_temp_norm = self._normalize(comp_temp, self.sand_min, self.sand_max)
                X[i, :, :, 1][roi_mask == 1] = comp_temp_norm
        
        return X
    
    def _extract_roi_predictions(self, predictions, sand_temps, component_idx):
        """
        Extract predictions at specific component's ROI locations.
        
        Args:
            predictions: Model output [n_frames, H, W]
            sand_temps: Ground truth [n_frames, n_components]
            component_idx: Which component to extract
        
        Returns:
            actual, predicted: Arrays of temperatures at ROI
        """
        roi_mask = self.roi_masks[component_idx]
        
        # Debug: Check mask
        mask_sum = np.sum(roi_mask)
        if self.verbose:
            print(f"mask_pixels={int(mask_sum)}, ", end="", flush=True)
        
        if mask_sum == 0:
            if self.verbose:
                print("empty_mask ", end="", flush=True)
            return np.array([]), np.array([])
        
        actual = []
        predicted = []
        
        for frame_idx in range(len(predictions)):
            actual_temp = sand_temps[frame_idx, component_idx]
            pred_temp = predictions[frame_idx][roi_mask > 0].mean()
            
            if not (np.isnan(actual_temp) or np.isnan(pred_temp)):
                actual.append(actual_temp)
                predicted.append(pred_temp)
        
        if self.verbose and len(actual) == 0:
            print(f"all_nan ", end="", flush=True)
        
        return np.array(actual), np.array(predicted)
    
    def _compute_naive_baseline(self, sand_temps, component_idx):
        """
        Compute naive baseline: average of all OTHER components' temps.
        
        Args:
            sand_temps: [n_frames, n_components]
            component_idx: Target component
        
        Returns:
            baseline_temps: [n_frames]
        """
        other_indices = [i for i in range(self.n_components) if i != component_idx]
        return np.mean(sand_temps[:, other_indices], axis=1)
    
    def validate_component(self, component_idx, n_samples=100):
        """
        Run LOCO validation for one component.
        
        Args:
            component_idx: Index of component to mask
            n_samples: Number of timestamp samples to evaluate (for speed)
        
        Returns:
            results: Dict with metrics and predictions
        """
        comp_name = self.component_names[component_idx]
        
        if self.verbose:
            print(f"  [{component_idx+1}/{self.n_components}] {comp_name}...", end=" ", flush=True)
        
        # Load subset of data (use frame_indices mapping)
        with h5py.File(self.dataset_path, 'r') as f:
            all_flir = f['flir_frames'][:]
            sand_temps = f['sand_temps'][:n_samples]
            frame_indices = f['frame_indices'][:n_samples]
            
            # Map timestamps to FLIR frames
            flir_frames = all_flir[frame_indices]
        
        # Build masked input (target component thermistor = 0)
        X_masked = self._build_input_masked(flir_frames, sand_temps, 
                                            mask_component_idx=component_idx)
        
        # Predict (this can be slow on CPU)
        if self.verbose:
            print(f"predicting...", end=" ", flush=True)
        predictions = self.trainer.model.predict(X_masked, verbose=0).squeeze()
        predictions = self._denormalize(predictions, self.sand_min, self.sand_max)
        
        # Extract at target ROI
        actual, predicted = self._extract_roi_predictions(predictions, sand_temps, component_idx)
        
        if len(actual) == 0:
            if self.verbose:
                print("SKIP (empty ROI)")
            return None
        
        # Compute metrics
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        # Naive baseline
        baseline = self._compute_naive_baseline(sand_temps, component_idx)
        baseline_rmse = np.sqrt(mean_squared_error(actual, baseline[:len(actual)]))
        
        if self.verbose:
            print(f"R²={r2:.3f}, RMSE={rmse:.2f}°C (baseline={baseline_rmse:.2f}°C)")
        
        return {
            'component': comp_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'baseline_rmse': baseline_rmse,
            'n_samples': len(actual),
            'actual': actual,
            'predicted': predicted
        }
    
    def run_full_validation(self, n_samples=100):
        """
        Run LOCO validation on all components.
        
        Args:
            n_samples: Timestamp samples to evaluate per component
        
        Returns:
            results: List of per-component results
        """
        if self.verbose:
            print("Running LOCO validation:\n")
        
        results = []
        for i in range(self.n_components):
            result = self.validate_component(i, n_samples=n_samples)
            if result is not None:
                results.append(result)
        
        if self.verbose:
            print(f"\n✓ Validated {len(results)}/{self.n_components} components\n")
        
        # Save results
        self._save_results(results)
        self._generate_visualizations(results)
        
        return results
    
    def _save_results(self, results):
        """Save summary metrics to text file."""
        summary_path = self.output_dir / "summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write("LEAVE-ONE-COMPONENT-OUT VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {self.model_path.name}\n")
            f.write(f"Dataset: {self.dataset_path.name}\n")
            f.write(f"Validated: {len(results)}/{self.n_components} components\n\n")
            
            f.write("PER-COMPONENT RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Component':<25} {'R²':>8} {'RMSE':>10} {'Baseline':>10} {'Improvement':>12}\n")
            f.write("-"*80 + "\n")
            
            for res in sorted(results, key=lambda x: x['r2'], reverse=True):
                improvement = ((res['baseline_rmse'] - res['rmse']) / res['baseline_rmse']) * 100
                f.write(f"{res['component']:<25} {res['r2']:>8.3f} "
                       f"{res['rmse']:>9.2f}°C {res['baseline_rmse']:>9.2f}°C "
                       f"{improvement:>11.1f}%\n")
            
            f.write("-"*80 + "\n\n")
            
            # Overall stats
            avg_r2 = np.mean([r['r2'] for r in results])
            avg_rmse = np.mean([r['rmse'] for r in results])
            avg_baseline = np.mean([r['baseline_rmse'] for r in results])
            avg_improvement = ((avg_baseline - avg_rmse) / avg_baseline) * 100
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Average R²: {avg_r2:.3f}\n")
            f.write(f"  Average RMSE: {avg_rmse:.2f}°C\n")
            f.write(f"  Naive Baseline RMSE: {avg_baseline:.2f}°C\n")
            f.write(f"  Improvement over baseline: {avg_improvement:.1f}%\n\n")
            
            if avg_improvement > 50:
                f.write("✓ Model shows strong thermal physics understanding\n")
            elif avg_improvement > 20:
                f.write("⚠ Model shows moderate thermal understanding\n")
            else:
                f.write("✗ Model may be overfitting to thermistor inputs\n")
        
        if self.verbose:
            print(f"✓ Summary saved: {summary_path}")
    
    def _generate_visualizations(self, results):
        """Create plots for LOCO validation."""
        
        # 1. Scatter plot grid (2x2 or 3x3 depending on component count)
        n_results = len(results)
        if n_results <= 4:
            rows, cols = 2, 2
        elif n_results <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, res in enumerate(results[:rows*cols]):
            ax = axes[idx]
            
            actual = res['actual']
            predicted = res['predicted']
            
            ax.scatter(actual, predicted, alpha=0.5, s=10, edgecolors='k', linewidth=0.3)
            
            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
            
            ax.set_xlabel('Actual (°C)', fontsize=8)
            ax.set_ylabel('Predicted (°C)', fontsize=8)
            ax.set_title(f"{res['component']}\nR²={res['r2']:.3f}, RMSE={res['rmse']:.2f}°C", 
                        fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        scatter_path = self.output_dir / "component_scatter_grid.png"
        plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✓ Scatter grid saved: {scatter_path}")
        
        # 2. Bar chart: RMSE comparison (LOCO vs Baseline)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = [r['component'] for r in results]
        loco_rmse = [r['rmse'] for r in results]
        baseline_rmse = [r['baseline_rmse'] for r in results]
        
        x = np.arange(len(components))
        width = 0.35
        
        ax.bar(x - width/2, loco_rmse, width, label='LOCO CNN', color='steelblue')
        ax.bar(x + width/2, baseline_rmse, width, label='Naive Baseline', color='lightcoral')
        
        ax.set_xlabel('Component', fontsize=11)
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_title('LOCO Validation: CNN vs Naive Baseline', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        bar_path = self.output_dir / "rmse_comparison.png"
        plt.savefig(bar_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✓ RMSE comparison saved: {bar_path}")


def main():
    """CLI interface for LOCO validation."""
    
    # Find most recent model
    results_dir = Path("ml_model/cnn_thermal_modeling/results")
    
    if not results_dir.exists():
        print("✗ ERROR: No results directory found")
        print(f"  Expected: {results_dir}")
        return
    
    # Find all analysis directories
    analysis_dirs = sorted([d for d in results_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('analysis_')],
                          reverse=True)
    
    if not analysis_dirs:
        print("✗ ERROR: No analysis directories found")
        return
    
    # Find model in most recent analysis
    model_path = None
    for analysis_dir in analysis_dirs:
        candidate = analysis_dir / "unet_hbridge.keras"
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        print("✗ ERROR: No trained model found")
        print(f"  Searched in: {', '.join([str(d) for d in analysis_dirs[:3]])}")
        return
    
    # Find dataset
    dataset_path = Path("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
    if not dataset_path.exists():
        print(f"✗ ERROR: Dataset not found: {dataset_path}")
        return
    
    # Run validation (use fewer samples for speed on CPU)
    validator = LOCOValidator(model_path, dataset_path, verbose=True)
    results = validator.run_full_validation(n_samples=50)  # Reduced from 100 for speed
    
    # Print summary
    print("\nVALIDATION SUMMARY:")
    print("="*80)
    avg_r2 = np.mean([r['r2'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_baseline = np.mean([r['baseline_rmse'] for r in results])
    improvement = ((avg_baseline - avg_rmse) / avg_baseline) * 100
    
    print(f"  Average R²: {avg_r2:.3f}")
    print(f"  Average RMSE: {avg_rmse:.2f}°C (baseline: {avg_baseline:.2f}°C)")
    print(f"  Improvement: {improvement:.1f}%\n")
    
    if improvement > 50:
        print("✓ Model demonstrates strong thermal physics understanding!")
        print("  → Can predict component temps from FLIR + other components")
    elif improvement > 20:
        print("⚠ Model shows moderate understanding")
        print("  → Better than naive average but room for improvement")
    else:
        print("✗ Model may be overfitting to thermistor inputs")
        print("  → Consider FLIR-only architecture or more training data")
    
    print(f"\n✓ Results saved to: {validator.output_dir}\n")


if __name__ == "__main__":
    main()
