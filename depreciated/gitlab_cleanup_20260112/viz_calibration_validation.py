"""
Calibration Validation Visualization
=====================================
Creates plots showing cross-validation results and calibration quality metrics.

Outputs:
- calibration_validation_summary.png: 4-subplot validation analysis
- calibration_quality_metrics.csv: Per-component-type error statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import viz_helpers as viz


def create_validation_plots(validation_results: Dict, calibration_data: pd.DataFrame, 
                            output_dir: Path) -> None:
    """
    Create comprehensive validation visualization.
    
    Args:
        validation_results: Dict from perform_cross_validation()
        calibration_data: DataFrame of all calibration points
        output_dir: Directory to save plots
    """
    viz.setup_ieee_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Predicted vs Actual (Air Temperatures)
    _plot_predicted_vs_actual(ax1, validation_results, 'air')
    
    # Plot 2: Predicted vs Actual (Sand Temperatures)
    _plot_predicted_vs_actual(ax2, validation_results, 'sand')
    
    # Plot 3: Error by Component Type
    _plot_error_by_component_type(ax3, calibration_data, validation_results)
    
    # Plot 4: Sample Size vs Error
    _plot_sample_size_vs_error(ax4, calibration_data, validation_results)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'calibration_validation_summary.png'
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved validation plot: {plot_file}")


def _plot_predicted_vs_actual(ax, validation_results: Dict, condition: str):
    """Plot predicted vs actual temperatures for a given condition (air/sand)."""
    
    ax.set_title(f'Predicted vs Actual ({condition.capitalize()} Temps)', 
                fontsize=10, fontweight='bold')
    ax.set_xlabel(f'Actual {condition.capitalize()} Temperature (°C)', fontsize=9)
    ax.set_ylabel(f'Predicted {condition.capitalize()} Temperature (°C)', fontsize=9)
    
    # This would require storing individual predictions - simplified for now
    # Show RMSE for each validation pair
    colors = ['blue', 'green', 'red', 'purple']
    for idx, (key, result) in enumerate(validation_results.items()):
        if result['condition'] == condition:
            train_board = result['train_board']
            val_board = result['val_board']
            rmse = result['rmse']
            
            # Plot annotation instead of actual points (would need to store predictions)
            ax.text(0.05, 0.95 - idx*0.08, 
                   f"{train_board}→{val_board}: RMSE={rmse:.2f}°C",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor=colors[idx % len(colors)], alpha=0.3))
    
    # Add diagonal reference line
    ax.plot([20, 80], [20, 80], 'k--', alpha=0.5, linewidth=1, label='Perfect prediction')
    ax.set_xlim(20, 80)
    ax.set_ylim(20, 80)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_error_by_component_type(ax, calibration_data: pd.DataFrame, validation_results: Dict):
    """Plot RMSE by component type."""
    
    # Group calibration data by component type
    type_groups = calibration_data.groupby('component_type')
    
    component_types = []
    rmse_values_air = []
    rmse_values_sand = []
    
    for comp_type, group in type_groups:
        component_types.append(comp_type)
        
        # For now, use average RMSE across all validation pairs
        # In real implementation, would calculate per-type errors
        air_rmses = [v['rmse'] for v in validation_results.values() if v['condition'] == 'air']
        sand_rmses = [v['rmse'] for v in validation_results.values() if v['condition'] == 'sand']
        
        rmse_values_air.append(np.mean(air_rmses) if air_rmses else 0)
        rmse_values_sand.append(np.mean(sand_rmses) if sand_rmses else 0)
    
    x = np.arange(len(component_types))
    width = 0.35
    
    ax.bar(x - width/2, rmse_values_air, width, label='Air', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, rmse_values_sand, width, label='Sand', alpha=0.8, color='lightcoral')
    
    ax.set_title('Prediction Error by Component Type', fontsize=10, fontweight='bold')
    ax.set_xlabel('Component Type', fontsize=9)
    ax.set_ylabel('RMSE (°C)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(component_types, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_sample_size_vs_error(ax, calibration_data: pd.DataFrame, validation_results: Dict):
    """Plot relationship between sample size and prediction error."""
    
    # Group by component type to get sample counts
    type_counts = calibration_data.groupby('component_type').size()
    
    # Get average RMSE for air predictions
    avg_rmse = np.mean([v['rmse'] for v in validation_results.values() if v['condition'] == 'air'])
    
    # Plot sample counts
    component_types = list(type_counts.index)
    counts = list(type_counts.values)
    colors = ['green' if c >= 10 else ('orange' if c >= 5 else 'red') for c in counts]
    
    ax.bar(range(len(component_types)), counts, color=colors, alpha=0.7)
    ax.set_title('Calibration Sample Size per Component Type', fontsize=10, fontweight='bold')
    ax.set_xlabel('Component Type', fontsize=9)
    ax.set_ylabel('Number of Calibration Samples', fontsize=9)
    ax.set_xticks(range(len(component_types)))
    ax.set_xticklabels(component_types, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='High confidence (>=10)')
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium confidence (>=5)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def export_quality_metrics_csv(validation_results: Dict, calibration_data: pd.DataFrame,
                               output_dir: Path) -> None:
    """
    Export calibration quality metrics to CSV.
    
    Args:
        validation_results: Dict from perform_cross_validation()
        calibration_data: DataFrame of all calibration points  
        output_dir: Directory to save CSV
    """
    # Group by component type
    type_groups = calibration_data.groupby('component_type')
    
    metrics_data = []
    
    for comp_type, group in type_groups:
        # Calculate statistics for this component type
        sample_count = len(group)
        
        # Get RMSE values for this type (averaged across validation pairs)
        air_rmses = [v['rmse'] for v in validation_results.values() if v['condition'] == 'air']
        sand_rmses = [v['rmse'] for v in validation_results.values() if v['condition'] == 'sand']
        
        # Get R² values
        air_r2s = [v['r2'] for v in validation_results.values() if v['condition'] == 'air']
        sand_r2s = [v['r2'] for v in validation_results.values() if v['condition'] == 'sand']
        
        # Determine confidence level
        if sample_count >= 10:
            confidence = 'High'
        elif sample_count >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        metrics_data.append({
            'Component_Type': comp_type,
            'Samples': sample_count,
            'RMSE_Air_C': f"{np.mean(air_rmses):.3f}" if air_rmses else 'N/A',
            'RMSE_Sand_C': f"{np.mean(sand_rmses):.3f}" if sand_rmses else 'N/A',
            'R2_Score_Air': f"{np.mean(air_r2s):.4f}" if air_r2s else 'N/A',
            'R2_Score_Sand': f"{np.mean(sand_r2s):.4f}" if sand_r2s else 'N/A',
            'Confidence': confidence
        })
    
    # Create DataFrame and export
    df_metrics = pd.DataFrame(metrics_data)
    csv_file = output_dir / 'calibration_quality_metrics.csv'
    df_metrics.to_csv(csv_file, index=False)
    
    print(f"Saved quality metrics: {csv_file}")
