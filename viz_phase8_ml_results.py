"""
===============================================================================
VISUALIZATION: PHASE 8 ML RESULTS
===============================================================================
Visualizations for machine learning thermal prediction model (Phase 8).

Creates IEEE-format plots showing:
    - Predicted vs Actual scatter plot with component type color coding
    - 1:1 reference line
    - R² performance metric annotation
    - Residual analysis (optional)

Plot Style:
    - Matches existing thermal_post_processing visualization standards
    - IEEE publication format
    - Consistent fonts, colors, grid styling

Created: January 7, 2026
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict


def create_predicted_vs_actual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_types: np.ndarray,
    output_dir: str,
    pcb_name: str = "Board",
    component_names: np.ndarray = None,
    metrics: Dict = None
) -> str:
    """
    Create predicted vs actual scatter plot for ML model evaluation.
    
    Generates IEEE-format scatter plot with:
        - Points colored by component type
        - 1:1 reference line (perfect prediction)
        - R² annotation
        - Legend showing component types
        - Consistent styling with existing plots
    
    Args:
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_types: Component type labels [n_samples]
        output_dir: Directory for output files
        pcb_name: PCB name for plot title and filename
        component_names: Optional component names for hover/labels
        metrics: Optional metrics dict with R², RMSE, MAE
    
    Returns:
        Path to saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colors for component types (matching existing thermal plots)
    type_colors = {
        'IC': '#1f77b4',           # Blue
        'Resistor': '#ff7f0e',     # Orange
        'PowerSupply': '#2ca02c',  # Green
        'LED': '#d62728',          # Red
        'Connector': '#9467bd',    # Purple
        'Capacitor': '#8c564b',    # Brown
        'Inductor': '#e377c2',     # Pink
        'Diode': '#7f7f7f',        # Gray
        'Other': '#bcbd22'         # Yellow-green
    }
    
    # Create figure (IEEE format: 8x6 inches)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique component types
    unique_types = np.unique(component_types)
    
    # Plot each component type separately for legend
    for comp_type in unique_types:
        mask = component_types == comp_type
        color = type_colors.get(comp_type, '#000000')
        
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            c=color,
            label=comp_type,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Calculate axis limits with some padding
    all_temps = np.concatenate([y_true, y_pred])
    temp_min = np.min(all_temps)
    temp_max = np.max(all_temps)
    temp_range = temp_max - temp_min
    axis_min = temp_min - 0.1 * temp_range
    axis_max = temp_max + 0.1 * temp_range
    
    # Plot 1:1 reference line (perfect prediction)
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        'k--',
        linewidth=1.5,
        alpha=0.5,
        label='Perfect Prediction (1:1)',
        zorder=0
    )
    
    # Add R² annotation if metrics provided
    if metrics and 'R²' in metrics:
        r_squared = metrics['R²']
        rmse = metrics.get('RMSE', np.nan)
        mae = metrics.get('MAE', np.nan)
        
        # Position text box in upper left
        textstr = f'$R^2$ = {r_squared:.4f}\n'
        if not np.isnan(rmse):
            textstr += f'RMSE = {rmse:.2f} °C\n'
        if not np.isnan(mae):
            textstr += f'MAE = {mae:.2f} °C'
        
        # Add text box with white background
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=props
        )
    
    # Formatting
    ax.set_xlabel('Actual Sand ΔT (°C)', fontsize=12)
    ax.set_ylabel('Predicted Sand ΔT (°C)', fontsize=12)
    ax.set_title(f'ML Model: Predicted vs Actual ({pcb_name})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Equal aspect ratio for fair comparison
    ax.set_aspect('equal', adjustable='box')
    
    # Legend
    ax.legend(
        loc='lower right',
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{pcb_name}_ml_predicted_vs_actual.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {plot_path}")
    
    return str(plot_path)


def create_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_types: np.ndarray,
    output_dir: str,
    pcb_name: str = "Board"
) -> str:
    """
    Create residual plot for ML model diagnostics.
    
    Plots residuals (actual - predicted) vs predicted values to check for:
        - Homoscedasticity (constant variance)
        - Bias patterns
        - Outliers
    
    Args:
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_types: Component type labels [n_samples]
        output_dir: Directory for output files
        pcb_name: PCB name for plot title and filename
    
    Returns:
        Path to saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Define colors for component types
    type_colors = {
        'IC': '#1f77b4',
        'Resistor': '#ff7f0e',
        'PowerSupply': '#2ca02c',
        'LED': '#d62728',
        'Connector': '#9467bd',
        'Capacitor': '#8c564b',
        'Inductor': '#e377c2',
        'Diode': '#7f7f7f',
        'Other': '#bcbd22'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot residuals for each component type
    unique_types = np.unique(component_types)
    for comp_type in unique_types:
        mask = component_types == comp_type
        color = type_colors.get(comp_type, '#000000')
        
        ax.scatter(
            y_pred[mask],
            residuals[mask],
            c=color,
            label=comp_type,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Add horizontal line at y=0 (perfect prediction)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Predicted Sand ΔT (°C)', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted) (°C)', fontsize=12)
    ax.set_title(f'ML Model: Residual Analysis ({pcb_name})', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(
        loc='best',
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{pcb_name}_ml_residual_plot.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved residual plot: {plot_path}")
    
    return str(plot_path)


def create_regression_feature_plot(
    flir_delta_t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_types: np.ndarray,
    output_dir: str,
    pcb_name: str = "Board",
    model_coefficients: Dict = None
) -> str:
    """
    Create feature plot showing regression relationship between FLIR delta T and Sand delta T.
    
    This plot visualizes the actual linear regression model by showing:
        - Scatter: Actual measurements (x=FLIR delta T, y=Sand delta T)
        - Lines: Model predictions for each component type
        - Shows how the model uses FLIR delta T to predict Sand delta T
    
    This is different from predicted vs actual - it shows the MODEL'S LEARNED RELATIONSHIP.
    
    Args:
        flir_delta_t: FLIR air delta T values [n_samples]
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_types: Component type labels [n_samples]
        output_dir: Directory for output files
        pcb_name: PCB name for plot title and filename
        model_coefficients: Optional dict with model intercept and coefficients
    
    Returns:
        Path to saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colors for component types
    type_colors = {
        'IC': '#1f77b4',
        'Resistor': '#ff7f0e',
        'PowerSupply': '#2ca02c',
        'LED': '#d62728',
        'Connector': '#9467bd',
        'Capacitor': '#8c564b',
        'Inductor': '#e377c2',
        'Diode': '#7f7f7f',
        'Other': '#bcbd22'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get unique component types
    unique_types = np.unique(component_types)
    
    # Plot actual measurements as scatter points
    for comp_type in unique_types:
        mask = component_types == comp_type
        color = type_colors.get(comp_type, '#000000')
        
        ax.scatter(
            flir_delta_t[mask],
            y_true[mask],
            c=color,
            label=f'{comp_type} (actual)',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            marker='o'
        )
    
    # Plot model predictions as regression lines
    # Create smooth x-axis for line plotting
    flir_min = flir_delta_t.min()
    flir_max = flir_delta_t.max()
    flir_range = flir_max - flir_min
    flir_line = np.linspace(flir_min - 0.1*flir_range, flir_max + 0.1*flir_range, 100)
    
    # For each component type, plot the regression line
    for comp_type in unique_types:
        mask = component_types == comp_type
        if not np.any(mask):
            continue
        
        color = type_colors.get(comp_type, '#000000')
        
        # Get predictions for this component type at various FLIR delta T values
        # Use the actual model predictions for points we have
        flir_vals = flir_delta_t[mask]
        pred_vals = y_pred[mask]
        
        # Sort for line plotting
        sort_idx = np.argsort(flir_vals)
        flir_sorted = flir_vals[sort_idx]
        pred_sorted = pred_vals[sort_idx]
        
        # Plot regression line for this component type
        ax.plot(
            flir_sorted,
            pred_sorted,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            linestyle='-',
            label=f'{comp_type} (model)'
        )
    
    # Formatting
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sand Embedded ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Regression Model: FLIR Air → Sand Prediction ({pcb_name})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend with two columns (actual and model predictions)
    ax.legend(
        loc='upper left',
        fontsize=9,
        framealpha=0.95,
        edgecolor='gray',
        ncol=2
    )
    
    # Add annotation explaining the plot
    annotation_text = (
        'Circles: Actual measurements\n'
        'Lines: Model predictions\n'
        'Shows how model uses FLIR to predict Sand temps'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
    ax.text(
        0.98, 0.02,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{pcb_name}_ml_regression_feature_plot.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved regression feature plot: {plot_path}")
    
    return str(plot_path)


def create_component_type_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_types: np.ndarray,
    output_dir: str,
    pcb_name: str = "Board"
) -> str:
    """
    Create bar plot comparing model performance by component type.
    
    Shows RMSE and MAE for each component type to identify which types
    are well-predicted vs poorly-predicted.
    
    Args:
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_types: Component type labels [n_samples]
        output_dir: Directory for output files
        pcb_name: PCB name for plot title and filename
    
    Returns:
        Path to saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics per component type
    unique_types = np.unique(component_types)
    type_metrics = []
    
    for comp_type in unique_types:
        mask = component_types == comp_type
        residuals = y_true[mask] - y_pred[mask]
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        count = np.sum(mask)
        
        type_metrics.append({
            'type': comp_type,
            'RMSE': rmse,
            'MAE': mae,
            'count': count
        })
    
    df_metrics = pd.DataFrame(type_metrics)
    df_metrics = df_metrics.sort_values('RMSE', ascending=False)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE bar plot
    bars1 = ax1.bar(
        range(len(df_metrics)),
        df_metrics['RMSE'],
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )
    ax1.set_xlabel('Component Type', fontsize=11)
    ax1.set_ylabel('RMSE (°C)', fontsize=11)
    ax1.set_title('Root Mean Squared Error by Type', fontsize=11)
    ax1.set_xticks(range(len(df_metrics)))
    ax1.set_xticklabels(df_metrics['type'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars1, df_metrics['count'])):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # MAE bar plot
    bars2 = ax2.bar(
        range(len(df_metrics)),
        df_metrics['MAE'],
        color='coral',
        alpha=0.7,
        edgecolor='black'
    )
    ax2.set_xlabel('Component Type', fontsize=11)
    ax2.set_ylabel('MAE (°C)', fontsize=11)
    ax2.set_title('Mean Absolute Error by Type', fontsize=11)
    ax2.set_xticks(range(len(df_metrics)))
    ax2.set_xticklabels(df_metrics['type'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars2, df_metrics['count'])):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Overall title
    fig.suptitle(f'ML Model Performance by Component Type ({pcb_name})', fontsize=12, y=1.02)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{pcb_name}_ml_type_comparison.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved type comparison: {plot_path}")
    
    return str(plot_path)


def create_cross_board_validation_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_types: np.ndarray,
    train_pcb: str,
    test_pcb: str,
    metrics: Dict,
    output_dir: str,
    pcb_name: str = "Board"
) -> str:
    """
    Create cross-board validation plot showing model generalization.
    
    Shows how well a model trained on one board predicts temperatures on another board.
    Uses same format as predicted_vs_actual but emphasizes cross-board validation.
    
    Args:
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_types: Component type labels [n_samples]
        train_pcb: Name of training board
        test_pcb: Name of test/validation board
        metrics: Metrics dict with R², RMSE, MAE
        output_dir: Directory for output files
        pcb_name: PCB name for filename
    
    Returns:
        Path to saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colors for component types
    type_colors = {
        'IC': '#1f77b4',
        'Resistor': '#ff7f0e',
        'PowerSupply': '#2ca02c',
        'LED': '#d62728',
        'Connector': '#9467bd',
        'Capacitor': '#8c564b',
        'Inductor': '#e377c2',
        'Diode': '#7f7f7f',
        'Other': '#bcbd22'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each component type
    unique_types = np.unique(component_types)
    for comp_type in unique_types:
        mask = component_types == comp_type
        color = type_colors.get(comp_type, '#000000')
        
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            c=color,
            label=comp_type,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Calculate axis limits
    all_temps = np.concatenate([y_true, y_pred])
    temp_min = np.min(all_temps)
    temp_max = np.max(all_temps)
    temp_range = temp_max - temp_min
    axis_min = temp_min - 0.1 * temp_range
    axis_max = temp_max + 0.1 * temp_range
    
    # Plot 1:1 reference line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        'k--',
        linewidth=1.5,
        alpha=0.5,
        label='Perfect Prediction (1:1)',
        zorder=0
    )
    
    # Add metrics annotation
    r_squared = metrics['R²']
    rmse = metrics['RMSE']
    mae = metrics['MAE']
    
    textstr = f'$R^2$ = {r_squared:.4f}\n'
    textstr += f'RMSE = {rmse:.2f} °C\n'
    textstr += f'MAE = {mae:.2f} °C\n'
    textstr += f'\nTrain: {train_pcb}\n'
    textstr += f'Test: {test_pcb}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    
    # Formatting
    ax.set_xlabel('Actual Sand ΔT (°C)', fontsize=12)
    ax.set_ylabel('Predicted Sand ΔT (°C)', fontsize=12)
    ax.set_title(f'Cross-Board Validation: {train_pcb} → {test_pcb}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend
    ax.legend(
        loc='lower right',
        fontsize=9,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{pcb_name}_ml_validation_plot.png"
    plot_path = output_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved validation plot: {plot_path}")
    
    return str(plot_path)


def create_validation_percent_error_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_names: np.ndarray,
    component_types: np.ndarray,
    test_pcb: str,
    output_dir: str
) -> dict:
    """
    Create comprehensive percent error analysis plots for cross-board validation.
    
    Generates 4 plots analyzing prediction accuracy:
        1. Per-component percent error bar chart (sorted by error magnitude)
        2. Percent error distribution histogram
        3. Percent error vs temperature magnitude scatter
        4. Component type error box plot
    
    Args:
        y_true: Actual sand delta T values [n_samples]
        y_pred: Predicted sand delta T values [n_samples]
        component_names: Component name labels [n_samples]
        component_types: Component type labels [n_samples]
        test_pcb: PCB name for plot titles
        output_dir: Directory for output files
    
    Returns:
        Dictionary with paths to all generated plot files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate percent errors (matching the logic in phase8_ml_training.py)
    MIN_DELTA_T_FOR_PERCENT = 1.0  # Only calculate % error for delta T >= 1°C
    
    percent_errors = np.zeros_like(y_true)
    abs_percent_errors = np.zeros_like(y_true)
    valid_for_percent = np.abs(y_true) >= MIN_DELTA_T_FOR_PERCENT
    
    if np.any(valid_for_percent):
        percent_errors[valid_for_percent] = ((y_pred[valid_for_percent] - y_true[valid_for_percent]) / y_true[valid_for_percent]) * 100
        abs_percent_errors[valid_for_percent] = np.abs(percent_errors[valid_for_percent])
    
    # For low delta T components, mark as NaN
    percent_errors[~valid_for_percent] = np.nan
    abs_percent_errors[~valid_for_percent] = np.nan
    
    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(percent_errors)
    
    if not np.any(valid_mask):
        print(f"  Warning: All components have ΔT < {MIN_DELTA_T_FOR_PERCENT}°C - skipping percent error plots")
        return {}
    
    # Use only valid data for plotting
    percent_errors_valid = percent_errors[valid_mask]
    abs_percent_errors_valid = abs_percent_errors[valid_mask]
    component_names_valid = component_names[valid_mask]
    component_types_valid = component_types[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Define colors for component types
    type_colors = {
        'IC': '#1f77b4',
        'Resistor': '#ff7f0e',
        'PowerSupply': '#2ca02c',
        'LED': '#d62728',
        'Connector': '#9467bd',
        'Capacitor': '#8c564b',
        'Inductor': '#e377c2',
        'Diode': '#7f7f7f',
        'LDO': '#bcbd22',
        'Other': '#17becf'
    }
    
    plot_paths = {}
    
    # =========================================================================
    # PLOT 1: Per-Component Percent Error Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by absolute error (descending)
    sort_idx = np.argsort(abs_percent_errors_valid)[::-1]
    sorted_names = component_names_valid[sort_idx]
    sorted_errors = percent_errors_valid[sort_idx]
    sorted_types = component_types_valid[sort_idx]
    sorted_actuals = y_true_valid[sort_idx]
    
    # Create color array
    colors = [type_colors.get(t, '#000000') for t in sorted_types]
    
    # Create bar chart
    x_pos = np.arange(len(sorted_names))
    bars = ax.bar(x_pos, sorted_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add horizontal threshold lines
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='±10% threshold')
    ax.axhline(y=-10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='±20% threshold')
    ax.axhline(y=-20, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Add actual delta T as text on bars
    for i, (bar, actual_t) in enumerate(zip(bars, sorted_actuals)):
        height = bar.get_height()
        y_pos = height + (2 if height > 0 else -5)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{actual_t:.1f}°C',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=7, rotation=90, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Component (sorted by error magnitude)', fontsize=12)
    ax.set_ylabel('Percent Error (%)', fontsize=12)
    ax.set_title(f'Prediction Percent Error by Component ({test_pcb})', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=9)
    
    # Add stats box
    mean_err = np.mean(percent_errors_valid)
    median_err = np.median(percent_errors_valid)
    mape = np.mean(abs_percent_errors_valid)
    within_10 = np.sum(abs_percent_errors_valid <= 10)
    within_20 = np.sum(abs_percent_errors_valid <= 20)
    
    textstr = f'Mean: {mean_err:+.1f}%\n'
    textstr += f'Median: {median_err:+.1f}%\n'
    textstr += f'MAPE: {mape:.1f}%\n'
    textstr += f'Within ±10%: {within_10}/{len(y_true_valid)}\n'
    textstr += f'Within ±20%: {within_20}/{len(y_true_valid)}\n'
    textstr += f'(Analyzing {len(y_true_valid)}/{len(y_true)} components)'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    plot1_path = output_path / f"{test_pcb}_ml_percent_error_by_component.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['per_component'] = str(plot1_path)
    print(f"  Saved percent error by component: {plot1_path}")
    
    # =========================================================================
    # PLOT 2: Percent Error Distribution Histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(percent_errors_valid, bins=20, color='steelblue', 
                                alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Overlay normal distribution
    from scipy import stats
    mu, sigma = np.mean(percent_errors_valid), np.std(percent_errors_valid)
    x = np.linspace(percent_errors_valid.min(), percent_errors_valid.max(), 100)
    y = stats.norm.pdf(x, mu, sigma) * len(percent_errors_valid) * (bins[1] - bins[0])
    ax.plot(x, y, 'r-', linewidth=2, label=f'Normal (μ={mu:.1f}%, σ={sigma:.1f}%)')
    
    # Add vertical lines for thresholds
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Zero error')
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(x=-10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='±10%')
    ax.axvline(x=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(x=-20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='±20%')
    
    # Formatting
    ax.set_xlabel('Percent Error (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Percent Error Distribution ({test_pcb})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add statistics box
    textstr = f'Mean: {mu:+.1f}%\n'
    textstr += f'Median: {np.median(percent_errors_valid):+.1f}%\n'
    textstr += f'Std Dev: {sigma:.1f}%\n'
    textstr += f'MAPE: {mape:.1f}%\n'
    textstr += f'Samples: {len(percent_errors_valid)}/{len(y_true)}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    plot2_path = output_path / f"{test_pcb}_ml_percent_error_distribution.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['distribution'] = str(plot2_path)
    print(f"  Saved percent error distribution: {plot2_path}")
    
    # =========================================================================
    # PLOT 3: Percent Error vs Temperature Magnitude Scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot by component type (using valid data)
    unique_types = np.unique(component_types_valid)
    for comp_type in unique_types:
        mask = component_types_valid == comp_type
        color = type_colors.get(comp_type, '#000000')
        
        ax.scatter(
            y_true_valid[mask],
            abs_percent_errors_valid[mask],
            c=color,
            label=comp_type,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Add threshold lines
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='10% error')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='20% error')
    
    # Add trend line
    z = np.polyfit(y_true_valid, abs_percent_errors_valid, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(y_true_valid.min(), y_true_valid.max(), 100)
    ax.plot(x_trend, p(x_trend), "k--", alpha=0.5, linewidth=2, 
            label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
    
    # Formatting
    ax.set_xlabel('Actual Sand ΔT (°C)', fontsize=12)
    ax.set_ylabel('Absolute Percent Error (%)', fontsize=12)
    ax.set_title(f'Prediction Error vs Temperature Magnitude ({test_pcb})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    plot3_path = output_path / f"{test_pcb}_ml_percent_error_vs_magnitude.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['vs_magnitude'] = str(plot3_path)
    print(f"  Saved percent error vs magnitude: {plot3_path}")
    
    # =========================================================================
    # PLOT 4: Component Type Error Box Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plot (using valid data)
    type_data = []
    type_labels = []
    type_sample_counts = []
    
    unique_types_valid = np.unique(component_types_valid)
    for comp_type in unique_types_valid:
        mask = component_types_valid == comp_type
        type_data.append(percent_errors_valid[mask])
        type_labels.append(comp_type)
        type_sample_counts.append(np.sum(mask))
    
    # Create box plot
    bp = ax.boxplot(type_data, labels=type_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True)
    
    # Color boxes
    for patch, comp_type in zip(bp['boxes'], type_labels):
        patch.set_facecolor(type_colors.get(comp_type, '#CCCCCC'))
        patch.set_alpha(0.7)
    
    # Add horizontal threshold lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='±10%')
    ax.axhline(y=-10, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='±20%')
    ax.axhline(y=-20, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Add sample counts
    for i, (label, count) in enumerate(zip(type_labels, type_sample_counts)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.9, f'n={count}',
                ha='center', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Component Type', fontsize=12)
    ax.set_ylabel('Percent Error (%)', fontsize=12)
    ax.set_title(f'Prediction Error Distribution by Component Type ({test_pcb})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot4_path = output_path / f"{test_pcb}_ml_percent_error_by_type.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths['by_type'] = str(plot4_path)
    print(f"  Saved percent error by type: {plot4_path}")
    
    return plot_paths
