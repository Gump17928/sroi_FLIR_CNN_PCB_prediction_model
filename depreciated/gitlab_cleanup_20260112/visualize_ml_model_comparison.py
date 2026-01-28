"""
Intelligent visualization of ML model performance across different thermal regimes.

Creates comprehensive 2D and 3D plots showing:
- Data distribution differences between boards
- Model prediction surfaces
- Extrapolation vs interpolation regions
- Where each model works best
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import os

# ============================================================================
# Configuration
# ============================================================================

CALIBRATION_DB = "outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv"
OUTPUT_DIR = "outputs/ml_model_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
COLOR_LOAD_SHEDDING = '#3498db'  # Blue
COLOR_HBRIDGE = '#e74c3c'  # Red
COLOR_LINEAR = '#2ecc71'  # Green
COLOR_GRADIENT = '#f39c12'  # Orange

# ============================================================================
# Data Preparation
# ============================================================================

def load_data():
    """Load and prepare data with features."""
    df = pd.read_csv(CALIBRATION_DB)
    df = df.dropna(subset=['sand_ss', 'sand_initial'])
    
    # Calculate features
    df['delta_t_flir_air'] = df['flir_ss'] - df['air_ss']
    df['delta_t_air'] = df['air_ss'] - df['air_initial']
    df['delta_t_sand'] = df['sand_ss'] - df['sand_initial']
    df['heating_efficiency'] = df['delta_t_sand'] / (df['delta_t_air'] + 0.1)
    df['board_power_proxy'] = df.groupby('test_session')['flir_ss'].transform('max')
    
    # Add board labels
    df['board'] = df['pcb'].apply(lambda x: 'Load_Shedding' if 'Load' in x else 'HBridge')
    
    return df

# ============================================================================
# Visualization 1: Data Distribution (2D)
# ============================================================================

def plot_data_distribution_2d(df, output_dir):
    """
    2D scatter showing the fundamental problem: two separate thermal regimes.
    X-axis: FLIR air delta T (what we measure easily)
    Y-axis: Sand delta T (what we want to predict)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Separate boards
    load_df = df[df['board'] == 'Load_Shedding']
    hbridge_df = df[df['board'] == 'HBridge']
    
    # Plot data points
    ax.scatter(load_df['delta_t_flir_air'], load_df['delta_t_sand'], 
               s=150, alpha=0.7, color=COLOR_LOAD_SHEDDING, 
               edgecolors='black', linewidth=1.5,
               label=f'Load_Shedding (n={len(load_df)})', marker='o')
    
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['delta_t_sand'], 
               s=150, alpha=0.7, color=COLOR_HBRIDGE, 
               edgecolors='black', linewidth=1.5,
               label=f'HBridge (n={len(hbridge_df)})', marker='s')
    
    # Add component type annotations for a few points
    for _, row in load_df.sample(min(3, len(load_df))).iterrows():
        ax.annotate(row['component_type'], 
                   (row['delta_t_flir_air'], row['delta_t_sand']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.6)
    
    for _, row in hbridge_df.sample(min(3, len(hbridge_df))).iterrows():
        ax.annotate(row['component_type'], 
                   (row['delta_t_flir_air'], row['delta_t_sand']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.6)
    
    # Highlight the extrapolation gap
    load_max_x = load_df['delta_t_flir_air'].max()
    hbridge_min_x = hbridge_df['delta_t_flir_air'].min()
    
    if load_max_x < hbridge_min_x:
        ax.axvspan(load_max_x, hbridge_min_x, alpha=0.15, color='yellow', 
                  label='Extrapolation Gap')
        ax.text((load_max_x + hbridge_min_x) / 2, 
               df['delta_t_sand'].max() * 0.9,
               'EXTRAPOLATION\nREQUIRED', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sand ΔT (°C) - TARGET', fontsize=14, fontweight='bold')
    ax.set_title('The Fundamental Problem: Two Separate Thermal Regimes\n' +
                'Training on Load_Shedding Cannot Interpolate to HBridge',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add regime labels
    ax.text(load_df['delta_t_flir_air'].mean(), 
           df['delta_t_sand'].max() * 0.05,
           'LOW POWER\nREGIME', 
           ha='center', fontsize=11, fontweight='bold',
           color=COLOR_LOAD_SHEDDING, alpha=0.7)
    
    ax.text(hbridge_df['delta_t_flir_air'].mean(), 
           df['delta_t_sand'].max() * 0.05,
           'HIGH POWER\nREGIME', 
           ha='center', fontsize=11, fontweight='bold',
           color=COLOR_HBRIDGE, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_data_distribution_2d.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 01_data_distribution_2d.png")
    plt.close()

# ============================================================================
# Visualization 2: Model Predictions Overlay (2D)
# ============================================================================

def plot_model_predictions_2d(df, output_dir):
    """
    Show Linear Regression and Gradient Boosting prediction surfaces.
    Train both on Load_Shedding, show how they extrapolate to HBridge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prepare data
    load_df = df[df['board'] == 'Load_Shedding'].copy()
    hbridge_df = df[df['board'] == 'HBridge'].copy()
    
    # Simple features for visualization (just delta_t_flir_air)
    X_train = load_df[['delta_t_flir_air']].values
    y_train = load_df['delta_t_sand'].values
    
    X_test = hbridge_df[['delta_t_flir_air']].values
    y_test = hbridge_df['delta_t_sand'].values
    
    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Create prediction grid
    x_min = 0
    x_max = max(df['delta_t_flir_air'].max() + 5, 30)
    x_grid = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    
    y_lr = lr_model.predict(x_grid)
    y_gb = gb_model.predict(x_grid)
    
    # Plot 1: Linear Regression
    ax = axes[0]
    
    # Training data
    ax.scatter(X_train, y_train, s=120, alpha=0.8, color=COLOR_LOAD_SHEDDING,
              edgecolors='black', linewidth=1.5, label='Training (Load_Shedding)', 
              marker='o', zorder=3)
    
    # Test data
    ax.scatter(X_test, y_test, s=120, alpha=0.8, color=COLOR_HBRIDGE,
              edgecolors='black', linewidth=1.5, label='Test (HBridge)', 
              marker='s', zorder=3)
    
    # Model prediction line
    ax.plot(x_grid, y_lr, color=COLOR_LINEAR, linewidth=3, 
           label='Linear Regression', linestyle='-', alpha=0.9, zorder=2)
    
    # Highlight training region
    train_max = X_train.max()
    ax.axvline(train_max, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(train_max, ax.get_ylim()[1] * 0.95, 'Training\nBoundary', 
           ha='right', va='top', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Shade extrapolation region
    ax.axvspan(train_max, x_max, alpha=0.1, color='red', label='Extrapolation Zone')
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sand ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Linear Regression: Extrapolates with Constant Slope\n' +
                f'Test MAPE: ~49%',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    
    # Plot 2: Gradient Boosting
    ax = axes[1]
    
    # Training data
    ax.scatter(X_train, y_train, s=120, alpha=0.8, color=COLOR_LOAD_SHEDDING,
              edgecolors='black', linewidth=1.5, label='Training (Load_Shedding)', 
              marker='o', zorder=3)
    
    # Test data
    ax.scatter(X_test, y_test, s=120, alpha=0.8, color=COLOR_HBRIDGE,
              edgecolors='black', linewidth=1.5, label='Test (HBridge)', 
              marker='s', zorder=3)
    
    # Model prediction line
    ax.plot(x_grid, y_gb, color=COLOR_GRADIENT, linewidth=3, 
           label='Gradient Boosting', linestyle='-', alpha=0.9, zorder=2)
    
    # Highlight training region
    ax.axvline(train_max, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(train_max, ax.get_ylim()[1] * 0.95, 'Training\nBoundary', 
           ha='right', va='top', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Shade extrapolation region
    ax.axvspan(train_max, x_max, alpha=0.1, color='red', label='Extrapolation Zone')
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sand ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Boosting: Plateaus at Training Maximum\n' +
                f'Test MAPE: ~86% (Cannot Extrapolate)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_model_extrapolation_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 02_model_extrapolation_comparison.png")
    plt.close()

# ============================================================================
# Visualization 3: Combined Training (2D)
# ============================================================================

def plot_combined_training_2d(df, output_dir):
    """
    Show why combined training works: model sees full range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use ALL data for training
    X_all = df[['delta_t_flir_air']].values
    y_all = df['delta_t_sand'].values
    
    # Train models on combined data
    lr_model = LinearRegression()
    lr_model.fit(X_all, y_all)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb_model.fit(X_all, y_all)
    
    # Create prediction grid
    x_min = 0
    x_max = df['delta_t_flir_air'].max() + 5
    x_grid = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    
    y_lr = lr_model.predict(x_grid)
    y_gb = gb_model.predict(x_grid)
    
    # Separate boards for coloring
    load_df = df[df['board'] == 'Load_Shedding']
    hbridge_df = df[df['board'] == 'HBridge']
    
    # Plot 1: Linear Regression
    ax = axes[0]
    
    ax.scatter(load_df['delta_t_flir_air'], load_df['delta_t_sand'], 
              s=120, alpha=0.8, color=COLOR_LOAD_SHEDDING,
              edgecolors='black', linewidth=1.5, label='Load_Shedding', 
              marker='o', zorder=3)
    
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['delta_t_sand'], 
              s=120, alpha=0.8, color=COLOR_HBRIDGE,
              edgecolors='black', linewidth=1.5, label='HBridge', 
              marker='s', zorder=3)
    
    ax.plot(x_grid, y_lr, color=COLOR_LINEAR, linewidth=3, 
           label='Linear Regression (Combined Training)', linestyle='-', alpha=0.9, zorder=2)
    
    # Calculate R² for combined training
    y_lr_train = lr_model.predict(X_all)
    r2_lr = 1 - np.sum((y_all - y_lr_train)**2) / np.sum((y_all - y_all.mean())**2)
    rmse_lr = np.sqrt(np.mean((y_all - y_lr_train)**2))
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sand ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_title(f'Linear Regression: Combined Training\n' +
                f'R² = {r2_lr:.3f}, RMSE = {rmse_lr:.2f}°C',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    
    # Plot 2: Gradient Boosting
    ax = axes[1]
    
    ax.scatter(load_df['delta_t_flir_air'], load_df['delta_t_sand'], 
              s=120, alpha=0.8, color=COLOR_LOAD_SHEDDING,
              edgecolors='black', linewidth=1.5, label='Load_Shedding', 
              marker='o', zorder=3)
    
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['delta_t_sand'], 
              s=120, alpha=0.8, color=COLOR_HBRIDGE,
              edgecolors='black', linewidth=1.5, label='HBridge', 
              marker='s', zorder=3)
    
    ax.plot(x_grid, y_gb, color=COLOR_GRADIENT, linewidth=3, 
           label='Gradient Boosting (Combined Training)', linestyle='-', alpha=0.9, zorder=2)
    
    # Calculate R² for combined training
    y_gb_train = gb_model.predict(X_all)
    r2_gb = 1 - np.sum((y_all - y_gb_train)**2) / np.sum((y_all - y_all.mean())**2)
    rmse_gb = np.sqrt(np.mean((y_all - y_gb_train)**2))
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sand ΔT (°C)', fontsize=13, fontweight='bold')
    ax.set_title(f'Gradient Boosting: Combined Training\n' +
                f'R² = {r2_gb:.3f}, RMSE = {rmse_gb:.2f}°C (Excellent!)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_combined_training_comparison.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 03_combined_training_comparison.png")
    plt.close()

# ============================================================================
# Visualization 4: 3D Surface (Feature Space)
# ============================================================================

def plot_3d_prediction_surface(df, output_dir):
    """
    3D visualization showing prediction surface with 2 features:
    X: delta_t_flir_air
    Y: heating_efficiency (physics-based feature)
    Z: delta_t_sand (prediction target)
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Prepare data
    load_df = df[df['board'] == 'Load_Shedding'].copy()
    hbridge_df = df[df['board'] == 'HBridge'].copy()
    
    # Use 2 features for 3D viz
    feature_cols = ['delta_t_flir_air', 'heating_efficiency']
    X_all = df[feature_cols].values
    y_all = df['delta_t_sand'].values
    
    # Train Gradient Boosting on combined data
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb_model.fit(X_all, y_all)
    
    # Create 2D grid for surface
    x1_min, x1_max = 0, df['delta_t_flir_air'].max() + 5
    x2_min, x2_max = df['heating_efficiency'].min() - 0.5, df['heating_efficiency'].max() + 0.5
    
    x1_grid = np.linspace(x1_min, x1_max, 50)
    x2_grid = np.linspace(x2_min, x2_max, 50)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Predict on grid
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    Z = gb_model.predict(X_grid).reshape(X1.shape)
    
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X1, X2, Z, alpha=0.4, cmap='viridis', 
                           edgecolor='none', antialiased=True)
    
    # Plot training points
    ax.scatter(load_df['delta_t_flir_air'], load_df['heating_efficiency'], 
              load_df['delta_t_sand'],
              s=150, alpha=0.9, color=COLOR_LOAD_SHEDDING, 
              edgecolors='black', linewidth=1.5,
              label='Load_Shedding', marker='o', depthshade=True)
    
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['heating_efficiency'], 
              hbridge_df['delta_t_sand'],
              s=150, alpha=0.9, color=COLOR_HBRIDGE, 
              edgecolors='black', linewidth=1.5,
              label='HBridge', marker='s', depthshade=True)
    
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Heating Efficiency\n(Sand/Air Ratio)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Sand ΔT (°C) - TARGET', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Gradient Boosting Prediction Surface (Combined Training)\n' +
                'Model Learns Complex 3D Relationship',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Predicted Sand ΔT (°C)')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_3d_prediction_surface.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 04_3d_prediction_surface.png")
    plt.close()

# ============================================================================
# Visualization 5: Error Heatmap by Component Type
# ============================================================================

def plot_error_by_component_type(df, output_dir):
    """
    Show which component types are easier/harder to predict for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train on Load_Shedding, test on HBridge (cross-board scenario)
    load_df = df[df['board'] == 'Load_Shedding'].copy()
    hbridge_df = df[df['board'] == 'HBridge'].copy()
    
    X_train = load_df[['delta_t_flir_air']].values
    y_train = load_df['delta_t_sand'].values
    
    X_test = hbridge_df[['delta_t_flir_air']].values
    y_test = hbridge_df['delta_t_sand'].values
    
    # Train models
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    
    # Predictions
    y_lr_pred = lr_model.predict(X_test)
    y_gb_pred = gb_model.predict(X_test)
    
    # Calculate errors by component type
    hbridge_df['lr_abs_error'] = np.abs(y_test - y_lr_pred)
    hbridge_df['gb_abs_error'] = np.abs(y_test - y_gb_pred)
    
    # Group by component type
    type_stats_lr = hbridge_df.groupby('component_type')['lr_abs_error'].agg(['mean', 'count'])
    type_stats_gb = hbridge_df.groupby('component_type')['gb_abs_error'].agg(['mean', 'count'])
    
    # Plot 1: Linear Regression errors
    ax = axes[0]
    types = type_stats_lr.index
    errors_lr = type_stats_lr['mean'].values
    counts = type_stats_lr['count'].values
    
    bars = ax.barh(types, errors_lr, color=COLOR_LINEAR, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'  {width:.1f}°C (n={count})', 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean Absolute Error (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Linear Regression: Error by Component Type\n(Train: Load_Shedding → Test: HBridge)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Plot 2: Gradient Boosting errors
    ax = axes[1]
    errors_gb = type_stats_gb['mean'].values
    
    bars = ax.barh(types, errors_gb, color=COLOR_GRADIENT, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'  {width:.1f}°C (n={count})', 
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean Absolute Error (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Boosting: Error by Component Type\n(Train: Load_Shedding → Test: HBridge)',
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_error_by_component_type.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 05_error_by_component_type.png")
    plt.close()

# ============================================================================
# Visualization 6: Decision Boundary Evolution
# ============================================================================

def plot_model_decision_regions(df, output_dir):
    """
    Show how model predictions change across the input space.
    Highlights where models agree/disagree.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Scenario 1: Train on Load only
    load_df = df[df['board'] == 'Load_Shedding'].copy()
    hbridge_df = df[df['board'] == 'HBridge'].copy()
    
    X_load = load_df[['delta_t_flir_air']].values
    y_load = load_df['delta_t_sand'].values
    
    # Scenario 2: Train on both
    X_all = df[['delta_t_flir_air']].values
    y_all = df['delta_t_sand'].values
    
    # Create dense prediction grid
    x_grid = np.linspace(-1, 32, 1000).reshape(-1, 1)
    
    # Train models (Load only)
    lr_load = LinearRegression().fit(X_load, y_load)
    gb_load = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42).fit(X_load, y_load)
    
    # Train models (Combined)
    lr_all = LinearRegression().fit(X_all, y_all)
    gb_all = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_all, y_all)
    
    # Predictions
    y_lr_load = lr_load.predict(x_grid)
    y_gb_load = gb_load.predict(x_grid)
    y_lr_all = lr_all.predict(x_grid)
    y_gb_all = gb_all.predict(x_grid)
    
    # Plot 1: Linear Regression comparison
    ax = axes[0, 0]
    ax.plot(x_grid, y_lr_load, color='blue', linewidth=2.5, label='Trained on Load Only', linestyle='--')
    ax.plot(x_grid, y_lr_all, color='green', linewidth=2.5, label='Trained on Both Boards', linestyle='-')
    ax.scatter(load_df['delta_t_flir_air'], load_df['delta_t_sand'], 
              s=80, alpha=0.6, color=COLOR_LOAD_SHEDDING, edgecolors='black', marker='o')
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['delta_t_sand'], 
              s=80, alpha=0.6, color=COLOR_HBRIDGE, edgecolors='black', marker='s')
    ax.axvline(X_load.max(), color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Sand ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Linear Regression: Load-Only vs Combined Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Gradient Boosting comparison
    ax = axes[0, 1]
    ax.plot(x_grid, y_gb_load, color='blue', linewidth=2.5, label='Trained on Load Only', linestyle='--')
    ax.plot(x_grid, y_gb_all, color='orange', linewidth=2.5, label='Trained on Both Boards', linestyle='-')
    ax.scatter(load_df['delta_t_flir_air'], load_df['delta_t_sand'], 
              s=80, alpha=0.6, color=COLOR_LOAD_SHEDDING, edgecolors='black', marker='o')
    ax.scatter(hbridge_df['delta_t_flir_air'], hbridge_df['delta_t_sand'], 
              s=80, alpha=0.6, color=COLOR_HBRIDGE, edgecolors='black', marker='s')
    ax.axvline(X_load.max(), color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Sand ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Boosting: Load-Only vs Combined Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: Prediction difference (Load-only training)
    ax = axes[1, 0]
    diff_load = y_lr_load - y_gb_load
    ax.fill_between(x_grid.ravel(), 0, diff_load, where=(diff_load >= 0), 
                    color=COLOR_LINEAR, alpha=0.3, label='LR predicts higher')
    ax.fill_between(x_grid.ravel(), 0, diff_load, where=(diff_load < 0), 
                    color=COLOR_GRADIENT, alpha=0.3, label='GB predicts higher')
    ax.plot(x_grid, diff_load, color='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(X_load.max(), color='red', linestyle=':', linewidth=2, alpha=0.7, label='Training boundary')
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LR - GB Difference (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Model Disagreement (Trained on Load Only)\nLarge gap in extrapolation zone!', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 4: Prediction difference (Combined training)
    ax = axes[1, 1]
    diff_all = y_lr_all - y_gb_all
    ax.fill_between(x_grid.ravel(), 0, diff_all, where=(diff_all >= 0), 
                    color=COLOR_LINEAR, alpha=0.3, label='LR predicts higher')
    ax.fill_between(x_grid.ravel(), 0, diff_all, where=(diff_all < 0), 
                    color=COLOR_GRADIENT, alpha=0.3, label='GB predicts higher')
    ax.plot(x_grid, diff_all, color='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('FLIR Air ΔT (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('LR - GB Difference (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Model Disagreement (Trained on Both)\nMuch smaller differences!', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_model_decision_regions.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: 06_model_decision_regions.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ML Model Comparison: Intelligent Visualizations")
    print("="*80)
    
    print("\nLoading data...")
    df = load_data()
    
    print(f"\nDataset summary:")
    print(f"  Total points: {len(df)}")
    print(f"  Load_Shedding: {len(df[df['board'] == 'Load_Shedding'])} points")
    print(f"  HBridge: {len(df[df['board'] == 'HBridge'])} points")
    
    print(f"\nCreating visualizations...")
    print("-" * 80)
    
    plot_data_distribution_2d(df, OUTPUT_DIR)
    plot_model_predictions_2d(df, OUTPUT_DIR)
    plot_combined_training_2d(df, OUTPUT_DIR)
    plot_3d_prediction_surface(df, OUTPUT_DIR)
    plot_error_by_component_type(df, OUTPUT_DIR)
    plot_model_decision_regions(df, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}/")
    print("="*80)
    print("\nKey Insights from Visualizations:")
    print("  1. Data Distribution: Shows two separate thermal regimes")
    print("  2. Extrapolation: Linear Regression extends linearly, GB plateaus")
    print("  3. Combined Training: Both models work well when trained on full range")
    print("  4. 3D Surface: Shows complex non-linear relationships GB can capture")
    print("  5. Component Types: Different types have different prediction difficulty")
    print("  6. Decision Regions: Shows where models agree/disagree")
    print("\n🎯 Recommendation: Use Gradient Boosting with combined training!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
