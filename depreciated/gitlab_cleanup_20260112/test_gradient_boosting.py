"""
Test script to compare Gradient Boosting vs Linear Regression for thermal prediction.

This is a standalone test to evaluate whether Gradient Boosting can improve
cross-board prediction accuracy without modifying the existing Phase 8 workflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ============================================================================
# Configuration
# ============================================================================

CALIBRATION_DB = "outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv"
OUTPUT_DIR = "outputs/gradient_boosting_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(csv_path):
    """Load calibration database and prepare features."""
    print(f"\n{'='*80}")
    print("LOADING CALIBRATION DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(csv_path)
    print(f"Total calibration points: {len(df)}")
    
    # Filter out points without sand data (NaN sand_ss or sand_initial)
    df = df.dropna(subset=['sand_ss', 'sand_initial'])
    print(f"Points with valid sand data: {len(df)}")
    
    # Calculate delta T
    df['delta_t_flir_air'] = df['flir_ss'] - df['air_ss']
    df['delta_t_air'] = df['air_ss'] - df['air_initial']
    df['delta_t_sand'] = df['sand_ss'] - df['sand_initial']
    
    # Add normalized features (physics-informed)
    df['heating_efficiency'] = df['delta_t_sand'] / (df['delta_t_air'] + 0.1)  # avoid div by 0
    df['board_power_proxy'] = df.groupby('test_session')['flir_ss'].transform('max')
    
    print(f"\nData by board:")
    for pcb in df['pcb'].unique():
        pcb_data = df[df['pcb'] == pcb]
        mean_sand_delta = pcb_data['delta_t_sand'].mean()
        print(f"  {pcb}: {len(pcb_data)} points, mean sand ΔT = {mean_sand_delta:.2f}°C")
    
    return df

def prepare_features(df, include_normalized=True):
    """Prepare feature matrix and target vector."""
    # One-hot encode component types
    type_dummies = pd.get_dummies(df['component_type'], prefix='type')
    
    # Base features
    feature_cols = ['delta_t_flir_air']
    X = df[feature_cols].copy()
    
    # Add one-hot encoded types
    X = pd.concat([X, type_dummies], axis=1)
    
    # Add normalized features if requested
    if include_normalized:
        X['heating_efficiency'] = df['heating_efficiency']
        X['board_power_proxy'] = df['board_power_proxy']
    
    # Target
    y = df['delta_t_sand'].values
    
    return X, y, df

# ============================================================================
# Model Training Functions
# ============================================================================

def train_linear_regression(X_train, y_train):
    """Train baseline Linear Regression (current Phase 8 approach)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting Regressor."""
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    """Train Support Vector Regressor with scaling."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'))
    ])
    model.fit(X_train, y_train)
    return model

def train_gaussian_process(X_train, y_train):
    """Train Gaussian Process Regressor."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0,
        n_restarts_optimizer=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_polynomial_ridge(X_train, y_train, degree=2):
    """Train Polynomial Ridge Regression."""
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    model.fit(X_train, y_train)
    return model

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Percent errors
    epsilon = 1.0  # Avoid division by near-zero values
    valid_mask = np.abs(y_test) >= epsilon
    if valid_mask.sum() > 0:
        percent_errors = ((y_pred[valid_mask] - y_test[valid_mask]) / y_test[valid_mask]) * 100
        mean_pe = np.mean(percent_errors)
        mape = np.mean(np.abs(percent_errors))
    else:
        mean_pe = np.nan
        mape = np.nan
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_percent_error': mean_pe,
        'mape': mape,
        'predictions': y_pred,
        'actuals': y_test
    }

# ============================================================================
# Experiment Functions
# ============================================================================

def experiment_1_combined_training(df):
    """
    Experiment 1: Train on ALL data (both boards), validate with cross-validation.
    This tests whether combining datasets helps model learn full thermal range.
    """
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Combined Training (All 43 Points)")
    print(f"{'='*80}")
    
    X, y, df_with_features = prepare_features(df, include_normalized=True)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Train all models
    models = {
        'Linear Regression': train_linear_regression(X, y),
        'Gradient Boosting': train_gradient_boosting(X, y),
        'Random Forest': train_random_forest(X, y),
        'SVR (RBF)': train_svr(X, y),
        'Polynomial Ridge (deg=2)': train_polynomial_ridge(X, y, degree=2),
    }
    
    # Evaluate on training data (just to see in-sample fit)
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X, y, name)
        results.append(result)
        print(f"\n{name}:")
        print(f"  R² = {result['r2']:.4f}")
        print(f"  RMSE = {result['rmse']:.2f}°C")
        print(f"  MAE = {result['mae']:.2f}°C")
    
    return models, results, X, y, df_with_features

def experiment_2_cross_board_validation(df):
    """
    Experiment 2: Train on one board, test on other (cross-board generalization).
    This is the critical test for your use case.
    """
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: Cross-Board Validation")
    print(f"{'='*80}")
    
    boards = df['pcb'].unique()
    all_results = []
    
    for train_board in boards:
        for test_board in boards:
            if train_board == test_board:
                continue
            
            print(f"\n--- Train on {train_board.split('.')[0]} → Test on {test_board.split('.')[0]} ---")
            
            # Split data
            train_df = df[df['pcb'] == train_board]
            test_df = df[df['pcb'] == test_board]
            
            print(f"Training samples: {len(train_df)}")
            print(f"Test samples: {len(test_df)}")
            
            # Prepare features
            X_train, y_train, _ = prepare_features(train_df, include_normalized=True)
            X_test, y_test, test_df_with_features = prepare_features(test_df, include_normalized=True)
            
            # Align columns (in case test set has different component types)
            X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
            
            # Train models
            models = {
                'Linear Regression': train_linear_regression(X_train, y_train),
                'Gradient Boosting': train_gradient_boosting(X_train, y_train),
                'Random Forest': train_random_forest(X_train, y_train),
                'SVR (RBF)': train_svr(X_train, y_train),
            }
            
            # Evaluate
            for name, model in models.items():
                result = evaluate_model(model, X_test_aligned, y_test, name)
                result['train_board'] = train_board
                result['test_board'] = test_board
                all_results.append(result)
                
                print(f"\n  {name}:")
                print(f"    R² = {result['r2']:.4f}")
                print(f"    RMSE = {result['rmse']:.2f}°C")
                print(f"    MAE = {result['mae']:.2f}°C")
                print(f"    Mean PE = {result['mean_percent_error']:.1f}%")
                print(f"    MAPE = {result['mape']:.1f}%")
    
    return all_results

def experiment_3_combined_vs_single(df):
    """
    Experiment 3: Train on combined data, test on each board separately.
    This shows whether combined training helps generalization.
    """
    print(f"\n{'='*80}")
    print("EXPERIMENT 3: Combined Training → Individual Board Testing")
    print(f"{'='*80}")
    
    boards = df['pcb'].unique()
    
    # Train on ALL data
    X_all, y_all, _ = prepare_features(df, include_normalized=True)
    
    models = {
        'Linear Regression': train_linear_regression(X_all, y_all),
        'Gradient Boosting': train_gradient_boosting(X_all, y_all),
        'Random Forest': train_random_forest(X_all, y_all),
        'SVR (RBF)': train_svr(X_all, y_all),
    }
    
    all_results = []
    
    for test_board in boards:
        print(f"\n--- Testing on {test_board.split('.')[0]} ---")
        
        test_df = df[df['pcb'] == test_board]
        X_test, y_test, _ = prepare_features(test_df, include_normalized=True)
        X_test_aligned = X_test.reindex(columns=X_all.columns, fill_value=0)
        
        for name, model in models.items():
            result = evaluate_model(model, X_test_aligned, y_test, name)
            result['test_board'] = test_board
            all_results.append(result)
            
            print(f"  {name}:")
            print(f"    R² = {result['r2']:.4f}")
            print(f"    RMSE = {result['rmse']:.2f}°C")
            print(f"    MAPE = {result['mape']:.1f}%")
    
    return all_results

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_cross_board_comparison(results, output_dir):
    """Create comparison plots for cross-board validation."""
    
    # Convert to DataFrame for easier plotting
    df_results = pd.DataFrame(results)
    
    # Filter to Load→HBridge direction (most critical)
    load_to_hbridge = df_results[
        (df_results['train_board'].str.contains('Load_Shedding')) &
        (df_results['test_board'].str.contains('H_Bridge'))
    ]
    
    if len(load_to_hbridge) == 0:
        print("No Load→HBridge results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Board Validation: Load_Shedding → HBridge', fontsize=14, fontweight='bold')
    
    models = load_to_hbridge['model'].unique()
    colors = plt.cm.Set2(range(len(models)))
    
    # Plot 1: RMSE comparison
    ax = axes[0, 0]
    rmse_values = [load_to_hbridge[load_to_hbridge['model'] == m]['rmse'].values[0] for m in models]
    bars = ax.barh(models, rmse_values, color=colors)
    ax.set_xlabel('RMSE (°C)', fontweight='bold')
    ax.set_title('Prediction Error (Lower is Better)')
    ax.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}°C', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: R² comparison
    ax = axes[0, 1]
    r2_values = [load_to_hbridge[load_to_hbridge['model'] == m]['r2'].values[0] for m in models]
    bars = ax.barh(models, r2_values, color=colors)
    ax.set_xlabel('R² Score', fontweight='bold')
    ax.set_title('Model Fit (Higher is Better)')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 3: MAPE comparison
    ax = axes[1, 0]
    mape_values = [load_to_hbridge[load_to_hbridge['model'] == m]['mape'].values[0] for m in models]
    bars = ax.barh(models, mape_values, color=colors)
    ax.set_xlabel('MAPE (%)', fontweight='bold')
    ax.set_title('Mean Absolute Percent Error (Lower is Better)')
    ax.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 4: Predicted vs Actual for best model
    ax = axes[1, 1]
    best_model_idx = np.argmin(mape_values)
    best_model = models[best_model_idx]
    best_result = load_to_hbridge[load_to_hbridge['model'] == best_model].iloc[0]
    
    y_test = best_result['actuals']
    y_pred = best_result['predictions']
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=80, color=colors[best_model_idx], 
               edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    ax.set_xlabel('Actual Sand ΔT (°C)', fontweight='bold')
    ax.set_ylabel('Predicted Sand ΔT (°C)', fontweight='bold')
    ax.set_title(f'Best Model: {best_model}\n(MAPE = {best_result["mape"]:.1f}%)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cross_board_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    plt.close()

def plot_feature_importance(model, feature_names, output_dir):
    """Plot feature importance for Gradient Boosting."""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontweight='bold')
    plt.title('Gradient Boosting: Top 15 Most Important Features', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature importance plot: {output_path}")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\n" + "="*80)
    print("GRADIENT BOOSTING TEST: Thermal Prediction Model Comparison")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(CALIBRATION_DB)
    
    # Run experiments
    print("\n" + "="*80)
    print("Running Experiments...")
    print("="*80)
    
    # Experiment 1: Combined training (all data)
    models_exp1, results_exp1, X_all, y_all, df_features = experiment_1_combined_training(df)
    
    # Save feature importance
    if 'Gradient Boosting' in models_exp1:
        plot_feature_importance(models_exp1['Gradient Boosting'], X_all.columns, OUTPUT_DIR)
    
    # Experiment 2: Cross-board validation (the critical test)
    results_exp2 = experiment_2_cross_board_validation(df)
    
    # Experiment 3: Combined training, test on each board
    results_exp3 = experiment_3_combined_vs_single(df)
    
    # Create comparison plots
    plot_cross_board_comparison(results_exp2, OUTPUT_DIR)
    
    # Save results to CSV
    df_exp2 = pd.DataFrame(results_exp2)
    df_exp2_summary = df_exp2[['model', 'train_board', 'test_board', 'rmse', 'mae', 'r2', 'mape']]
    output_csv = os.path.join(OUTPUT_DIR, 'cross_board_results.csv')
    df_exp2_summary.to_csv(output_csv, index=False)
    print(f"\nSaved detailed results: {output_csv}")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Find best model for Load→HBridge
    load_to_hbridge = df_exp2[
        (df_exp2['train_board'].str.contains('Load_Shedding')) &
        (df_exp2['test_board'].str.contains('H_Bridge'))
    ]
    
    if len(load_to_hbridge) > 0:
        print("\nLoad_Shedding → HBridge (Critical Test):")
        print("-" * 80)
        for _, row in load_to_hbridge.iterrows():
            print(f"\n{row['model']}:")
            print(f"  RMSE: {row['rmse']:.2f}°C")
            print(f"  R²: {row['r2']:.4f}")
            print(f"  MAPE: {row['mape']:.1f}%")
        
        best_idx = load_to_hbridge['mape'].idxmin()
        best_model = load_to_hbridge.loc[best_idx]
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best_model['model']}")
        print(f"   MAPE: {best_model['mape']:.1f}% (vs Linear Regression baseline)")
        print("="*80)
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}/")
    print(f"   - cross_board_comparison.png")
    print(f"   - feature_importance.png")
    print(f"   - cross_board_results.csv")

if __name__ == "__main__":
    main()
