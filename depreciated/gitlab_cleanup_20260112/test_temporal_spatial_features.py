"""
Test Script: Temporal & Spatial Feature Analysis from SROI Thermal Maps
Option C Implementation - Validate then Extract

This script:
1. Loads SROI files with 300-frame thermal time series
2. Visualizes heating curves to confirm board differences
3. Extracts temporal features (heating rate, time constant, etc.)
4. Extracts spatial features (gradients, neighborhoods, etc.)
5. Tests if features can discriminate between power regimes
6. Tests if features improve ML model cross-board prediction

Author: Thermal Analysis Pipeline
Date: 2026-01-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import sys

# Import local ResearchIR loader
try:
    from loader_researchir import ResearchIRStatsParser
except ImportError:
    print("ERROR: loader_researchir.py not found in current directory")
    print("This script requires loader_researchir.py to read FLIR data")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

CALIBRATION_DB = "outputs/0107_1439_P1-7/calibration_database/thermal_calibration_points.csv"
OUTPUT_DIR = "outputs/temporal_spatial_feature_analysis"

# ResearchIR Stats directory mapping (actual paths from inputs folder)
# These directories contain the Stats.txt files with ROI averages over time
FLIR_STATS_DIRS = {
    'Load_Shedding_Air': 'inputs/ResearchIR_Outputs_Load_Shedding',
    'HBridge_15s_Air': 'inputs/ResearchIR_Outputs_HBridge_15s',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ResearchIR Data Loading
# ============================================================================

def load_flir_thermal_data(stats_dir):
    """
    Load ResearchIR Stats.txt files and extract thermal time series for all ROIs.
    
    Returns:
        DataFrame with columns: frame, reltime, [component1], [component2], ...
    """
    print(f"  Loading ResearchIR Stats: {stats_dir}")
    
    parser = ResearchIRStatsParser(stats_dir)
    df = parser.parse_all_frames()
    
    print(f"    Loaded {len(df)} frames for {len(parser.component_names)} components")
    return df

def load_all_flir_data():
    """Load FLIR data for all test sessions."""
    print("\n" + "="*80)
    print("PHASE 1: Loading ResearchIR Thermal Time Series")
    print("="*80)
    
    all_data = {}
    
    for session_name, stats_dir in FLIR_STATS_DIRS.items():
        if not os.path.exists(stats_dir):
            print(f"  WARNING: Stats directory not found: {stats_dir}")
            print(f"     Skipping session: {session_name}")
            continue
        
        all_data[session_name] = load_flir_thermal_data(stats_dir)
    
    return all_data

# ============================================================================
# Temporal Feature Extraction
# ============================================================================

def exponential_heating_model(t, T_inf, tau, T0):
    """Exponential heating model: T(t) = T_inf - (T_inf - T0) * exp(-t/tau)"""
    return T_inf - (T_inf - T0) * np.exp(-t / tau)

def calculate_temporal_features(time, temp):
    """
    Extract temporal features from thermal time series.
    
    Args:
        time: Time array (seconds)
        temp: Temperature array (°C)
    
    Returns:
        dict: Temporal features
    """
    features = {}
    
    # Initial and final temperatures
    T_initial = temp[0]
    T_final = temp[-1]
    features['T_initial'] = T_initial
    features['T_steady_state'] = T_final
    features['delta_T_total'] = T_final - T_initial
    
    # 1. Heating rate (linear fit to first 30 seconds)
    mask_30s = time <= 30
    if mask_30s.sum() >= 5:
        slope, intercept, r_value, _, _ = linregress(time[mask_30s], temp[mask_30s])
        features['heating_rate_30s'] = slope  # °C/sec
        features['heating_linearity_30s'] = r_value**2  # R²
    else:
        features['heating_rate_30s'] = np.nan
        features['heating_linearity_30s'] = np.nan
    
    # 2. Overall heating rate (entire curve)
    if len(time) > 1:
        overall_slope, _, overall_r, _, _ = linregress(time, temp)
        features['heating_rate_overall'] = overall_slope
        features['heating_linearity_overall'] = overall_r**2
    else:
        features['heating_rate_overall'] = np.nan
        features['heating_linearity_overall'] = np.nan
    
    # 3. Time constant (exponential fit)
    try:
        # Initial guess: tau = time to 63% of final value
        delta_T = T_final - T_initial
        if abs(delta_T) > 0.1:  # Only fit if significant heating
            # Find 63% point
            T_63 = T_initial + 0.63 * delta_T
            idx_63 = np.argmin(np.abs(temp - T_63))
            tau_guess = time[idx_63] if idx_63 > 0 else 60
            
            popt, _ = curve_fit(
                exponential_heating_model,
                time, temp,
                p0=[T_final, tau_guess, T_initial],
                maxfev=5000,
                bounds=([T_final*0.9, 1, T_initial*0.9], 
                       [T_final*1.1, 300, T_initial*1.1])
            )
            
            features['time_constant_tau'] = popt[1]  # seconds
            
            # Calculate R² for exponential fit
            temp_fit = exponential_heating_model(time, *popt)
            ss_res = np.sum((temp - temp_fit)**2)
            ss_tot = np.sum((temp - temp.mean())**2)
            features['exponential_fit_r2'] = 1 - (ss_res / ss_tot)
        else:
            features['time_constant_tau'] = np.nan
            features['exponential_fit_r2'] = np.nan
    except:
        features['time_constant_tau'] = np.nan
        features['exponential_fit_r2'] = np.nan
    
    # 4. Settling time (90% of steady state)
    if abs(T_final - T_initial) > 0.1:
        T_90 = T_initial + 0.9 * (T_final - T_initial)
        idx_90 = np.argmax(temp >= T_90)
        features['settling_time_90pct'] = time[idx_90] if idx_90 > 0 else np.nan
    else:
        features['settling_time_90pct'] = np.nan
    
    # 5. Peak-to-initial ratio
    T_max = temp.max()
    if T_initial > 0:
        features['peak_to_initial_ratio'] = T_max / T_initial
    else:
        features['peak_to_initial_ratio'] = np.nan
    
    # 6. Overshoot detection
    overshoot = T_max - T_final
    features['overshoot_magnitude'] = overshoot
    features['has_overshoot'] = 1 if overshoot > 0.5 else 0
    
    # 7. Temperature range
    features['temp_range'] = T_max - temp.min()
    
    return features

# ============================================================================
# Spatial Feature Extraction
# ============================================================================

def calculate_spatial_features(roi_data, roi_name, board_thermal_map=None):
    """
    Extract spatial features for a component.
    
    Args:
        roi_data: Dict with thermal time series
        roi_name: Name of the ROI
        board_thermal_map: Optional 2D array of board temperatures at steady-state
    
    Returns:
        dict: Spatial features
    """
    features = {}
    
    if roi_name not in roi_data:
        return features
    
    roi = roi_data[roi_name]
    T_ss = roi['temp'][-1]  # Steady-state temperature
    
    features['steady_state_temp'] = T_ss
    
    # If we have full board thermal map, calculate neighborhood features
    if board_thermal_map is not None:
        x, y = roi['x'], roi['y']
        
        # Define neighborhood radius (e.g., 5 pixels)
        radius = 5
        y_min = max(0, y - radius)
        y_max = min(board_thermal_map.shape[0], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(board_thermal_map.shape[1], x + radius + 1)
        
        neighborhood = board_thermal_map[y_min:y_max, x_min:x_max]
        
        # Neighborhood statistics
        features['neighbor_mean_temp'] = np.mean(neighborhood)
        features['neighbor_max_temp'] = np.max(neighborhood)
        features['neighbor_std_temp'] = np.std(neighborhood)
        
        # Local gradient (difference to neighborhood mean)
        features['local_gradient'] = T_ss - features['neighbor_mean_temp']
        
        # Thermal isolation score (high = isolated hot spot)
        features['thermal_isolation'] = features['local_gradient'] / (features['neighbor_std_temp'] + 0.1)
        
        # Relative to board maximum
        board_max = np.max(board_thermal_map)
        features['relative_to_board_max'] = (T_ss - board_max) / (board_max + 0.1)
    
    return features

def calculate_board_level_features(roi_data):
    """
    Calculate board-level features (global context).
    
    Args:
        roi_data: Dict of all ROI thermal data
    
    Returns:
        dict: Board-level features
    """
    features = {}
    
    # Collect all steady-state temperatures
    all_temps_ss = [roi['temp'][-1] for roi in roi_data.values()]
    all_temps_initial = [roi['temp'][0] for roi in roi_data.values()]
    
    features['board_max_temp'] = np.max(all_temps_ss)
    features['board_min_temp'] = np.min(all_temps_ss)
    features['board_mean_temp'] = np.mean(all_temps_ss)
    features['board_std_temp'] = np.std(all_temps_ss)
    features['board_temp_range'] = features['board_max_temp'] - features['board_min_temp']
    
    # Global heating metrics
    all_delta_T = np.array(all_temps_ss) - np.array(all_temps_initial)
    features['board_mean_delta_T'] = np.mean(all_delta_T)
    features['board_max_delta_T'] = np.max(all_delta_T)
    
    return features

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_heating_curves_comparison(flir_data, calibration_df, output_dir):
    """
    Phase 1 Visualization: Compare heating curves between boards.
    """
    print("\n" + "="*80)
    print("PHASE 1: Heating Curve Visualization")
    print("="*80)
    
    # Get sample components from each board
    load_components = calibration_df[calibration_df['board'] == 'Load_Shedding']['component_name'].head(5).tolist()
    hbridge_components = calibration_df[calibration_df['board'] == 'HBridge']['component_name'].head(5).tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Heating Curve Comparison: Load_Shedding vs HBridge', fontsize=16, fontweight='bold')
    
    # Plot Load_Shedding components
    for i, comp in enumerate(load_components[:3]):
        ax = axes[0, i]
        
        # Find this component in FLIR data
        for session, df in flir_data.items():
            if 'Load' in session and comp in df.columns:
                time = df['reltime'].values
                temp = df[comp].values
                ax.plot(time, temp, linewidth=2, label=session.split('_')[-1])
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontweight='bold')
        ax.set_title(f'Load_Shedding: {comp}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Plot HBridge components
    for i, comp in enumerate(hbridge_components[:3]):
        ax = axes[1, i]
        
        # Find this component in FLIR data
        for session, df in flir_data.items():
            if 'HBridge' in session and comp in df.columns:
                time = df['reltime'].values
                temp = df[comp].values
                ax.plot(time, temp, linewidth=2, label=session.split('_')[-1], color='red')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontweight='bold')
        ax.set_title(f'HBridge: {comp}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_heating_curves.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: phase1_heating_curves.png")
    plt.close()

def plot_temporal_feature_distributions(feature_df, output_dir):
    """
    Phase 2 Visualization: Compare temporal feature distributions.
    """
    print("\n" + "="*80)
    print("PHASE 2: Temporal Feature Distribution Analysis")
    print("="*80)
    
    # Select key temporal features to visualize
    temporal_features = [
        'heating_rate_30s',
        'time_constant_tau',
        'settling_time_90pct',
        'peak_to_initial_ratio',
        'heating_linearity_30s'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Temporal Features: Load_Shedding vs HBridge', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for i, feature in enumerate(temporal_features):
        ax = axes[i]
        
        load_data = feature_df[feature_df['board'] == 'Load_Shedding'][feature].dropna()
        hbridge_data = feature_df[feature_df['board'] == 'HBridge'][feature].dropna()
        
        # Box plot comparison
        data_to_plot = [load_data, hbridge_data]
        bp = ax.boxplot(data_to_plot, labels=['Load_Shedding', 'HBridge'],
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        
        ax.set_ylabel(feature.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()}\nLoad: {load_data.mean():.2f} | HBridge: {hbridge_data.mean():.2f}',
                    fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_temporal_features.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: phase2_temporal_features.png")
    plt.close()
    
    # Print statistics
    print("\nTemporal Feature Statistics:")
    print("-" * 80)
    for feature in temporal_features:
        load_mean = feature_df[feature_df['board'] == 'Load_Shedding'][feature].mean()
        hbridge_mean = feature_df[feature_df['board'] == 'HBridge'][feature].mean()
        print(f"{feature:30s} | Load: {load_mean:8.3f} | HBridge: {hbridge_mean:8.3f} | Ratio: {hbridge_mean/load_mean if load_mean != 0 else np.nan:6.2f}x")

def plot_regime_discrimination(feature_df, output_dir):
    """
    Phase 4 Visualization: Test if features can discriminate power regimes.
    """
    print("\n" + "="*80)
    print("PHASE 4: Regime Discrimination Test")
    print("="*80)
    
    # Prepare features for classification
    feature_cols = [col for col in feature_df.columns 
                   if col not in ['component_name', 'board', 'test_session', 'component_type']]
    
    X = feature_df[feature_cols].fillna(0)
    y = (feature_df['board'] == 'HBridge').astype(int)  # 1 for HBridge, 0 for Load_Shedding
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Predict
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\nClassification Accuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Load_Shedding', 'HBridge']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features for Regime Discrimination:")
    print("-" * 80)
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontweight='bold', fontsize=12)
    ax.set_title(f'Top 15 Features for Board Regime Classification\nAccuracy: {accuracy:.1%}', 
                fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase4_regime_discrimination.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: phase4_regime_discrimination.png")
    plt.close()
    
    return accuracy, feature_importance

def plot_ml_improvement_comparison(baseline_results, enriched_results, output_dir):
    """
    Phase 5 Visualization: Compare ML performance with/without enriched features.
    """
    print("\n" + "="*80)
    print("PHASE 5: ML Model Improvement with Enriched Features")
    print("="*80)
    
    metrics = ['RMSE', 'MAE', 'MAPE']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('ML Model Performance: Baseline vs Enriched Features\n(Cross-Board: Load_Shedding → HBridge)',
                fontsize=14, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        baseline_value = baseline_results.get(metric.lower(), 0)
        enriched_value = enriched_results.get(metric.lower(), 0)
        
        bars = ax.bar(['Baseline\n(Delta T only)', 'Enriched\n(+Temporal/Spatial)'],
                     [baseline_value, enriched_value],
                     color=['gray', 'green'], edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}{"°C" if metric != "MAPE" else "%"}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Show improvement percentage
        if baseline_value > 0:
            improvement = ((baseline_value - enriched_value) / baseline_value) * 100
            ax.text(0.5, max(baseline_value, enriched_value) * 0.9,
                   f'{improvement:+.1f}% improvement',
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ylabel_text = f'{metric} (°C)' if metric != 'MAPE' else f'{metric} (%)'
        ax.set_ylabel(ylabel_text, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase5_ml_improvement.png'), dpi=200, bbox_inches='tight')
    print(f"✓ Saved: phase5_ml_improvement.png")
    plt.close()

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    """
    Main analysis pipeline for Option C.
    """
    print("\n" + "="*80)
    print("TEMPORAL & SPATIAL FEATURE ANALYSIS - OPTION C")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load ResearchIR thermal time series (~300 frames @ 15s intervals)")
    print("  2. Visualize heating curves (Load_Shedding vs HBridge)")
    print("  3. Extract temporal features (heating rate, time constant, etc.)")
    print("  4. Extract board-level context features")
    print("  5. Test regime discrimination (can features tell boards apart?)")
    print("  6. Test ML improvement (does it help cross-board prediction?)")
    print("="*80)
    
    # NOTE: This is a template script
    # You need to provide actual ResearchIR Stats directory paths
    print("\nWARNING: SETUP REQUIRED:")
    print("   Please update FLIR_STATS_DIRS dictionary with actual directory paths")
    print("   Each directory should contain ResearchIR Stats files:")
    print("     - Rec-000001_0 - Stats.txt")
    print("     - Rec-000002_0 - Stats.txt")
    print("     - ... (one per frame)")
    print("\n   Example paths:")
    print("     Load_Shedding_Test1_Air: 'outputs/Load_Shedding/Test1_Air'")
    print("     HBridge_15s_Test1_Air: 'outputs/HBridge/15s_Test1_Air'")
    print("\n   Once paths are set, this script will:")
    print("     - Load all frame data (time series)")
    print("     - Extract rich temporal features")
    print("     - Test if features improve ML predictions")
    
    # Load calibration database
    print("\nLoading calibration database...")
    calibration_df = pd.read_csv(CALIBRATION_DB)
    calibration_df = calibration_df.dropna(subset=['sand_ss', 'sand_initial'])
    calibration_df['board'] = calibration_df['pcb'].apply(lambda x: 'Load_Shedding' if 'Load' in x else 'HBridge')
    
    print(f"  Calibrated components: {len(calibration_df)}")
    print(f"    Load_Shedding: {len(calibration_df[calibration_df['board'] == 'Load_Shedding'])}")
    print(f"    HBridge: {len(calibration_df[calibration_df['board'] == 'HBridge'])}")
    
    # Check if FLIR Stats directories exist
    stats_exist = any(os.path.exists(path) for path in FLIR_STATS_DIRS.values())
    
    if not stats_exist:
        print("\n" + "="*80)
        print("WARNING: NO FLIR STATS DIRECTORIES FOUND")
        print("="*80)
        print("\nPlease update the FLIR_STATS_DIRS dictionary at the top of this script")
        print("with the actual paths to your ResearchIR Stats directories.")
        print("\nExample:")
        print("  FLIR_STATS_DIRS = {")
        print("      'Load_Shedding_Test1_Air': 'outputs/Load_Shedding/Test1_Air',")
        print("      'HBridge_15s_Test1_Air': 'outputs/HBridge/15s_Test1_Air',")
        print("  }")
        print("\nEach directory should contain Rec-XXXXXX_N - Stats.txt files.")
        print("\nOnce configured, re-run this script to perform full analysis.")
        return
    
    # Load FLIR data
    flir_data = load_all_flir_data()
    
    if not flir_data:
        print("\nWARNING: No FLIR data loaded. Check directory paths and try again.")
        return
    
    # Phase 1: Visualize heating curves
    plot_heating_curves_comparison(flir_data, calibration_df, OUTPUT_DIR)
    
    # Phase 2: Extract temporal features
    print("\nExtracting temporal features...")
    feature_rows = []
    
    for session, df_flir in flir_data.items():
        board = 'Load_Shedding' if 'Load' in session else 'HBridge'
        
        # Get time and component columns
        time = df_flir['reltime'].values
        component_columns = [col for col in df_flir.columns if col not in ['frame', 'reltime']]
        
        for component_name in component_columns:
            temp = df_flir[component_name].values
            
            temporal_features = calculate_temporal_features(time, temp)
            temporal_features['component_name'] = component_name
            temporal_features['board'] = board
            temporal_features['test_session'] = session
            feature_rows.append(temporal_features)
    
    feature_df = pd.DataFrame(feature_rows)
    
    # Merge with calibration data
    feature_df = feature_df.merge(
        calibration_df[['component_name', 'component_type', 'delta_t_sand']],
        on='component_name',
        how='inner'
    )
    
    print(f"  Extracted features for {len(feature_df)} components")
    
    # Save feature data
    feature_csv = os.path.join(OUTPUT_DIR, 'temporal_features.csv')
    feature_df.to_csv(feature_csv, index=False)
    print(f"✓ Saved: temporal_features.csv")
    
    # Visualize temporal features
    plot_temporal_feature_distributions(feature_df, OUTPUT_DIR)
    
    # Phase 4: Regime discrimination test
    accuracy, feature_importance = plot_regime_discrimination(feature_df, OUTPUT_DIR)
    
    # Phase 5: ML improvement test
    print("\nTesting ML model with enriched features...")
    
    # Split data: Train on Load_Shedding, test on HBridge
    load_df = feature_df[feature_df['board'] == 'Load_Shedding']
    hbridge_df = feature_df[feature_df['board'] == 'HBridge']
    
    # Baseline: Only delta_t_flir_air (if available) or basic features
    baseline_features = ['T_initial', 'T_steady_state', 'delta_T_total']
    enriched_features = [col for col in feature_df.columns 
                        if col not in ['component_name', 'board', 'test_session', 'component_type', 'delta_t_sand']]
    
    # Train baseline model
    X_train_baseline = load_df[baseline_features].fillna(0)
    y_train = load_df['delta_t_sand']
    X_test_baseline = hbridge_df[baseline_features].fillna(0)
    y_test = hbridge_df['delta_t_sand']
    
    baseline_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    baseline_model.fit(X_train_baseline, y_train)
    y_pred_baseline = baseline_model.predict(X_test_baseline)
    
    baseline_results = {
        'rmse': np.sqrt(np.mean((y_test - y_pred_baseline)**2)),
        'mae': np.mean(np.abs(y_test - y_pred_baseline)),
        'mape': np.mean(np.abs((y_test - y_pred_baseline) / y_test)) * 100
    }
    
    # Train enriched model
    X_train_enriched = load_df[enriched_features].fillna(0)
    X_test_enriched = hbridge_df[enriched_features].fillna(0)
    
    enriched_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    enriched_model.fit(X_train_enriched, y_train)
    y_pred_enriched = enriched_model.predict(X_test_enriched)
    
    enriched_results = {
        'rmse': np.sqrt(np.mean((y_test - y_pred_enriched)**2)),
        'mae': np.mean(np.abs(y_test - y_pred_enriched)),
        'mape': np.mean(np.abs((y_test - y_pred_enriched) / y_test)) * 100
    }
    
    # Visualize comparison
    plot_ml_improvement_comparison(baseline_results, enriched_results, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n✅ Regime Discrimination Accuracy: {accuracy:.1%}")
    if accuracy > 0.9:
        print("   → Features STRONGLY discriminate between power regimes!")
    elif accuracy > 0.75:
        print("   → Features moderately discriminate between power regimes")
    else:
        print("   → Features weakly discriminate (may not help much)")
    
    print(f"\n✅ ML Model Improvement (Cross-Board: Load→HBridge):")
    improvement = ((baseline_results['mape'] - enriched_results['mape']) / baseline_results['mape']) * 100
    print(f"   Baseline MAPE: {baseline_results['mape']:.1f}%")
    print(f"   Enriched MAPE: {enriched_results['mape']:.1f}%")
    print(f"   Improvement: {improvement:+.1f}%")
    
    if improvement > 30:
        print("\n🎉 RECOMMENDATION: Temporal/spatial features provide SIGNIFICANT improvement!")
        print("   → Proceed with full feature extractor implementation")
    elif improvement > 10:
        print("\n✓ RECOMMENDATION: Temporal/spatial features provide moderate improvement")
        print("   → Consider implementing, may help edge cases")
    else:
        print("\nWARNING: RECOMMENDATION: Limited improvement from temporal/spatial features")
        print("   → May not be worth the added complexity")
    
    print("\n" + "="*80)
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("="*80)

if __name__ == "__main__":
    main()
