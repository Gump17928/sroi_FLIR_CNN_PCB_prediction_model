"""
===============================================================================
PHASE 8: MACHINE LEARNING THERMAL PREDICTION
===============================================================================
Train linear regression model to predict sand embedded temperatures from
FLIR air measurements using Ordinary Least Squares (OLS) regression.

Model Structure:
    ΔT_sand = β₀ + β₁·ΔT_flir_air + β₂·is_IC + β₃·is_Resistor + ... + ε

Where:
    - ΔT_sand: Sand thermistor temperature rise (max - min)
    - ΔT_flir_air: FLIR air measurement temperature rise (max - min)
    - is_IC, is_Resistor, etc.: One-hot encoded component types

Features:
    - FLIR air delta T (continuous)
    - Component type (categorical, one-hot encoded)

Target:
    - Sand thermistor delta T (continuous)

Workflow:
    1. Load thermal_calibration_points.csv (from Phase 6)
    2. Filter to specific PCB (e.g., Load_Shedding)
    3. Calculate delta T for each component (max - min)
    4. One-hot encode component types
    5. Train OLS linear regression model
    6. Export trained model (pickle)
    7. Export metrics (R², RMSE, MAE, coefficients)
    8. Generate predictions for visualization

Usage:
    from phase8_ml_training import ThermalMLPredictor
    
    predictor = ThermalMLPredictor(output_dir="outputs/phase8")
    predictor.load_training_data("thermal_calibration_points.csv", pcb_filter="Load_Shedding")
    predictor.train_model()
    predictor.export_model()
    predictor.export_metrics()

Created: January 7, 2026
===============================================================================
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ThermalMLPredictor:
    """
    Machine learning predictor for embedded thermal behavior using OLS regression.
    
    Predicts sand embedded temperatures from FLIR air measurements and component type.
    Uses Ordinary Least Squares linear regression with one-hot encoded component types.
    """
    
    # Standard component types for one-hot encoding
    COMPONENT_TYPES = ['IC', 'Resistor', 'PowerSupply', 'LED', 'Connector', 'Capacitor', 'Inductor', 'Diode']
    
    def __init__(self, output_dir: str = "outputs/phase8", verbose: bool = True):
        """
        Initialize ML predictor.
        
        Args:
            output_dir: Directory for output files (model, metrics, plots)
            verbose: Enable verbose output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Training data
        self.df_raw = None
        self.df_features = None
        self.X = None  # Feature matrix
        self.y = None  # Target vector
        self.component_names = None
        self.component_types = None
        
        # Trained model (sklearn LinearRegression)
        self.model = None
        
        # Performance metrics
        self.metrics = {}
        
    def load_training_data(self, calibration_csv: str, pcb_filter: str = None):
        """
        Load and prepare training data from calibration database.
        
        Loads thermal_calibration_points.csv from Phase 6, filters to specific PCB,
        and extracts FLIR air and sand thermistor measurements for delta T calculation.
        
        Args:
            calibration_csv: Path to thermal_calibration_points.csv
            pcb_filter: Filter to specific PCB name (e.g., "Load_Shedding")
        """
        if self.verbose:
            print(f"\n[PHASE 8 - DATA LOADING]")
            print(f"  Loading calibration data: {calibration_csv}")
        
        # Load calibration database
        self.df_raw = pd.read_csv(calibration_csv)
        
        if self.verbose:
            print(f"  Total calibration points: {len(self.df_raw)}")
        
        # Filter to specific PCB if requested
        if pcb_filter:
            # Try matching against multiple columns (flexible matching)
            # Extract key parts from filter (e.g., "Load_Shedding" -> "Load")
            filter_parts = pcb_filter.replace('_', ' ').split()
            
            if 'board_name' in self.df_raw.columns:
                # Match against board_name (e.g., "Load" matches "Load_Shedding")
                mask = self.df_raw['board_name'].apply(
                    lambda x: any(part.lower() in str(x).lower() for part in filter_parts) if pd.notna(x) else False
                )
            elif 'pcb' in self.df_raw.columns:
                # Fallback to 'pcb' column
                mask = self.df_raw['pcb'].str.contains(pcb_filter, case=False, na=False)
            else:
                raise ValueError("Cannot find 'board_name' or 'pcb' column for filtering")
            
            self.df_raw = self.df_raw[mask].copy()
            if self.verbose:
                print(f"  Filtered to {pcb_filter}: {len(self.df_raw)} points")
        
        # Check required columns exist (using actual column names from calibration database)
        required_cols = ['component_name', 'component_type', 'flir_ss', 'air_ss', 'sand_ss']
        missing_cols = [col for col in required_cols if col not in self.df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in calibration data: {missing_cols}")
        
        # Filter to only components with both FLIR and sand data
        # Remove rows where sand_ss is NaN
        initial_count = len(self.df_raw)
        self.df_raw = self.df_raw.dropna(subset=['sand_ss']).copy()
        removed_count = initial_count - len(self.df_raw)
        
        if self.verbose and removed_count > 0:
            print(f"  Removed {removed_count} points without sand data")
            print(f"  Final dataset: {len(self.df_raw)} points")
        
        if len(self.df_raw) == 0:
            raise ValueError("No valid training data after filtering (need both FLIR and sand measurements)")
    
    def calculate_delta_t(self):
        """
        Calculate delta T (temperature rise above ambient) for each measurement.
        
        For each test session, finds the minimum temperature as ambient reference,
        then calculates temperature rise for each component.
        Creates feature dataset with delta_t_flir_air, delta_t_sand, component_type.
        """
        if self.verbose:
            print(f"\n[PHASE 8 - DELTA T CALCULATION]")
        
        # Calculate delta T for each row (measurement) as temperature rise above initial baseline
        component_deltas = []
        
        # Check if initial temperature columns exist
        has_initial_temps = 'air_initial' in self.df_raw.columns and 'sand_initial' in self.df_raw.columns
        
        if not has_initial_temps:
            print("\nWarning: air_initial and sand_initial columns not found in calibration database.")
            print("Using legacy method (steady-state minimum as ambient).")
            print("For accurate results, regenerate calibration database with Phase 6.\n")
        
        # Group by test session to calculate delta T
        for test_session, test_group in self.df_raw.groupby('test_session'):
            for _, row in test_group.iterrows():
                comp_name = row['component_name']
                comp_type = row['component_type']
                
                if has_initial_temps:
                    # NEW METHOD: Use initial temperatures as ambient baseline
                    # delta T = steady_state - initial (temperature rise during test)
                    delta_t_air = row['air_ss'] - row['air_initial']
                    delta_t_sand = row['sand_ss'] - row['sand_initial']
                    # Note: FLIR doesn't have initial temp, use air as proxy
                    delta_t_flir_air = delta_t_air
                else:
                    # LEGACY METHOD: Use minimum steady-state as ambient (INCORRECT)
                    ambient_flir = test_group['flir_ss'].min()
                    ambient_air = test_group['air_ss'].min()
                    ambient_sand = test_group['sand_ss'].min()
                    delta_t_flir_air = row['air_ss'] - ambient_air
                    delta_t_sand = row['sand_ss'] - ambient_sand
                
                component_deltas.append({
                    'component': comp_name,
                    'component_type': comp_type,
                    'delta_t_flir_air': delta_t_flir_air,
                    'delta_t_sand': delta_t_sand,
                    'n_measurements': 1,
                    'test_session': test_session
                })
        
        self.df_features = pd.DataFrame(component_deltas)
        
        if self.verbose:
            print(f"  Calculated delta T for {len(self.df_features)} components")
            print(f"  Component types: {self.df_features['component_type'].unique()}")
            print(f"\n  Delta T Statistics:")
            print(f"    FLIR Air:  {self.df_features['delta_t_flir_air'].mean():.1f} ± {self.df_features['delta_t_flir_air'].std():.1f} °C")
            print(f"    Sand:      {self.df_features['delta_t_sand'].mean():.1f} ± {self.df_features['delta_t_sand'].std():.1f} °C")
    
    def prepare_feature_matrix(self):
        """
        Prepare feature matrix X and target vector y with one-hot encoded component types.
        
        Creates feature matrix with:
            - Column 0: delta_t_flir_air (continuous)
            - Columns 1-N: One-hot encoded component types
        
        Target vector y: delta_t_sand (continuous)
        """
        if self.verbose:
            print(f"\n[PHASE 8 - FEATURE ENGINEERING]")
        
        # Extract component names and types
        self.component_names = self.df_features['component'].values
        self.component_types = self.df_features['component_type'].values
        
        # Create one-hot encoding for component types
        n_samples = len(self.df_features)
        n_types = len(self.COMPONENT_TYPES)
        
        # Initialize feature matrix: [delta_t_flir_air, type_0, type_1, ..., type_N]
        X_list = []
        feature_names = ['delta_t_flir_air']
        
        # Add FLIR delta T as first feature
        X_list.append(self.df_features['delta_t_flir_air'].values.reshape(-1, 1))
        
        # Add one-hot encoded component types
        for comp_type in self.COMPONENT_TYPES:
            is_type = (self.component_types == comp_type).astype(float).reshape(-1, 1)
            X_list.append(is_type)
            feature_names.append(f'type_{comp_type}')
        
        # Combine into feature matrix
        self.X = np.hstack(X_list)
        self.feature_names = feature_names
        
        # Extract target vector
        self.y = self.df_features['delta_t_sand'].values
        
        if self.verbose:
            print(f"  Feature matrix shape: {self.X.shape}")
            print(f"  Features: {self.feature_names}")
            print(f"  Target vector shape: {self.y.shape}")
            print(f"\n  Component Type Distribution:")
            for comp_type in self.COMPONENT_TYPES:
                count = np.sum(self.component_types == comp_type)
                if count > 0:
                    print(f"    {comp_type}: {count} components")
    
    def train_model(self):
        """
        Train scikit-learn Linear Regression model using Ordinary Least Squares.
        
        Uses sklearn.linear_model.LinearRegression which implements OLS regression.
        
        Model equation:
            y = X β + ε
            
        Where:
            - y: delta_t_sand (target)
            - X: [delta_t_flir_air, type_IC, type_Resistor, ...] (features)
            - β: model coefficients
            - ε: residuals
        """
        if self.verbose:
            print(f"\n[PHASE 8 - MODEL TRAINING]")
            print(f"  Training sklearn Linear Regression (OLS)...")
        
        # Initialize sklearn linear regression model
        self.model = LinearRegression()
        
        # Fit model: model.fit(X, y)
        self.model.fit(self.X, self.y)
        
        if self.verbose:
            print(f"  Model trained successfully")
            print(f"\n  Model Equation:")
            print(f"    ΔT_sand = {self.model.intercept_:.3f}", end='')
            for i, (coef, feat_name) in enumerate(zip(self.model.coef_, self.feature_names)):
                sign = '+' if coef >= 0 else ''
                print(f" {sign}{coef:.3f}·{feat_name}", end='')
            print()
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate model performance metrics using sklearn.
        
        Computes:
            - R²: Coefficient of determination
            - RMSE: Root mean squared error
            - MAE: Mean absolute error
        
        Returns:
            Dictionary with metric names and values
        """
        # Generate predictions
        y_pred = self.model.predict(self.X)
        
        # Calculate metrics using sklearn
        r_squared = r2_score(self.y, y_pred)
        mse = mean_squared_error(self.y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y, y_pred)
        
        self.metrics = {
            'R²': r_squared,
            'RMSE': rmse,
            'MAE': mae,
            'n_samples': len(self.y),
            'n_features': len(self.feature_names)
        }
        
        if self.verbose:
            print(f"\n[PHASE 8 - MODEL PERFORMANCE]")
            print(f"  R² Score:     {r_squared:.4f}")
            print(f"  RMSE:         {rmse:.2f} °C")
            print(f"  MAE:          {mae:.2f} °C")
            print(f"  Samples:      {len(self.y)}")
            print(f"  Features:     {len(self.feature_names)}")
        
        return self.metrics
    
    def load_trained_model(self, model_path: str):
        """
        Load previously trained model from pickle file.
        
        Args:
            model_path: Path to .pkl model file
        """
        if self.verbose:
            print(f"\n[PHASE 8 - MODEL LOADING]")
            print(f"  Loading trained model: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.COMPONENT_TYPES = model_data['component_types']
        self.metrics = model_data.get('metrics', {})
        
        if self.verbose:
            print(f"  Model loaded successfully")
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Component types: {len(self.COMPONENT_TYPES)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained sklearn model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
        
        Returns:
            Predictions [n_samples]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return self.model.predict(X)
    
    def predict_sand_delta_t(self, flir_delta_t: float, component_type: str) -> float:
        """
        Predict sand delta T for a single component.
        
        Convenience function for making predictions on new data.
        
        Args:
            flir_delta_t: FLIR air temperature rise (°C)
            component_type: Component type (e.g., 'IC', 'Resistor')
        
        Returns:
            Predicted sand delta T (°C)
        """
        # Build feature vector
        features = [flir_delta_t]
        for comp_type in self.COMPONENT_TYPES:
            features.append(1.0 if comp_type == component_type else 0.0)
        
        X_single = np.array([features])
        return self.predict(X_single)[0]
    
    def export_model(self, filename: str = None):
        """
        Export trained sklearn model to pickle file.
        
        Saves complete sklearn LinearRegression model object including
        coefficients, intercept, and feature names for later use.
        
        Args:
            filename: Output filename (default: auto-generated from PCB name)
        """
        if filename is None:
            # Auto-generate filename from PCB name if available
            if 'pcb' in self.df_raw.columns:
                pcb_name = self.df_raw['pcb'].iloc[0]
                filename = f"{pcb_name}_thermal_ml_model.pkl"
            else:
                filename = "thermal_ml_model.pkl"
        
        model_path = self.output_dir / filename
        
        # Package model data (save sklearn model + metadata)
        model_data = {
            'model': self.model,  # sklearn LinearRegression object
            'feature_names': self.feature_names,
            'component_types': self.COMPONENT_TYPES,
            'metrics': self.metrics
        }
        
        # Save to pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"\n[PHASE 8 - MODEL EXPORT]")
            print(f"  Saved model: {model_path}")
    
    def export_metrics(self, filename: str = None):
        """
        Export model metrics to CSV file.
        
        Creates CSV with:
            - R², RMSE, MAE
            - Model intercept
            - All feature coefficients
        
        Args:
            filename: Output filename (default: auto-generated from PCB name)
        """
        if filename is None:
            # Auto-generate filename from PCB name if available
            if 'pcb' in self.df_raw.columns:
                pcb_name = self.df_raw['pcb'].iloc[0]
                filename = f"{pcb_name}_ml_metrics.csv"
            else:
                filename = "ml_metrics.csv"
        
        metrics_path = self.output_dir / filename
        
        # Build metrics dataframe
        rows = []
        
        # Add performance metrics
        rows.append({'Metric': 'R²', 'Value': f"{self.metrics['R²']:.6f}"})
        rows.append({'Metric': 'RMSE', 'Value': f"{self.metrics['RMSE']:.4f}"})
        rows.append({'Metric': 'MAE', 'Value': f"{self.metrics['MAE']:.4f}"})
        rows.append({'Metric': 'n_samples', 'Value': str(self.metrics['n_samples'])})
        rows.append({'Metric': 'n_features', 'Value': str(self.metrics['n_features'])})
        
        # Add model coefficients
        rows.append({'Metric': 'Intercept', 'Value': f"{self.model.intercept_:.6f}"})
        for feat_name, coef in zip(self.feature_names, self.model.coef_):
            rows.append({'Metric': f'Coef_{feat_name}', 'Value': f"{coef:.6f}"})
        
        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(metrics_path, index=False)
        
        if self.verbose:
            print(f"  Saved metrics: {metrics_path}")
    
    def export_training_data(self, filename: str = None):
        """
        Export feature dataset to CSV for debugging/inspection.
        
        Saves component names, types, delta T values, and predictions.
        
        Args:
            filename: Output filename (default: auto-generated from PCB name)
        """
        if filename is None:
            # Auto-generate filename from PCB name if available
            if 'pcb' in self.df_raw.columns:
                pcb_name = self.df_raw['pcb'].iloc[0]
                filename = f"{pcb_name}_ml_training_data.csv"
            else:
                filename = "ml_training_data.csv"
        
        data_path = self.output_dir / filename
        
        # Add predictions to feature dataframe
        self.df_features['predicted_sand_delta_t'] = self.predict(self.X)
        self.df_features['residual'] = self.y - self.df_features['predicted_sand_delta_t']
        
        # Export
        self.df_features.to_csv(data_path, index=False)
        
        if self.verbose:
            print(f"  Saved training data: {data_path}")
    
    def get_predictions_for_visualization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions and actual values for visualization.
        
        Returns:
            Tuple of (y_true, y_pred, component_types)
        """
        y_pred = self.predict(self.X)
        return self.y, y_pred, self.component_types
    
    def predict_on_board(self, calibration_file: str, target_pcb: str) -> pd.DataFrame:
        """
        Predict sand temps for a different board using trained model.
        
        Args:
            calibration_file: Path to thermal_calibration_points.csv
            target_pcb: PCB name to predict (e.g., "HBridge")
        
        Returns:
            DataFrame with predictions, actuals, and residuals
        """
        if self.model is None:
            raise ValueError("No trained model loaded. Call load_trained_model() or train_model() first.")
        
        if self.verbose:
            print(f"\n[PHASE 8 - CROSS-BOARD PREDICTION]")
            print(f"  Target board: {target_pcb}")
        
        # Load target board calibration data (reuse existing method)
        original_verbose = self.verbose
        self.verbose = False  # Suppress loading messages
        
        # Save current training data
        train_df_raw = self.df_raw
        
        # Load test board data
        self.load_training_data(calibration_file, pcb_filter=target_pcb)
        test_df_raw = self.df_raw
        
        # Calculate delta T for test board
        self.calculate_delta_t()
        test_df_features = self.df_features
        
        # Prepare feature matrix for test board
        self.prepare_feature_matrix()
        X_test = self.X
        y_test = self.y
        test_component_names = self.component_names
        test_component_types = self.component_types
        
        # Restore training data
        self.df_raw = train_df_raw
        self.verbose = original_verbose
        
        if self.verbose:
            print(f"  Test samples: {len(X_test)}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Build results dataframe
        results = pd.DataFrame({
            'component': test_component_names,
            'component_type': test_component_types,
            'delta_t_flir_air': X_test[:, 0],  # First column is delta_t_flir_air
            'actual_sand_delta_t': y_test,
            'predicted_sand_delta_t': y_pred,
            'residual': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        
        # Calculate validation metrics
        val_metrics = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'n_samples': len(y_test)
        }
        
        if self.verbose:
            print(f"\n[VALIDATION METRICS]")
            print(f"  R² Score:  {val_metrics['R²']:.4f}")
            print(f"  RMSE:      {val_metrics['RMSE']:.2f} °C")
            print(f"  MAE:       {val_metrics['MAE']:.2f} °C")
            print(f"  Samples:   {val_metrics['n_samples']}")
        
        return results, val_metrics, y_test, y_pred, test_component_types


def load_pretrained_model(model_path: str) -> Dict:
    """
    Load pretrained model from pickle file.
    
    Args:
        model_path: Path to .pkl model file
    
    Returns:
        Dictionary with model data (intercept, coefficients, feature_names, etc.)
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_with_model(model_data: Dict, flir_delta_t: float, component_type: str) -> float:
    """
    Make prediction using loaded sklearn model.
    
    Args:
        model_data: Model dictionary from load_pretrained_model()
        flir_delta_t: FLIR air temperature rise (°C)
        component_type: Component type (e.g., 'IC', 'Resistor')
    
    Returns:
        Predicted sand delta T (°C)
    """
    # Build feature vector
    features = [flir_delta_t]
    for comp_type in model_data['component_types']:
        features.append(1.0 if comp_type == component_type else 0.0)
    
    X = np.array([features])
    
    # Predict using sklearn model
    y_pred = model_data['model'].predict(X)
    
    return y_pred[0]


def validate_cross_board(
    trained_model_path: str,
    calibration_db_path: str,
    test_pcb: str,
    output_dir: str,
    verbose: bool = True
) -> Dict:
    """
    Validate trained model on different PCB (cross-board validation).
    
    Loads model trained on one board, predicts temperatures on another board,
    and compares to actual measurements for validation.
    
    Args:
        trained_model_path: Path to trained model .pkl file
        calibration_db_path: Path to thermal_calibration_points.csv
        test_pcb: PCB name to test on (e.g., "HBridge", "Load_Shedding")
        output_dir: Directory for validation output files
        verbose: Enable verbose output
    
    Returns:
        Dictionary with predictions, metrics, and output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"[PHASE 8 - CROSS-BOARD VALIDATION]")
        print(f"{'='*80}")
        print(f"Trained model: {trained_model_path}")
        print(f"Test PCB: {test_pcb}")
        print(f"Calibration data: {calibration_db_path}")
        print(f"{'='*80}\n")
    
    # Load trained model
    if verbose:
        print(f"[1] Loading trained model...")
    
    with open(trained_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    component_types = model_data['component_types']
    training_metrics = model_data.get('metrics', {})
    
    if verbose:
        print(f"  Model loaded: {len(feature_names)} features")
        print(f"  Training R²: {training_metrics.get('R²', 'N/A'):.4f}" if 'R²' in training_metrics else "")
    
    # Load calibration database
    if verbose:
        print(f"\n[2] Loading test board calibration data...")
    
    df_cal = pd.read_csv(calibration_db_path)
    
    # Filter to test PCB
    initial_count = len(df_cal)
    
    # Flexible PCB filtering (handle "Load_Shedding" matching "Load" in board_name)
    filter_parts = test_pcb.split('_')
    if 'board_name' in df_cal.columns:
        mask = df_cal['board_name'].apply(
            lambda x: any(part in str(x) for part in filter_parts)
        )
        df_test = df_cal[mask].copy()
    elif 'pcb' in df_cal.columns:
        mask = df_cal['pcb'].apply(
            lambda x: any(part in str(x) for part in filter_parts)
        )
        df_test = df_cal[mask].copy()
    else:
        raise ValueError("Calibration database missing board_name or pcb column")
    
    if verbose:
        print(f"  Total calibration points: {initial_count}")
        print(f"  Filtered to {test_pcb}: {len(df_test)} points")
    
    # Remove rows without sand data
    df_test = df_test.dropna(subset=['sand_ss']).copy()
    
    if len(df_test) == 0:
        raise ValueError(f"No valid test data for {test_pcb} after filtering")
    
    if verbose:
        print(f"  Valid test samples (with sand data): {len(df_test)}")
    
    # Calculate delta T for test board (same method as training)
    if verbose:
        print(f"\n[3] Calculating delta T for test samples...")
    
    # Check if initial temperature columns exist
    has_initial_temps = 'air_initial' in df_test.columns and 'sand_initial' in df_test.columns
    
    if not has_initial_temps and verbose:
        print("  Warning: air_initial and sand_initial columns not found.")
        print("  Using legacy method (steady-state minimum as ambient).")
    
    test_deltas = []
    
    for test_session, test_group in df_test.groupby('test_session'):
        # Calculate delta T for each component
        for _, row in test_group.iterrows():
            if has_initial_temps:
                # NEW METHOD: Use initial temperatures as ambient baseline
                delta_t_air = row['air_ss'] - row['air_initial']
                delta_t_sand = row['sand_ss'] - row['sand_initial']
                delta_t_flir_air = delta_t_air
            else:
                # LEGACY METHOD: Use minimum steady-state as ambient (INCORRECT)
                ambient_air = test_group['air_ss'].min()
                ambient_sand = test_group['sand_ss'].min()
                delta_t_air = row['air_ss'] - ambient_air
                delta_t_sand = row['sand_ss'] - ambient_sand
                delta_t_flir_air = delta_t_air
            
            test_deltas.append({
                'component': row['component_name'],
                'component_type': row['component_type'],
                'delta_t_flir_air': delta_t_flir_air,
                'delta_t_sand': delta_t_sand,
                'test_session': test_session
            })
    
    df_test_features = pd.DataFrame(test_deltas)
    
    if verbose:
        print(f"  Test samples prepared: {len(df_test_features)}")
        print(f"  Component types: {df_test_features['component_type'].unique()}")
    
    # Prepare feature matrix (same one-hot encoding as training)
    if verbose:
        print(f"\n[4] Preparing feature matrix...")
    
    X_test_list = []
    
    for _, row in df_test_features.iterrows():
        features = [row['delta_t_flir_air']]
        
        # One-hot encode component type (use same order as training)
        for comp_type in component_types:
            features.append(1.0 if row['component_type'] == comp_type else 0.0)
        
        X_test_list.append(features)
    
    X_test = np.array(X_test_list)
    y_test = df_test_features['delta_t_sand'].values
    
    if verbose:
        print(f"  Feature matrix: {X_test.shape}")
        print(f"  Target vector: {y_test.shape}")
    
    # Make predictions
    if verbose:
        print(f"\n[5] Making predictions...")
    
    y_pred = model.predict(X_test)
    
    # Calculate validation metrics
    val_metrics = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'n_samples': len(y_test),
        'train_R²': training_metrics.get('R²', np.nan),
        'train_RMSE': training_metrics.get('RMSE', np.nan),
        'train_MAE': training_metrics.get('MAE', np.nan)
    }
    
    if verbose:
        print(f"\n[VALIDATION RESULTS]")
        print(f"  Test R²:      {val_metrics['R²']:.4f}")
        print(f"  Test RMSE:    {val_metrics['RMSE']:.2f} °C")
        print(f"  Test MAE:     {val_metrics['MAE']:.2f} °C")
        print(f"  Samples:      {val_metrics['n_samples']}")
        
        if not np.isnan(val_metrics['train_R²']):
            r2_degradation = val_metrics['train_R²'] - val_metrics['R²']
            print(f"\n[GENERALIZATION]")
            print(f"  Training R²:  {val_metrics['train_R²']:.4f}")
            print(f"  Test R²:      {val_metrics['R²']:.4f}")
            print(f"  Degradation:  {r2_degradation:.4f} ({r2_degradation/val_metrics['train_R²']*100:.1f}%)")
    
    # Calculate percent errors
    # For components with very small delta T (<1°C), percent error is not meaningful
    # Use a minimum threshold to avoid astronomical percent errors
    MIN_DELTA_T_FOR_PERCENT = 1.0  # Only calculate % error for delta T >= 1°C
    
    # Calculate percent errors (only for components above threshold)
    percent_errors = np.zeros_like(y_test)
    abs_percent_errors = np.zeros_like(y_test)
    valid_for_percent = np.abs(y_test) >= MIN_DELTA_T_FOR_PERCENT
    
    if np.any(valid_for_percent):
        percent_errors[valid_for_percent] = ((y_pred[valid_for_percent] - y_test[valid_for_percent]) / y_test[valid_for_percent]) * 100
        abs_percent_errors[valid_for_percent] = np.abs(percent_errors[valid_for_percent])
    
    # For low delta T components, mark as NaN (not applicable)
    percent_errors[~valid_for_percent] = np.nan
    abs_percent_errors[~valid_for_percent] = np.nan
    
    # Build results dataframe
    results_df = df_test_features.copy()
    results_df['predicted_sand_delta_t'] = y_pred
    results_df['residual'] = y_test - y_pred
    results_df['abs_error'] = np.abs(y_test - y_pred)
    results_df['percent_error'] = percent_errors
    results_df['abs_percent_error'] = abs_percent_errors
    results_df['within_10pct'] = abs_percent_errors <= 10
    results_df['within_20pct'] = abs_percent_errors <= 20
    
    # Export predictions CSV
    predictions_file = output_path / f"{test_pcb}_ml_validation_predictions.csv"
    results_df.to_csv(predictions_file, index=False)
    
    if verbose:
        print(f"\n[6] Exporting results...")
        print(f"  Saved predictions: {predictions_file}")
    
    # Export detailed errors CSV
    detailed_errors_file = output_path / f"{test_pcb}_ml_validation_detailed_errors.csv"
    results_df.to_csv(detailed_errors_file, index=False)
    
    if verbose:
        print(f"  Saved detailed errors: {detailed_errors_file}")
    
    # Calculate and export error summary statistics (excluding NaN values from low delta T)
    valid_percent_mask = ~np.isnan(percent_errors)
    n_valid_percent = np.sum(valid_percent_mask)
    
    if n_valid_percent > 0:
        error_summary = {
            'Mean_Percent_Error': np.nanmean(percent_errors),
            'Std_Percent_Error': np.nanstd(percent_errors),
            'Median_Percent_Error': np.nanmedian(percent_errors),
            'MAPE': np.nanmean(abs_percent_errors),
            'Components_Analyzed': f"{n_valid_percent}/{len(y_test)}",
            'Components_Below_Threshold': f"{len(y_test) - n_valid_percent} (ΔT < {MIN_DELTA_T_FOR_PERCENT}°C)",
            'Components_Within_10pct': f"{np.nansum(abs_percent_errors <= 10)}/{n_valid_percent}",
            'Percent_Within_10pct': f"{np.nansum(abs_percent_errors <= 10) / n_valid_percent * 100:.1f}%",
            'Components_Within_20pct': f"{np.nansum(abs_percent_errors <= 20)}/{n_valid_percent}",
            'Percent_Within_20pct': f"{np.nansum(abs_percent_errors <= 20) / n_valid_percent * 100:.1f}%",
            'Max_Error_Component': df_test_features['component'].iloc[np.nanargmax(abs_percent_errors)],
            'Max_Error_Value': f"{np.nanmax(abs_percent_errors):.1f}%"
        }
    else:
        # All components below threshold
        error_summary = {
            'Mean_Percent_Error': 'N/A',
            'Std_Percent_Error': 'N/A',
            'Median_Percent_Error': 'N/A',
            'MAPE': 'N/A',
            'Components_Analyzed': f"0/{len(y_test)}",
            'Components_Below_Threshold': f"{len(y_test)} (all below {MIN_DELTA_T_FOR_PERCENT}°C)",
            'Components_Within_10pct': 'N/A',
            'Percent_Within_10pct': 'N/A',
            'Components_Within_20pct': 'N/A',
            'Percent_Within_20pct': 'N/A',
            'Max_Error_Component': 'N/A',
            'Max_Error_Value': 'N/A'
        }
    
    error_summary_file = output_path / f"{test_pcb}_ml_validation_error_summary.csv"
    error_summary_df = pd.DataFrame([error_summary])
    error_summary_df.to_csv(error_summary_file, index=False)
    
    if verbose:
        print(f"  Saved error summary: {error_summary_file}")
        if n_valid_percent > 0:
            print(f"\n[PERCENT ERROR ANALYSIS] ({n_valid_percent}/{len(y_test)} components with ΔT >= {MIN_DELTA_T_FOR_PERCENT}°C)")
            print(f"  Mean Error:       {error_summary['Mean_Percent_Error']:+.1f}%")
            print(f"  Median Error:     {error_summary['Median_Percent_Error']:+.1f}%")
            print(f"  MAPE:             {error_summary['MAPE']:.1f}%")
            print(f"  Within ±10%:      {error_summary['Components_Within_10pct']} ({error_summary['Percent_Within_10pct']})")
            print(f"  Within ±20%:      {error_summary['Components_Within_20pct']} ({error_summary['Percent_Within_20pct']})")
            print(f"  Worst prediction: {error_summary['Max_Error_Component']} ({error_summary['Max_Error_Value']})")
        else:
            print(f"\n[PERCENT ERROR ANALYSIS] All components have ΔT < {MIN_DELTA_T_FOR_PERCENT}°C - using absolute error instead")
    
    # Export metrics CSV
    metrics_file = output_path / f"{test_pcb}_ml_validation_metrics.csv"
    metrics_df = pd.DataFrame([val_metrics])
    metrics_df.to_csv(metrics_file, index=False)
    
    if verbose:
        print(f"  Saved metrics: {metrics_file}")
    
    # Return results
    return {
        'predictions': results_df,
        'metrics': val_metrics,
        'error_summary': error_summary,
        'y_true': y_test,
        'y_pred': y_pred,
        'component_types': df_test_features['component_type'].values,
        'component_names': df_test_features['component'].values,
        'predictions_file': str(predictions_file),
        'metrics_file': str(metrics_file)
    }
