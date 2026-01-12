"""
Phase 7: Thermal Prediction Module
===================================
Applies learned thermal calibrations to predict component temperatures from FLIR-only data.

This module:
1. Loads calibration database from Phase 6
2. Applies FLIR correction offsets to camera measurements
3. Predicts sand cooling temperatures from air measurements
4. Handles component type mapping and unknown component types
5. Exports predicted temperature profiles for all PCB components

Key Inputs:
- thermal_calibration_by_type.csv: Component type calibration data
- FLIR camera CSV export (air cooling)
- Component type mapping (from SROI or manual configuration)

Key Outputs:
- predicted_temperatures.csv: Corrected and predicted temps for all components
- prediction_summary.json: Prediction metadata and confidence metrics
- prediction_visualization.png: Before/after correction plots

Author: Thermal Analysis Pipeline
Date: November 25, 2025
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import re

# Import data loaders
import sys
sys.path.insert(0, str(Path(__file__).parent))
from loader_thermistor import ThermalDataLoader, SteadyStateAnalyzer


class ThermalPredictor:
    """Predicts component temperatures using learned calibrations."""
    
    def __init__(self, calibration_file: Path, output_dir: Optional[Path] = None, verbose: bool = False):
        """
        Initialize thermal predictor.
        
        Args:
            calibration_file: Path to thermal_calibration_by_type.csv
            output_dir: Directory for prediction output files
            verbose: If True, print detailed per-component output
            output_dir: Directory for prediction outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Load calibration database
        self.calibration_db = pd.read_csv(calibration_file, index_col='component_type')
        print(f"\nLoaded calibration database: {calibration_file}")
        print(f"  Component types: {', '.join(self.calibration_db.index.tolist())}")
        
        self.predictions = []
        self.component_mapping = {}
    
    def set_component_type_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set mapping from FLIR channel names/patterns to component types.
        
        Args:
            mapping: Dict mapping FLIR pattern -> component type
                     e.g., {'Cursor 1': 'Resistor', 'Cursor 2': 'IC', ...}
        """
        self.component_mapping = mapping
        print(f"\nComponent type mapping configured:")
        for pattern, comp_type in mapping.items():
            print(f"  {pattern} -> {comp_type}")
    
    def predict_from_flir(self,
                         flir_file: Path,
                         pcb_name: str = "Unknown",
                         use_median_offset: bool = True) -> pd.DataFrame:
        """
        Predict corrected temperatures from FLIR measurements.
        
        Args:
            flir_file: Path to FLIR camera CSV export
            pcb_name: Name of PCB for tracking
            use_median_offset: Use median instead of mean for offset (more robust)
            
        Returns:
            DataFrame with predictions for all components
        """
        print(f"\n{'='*80}")
        print(f"Predicting Temperatures: {pcb_name}")
        print(f"{'='*80}")
        
        # Load FLIR data
        loader = ThermalDataLoader()
        x_flir, y_flir, labels_flir, meta_flir = loader.load_temp_csv(flir_file)
        
        print(f"\nLoaded FLIR data: {len(x_flir)} samples, {y_flir.shape[1]} channels")
        if self.verbose:
            print(f"  Channels: {', '.join(labels_flir[:5])}{'...' if len(labels_flir) > 5 else ''}")
        
        # Process each FLIR channel
        predictions = []
        total_components = len(labels_flir)
        unknown_count = 0
        uncalibrated_count = 0
        
        for i, label in enumerate(labels_flir):
            # Progress indicator (every 25 components or at end)
            if not self.verbose and (i % 25 == 0 or i == total_components - 1):
                print(f"\r  Processing components... {i+1}/{total_components}", end='', flush=True)
            
            if self.verbose:
                print(f"\n  Processing: {label}")
            
            # Calculate steady-state from FLIR
            ss_flir = SteadyStateAnalyzer.steady_value(x_flir, y_flir[:, i])
            flir_ss_temp = ss_flir['value']
            
            # Determine component type
            comp_type = self._determine_component_type(label)
            
            if comp_type is None:
                if self.verbose:
                    print(f"    Warning: Unknown component type, using default calibration")
                comp_type = 'Unknown'
                unknown_count += 1
            
            if self.verbose:
                print(f"    Component Type: {comp_type}")
                print(f"    FLIR SS Temp: {flir_ss_temp:.2f}°C")
            
            # Get calibration parameters
            if comp_type in self.calibration_db.index:
                cal = self.calibration_db.loc[comp_type]
                
                # Choose offset method
                offset = cal['flir_offset_median'] if use_median_offset else cal['flir_offset_mean']
                cooling_benefit = cal['cooling_benefit_mean']
                cooling_ratio = cal['cooling_ratio_mean']
                
                # Apply corrections
                corrected_air_temp = flir_ss_temp - offset  # Subtract offset to get true temp
                
                # Predict sand cooling (two methods)
                if pd.notna(cooling_benefit):
                    predicted_sand_temp_delta = corrected_air_temp - cooling_benefit
                else:
                    predicted_sand_temp_delta = None
                
                if pd.notna(cooling_ratio):
                    predicted_sand_temp_ratio = corrected_air_temp * cooling_ratio
                else:
                    predicted_sand_temp_ratio = None
                
                # Use delta method as primary, ratio as secondary
                if predicted_sand_temp_delta is not None:
                    predicted_sand_temp = predicted_sand_temp_delta
                elif predicted_sand_temp_ratio is not None:
                    predicted_sand_temp = predicted_sand_temp_ratio
                else:
                    predicted_sand_temp = None
                
                confidence = 'High' if pd.notna(cal['flir_offset_std']) and cal['flir_offset_std'] < 2.0 else 'Medium'
                
                if self.verbose:
                    print(f"    FLIR Offset Applied: {offset:+.2f}°C")
                    print(f"    Corrected Air Temp: {corrected_air_temp:.2f}°C")
                    if predicted_sand_temp is not None:
                        print(f"    Predicted Sand Temp: {predicted_sand_temp:.2f}°C")
                        print(f"    Predicted Cooling: {corrected_air_temp - predicted_sand_temp:+.2f}°C")
                
            else:
                # Fallback: use average calibration from all types
                if self.verbose:
                    print(f"    Warning: No calibration for '{comp_type}', using average")
                avg_offset = self.calibration_db['flir_offset_median'].mean() if use_median_offset else self.calibration_db['flir_offset_mean'].mean()
                corrected_air_temp = flir_ss_temp - avg_offset
                predicted_sand_temp = None
                offset = avg_offset
                confidence = 'Low'
                uncalibrated_count += 1
                
                if self.verbose:
                    print(f"    Average Offset Applied: {offset:+.2f}°C")
                    print(f"    Corrected Air Temp: {corrected_air_temp:.2f}°C")
            
            # Store prediction
            pred = {
                'pcb': pcb_name,
                'component_name': label,
                'component_type': comp_type,
                'flir_raw_ss': flir_ss_temp,
                'flir_offset_applied': offset if comp_type in self.calibration_db.index or comp_type == 'Unknown' else avg_offset,
                'predicted_air_ss': corrected_air_temp,
                'predicted_sand_ss': predicted_sand_temp,
                'predicted_temp_drop': (corrected_air_temp - predicted_sand_temp) if predicted_sand_temp else None,
                'confidence': confidence
            }
            predictions.append(pred)
        
        # Clear progress line and print summary
        if not self.verbose:
            print()  # New line after progress indicator
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print(f"Prediction Summary")
        print(f"{'='*80}")
        print(f"  Total components processed: {total_components}")
        print(f"  Components with calibration: {total_components - uncalibrated_count}")
        print(f"  Unknown component types: {unknown_count}")
        if uncalibrated_count > 0:
            print(f"  Uncalibrated (using average): {uncalibrated_count}")
        
        # Show hottest components
        df_temp = pd.DataFrame(predictions)
        top_5 = df_temp.nlargest(5, 'predicted_air_ss')[['component_name', 'component_type', 'predicted_air_ss']]
        print(f"\nHottest Components (Top 5):")
        for idx, row in top_5.iterrows():
            print(f"  {row['component_name']}: {row['predicted_air_ss']:.2f}°C ({row['component_type']})")
        
        # Convert to DataFrame
        df_predictions = pd.DataFrame(predictions)
        self.predictions.append(df_predictions)
        
        return df_predictions
    
    def export_predictions(self, df_predictions: pd.DataFrame, prefix: str = "thermal_prediction") -> None:
        """
        Export prediction results to CSV and visualizations.
        
        Args:
            df_predictions: DataFrame with prediction results
            prefix: Filename prefix for outputs
        """
        print(f"\n{'='*80}")
        print("Exporting Predictions")
        print(f"{'='*80}")
        
        # Export CSV
        csv_file = self.output_dir / f"{prefix}_{df_predictions['pcb'].iloc[0]}.csv"
        df_predictions.to_csv(csv_file, index=False)
        print(f"\nSaved: {csv_file}")
        print(f"  {len(df_predictions)} component predictions")
        
        # Export summary statistics
        summary = {
            'prediction_date': datetime.now().isoformat(),
            'pcb_name': df_predictions['pcb'].iloc[0],
            'num_components': len(df_predictions),
            'component_types': df_predictions['component_type'].value_counts().to_dict(),
            'temperature_statistics': {
                'flir_raw_mean': float(df_predictions['flir_raw_ss'].mean()),
                'flir_raw_std': float(df_predictions['flir_raw_ss'].std()),
                'predicted_air_mean': float(df_predictions['predicted_air_ss'].mean()),
                'predicted_air_std': float(df_predictions['predicted_air_ss'].std()),
                'predicted_sand_mean': float(df_predictions['predicted_sand_ss'].mean()),
                'predicted_sand_std': float(df_predictions['predicted_sand_ss'].std()),
                'avg_cooling_benefit': float(df_predictions['predicted_temp_drop'].mean()),
            },
            'confidence_distribution': df_predictions['confidence'].value_counts().to_dict()
        }
        
        json_file = self.output_dir / f"{prefix}_{df_predictions['pcb'].iloc[0]}_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {json_file}")
        
        # Create visualization
        self._create_prediction_plots(df_predictions, prefix)
    
    def _create_prediction_plots(self, df: pd.DataFrame, prefix: str) -> None:
        """Create visualization of prediction results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Raw FLIR vs Corrected temperatures
        ax = axes[0, 0]
        components = df['component_name'].tolist()
        x_pos = np.arange(len(components))
        ax.scatter(x_pos, df['flir_raw_ss'], label='FLIR Raw', alpha=0.7, s=50)
        ax.scatter(x_pos, df['predicted_air_ss'], label='Corrected (Air)', alpha=0.7, s=50)
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('FLIR Raw vs Corrected Temperatures')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Temperature correction by component type
        ax = axes[0, 1]
        for comp_type in df['component_type'].unique():
            type_data = df[df['component_type'] == comp_type]
            ax.scatter(type_data['flir_raw_ss'], type_data['predicted_air_ss'], 
                      label=comp_type, alpha=0.7, s=80)
        ax.plot([df['flir_raw_ss'].min(), df['flir_raw_ss'].max()],
               [df['flir_raw_ss'].min(), df['flir_raw_ss'].max()],
               'k--', alpha=0.3, label='No correction')
        ax.set_xlabel('FLIR Raw (°C)')
        ax.set_ylabel('Corrected Air (°C)')
        ax.set_title('Calibration Correction Applied')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Air vs Sand predictions
        ax = axes[1, 0]
        sand_data = df.dropna(subset=['predicted_sand_ss'])
        if len(sand_data) > 0:
            for comp_type in sand_data['component_type'].unique():
                type_data = sand_data[sand_data['component_type'] == comp_type]
                ax.scatter(type_data['predicted_air_ss'], type_data['predicted_sand_ss'],
                          label=comp_type, alpha=0.7, s=80)
            ax.plot([sand_data['predicted_air_ss'].min(), sand_data['predicted_air_ss'].max()],
                   [sand_data['predicted_air_ss'].min(), sand_data['predicted_air_ss'].max()],
                   'k--', alpha=0.3, label='No cooling')
            ax.set_xlabel('Predicted Air (°C)')
            ax.set_ylabel('Predicted Sand (°C)')
            ax.set_title('Air vs Sand Cooling Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Sand Cooling Predictions', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Air vs Sand Cooling Prediction')
        
        # Plot 4: Temperature distribution by condition
        ax = axes[1, 1]
        positions = [1, 2, 3]
        data_to_plot = [
            df['flir_raw_ss'].values,
            df['predicted_air_ss'].values,
            df.dropna(subset=['predicted_sand_ss'])['predicted_sand_ss'].values
        ]
        labels_plot = ['FLIR Raw', 'Corrected Air', 'Predicted Sand']
        bp = ax.boxplot(data_to_plot, positions=positions, labels=labels_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Distribution by Condition')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = self.output_dir / f"{prefix}_{df['pcb'].iloc[0]}_visualization.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved: {plot_file}")
    
    def _determine_component_type(self, label: str) -> Optional[str]:
        """Determine component type from FLIR channel label."""
        # First check explicit mapping
        for pattern, comp_type in self.component_mapping.items():
            if re.search(pattern, label, re.IGNORECASE):
                return comp_type
        
        # Fallback: try to infer from label patterns
        label_upper = label.upper()
        
        # Common patterns
        if any(x in label_upper for x in ['LDO', 'REGULATOR', 'REG']):
            return 'LDO'
        elif any(x in label_upper for x in ['PS', 'POWER SUPPLY', 'SUPPLY']):
            return 'PowerSupply'
        elif any(x in label_upper for x in ['LED', 'LIGHT']):
            return 'LED'
        elif any(x in label_upper for x in ['RES', 'RESISTOR', 'R']):
            return 'Resistor'
        elif any(x in label_upper for x in ['CAP', 'CAPACITOR', 'C']):
            return 'Capacitor'
        elif any(x in label_upper for x in ['IC', 'CHIP', 'U']):
            return 'IC'
        elif any(x in label_upper for x in ['DIODE', 'D']):
            return 'Diode'
        elif any(x in label_upper for x in ['INDUCTOR', 'L']):
            return 'Inductor'
        
        return None


def main():
    """
    Main execution for thermal prediction.
    
    Applies calibration database to FLIR-only measurements to predict
    corrected air temperatures and sand cooling performance.
    """
    print("="*80)
    print("THERMAL PREDICTION - PHASE 7")
    print("="*80)
    print("\nApplying calibration to predict component temperatures...")
    
    # Define paths
    input_dir = Path(__file__).parent / "inputs"
    output_dir = Path(__file__).parent / "outputs"
    calibration_file = output_dir / "thermal_calibration_by_type.csv"
    
    # Check if calibration exists
    if not calibration_file.exists():
        print(f"\nERROR: Calibration file not found: {calibration_file}")
        print("Please run phase6_thermal_calibration.py first!")
        return
    
    # Initialize predictor
    predictor = ThermalPredictor(calibration_file=calibration_file, output_dir=output_dir)
    
    # Component type mapping for H-Bridge PCB
    # Pattern matching component designators to types
    component_mapping = {
        # Power supply components
        r'^PS\d+$': 'PowerSupply',           # PS1, PS2, PS3
        r'^CONV$': 'PowerSupply',            # Converter
        
        # Integrated circuits
        r'^U\d+$': 'IC',                     # U1-U40, U200
        r'^IC\d+$': 'IC',                    # IC1, IC2
        
        # Resistors
        r'^R\d+$': 'Resistor',               # R1-R140
        
        # LEDs and diodes
        r'^DL\d+$': 'LED',                   # DL1-DL16
        r'^CR\d+$': 'LED',                   # CR3, CR4, CR5 (diodes)
        
        # Capacitors (not in calibration, will be marked unknown)
        r'^C\d+$': 'Unknown',                # C18-C127
        
        # Other components (not in calibration)
        r'^[HJLF]\d+$': 'Unknown',           # J, H, L, F components
        r'^VR\d+$': 'Unknown',               # VR1-VR6
        r'^SW\d+$': 'Unknown',               # SW1, SW2
    }
    
    predictor.set_component_type_mapping(component_mapping)
    
    # Load H-Bridge FLIR data from parsed ResearchIR output
    hbridge_flir_file = output_dir / "HBridge_FLIR_AllComponents.csv"
    
    if not hbridge_flir_file.exists():
        print(f"\nERROR: H-Bridge FLIR file not found: {hbridge_flir_file}")
        print("The ResearchIR stats parsing may have failed.")
        print("Expected file should contain 212 H-Bridge components.")
        return
    
    print(f"\nLoading H-Bridge FLIR data: {hbridge_flir_file}")
    
    # Predict temperatures
    df_predictions = predictor.predict_from_flir(
        flir_file=hbridge_flir_file,
        pcb_name="H_Bridge_Sensing",
        use_median_offset=True  # More robust to outliers
    )
    
    # Export results
    predictor.export_predictions(df_predictions, prefix="thermal_prediction_HBridge")
    
    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nPredicted temperatures for {len(df_predictions.columns)-2} components")
    print(f"Output directory: {output_dir}")
    print("\nCalibration applied to H-Bridge components:")
    print("  - PowerSupply (PS1, PS2, PS3, CONV)")
    print("  - IC (U1-U40, U200, IC1, IC2)")
    print("  - Resistor (R1-R140)")
    print("  - LED (DL1-DL16, CR3-CR5)")
    print("\nNext steps:")
    print("  1. Review thermal_prediction_HBridge.csv for component temperature predictions")
    print("  2. Check thermal_prediction_HBridge_visualization.png for analysis plots")
    print("  3. Components marked 'Unknown' use raw FLIR data (no calibration available)")


if __name__ == "__main__":
    main()
