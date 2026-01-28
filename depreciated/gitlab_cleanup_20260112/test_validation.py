"""
Test script for validation workflow.
Tests the new cross-validation and visualization functionality.
"""

import sys
from pathlib import Path
import json

# Add thermal_post_processing to path
sys.path.insert(0, str(Path(__file__).parent))

from phase6_thermal_calibration import ThermalCalibrator

def test_validation_workflow():
    """Test the validation workflow with existing calibration data."""
    
    # Find most recent calibration database
    outputs_dir = Path(__file__).parent / "outputs"
    
    # Look for calibration databases
    cal_dirs = list(outputs_dir.glob("*/calibration_database"))
    
    if not cal_dirs:
        print("No calibration database found in outputs/")
        print("Please run the thermal modeling workflow first:")
        print("  python researchir_post_processor.py --thermal_modeling --multi_session_config All_Boards_multi_test_config.json")
        return
    
    # Use most recent
    cal_dir = sorted(cal_dirs, key=lambda p: p.parent.name)[-1]
    print(f"Using calibration database: {cal_dir}")
    
    # Check if calibration points file exists
    points_file = cal_dir / "thermal_calibration_points.csv"
    if not points_file.exists():
        print(f"Calibration points file not found: {points_file}")
        return
    
    print(f"\nFound calibration data: {points_file}")
    
    # Load calibration data
    calibrator = ThermalCalibrator(output_dir=cal_dir)
    calibrator.load_existing_calibration(points_file)
    
    print(f"Loaded {len(calibrator.calibration_pairs)} calibration points")
    
    # Compute calibrations
    print("\nComputing component-type calibrations...")
    calibrator.compute_component_type_calibrations()
    
    # Run validation workflow
    print("\nRunning validation workflow...")
    calibrator.run_validation_workflow()
    
    print("\n" + "="*80)
    print("VALIDATION TEST COMPLETE")
    print("="*80)
    print(f"\nCheck outputs in: {cal_dir}")
    print("  - calibration_validation_summary.png")
    print("  - calibration_quality_metrics.csv")


if __name__ == "__main__":
    test_validation_workflow()
