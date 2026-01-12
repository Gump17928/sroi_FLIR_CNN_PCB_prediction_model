"""
Quick test for multi-device loading (Air only, no Sand for speed).
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    # Create simple config with just Air data
    config = {
        "sessions": [
            {
                "name": "Quick_MultiDevice_Test",
                "pcb": "LoadShedding_AC_Switch",
                "flir_file": "inputs/Test_3_FLIR_Camera_Results_5_of_7.csv",
                "therm_air_file": [
                    "inputs/Test_Air_Load_Shedding/USB-TEMP (Device 0) - Analog - 12-4-2025 12-13-19.6 PM.csv",
                    "inputs/Test_Air_Load_Shedding/USB-TEMP-AI (Device 1) - Analog - 12-4-2025 12-13-19.6 PM.csv"
                ],
                "therm_sand_file": None,  # Skip sand for speed
                "pairs": [
                    ["Box 1", "Device0_AI0", "Test_U2", "IC"],
                    ["Box 3", "Device0_AI1", "Test_R5", "Resistor"],
                    ["SMD RES1", "Device0_AI4", "Test_C3", "Capacitor"]
                ]
            }
        ]
    }
    
    print("="*80)
    print("QUICK MULTI-DEVICE TEST (Air only)")
    print("="*80)
    
    # Initialize calibrator
    calibrator = ThermalCalibrator(output_dir=Path('outputs/test_quick'))
    
    try:
        calibrator.add_measurement_sessions(config['sessions'])
        
        print("\n" + "="*80)
        print("✓ Multi-device processing successful!")
        print(f"Calibration points: {len(calibrator.calibration_pairs)}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
