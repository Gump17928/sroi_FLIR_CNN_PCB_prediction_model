"""
Test script for Batch 1 single-device calibration processing.
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    # Load config
    with open('multi_batch_loadshedding_config.json', 'r') as f:
        config = json.load(f)
    
    # Process only Batch 1 session
    batch1_session = config['sessions'][0]
    print(f"Testing: {batch1_session['name']}")
    print(f"PCB: {batch1_session['pcb']}")
    print(f"Components: {len(batch1_session['pairs'])}")
    print(f"Air file: {batch1_session['therm_air_file']}")
    print(f"Sand file: {batch1_session['therm_sand_file']}")
    
    # Initialize calibrator
    calibrator = ThermalCalibrator(output_dir=Path('outputs/test_batch1'))
    
    # Process single session
    try:
        calibrator.add_measurement_sessions([batch1_session])
        print('\n✓ Batch 1 processing successful!')
        print(f'Calibration points collected: {len(calibrator.calibration_pairs)}')
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
