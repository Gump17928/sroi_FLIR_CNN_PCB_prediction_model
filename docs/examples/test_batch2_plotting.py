"""
Test script for Batch 2 multi-device calibration with full plotting.
Tests raw input overview + detailed comparison plots for 8 merged channels.
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    # Load config
    with open('multi_batch_loadshedding_config.json', 'r') as f:
        config = json.load(f)
    
    # Process only Batch 2 session (multi-device)
    batch2_session = config['sessions'][1]
    print(f"Testing: {batch2_session['name']}")
    print(f"PCB: {batch2_session['pcb']}")
    print(f"Components: {len(batch2_session['pairs'])}")
    print(f"Air files: {batch2_session['therm_air_file']}")
    print(f"Sand files: {batch2_session['therm_sand_file']}")
    
    # Initialize calibrator
    calibrator = ThermalCalibrator(output_dir=Path('outputs/test_batch2'))
    
    # Process single session
    try:
        calibrator.add_measurement_sessions([batch2_session])
        print('\n✓ Batch 2 multi-device processing successful!')
        print(f'Calibration points collected: {len(calibrator.calibration_pairs)}')
        
        # Check output structure
        session_dir = Path('outputs/test_batch2/calibration_database') / batch2_session['name']
        if session_dir.exists():
            print(f'\n✓ Session folder created: {session_dir}')
            if (session_dir / 'raw_inputs_overview.png').exists():
                print(f'  ✓ Raw input overview plot exists')
            if (session_dir / 'detailed_plots').exists():
                detailed_count = len(list((session_dir / 'detailed_plots').glob('*.png')))
                print(f'  ✓ Detailed plots folder exists ({detailed_count} PNG files)')
        
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
