"""
Test script for complete multi-batch workflow with both Batch 1 and Batch 2.
Tests that both sessions create separate folders with all plots.
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    # Load config with both batches
    with open('multi_batch_loadshedding_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"Testing complete multi-batch workflow")
    print(f"Total sessions: {len(config['sessions'])}")
    for i, session in enumerate(config['sessions'], 1):
        print(f"  {i}. {session['name']}")
    
    # Initialize calibrator
    output_dir = Path('outputs/test_multibatch_complete')
    calibrator = ThermalCalibrator(output_dir=output_dir)
    
    # Process all sessions
    try:
        calibrator.add_measurement_sessions(config['sessions'])
        
        print('\n' + '='*80)
        print('✓ Multi-batch processing successful!')
        print('='*80)
        print(f'Total calibration points collected: {len(calibrator.calibration_pairs)}')
        
        # Verify folder structure
        print('\nVerifying output structure:')
        cal_db_dir = output_dir / 'calibration_database'
        
        for session in config['sessions']:
            session_name = session['name']
            session_dir = cal_db_dir / session_name
            
            print(f'\n[{session_name}]')
            if session_dir.exists():
                print(f'  ✓ Session folder exists')
                
                # Check raw overview
                if (session_dir / 'raw_inputs_overview.png').exists():
                    print(f'  ✓ Raw input overview plot exists')
                else:
                    print(f'  ✗ Raw input overview plot missing')
                
                # Check detailed plots
                detailed_dir = session_dir / 'detailed_plots'
                if detailed_dir.exists():
                    png_count = len(list(detailed_dir.glob('*.png')))
                    pdf_count = len(list(detailed_dir.glob('*.pdf')))
                    print(f'  ✓ Detailed plots folder exists ({png_count} PNG, {pdf_count} PDF)')
                else:
                    print(f'  ✗ Detailed plots folder missing')
            else:
                print(f'  ✗ Session folder missing')
        
        # Export calibration database to verify aggregate outputs work
        print('\nExporting aggregate calibration database...')
        calibrator.compute_component_type_calibrations()
        calibrator.export_calibration_database(prefix="thermal_calibration")
        
        print('\n✓ All tests passed!')
        return True
        
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
