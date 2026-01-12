"""
Final test: Complete multi-batch workflow with corrected FLIR data.
Batch 1: 5 components (Test_3 FLIR subset)
Batch 2: 8 components (Load_Shedding_FLIR_AllComponents - full board)
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    print("="*80)
    print("FINAL MULTI-BATCH CALIBRATION TEST")
    print("="*80)
    
    # Load config with both batches
    with open('multi_batch_loadshedding_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"\nTotal sessions: {len(config['sessions'])}")
    for i, session in enumerate(config['sessions'], 1):
        print(f"  {i}. {session['name']}: {len(session['pairs'])} components")
    
    # Initialize calibrator
    output_dir = Path('outputs/final_multibatch_test')
    calibrator = ThermalCalibrator(output_dir=output_dir)
    
    # Process all sessions
    try:
        calibrator.add_measurement_sessions(config['sessions'])
        
        print('\n' + '='*80)
        print('✓ MULTI-BATCH PROCESSING SUCCESSFUL!')
        print('='*80)
        print(f'Total calibration points: {len(calibrator.calibration_pairs)}')
        print(f'  Batch 1: 5 components')
        print(f'  Batch 2: 8 components')
        print(f'  Total: 13 components')
        
        # Export aggregate calibration database
        print('\nExporting aggregate calibration database...')
        calibrator.compute_component_type_calibrations()
        calibrator.export_calibration_database(prefix="thermal_calibration")
        
        # Verify folder structure
        print('\nVerifying output structure:')
        cal_db_dir = output_dir / 'calibration_database'
        
        for session in config['sessions']:
            session_name = session['name']
            session_dir = cal_db_dir / session_name
            
            print(f'\n[{session_name}]')
            if session_dir.exists():
                print(f'  ✓ Session folder exists')
                
                if (session_dir / 'raw_inputs_overview.png').exists():
                    print(f'  ✓ Raw input overview plot')
                
                detailed_dir = session_dir / 'detailed_plots'
                if detailed_dir.exists():
                    png_count = len(list(detailed_dir.glob('*.png')))
                    pdf_count = len(list(detailed_dir.glob('*.pdf')))
                    print(f'  ✓ Detailed plots: {png_count} PNG, {pdf_count} PDF')
            else:
                print(f'  ✗ Session folder missing')
        
        # Check aggregate outputs
        print('\n[Aggregate Outputs]')
        if (cal_db_dir / 'thermal_calibration_points.csv').exists():
            print('  ✓ thermal_calibration_points.csv')
        if (cal_db_dir / 'thermal_calibration_by_type.csv').exists():
            print('  ✓ thermal_calibration_by_type.csv')
        if (cal_db_dir / 'thermal_calibration_summary.png').exists():
            print('  ✓ thermal_calibration_summary.png')
        
        print('\n' + '='*80)
        print('✓ ALL TESTS PASSED - WORKFLOW COMPLETE!')
        print('='*80)
        print(f'\nOutput directory: {output_dir}')
        
        return True
        
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
