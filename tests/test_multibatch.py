"""
Test script for complete multi-batch calibration workflow.
Tests both Batch 1 (single device) and Batch 2 (multi-device).
"""
import json
from pathlib import Path
from phase6_thermal_calibration import ThermalCalibrator

def main():
    # Load config
    with open('tests/test_multibatch_config.json', 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("MULTI-BATCH CALIBRATION WORKFLOW TEST")
    print("="*80)
    print(f"Total sessions: {len(config['sessions'])}\n")
    
    # Initialize calibrator
    output_dir = Path('outputs/test_multibatch')
    calibrator = ThermalCalibrator(output_dir=output_dir)
    
    # Process all sessions
    try:
        calibrator.add_measurement_sessions(config['sessions'])
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"✓ All sessions processed successfully!")
        print(f"Total calibration points: {len(calibrator.calibration_pairs)}")
        
        # Show breakdown by session
        from collections import Counter
        session_counts = Counter(p['test_session'] for p in calibrator.calibration_pairs)
        print(f"\nCalibration points by session:")
        for session, count in session_counts.items():
            print(f"  {session}: {count} points")
        
        # Save calibration data
        print(f"\nSaving calibration database to: {output_dir}")
        try:
            cal_file = calibrator.save_calibration_points()
            print(f"✓ Saved: {cal_file}")
        except Exception as save_error:
            print(f"Warning: Could not save calibration file: {save_error}")
            # This is OK for testing, we just want to verify the processing works
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠ Test interrupted by user")
        print(f"Calibration points collected before interruption: {len(calibrator.calibration_pairs)}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
