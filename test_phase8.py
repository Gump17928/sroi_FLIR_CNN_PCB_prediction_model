#!/usr/bin/env python3
"""
Quick test script for Phase 8 ML Training.

Runs Phase 8 without needing to execute the entire pipeline.

Usage:
    # Component holdout (default):
    python test_phase8.py --board HBridge --validation_mode component_holdout
    
    # Cross-PCB validation:
    python test_phase8.py --validation_mode cross_pcb
    
    # Build dataset before training:
    python test_phase8.py --build_dataset --board LoadShedding
    
    # Legacy: specify session name
    python test_phase8.py 0121_1804_P1-7
"""

import sys
from pathlib import Path
import json
import argparse

# Add current directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import phase8_ml_training


def main():
    """Run Phase 8 with latest session data."""
    
    # Check if session specified on command line
    if len(sys.argv) > 1:
        session_name = sys.argv[1]
        session_dir = project_root / "outputs" / session_name
        if not session_dir.exists():
            print(f"❌ Session not found: {session_dir}")
            return
    else:
        # Auto-detect sessions and show options
        outputs_dir = project_root / "outputs"
        session_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("0")]
        
        if not session_dirs:
            print("❌ No session directories found in outputs/")
            return
        
        # Sort by name (timestamp)
        session_dirs = sorted(session_dirs, key=lambda d: d.name, reverse=True)
        
        print("="*80)
        print("AVAILABLE SESSIONS")
        print("="*80)
        for i, d in enumerate(session_dirs[:5], 1):  # Show top 5
            # Check if thermistor CSV exists
            therm_csv = d / "HBridge_15s_thermistor_timeseries.csv"
            status = "✓" if therm_csv.exists() else "✗"
            print(f"  [{i}] {d.name} {status}")
        
        print("\nUsage: python test_phase8.py [session_name]")
        print("Example: python test_phase8.py 0121_1804_P1-7")
        
        # Default to most recent with valid thermistor data
        for session_dir in session_dirs:
            therm_csv = session_dir / "HBridge_15s_thermistor_timeseries.csv"
            if therm_csv.exists():
                break
        else:
            print("\n❌ No valid sessions found with thermistor data")
            return
    
    print("="*80)
    print("PHASE 8 ML TRAINING - STANDALONE TEST")
    print("="*80)
    print(f"\nUsing session: {session_dir.name}")
    print(f"Session path: {session_dir}\n")
    
    # Create minimal config
    config = {
        'output_dir': str(session_dir),
        'debug': True,
    }
    
    # Run Phase 8
    try:
        results = phase8_ml_training.run_phase8_ml_training(
            session_dir=session_dir,
            config=config,
            board_name='HBridge'
        )
        
        if not results.get('skipped'):
            print("\n" + "="*80)
            print("✅ PHASE 8 COMPLETE")
            print("="*80)
            if 'unet' in results:
                print(f"U-Net CNN: {results['unet'].get('status', 'N/A')}")
            if 'linear' in results:
                print(f"Linear Regression: {results['linear'].get('status', 'N/A')}")
        else:
            print("\n⏭️  Phase 8 skipped")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phase 8 ML Training")
    parser.add_argument('session', nargs='?', default=None,
                        help='Session name (legacy mode, e.g., 0121_1804_P1-7)')
    parser.add_argument('--board', type=str, choices=['HBridge', 'LoadShedding'],
                        default='HBridge',
                        help='Board to use for training (default: HBridge)')
    parser.add_argument('--validation_mode', type=str,
                        choices=['component_holdout', 'cross_pcb'],
                        default='component_holdout',
                        help='Validation mode (default: component_holdout)')
    parser.add_argument('--build_dataset', action='store_true',
                        help='Build dataset(s) before training')
    parser.add_argument('--use_cnn_pipeline', action='store_true',
                        help='Use train_hbridge_model.py directly instead of phase8_ml_training')
    
    args = parser.parse_args()
    
    # If using CNN pipeline directly, delegate to train_hbridge_model.py
    if args.use_cnn_pipeline:
        print("="*80)
        print("USING DIRECT CNN TRAINING PIPELINE")
        print("="*80)
        
        cnn_script = project_root / "ml_model" / "cnn_thermal_modeling" / "train_hbridge_model.py"
        cmd_args = [
            sys.executable, str(cnn_script),
            '--board', args.board,
            '--validation_mode', args.validation_mode
        ]
        
        if args.build_dataset:
            cmd_args.append('--build_dataset')
        
        import subprocess
        result = subprocess.run(cmd_args)
        sys.exit(result.returncode)
    
    # Otherwise use legacy Phase 8 pipeline
    main()
    main()
