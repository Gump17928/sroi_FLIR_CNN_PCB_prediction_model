#!/usr/bin/env python3
"""
Verify ROI pixel map positions by visualizing on FLIR frame.

STANDALONE WRAPPER: This is a lightweight wrapper around viz_phase8_dataset_validation.py
                   for quick standalone verification outside the main pipeline.

This script overlays the ROI pixel map on an actual FLIR thermal frame
to verify that the positions are correct before building the HDF5 dataset.

Usage:
    python verify_roi_pixel_map.py --board HBridge
    python verify_roi_pixel_map.py --board HBridge --frame 150
    python verify_roi_pixel_map.py --board HBridge --compare-hdf5
    
For pipeline integration, use: viz_phase8_dataset_validation.py
"""

import argparse
from pathlib import Path
from viz_phase8_dataset_validation import (
    visualize_roi_pixel_map,
    visualize_hdf5_validation
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ROI pixel map locations on FLIR frame",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--board',
        type=str,
        default='HBridge',
        help='Board name (default: HBridge)'
    )
    
    parser.add_argument(
        '--frame',
        type=int,
        default=None,
        help='Frame index to visualize (default: middle frame)'
    )
    
    parser.add_argument(
        '--compare-hdf5',
        action='store_true',
        help='Also create comparison with HDF5 dataset ROI masks'
    )
    
    args = parser.parse_args()
    
    # For standalone use, output goes to roi_pixel_maps folder (backward compatible)
    # Session ID set to "standalone" to differentiate from pipeline runs
    session_id = "standalone"
    
    try:
        print("="*80)
        print("ROI PIXEL MAP VERIFICATION (Standalone Mode)")
        print("="*80)
        print(f"\nBoard: {args.board}")
        print(f"Frame: {args.frame if args.frame is not None else 'middle frame'}")
        
        # Generate ROI verification
        print("\n1. Generating ROI Pixel Map Visualization...")
        roi_path = visualize_roi_pixel_map(args.board, session_id, args.frame)
        print(f"   ✓ Saved: {roi_path}")
        
        # Generate HDF5 comparison if requested
        if args.compare_hdf5:
            print("\n2. Generating HDF5 Dataset Comparison...")
            hdf5_path = visualize_hdf5_validation(args.board, session_id, args.frame)
            print(f"   ✓ Saved: {hdf5_path}")
        
        print("\n" + "="*80)
        print("VERIFICATION COMPLETE")
        print("="*80)
        print(f"\nVisualizations saved to:")
        print(f"  outputs/{session_id}/visualizations/")
        print("\nCheck the images to verify:")
        print("  1. ROI markers align with actual components")
        print("  2. ROIs are not all clustered in one corner")
        print("  3. Component labels match expected locations")
        print("\nIf misaligned: Regenerate SROI with corrected corner coordinates")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
