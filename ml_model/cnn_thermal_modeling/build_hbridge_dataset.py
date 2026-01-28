"""
Build HDF5 training dataset for HBridge CNN thermal modeling.

This script:
1. Loads FLIR thermal frame sequences (640x480 pixels, 15s intervals)
2. Loads thermistor ground truth data from Phase 7 outputs
3. Performs temporal alignment via cross-correlation
4. Generates ROI masks from pixel maps
5. Splits components into train/val sets (80/20, random)
6. Exports aligned HDF5 dataset for U-Net training

Component-Level Split (NEW):
    - Random 80/20 split of components (default: seed=42)
    - Training: 80% of components in input Channel 1
    - Validation: Same 80% in input, predict held-out 20%
    - Tests if model learns thermal coupling vs copying inputs

Author: CNN Pipeline
Date: 2025-01-21
Updated: 2026-01-23 - Component-level validation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import from root directory where files actually are
from cnn_data_preprocessor import CNNDataPreprocessor


def build_hbridge_dataset(flir_folder=None, thermistor_csv=None, pixel_map_csv=None, output_h5=None, component_val_split=0.2):
    """Build HDF5 training dataset for HBridge board.
    
    Args:
        flir_folder: Path to FLIR frames folder (if None, uses default)
        thermistor_csv: Path to thermistor CSV (if None, uses default)
        pixel_map_csv: Path to ROI pixel map (if None, uses default)
        output_h5: Path to output HDF5 file (if None, uses default)
        component_val_split: Fraction of components to reserve for validation (0.0-1.0)
    """
    
    print("="*80)
    print("Building HBridge CNN Training Dataset")
    print("="*80)
    
    # Use provided paths or fall back to defaults
    if flir_folder is None:
        board_name = "HBridge_15s"
        flir_folder_filtered = parent_dir / "inputs" / f"ResearchIR_Outputs_{board_name}_filtered"
        flir_folder_raw = parent_dir / "inputs" / f"ResearchIR_Outputs_{board_name}"
        
        if flir_folder_filtered.exists():
            flir_folder = flir_folder_filtered
        else:
            flir_folder = flir_folder_raw
    else:
        flir_folder = Path(flir_folder)
    
    if thermistor_csv is None:
        # Find most recent output directory with thermistor data
        output_dirs = sorted([d for d in (parent_dir / "outputs").glob("*_P1-7") if d.is_dir()], reverse=True)
        thermistor_csv = None
        for output_dir in output_dirs:
            candidate = output_dir / "HBridge_15s_thermistor_timeseries.csv"
            if candidate.exists():
                thermistor_csv = candidate
                break
        if thermistor_csv is None:
            raise FileNotFoundError("No HBridge_15s_thermistor_timeseries.csv found in recent output directories")
    else:
        thermistor_csv = Path(thermistor_csv)
    
    if pixel_map_csv is None:
        pixel_map_csv = parent_dir / "outputs" / "roi_pixel_maps" / "HBridge_roi_pixel_map.csv"
    else:
        pixel_map_csv = Path(pixel_map_csv)
    
    if output_h5 is None:
        output_h5 = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
    else:
        output_h5 = Path(output_h5)
    
    # Print what we're using
    print(f"\n✓ Using FLIR frames: {flir_folder}")
    print(f"✓ Using thermistor CSV: {thermistor_csv}")
    print(f"✓ Using pixel map: {pixel_map_csv}")
    
    # Create output directory
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify inputs exist
    print("\n1. Verifying Input Files")
    print("-" * 40)
    if not flir_folder.exists():
        raise FileNotFoundError(f"FLIR folder not found: {flir_folder}")
    print(f"✓ FLIR folder: {flir_folder}")
    
    if not thermistor_csv.exists():
        raise FileNotFoundError(f"Thermistor CSV not found: {thermistor_csv}")
    print(f"✓ Thermistor CSV: {thermistor_csv}")
    
    if not pixel_map_csv.exists():
        raise FileNotFoundError(f"Pixel map CSV not found: {pixel_map_csv}")
    print(f"✓ Pixel map CSV: {pixel_map_csv}")
    
    # Initialize preprocessor
    print("\n2. Initializing CNN Data Preprocessor")
    print("-" * 40)
    preprocessor = CNNDataPreprocessor()
    
    # Build dataset
    print("\n3. Building HDF5 Dataset")
    print("-" * 40)
    print("This will:")
    print("  - Load FLIR thermal frame sequences")
    print("  - Load thermistor ground truth data")
    print("  - Detect temporal offset via cross-correlation")
    print("  - Align and interpolate thermistor to FLIR timestamps")
    print("  - Generate ROI masks from pixel maps")
    print("  - Export to HDF5 format")
    print()
    
    # NOTE: Thermistor CSV is already in correct format from Phase 6:
    # Time (s), Component1, Component2, Component3, ...
    # No preprocessing needed - use directly!
    
    print(f"\n✓ Using thermistor CSV from Phase 6: {thermistor_csv.name}")
    
    # MANUAL TIME OFFSET: FLIR leads thermistor by 20 seconds
    MANUAL_OFFSET = 20.0  # seconds (positive = FLIR started before thermistor)
    print(f"✓ Using manual time offset: {MANUAL_OFFSET}s (FLIR leads)\n")
    
    # Build dataset with actual thermistor ground truth
    preprocessor.build_dataset(
        flir_folder=str(flir_folder),
        thermistor_csv=str(thermistor_csv),
        pixel_map_csv=str(pixel_map_csv),
        output_h5=str(output_h5),
        max_frames=None,  # Use all available frames (~300)
        time_column="Time (s)",  # Standard time column name
        component_prefix="",  # No prefix - component names are direct column names
        manual_time_offset=MANUAL_OFFSET,  # Use manual offset instead of cross-correlation
        component_val_split=component_val_split  # Random 80/20 component split
    )
    
    print("\n" + "="*80)
    print("✓ HBridge Dataset Build Complete")
    print("="*80)
    print(f"\nOutput saved to: {output_h5}")
    print(f"File size: {output_h5.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print("-" * 40)
    import h5py
    with h5py.File(output_h5, 'r') as f:
        print(f"FLIR frames shape: {f['flir_frames'].shape}")
        print(f"Sand temps shape: {f['sand_temps'].shape}")
        print(f"ROI masks shape: {f['roi_masks'].shape}")
        print(f"Timestamps shape: {f['timestamps'].shape}")
        print(f"\nMetadata:")
        for key, val in f['metadata'].attrs.items():
            print(f"  {key}: {val}")
    
    return output_h5


if __name__ == "__main__":
    try:
        output_file = build_hbridge_dataset()
        print(f"\n✓ SUCCESS: Dataset ready for training at {output_file}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
