"""
Generic HDF5 dataset builder for CNN thermal modeling.

This script builds HDF5 datasets for any board (HBridge, LoadShedding, etc.)

Usage:
    python build_dataset.py HBridge [--component_val_split 0.2]
    python build_dataset.py LoadShedding [--component_val_split 0.0]

Component-Level Split:
    - Random split of components for validation
    - component_val_split=0.2: 80% train, 20% validation (component holdout)
    - component_val_split=0.0: All components for training (cross-PCB validation)

Author: CNN Pipeline
Date: 2026-01-28
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from cnn_data_preprocessor import CNNDataPreprocessor


# Board-specific configurations
BOARD_CONFIGS = {
    "HBridge": {
        "flir_folder_name": "ResearchIR_Outputs_HBridge_15s_filtered",
        "thermistor_pattern": "HBridge_15s_thermistor_timeseries.csv",
        "pixel_map": "HBridge_roi_pixel_map.csv",
        "manual_offset": 20.0,  # FLIR leads thermistor by 20s
        "flip_vertical": False,
        "flip_horizontal": False
    },
    "LoadShedding": {
        "flir_folder_name": "ResearchIR_Outputs_Load_Shedding_filtered",
        "thermistor_pattern": "Load_Shedding_thermistor_timeseries.csv",
        "pixel_map": "loadshedding_ac_switch_full_enhanced_roi_pixel_map.csv",
        "manual_offset": 20.0,  # FLIR leads thermistor by 20s
        "flip_vertical": False,
        "flip_horizontal": False
    }
}


def find_most_recent_thermistor(board_name, pattern):
    """Find most recent thermistor timeseries file for a board."""
    output_dirs = sorted([d for d in (parent_dir / "outputs").glob("*_P1-7") if d.is_dir()], reverse=True)
    for output_dir in output_dirs:
        candidate = output_dir / pattern
        if candidate.exists():
            return candidate
    return None


def build_dataset(board_name, component_val_split=0.2, flir_folder=None, thermistor_csv=None, 
                  pixel_map_csv=None, output_h5=None):
    """Build HDF5 training dataset for specified board.
    
    Args:
        board_name: Name of board (HBridge, LoadShedding, etc.)
        component_val_split: Fraction of components for validation (0.0=all train, 0.2=80/20 split)
        flir_folder: Override FLIR folder path
        thermistor_csv: Override thermistor CSV path
        pixel_map_csv: Override pixel map path
        output_h5: Override output HDF5 path
    """
    
    print("="*80)
    print(f"Building {board_name} CNN Training Dataset")
    print("="*80)
    
    # Get board configuration
    if board_name not in BOARD_CONFIGS:
        raise ValueError(f"Unknown board: {board_name}. Available: {list(BOARD_CONFIGS.keys())}")
    
    config = BOARD_CONFIGS[board_name]
    
    # Determine paths
    if flir_folder is None:
        flir_folder_filtered = parent_dir / "inputs" / config["flir_folder_name"]
        flir_folder_raw = parent_dir / "inputs" / config["flir_folder_name"].replace("_filtered", "")
        
        if flir_folder_filtered.exists():
            flir_folder = flir_folder_filtered
        else:
            flir_folder = flir_folder_raw
    else:
        flir_folder = Path(flir_folder)
    
    if thermistor_csv is None:
        # For LoadShedding, look in sroi_generation outputs first
        if board_name == "LoadShedding":
            sroi_pixel_map_dir = Path("/home/forrest/home/modeling/sroi_generation_ResearchIR/outputs")
            candidate = sroi_pixel_map_dir / config["thermistor_pattern"]
            if candidate.exists():
                thermistor_csv = candidate
        
        # If not found, search in thermal_post_processing outputs
        if thermistor_csv is None:
            thermistor_csv = find_most_recent_thermistor(board_name, config["thermistor_pattern"])
        
        if thermistor_csv is None:
            raise FileNotFoundError(f"No {config['thermistor_pattern']} found in recent output directories")
    else:
        thermistor_csv = Path(thermistor_csv)
    
    if pixel_map_csv is None:
        # Look in sroi_generation outputs first
        sroi_pixel_map_dir = Path("/home/forrest/home/modeling/sroi_generation_ResearchIR/outputs")
        candidate = sroi_pixel_map_dir / config["pixel_map"]
        
        if candidate.exists():
            pixel_map_csv = candidate
        else:
            # Fall back to thermal_post_processing outputs
            pixel_map_csv = parent_dir / "outputs" / "roi_pixel_maps" / config["pixel_map"]
    else:
        pixel_map_csv = Path(pixel_map_csv)
    
    if output_h5 is None:
        output_h5 = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / f"{board_name}_cnn_dataset.h5"
    else:
        output_h5 = Path(output_h5)
    
    # Print configuration
    print(f"\n✓ Board: {board_name}")
    print(f"✓ FLIR folder: {flir_folder}")
    print(f"✓ Thermistor CSV: {thermistor_csv}")
    print(f"✓ Pixel map: {pixel_map_csv}")
    print(f"✓ Output HDF5: {output_h5}")
    print(f"✓ Component validation split: {component_val_split:.1%}")
    if config["manual_offset"] is not None:
        print(f"✓ Manual time offset: {config['manual_offset']}s (FLIR leads)")
    else:
        print("✓ Time offset: Auto-detect via cross-correlation")
    
    # Create output directory
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify inputs exist
    print("\n1. Verifying Input Files")
    print("-" * 40)
    if not flir_folder.exists():
        raise FileNotFoundError(f"FLIR folder not found: {flir_folder}")
    print(f"✓ FLIR folder exists")
    
    if not thermistor_csv.exists():
        raise FileNotFoundError(f"Thermistor CSV not found: {thermistor_csv}")
    print(f"✓ Thermistor CSV exists")
    
    if not pixel_map_csv.exists():
        raise FileNotFoundError(f"Pixel map CSV not found: {pixel_map_csv}")
    print(f"✓ Pixel map CSV exists")
    
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
    if config["manual_offset"] is not None:
        print(f"  - Apply manual time offset: {config['manual_offset']}s")
    else:
        print("  - Detect temporal offset via cross-correlation")
    print("  - Align and interpolate thermistor to FLIR timestamps")
    print("  - Generate ROI masks from pixel maps")
    print("  - Export to HDF5 format")
    print()
    
    preprocessor.build_dataset(
        flir_folder=str(flir_folder),
        thermistor_csv=str(thermistor_csv),
        pixel_map_csv=str(pixel_map_csv),
        output_h5=str(output_h5),
        max_frames=None,  # Use all available frames
        time_column="Time (s)",  # Standard time column name
        component_prefix="",  # No prefix - component names are direct column names
        manual_time_offset=config["manual_offset"],  # Board-specific offset
        component_val_split=component_val_split  # Validation split
    )
    
    print("\n" + "="*80)
    print(f"✓ Dataset built successfully: {output_h5}")
    print(f"✓ Component validation split: {component_val_split:.1%}")
    print("="*80)
    
    return output_h5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HDF5 dataset for CNN thermal modeling")
    parser.add_argument("board", type=str, choices=list(BOARD_CONFIGS.keys()),
                        help="Board name (HBridge, LoadShedding, etc.)")
    parser.add_argument("--component_val_split", type=float, default=0.0,
                        help="Fraction of components for validation (0.0=all train, 0.2=80/20 split)")
    parser.add_argument("--flir_folder", type=str, default=None,
                        help="Override FLIR folder path")
    parser.add_argument("--thermistor_csv", type=str, default=None,
                        help="Override thermistor CSV path")
    parser.add_argument("--pixel_map_csv", type=str, default=None,
                        help="Override pixel map CSV path")
    parser.add_argument("--output_h5", type=str, default=None,
                        help="Override output HDF5 path")
    
    args = parser.parse_args()
    
    build_dataset(
        board_name=args.board,
        component_val_split=args.component_val_split,
        flir_folder=args.flir_folder,
        thermistor_csv=args.thermistor_csv,
        pixel_map_csv=args.pixel_map_csv,
        output_h5=args.output_h5
    )
