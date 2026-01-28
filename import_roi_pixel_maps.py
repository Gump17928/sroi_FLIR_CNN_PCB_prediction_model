#!/usr/bin/env python3
"""
Import ROI Pixel Maps from SROI Generation Pipeline

This script copies pixel map CSV files from the SROI generation outputs
to the thermal_post_processing expected location for ML training.

Usage:
    python import_roi_pixel_maps.py
    python import_roi_pixel_maps.py --board HBridge
    python import_roi_pixel_maps.py --board LoadShedding --force

Created: January 16, 2026
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_sroi_pixel_maps(sroi_outputs_dir):
    """
    Find all pixel map CSV files in SROI outputs directory.
    
    Args:
        sroi_outputs_dir: Path to sroi_generation_ResearchIR/outputs/
    
    Returns:
        List of (board_name, pixel_map_path) tuples
    """
    pixel_maps = []
    
    if not sroi_outputs_dir.exists():
        return pixel_maps
    
    # Look for *_roi_pixel_map.csv files
    for csv_file in sroi_outputs_dir.glob('*_roi_pixel_map.csv'):
        # Extract board name from filename
        # e.g., "hbridge_full_enhanced_roi_pixel_map.csv" -> "HBridge"
        board_name = csv_file.stem.replace('_roi_pixel_map', '').replace('_full_enhanced', '')
        board_name = board_name.replace('_enhanced_scaled', '').replace('hbridge', 'HBridge')
        board_name = board_name.replace('loadshedding', 'LoadShedding')
        
        pixel_maps.append((board_name, csv_file))
    
    return pixel_maps


def import_pixel_maps(board_filter=None, force=False):
    """
    Import pixel maps from SROI generation to thermal_post_processing.
    
    Args:
        board_filter: Only import this board (None = all boards)
        force: Overwrite existing pixel maps
    """
    print("="*80)
    print("ROI PIXEL MAP IMPORT")
    print("="*80)
    
    # Determine paths
    thermal_dir = Path(__file__).parent
    sroi_dir = thermal_dir.parent / "sroi_generation_ResearchIR"
    sroi_outputs = sroi_dir / "outputs"
    
    # Destination directory
    dest_dir = thermal_dir / "outputs" / "roi_pixel_maps"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource: {sroi_outputs}")
    print(f"Destination: {dest_dir}")
    
    # Find pixel maps
    pixel_maps = find_sroi_pixel_maps(sroi_outputs)
    
    if not pixel_maps:
        print("\n❌ No pixel map CSV files found in SROI outputs!")
        print(f"   Expected location: {sroi_outputs}/*_roi_pixel_map.csv")
        print("\nGenerate pixel maps first:")
        print(f"   cd {sroi_dir}")
        print("   python interactive_pipeline.py")
        return 0
    
    print(f"\n✓ Found {len(pixel_maps)} pixel map(s)")
    
    # Filter by board if specified
    if board_filter:
        pixel_maps = [(name, path) for name, path in pixel_maps 
                      if board_filter.lower() in name.lower()]
        if not pixel_maps:
            print(f"\n❌ No pixel maps found for board: {board_filter}")
            return 0
    
    # Import each pixel map
    imported = 0
    skipped = 0
    
    print("\n" + "-"*80)
    print("IMPORTING PIXEL MAPS")
    print("-"*80)
    
    for board_name, src_path in pixel_maps:
        dest_filename = f"{board_name}_roi_pixel_map.csv"
        dest_path = dest_dir / dest_filename
        
        # Check if already exists
        if dest_path.exists() and not force:
            print(f"\n⏭️  {board_name}: Already exists (use --force to overwrite)")
            print(f"   {dest_path}")
            skipped += 1
            continue
        
        # Copy file
        try:
            shutil.copy2(src_path, dest_path)
            print(f"\n✅ {board_name}: Imported successfully")
            print(f"   Source: {src_path.name}")
            print(f"   Dest:   {dest_path}")
            imported += 1
            
        except Exception as e:
            print(f"\n❌ {board_name}: Import failed - {e}")
    
    # Summary
    print("\n" + "="*80)
    print("IMPORT COMPLETE")
    print("="*80)
    print(f"\n  Imported: {imported}")
    print(f"  Skipped:  {skipped}")
    print(f"  Total:    {len(pixel_maps)}")
    
    if imported > 0:
        print(f"\n✓ Pixel maps are ready for ML pipeline")
        print(f"  Location: {dest_dir}/")
        print(f"\nNext steps:")
        print(f"  1. Run thermal_post_processing pipeline (Phases 1-7)")
        print(f"  2. Run Phase 8 ML training")
        print(f"  3. Dataset builder will automatically use these pixel maps")
    
    return imported


def main():
    parser = argparse.ArgumentParser(
        description="Import ROI pixel maps from SROI generation to thermal_post_processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all pixel maps
  python import_roi_pixel_maps.py
  
  # Import only HBridge pixel map
  python import_roi_pixel_maps.py --board HBridge
  
  # Force overwrite existing pixel maps
  python import_roi_pixel_maps.py --force
  
  # Import specific board and overwrite
  python import_roi_pixel_maps.py --board LoadShedding --force
        """
    )
    
    parser.add_argument(
        '--board',
        type=str,
        default=None,
        help='Only import pixel map for this board (e.g., HBridge, LoadShedding)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing pixel maps'
    )
    
    args = parser.parse_args()
    
    try:
        import_pixel_maps(board_filter=args.board, force=args.force)
    except KeyboardInterrupt:
        print("\n\nImport cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
