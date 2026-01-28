#!/usr/bin/env python3
"""
Diagnose ROI alignment between pixel map CSV and HDF5 dataset.

Checks if the ROI masks in the HDF5 were correctly generated from the pixel map.
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import ast


def diagnose_roi_alignment(board_name='HBridge'):
    """
    Compare pixel positions between pixel map CSV and HDF5 ROI masks.
    
    Args:
        board_name: Board name (HBridge, LoadShedding, etc.)
    """
    thermal_dir = Path(__file__).parent
    
    # Paths
    pixel_map_csv = thermal_dir / "outputs" / "roi_pixel_maps" / f"{board_name}_roi_pixel_map.csv"
    hdf5_file = thermal_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / f"{board_name}_cnn_dataset.h5"
    
    print("="*80)
    print("ROI ALIGNMENT DIAGNOSTIC")
    print("="*80)
    
    # Load pixel map CSV
    print(f"\n1. Loading Pixel Map CSV")
    print("-"*40)
    if not pixel_map_csv.exists():
        print(f"❌ Pixel map not found: {pixel_map_csv}")
        return
    
    pixel_df = pd.read_csv(pixel_map_csv)
    print(f"✓ Loaded: {pixel_map_csv.name}")
    print(f"  Total components in CSV: {len(pixel_df)}")
    
    # Parse pixel lists
    pixel_df['pixel_list_parsed'] = pixel_df['pixel_list'].apply(ast.literal_eval)
    
    # Create lookup by component name
    csv_pixel_map = {}
    for _, row in pixel_df.iterrows():
        comp_name = row['component_name']
        pixels = set(tuple(p) for p in row['pixel_list_parsed'])
        csv_pixel_map[comp_name] = {
            'pixels': pixels,
            'center': (row['center_x_px'], row['center_y_px']),
            'count': row['pixel_count']
        }
    
    # Load HDF5 dataset
    print(f"\n2. Loading HDF5 Dataset")
    print("-"*40)
    if not hdf5_file.exists():
        print(f"❌ HDF5 file not found: {hdf5_file}")
        return
    
    with h5py.File(hdf5_file, 'r') as f:
        roi_masks = f['roi_masks'][:]
        component_names = [c.decode() for c in f['metadata']['component_names'][:]]
    
    print(f"✓ Loaded: {hdf5_file.name}")
    print(f"  Components in HDF5: {len(component_names)}")
    print(f"  ROI masks shape: {roi_masks.shape}")
    
    # Extract pixels from HDF5 masks
    print(f"\n3. Extracting Pixels from HDF5 Masks")
    print("-"*40)
    
    hdf5_pixel_map = {}
    for idx, comp_name in enumerate(component_names):
        mask = roi_masks[idx]
        y_coords, x_coords = np.where(mask > 0)
        pixels = set(zip(x_coords, y_coords))
        
        if len(pixels) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
        else:
            center_x, center_y = 0, 0
        
        hdf5_pixel_map[comp_name] = {
            'pixels': pixels,
            'center': (center_x, center_y),
            'count': len(pixels)
        }
    
    print(f"✓ Extracted pixel positions from {len(component_names)} components")
    
    # Compare pixel positions
    print(f"\n4. Comparing Pixel Positions")
    print("="*80)
    
    perfect_matches = 0
    mismatches = []
    missing_in_csv = []
    
    for comp_name in component_names:
        hdf5_data = hdf5_pixel_map[comp_name]
        
        if comp_name not in csv_pixel_map:
            missing_in_csv.append(comp_name)
            continue
        
        csv_data = csv_pixel_map[comp_name]
        
        # Compare pixel sets
        hdf5_pixels = hdf5_data['pixels']
        csv_pixels = csv_data['pixels']
        
        if hdf5_pixels == csv_pixels:
            perfect_matches += 1
        else:
            # Calculate differences
            only_in_hdf5 = hdf5_pixels - csv_pixels
            only_in_csv = csv_pixels - hdf5_pixels
            
            mismatches.append({
                'component': comp_name,
                'hdf5_count': len(hdf5_pixels),
                'csv_count': len(csv_pixels),
                'hdf5_center': hdf5_data['center'],
                'csv_center': csv_data['center'],
                'only_in_hdf5': len(only_in_hdf5),
                'only_in_csv': len(only_in_csv),
                'extra_pixels_hdf5': only_in_hdf5,
                'extra_pixels_csv': only_in_csv
            })
    
    # Print results
    print(f"\nAlignment Results:")
    print(f"  Total components checked: {len(component_names)}")
    print(f"  Perfect matches: {perfect_matches}")
    print(f"  Mismatches: {len(mismatches)}")
    print(f"  Missing in CSV: {len(missing_in_csv)}")
    
    if missing_in_csv:
        print(f"\n⚠️  Components in HDF5 but NOT in pixel map CSV:")
        for comp in missing_in_csv:
            print(f"    - {comp}")
    
    if mismatches:
        print(f"\n❌ PIXEL POSITION MISMATCHES FOUND:")
        print("="*80)
        for m in mismatches[:10]:  # Show first 10
            print(f"\nComponent: {m['component']}")
            print(f"  CSV:  {m['csv_count']} pixels at ({m['csv_center'][0]:.1f}, {m['csv_center'][1]:.1f})")
            print(f"  HDF5: {m['hdf5_count']} pixels at ({m['hdf5_center'][0]:.1f}, {m['hdf5_center'][1]:.1f})")
            print(f"  Pixels only in HDF5: {m['only_in_hdf5']}")
            print(f"  Pixels only in CSV:  {m['only_in_csv']}")
            
            # Show first few mismatched pixels
            if m['only_in_csv']:
                sample = list(m['extra_pixels_csv'])[:3]
                print(f"    Sample pixels missing from HDF5: {sample}")
            if m['only_in_hdf5']:
                sample = list(m['extra_pixels_hdf5'])[:3]
                print(f"    Sample pixels extra in HDF5: {sample}")
        
        if len(mismatches) > 10:
            print(f"\n  ... and {len(mismatches) - 10} more mismatches")
    else:
        print(f"\n✅ ALL PIXEL POSITIONS MATCH PERFECTLY!")
        print(f"   The HDF5 ROI masks were correctly generated from the pixel map CSV.")
    
    # Additional statistics
    print(f"\n5. Overall Statistics")
    print("="*80)
    
    total_csv_pixels = sum(data['count'] for data in csv_pixel_map.values())
    total_hdf5_pixels = sum(data['count'] for data in hdf5_pixel_map.values())
    
    print(f"Total pixels in CSV:  {total_csv_pixels}")
    print(f"Total pixels in HDF5: {total_hdf5_pixels}")
    print(f"Difference: {total_hdf5_pixels - total_csv_pixels}")
    
    # Show component-by-component comparison table
    print(f"\n6. Component-by-Component Comparison")
    print("="*80)
    print(f"{'Component':<20} {'CSV Pixels':<12} {'HDF5 Pixels':<12} {'Match':<8}")
    print("-"*80)
    
    for comp_name in sorted(component_names):
        if comp_name in csv_pixel_map:
            csv_count = csv_pixel_map[comp_name]['count']
            hdf5_count = hdf5_pixel_map[comp_name]['count']
            match = "✓" if csv_count == hdf5_count else "✗"
            print(f"{comp_name:<20} {csv_count:<12} {hdf5_count:<12} {match:<8}")
        else:
            hdf5_count = hdf5_pixel_map[comp_name]['count']
            print(f"{comp_name:<20} {'N/A':<12} {hdf5_count:<12} {'✗':<8}")
    
    print("="*80)
    
    if mismatches:
        print("\n⚠️  ISSUE DETECTED: HDF5 dataset was NOT correctly built from pixel map!")
        print("   Recommended action: Rebuild HDF5 dataset")
        return False
    else:
        print("\n✅ SUCCESS: HDF5 dataset correctly uses pixel map positions")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose ROI alignment between pixel map and HDF5")
    parser.add_argument('--board', type=str, default='HBridge', help='Board name')
    args = parser.parse_args()
    
    diagnose_roi_alignment(args.board)
