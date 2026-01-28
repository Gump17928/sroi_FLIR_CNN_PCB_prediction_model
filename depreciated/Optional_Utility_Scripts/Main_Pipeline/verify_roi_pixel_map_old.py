#!/usr/bin/env python3
"""
Visualize ROI Pixel Map Locations on FLIR Frame

Overlays ROI pixel positions from the imported pixel map CSV onto a FLIR frame
to verify that corner calibration was correct and pixels align with components.

Usage:
    python verify_roi_pixel_map.py
    python verify_roi_pixel_map.py --board HBridge
    python verify_roi_pixel_map.py --board LoadShedding --frame 100

Created: January 16, 2026
"""

import argparse
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import h5py


def load_flir_frame(flir_folder, frame_idx=0):
    """
    Load a single FLIR frame from ResearchIR exports.
    
    Args:
        flir_folder: Path to ResearchIR_Outputs_* folder
        frame_idx: Frame index to load
    
    Returns:
        numpy array of shape (480, 640) with temperatures
    """
    flir_path = Path(flir_folder)
    
    # Get list of CSV files
    csv_files = sorted(flir_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {flir_folder}")
    
    # Filter out metadata files
    data_files = [f for f in csv_files if 'metadata' not in f.name.lower()]
    
    if frame_idx >= len(data_files):
        frame_idx = len(data_files) // 2  # Use middle frame
        print(f"  Frame index too high, using frame {frame_idx}")
    
    # Load frame
    frame_file = data_files[frame_idx]
    print(f"  Loading: {frame_file.name}")
    
    # FLIR CSV format: 5-line header, then 480 rows x 640 columns
    df = pd.read_csv(frame_file, skiprows=5, header=None)
    frame = df.values
    
    return frame, frame_file.stem


def load_pixel_map(pixel_map_csv):
    """
    Load ROI pixel map CSV.
    
    Args:
        pixel_map_csv: Path to pixel map CSV
    
    Returns:
        DataFrame with component ROI data
    """
    df = pd.read_csv(pixel_map_csv)
    
    # Parse pixel_list strings into actual lists
    df['pixel_list_parsed'] = df['pixel_list'].apply(ast.literal_eval)
    
    return df


def visualize_roi_pixel_map(board_name='HBridge', frame_idx=None, output_dir=None, compare_hdf5=False):
    """
    Create visualization of ROI pixel map overlaid on FLIR frame.
    
    Args:
        board_name: Board name (HBridge, LoadShedding, etc.)
        frame_idx: Frame index to visualize (None = middle frame)
        output_dir: Output directory for visualization (None = auto)
        compare_hdf5: If True, create comparison with HDF5 dataset ROI masks
    """
    print("="*80)
    print("ROI PIXEL MAP VERIFICATION")
    print("="*80)
    
    # Paths
    thermal_dir = Path(__file__).parent
    pixel_map_csv = thermal_dir / "outputs" / "roi_pixel_maps" / f"{board_name}_roi_pixel_map.csv"
    
    # Find FLIR folder
    flir_folder_filtered = thermal_dir / "inputs" / f"ResearchIR_Outputs_{board_name}_15s_filtered"
    flir_folder_raw = thermal_dir / "inputs" / f"ResearchIR_Outputs_{board_name}_15s"
    
    if flir_folder_filtered.exists():
        flir_folder = flir_folder_filtered
        print(f"\n✓ Using filtered FLIR frames")
    elif flir_folder_raw.exists():
        flir_folder = flir_folder_raw
        print(f"\n✓ Using raw FLIR frames")
    else:
        print(f"\n❌ FLIR folder not found!")
        print(f"   Tried: {flir_folder_filtered}")
        print(f"   Tried: {flir_folder_raw}")
        return
    
    # Check pixel map exists
    if not pixel_map_csv.exists():
        print(f"\n❌ Pixel map not found: {pixel_map_csv}")
        print(f"\nGenerate pixel map first:")
        print(f"  cd ../sroi_generation_ResearchIR")
        print(f"  python interactive_pipeline.py")
        print(f"  cd ../thermal_post_processing")
        print(f"  python import_roi_pixel_maps.py")
        return
    
    print(f"✓ Pixel map: {pixel_map_csv.name}")
    print(f"✓ FLIR folder: {flir_folder}")
    
    # Load data
    print(f"\n1. Loading ROI Pixel Map")
    print("-" * 40)
    pixel_map = load_pixel_map(pixel_map_csv)
    print(f"  Components: {len(pixel_map)}")
    print(f"  Total pixels: {pixel_map['pixel_count'].sum()}")
    
    print(f"\n2. Loading FLIR Frame")
    print("-" * 40)
    if frame_idx is None:
        # Count frames
        csv_files = list(flir_folder.glob('*.csv'))
        data_files = [f for f in csv_files if 'metadata' not in f.name.lower()]
        frame_idx = len(data_files) // 2  # Middle frame
        print(f"  Using middle frame: {frame_idx} of {len(data_files)}")
    
    frame, frame_name = load_flir_frame(flir_folder, frame_idx)
    print(f"  Frame shape: {frame.shape}")
    print(f"  Temp range: {np.nanmin(frame):.1f}°C to {np.nanmax(frame):.1f}°C")
    
    # Create visualization
    print(f"\n3. Creating Visualization")
    print("-" * 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Raw FLIR frame
    im1 = ax1.imshow(frame, cmap='hot', interpolation='nearest', origin='upper')
    ax1.set_title(f'{board_name} FLIR Frame\n{frame_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
    
    # Right: ROI overlay
    im2 = ax2.imshow(frame, cmap='hot', interpolation='nearest', alpha=0.6, origin='upper')
    ax2.set_title(f'ROI Pixel Map Overlay\n{len(pixel_map)} Components', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=12)
    ax2.set_ylabel('Y (pixels)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Generate distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(pixel_map)))
    
    # Overlay ROIs
    for idx, row in pixel_map.iterrows():
        comp_name = row['component_name']
        pixels = row['pixel_list_parsed']
        roi_type = row['roi_type']
        
        if not pixels:
            continue
        
        # Extract pixel coordinates
        x_coords = [p[0] for p in pixels]
        y_coords = [p[1] for p in pixels]
        
        # Plot pixels
        ax2.scatter(x_coords, y_coords, c=[colors[idx]], s=50, alpha=0.8, 
                   edgecolors='white', linewidth=1, marker='s')
        
        # Add label at center
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Small text box for label
        ax2.text(center_x, center_y, comp_name, 
                color='white', fontsize=7, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], 
                         alpha=0.8, edgecolor='white', linewidth=1.5))
    
    plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = thermal_dir / "outputs" / "roi_pixel_maps"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{board_name}_roi_verification.png"
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # Print ROI statistics
    print(f"\n4. ROI Statistics")
    print("-" * 40)
    print(f"{'Component':<20} {'Type':<12} {'Center (x,y)':<18} {'Pixels':<8} {'Avg Temp'}")
    print("-" * 80)
    
    for idx, row in pixel_map.iterrows():
        comp_name = row['component_name']
        roi_type = row['roi_type']
        pixels = row['pixel_list_parsed']
        
        if not pixels:
            print(f"{comp_name:<20} {roi_type:<12} {'N/A':<18} {'0':<8} {'N/A'}")
            continue
        
        x_coords = [p[0] for p in pixels]
        y_coords = [p[1] for p in pixels]
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Get average temperature at ROI
        temps = [frame[y, x] for x, y in pixels if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]]
        avg_temp = np.nanmean(temps) if temps else np.nan
        
        print(f"{comp_name:<20} {roi_type:<12} ({center_x:5.1f}, {center_y:5.1f})  {len(pixels):<8} {avg_temp:6.1f}°C")
    
    # Spatial distribution check
    print(f"\n5. Spatial Distribution Check")
    print("-" * 40)
    
    all_x = []
    all_y = []
    for idx, row in pixel_map.iterrows():
        pixels = row['pixel_list_parsed']
        if pixels:
            all_x.extend([p[0] for p in pixels])
            all_y.extend([p[1] for p in pixels])
    
    if all_x:
        x_span = max(all_x) - min(all_x)
        y_span = max(all_y) - min(all_y)
        
        print(f"  X range: {min(all_x)} to {max(all_x)} (span = {x_span} pixels)")
        print(f"  Y range: {min(all_y)} to {max(all_y)} (span = {y_span} pixels)")
        print(f"  Image size: {frame.shape[1]} × {frame.shape[0]} pixels")
        
        # Warnings
        if x_span < frame.shape[1] * 0.3:
            print(f"  ⚠️  WARNING: ROIs clustered in narrow X region ({x_span/frame.shape[1]*100:.1f}% of width)")
        else:
            print(f"  ✓ ROIs spread across {x_span/frame.shape[1]*100:.1f}% of image width")
            
        if y_span < frame.shape[0] * 0.3:
            print(f"  ⚠️  WARNING: ROIs clustered in narrow Y region ({y_span/frame.shape[0]*100:.1f}% of height)")
        else:
            print(f"  ✓ ROIs spread across {y_span/frame.shape[0]*100:.1f}% of image height")
    
    print(f"\n" + "="*80)
    print(f"VERIFICATION COMPLETE")
    print(f"="*80)
    print(f"\nVisualization saved: {output_file}")
    print(f"\nCheck the image to verify:")
    print(f"  1. ROI markers align with actual components")
    print(f"  2. ROIs are not all clustered in one corner")
    print(f"  3. Component labels match expected locations")
    print(f"\nIf misaligned: Regenerate SROI with corrected corner coordinates")
    print(f"="*80)
    
    # If compare mode, create HDF5 comparison
    if compare_hdf5:
        print(f"\n" + "="*80)
        print(f"HDF5 DATASET COMPARISON")
        print(f"="*80)
        visualize_hdf5_comparison(board_name, frame_idx, output_dir)


def visualize_hdf5_comparison(board_name='HBridge', frame_idx=None, output_dir=None):
    """
    Create comparison visualization showing pixel map vs HDF5 dataset ROIs.
    
    Args:
        board_name: Board name
        frame_idx: Frame index to visualize
        output_dir: Output directory
    """
    thermal_dir = Path(__file__).parent
    
    # Find HDF5 dataset
    dataset_file = thermal_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / f"{board_name}_cnn_dataset.h5"
    
    if not dataset_file.exists():
        print(f"\n⚠️  HDF5 dataset not found: {dataset_file}")
        print(f"   Build dataset first with Phase 8")
        return
    
    print(f"\n✓ Loading HDF5 dataset: {dataset_file.name}")
    
    # Load HDF5 data
    with h5py.File(dataset_file, 'r') as f:
        flir_frames = f['flir_frames'][:]
        roi_masks = f['roi_masks'][:]
        component_names = [c.decode() for c in f['metadata']['component_names'][:]]
    
    # Use middle frame if not specified
    if frame_idx is None:
        frame_idx = len(flir_frames) // 2
    
    frame = flir_frames[frame_idx]
    
    print(f"  Frame {frame_idx}: {np.nanmin(frame):.1f}°C to {np.nanmax(frame):.1f}°C")
    print(f"  Components: {len(component_names)}")
    print(f"  ROI masks shape: {roi_masks.shape}")
    
    # Create 2x2 comparison figure
    fig = plt.figure(figsize=(20, 16))
    
    # Generate colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(component_names)))
    
    # Plot 1: Raw FLIR frame
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(frame, cmap='hot', interpolation='nearest', origin='upper')
    ax1.set_title(f'{board_name} FLIR Frame #{frame_idx}\n(Raw Thermal Image)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)', fraction=0.046)
    
    # Plot 2: HDF5 ROI masks overlay
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(frame, cmap='hot', interpolation='nearest', alpha=0.6, origin='upper')
    ax2.set_title(f'HDF5 Dataset ROI Masks\n(What ML Model Uses)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=12)
    ax2.set_ylabel('Y (pixels)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Overlay HDF5 ROI masks
    for i, (comp_name, mask) in enumerate(zip(component_names, roi_masks)):
        if np.sum(mask) > 0:
            # Draw mask contour
            ax2.contour(mask, levels=[0.5], colors=[colors[i]], linewidths=2, alpha=0.8)
            
            # Get centroid
            y_coords, x_coords = np.where(mask > 0)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Add label
            ax2.text(center_x, center_y, comp_name, 
                    color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                             alpha=0.8, edgecolor='white', linewidth=1.5))
    
    plt.colorbar(im2, ax=ax2, label='Temperature (°C)', fraction=0.046)
    
    # Plot 3: HDF5 ROI masks only (no thermal)
    ax3 = plt.subplot(2, 2, 3)
    
    # Create composite mask visualization
    composite = np.zeros((*frame.shape, 3))  # RGB image
    for i, (comp_name, mask) in enumerate(zip(component_names, roi_masks)):
        if np.sum(mask) > 0:
            color_rgb = colors[i][:3]  # Get RGB from colormap
            for c in range(3):
                composite[:, :, c] += mask * color_rgb[c]
    
    composite = np.clip(composite, 0, 1)  # Ensure valid RGB range
    
    ax3.imshow(composite, origin='upper')
    ax3.set_title(f'ROI Mask Locations\n(Dataset Ground Truth Regions)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (pixels)', fontsize=12)
    ax3.set_ylabel('Y (pixels)', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='white')
    
    # Add labels
    for i, (comp_name, mask) in enumerate(zip(component_names, roi_masks)):
        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask > 0)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            ax3.text(center_x, center_y, comp_name, 
                    color='black', fontsize=6, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             alpha=0.7, edgecolor=colors[i], linewidth=2))
    
    # Plot 4: Statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Prepare statistics
    stats_data = []
    for i, (comp_name, mask) in enumerate(zip(component_names, roi_masks)):
        if np.sum(mask) > 0:
            y_coords, x_coords = np.where(mask > 0)
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            pixel_count = np.sum(mask)
            
            # Get temperature at ROI
            roi_temp = frame[mask > 0].mean()
            
            stats_data.append([
                comp_name,
                f"({center_x:.0f}, {center_y:.0f})",
                f"{pixel_count:.0f}",
                f"{roi_temp:.1f}°C"
            ])
    
    # Create table
    table = ax4.table(
        cellText=stats_data,
        colLabels=['Component', 'Center (x,y)', 'Pixels', 'Avg Temp'],
        cellLoc='left',
        loc='center',
        colWidths=[0.3, 0.25, 0.15, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('HDF5 Dataset ROI Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save comparison figure
    if output_dir is None:
        output_dir = thermal_dir / "outputs" / "roi_pixel_maps"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{board_name}_hdf5_comparison.png"
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n  ✓ Saved HDF5 comparison: {output_file}")
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Total components: {len(component_names)}")
    print(f"    Components with ROIs: {sum(1 for mask in roi_masks if np.sum(mask) > 0)}")
    print(f"    Total ROI pixels: {sum(np.sum(mask) for mask in roi_masks):.0f}")
    
    # Verify consistency
    empty_rois = [comp_name for comp_name, mask in zip(component_names, roi_masks) if np.sum(mask) == 0]
    if empty_rois:
        print(f"\n  ⚠️  Warning: {len(empty_rois)} components have no ROI pixels:")
        for comp in empty_rois:
            print(f"      - {comp}")


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
        '--output',
        type=str,
        default=None,
        help='Output directory for visualization'
    )
    
    parser.add_argument(
        '--compare-hdf5',
        action='store_true',
        help='Create comparison with HDF5 dataset ROI masks'
    )
    
    args = parser.parse_args()
    
    try:
        visualize_roi_pixel_map(
            board_name=args.board,
            frame_idx=args.frame,
            output_dir=args.output,
            compare_hdf5=args.compare_hdf5
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
