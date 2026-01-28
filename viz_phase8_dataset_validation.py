#!/usr/bin/env python3
"""
===============================================================================
Phase 8: ML Dataset Validation Visualization
===============================================================================

PURPOSE:
    Visualize ROI pixel map positions and HDF5 dataset alignment to validate
    that ML training data has correct spatial positioning.

VISUALIZATIONS:
    1. ROI Verification: Pixel map overlaid on FLIR frame
    2. HDF5 Validation: Dataset ROI masks overlaid on FLIR frame
    3. Combined Report: Side-by-side comparison with statistics

OUTPUT LOCATION:
    outputs/{session_id}/visualizations/
    - {board}_roi_verification.png
    - {board}_hdf5_validation.png
    - {board}_dataset_validation_report.png

WHEN TO USE:
    - After pixel map import (before HDF5 build) - verify positions
    - After HDF5 dataset build - validate final dataset
    - Anytime you need to debug ROI positioning issues

CONTROL FLAGS (Future):
    TODO: Add configuration flags to control which visualizations are generated
    This will help manage image bloat when running pipeline repeatedly.
    Planned flags:
        ENABLE_PHASE8_ROI_VERIFICATION = True/False
        ENABLE_PHASE8_HDF5_VALIDATION = True/False
        ENABLE_PHASE8_COMBINED_REPORT = True/False

===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import ast
from pathlib import Path
from typing import Optional, Tuple


def load_flir_frame(flir_folder: Path, frame_idx: int = 0) -> Tuple[np.ndarray, str]:
    """
    Load a single FLIR frame from ResearchIR exports.
    
    Args:
        flir_folder: Path to ResearchIR_Outputs_* folder
        frame_idx: Frame index to load
    
    Returns:
        Tuple of (frame array, frame filename)
    """
    # Get list of CSV files
    csv_files = sorted(flir_folder.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {flir_folder}")
    
    # Filter out metadata files
    data_files = [f for f in csv_files if 'metadata' not in f.name.lower()]
    
    if frame_idx >= len(data_files):
        frame_idx = len(data_files) // 2  # Use middle frame
        print(f"  Frame index too high, using frame {frame_idx}")
    
    # Load frame (FLIR CSV: 5-line header, then 480 rows x 640 columns)
    frame_file = data_files[frame_idx]
    df = pd.read_csv(frame_file, skiprows=5, header=None)
    frame = df.values
    
    return frame, frame_file.stem


def load_pixel_map(pixel_map_csv: Path) -> pd.DataFrame:
    """
    Load ROI pixel map CSV.
    
    Args:
        pixel_map_csv: Path to pixel map CSV
    
    Returns:
        DataFrame with component ROI data and parsed pixel lists
    """
    df = pd.read_csv(pixel_map_csv)
    
    # Parse pixel_list strings into actual lists
    df['pixel_list_parsed'] = df['pixel_list'].apply(ast.literal_eval)
    
    return df


def visualize_roi_pixel_map(board_name: str, session_id: str, 
                            frame_idx: Optional[int] = None,
                            thermal_dir: Optional[Path] = None) -> Path:
    """
    Create ROI pixel map verification visualization.
    
    Shows pixel map overlaid on FLIR thermal frame to verify positions
    are correct before building HDF5 dataset.
    
    Args:
        board_name: Board name (HBridge, LoadShedding, etc.)
        session_id: Session ID for output organization
        frame_idx: Frame index to visualize (None = middle frame)
        thermal_dir: Thermal processing directory (None = auto-detect)
    
    Returns:
        Path to saved visualization
    """
    if thermal_dir is None:
        thermal_dir = Path(__file__).parent
    
    # Paths
    pixel_map_csv = thermal_dir / "outputs" / "roi_pixel_maps" / f"{board_name}_roi_pixel_map.csv"
    
    # Find FLIR folder
    flir_folder_filtered = thermal_dir / "inputs" / f"ResearchIR_Outputs_{board_name}_15s_filtered"
    flir_folder_raw = thermal_dir / "inputs" / f"ResearchIR_Outputs_{board_name}_15s"
    
    if flir_folder_filtered.exists():
        flir_folder = flir_folder_filtered
    elif flir_folder_raw.exists():
        flir_folder = flir_folder_raw
    else:
        raise FileNotFoundError(f"FLIR folder not found for {board_name}")
    
    # Check pixel map exists
    if not pixel_map_csv.exists():
        raise FileNotFoundError(f"Pixel map not found: {pixel_map_csv}")
    
    # Load data
    pixel_map = load_pixel_map(pixel_map_csv)
    
    # Load FLIR frame
    if frame_idx is None:
        csv_files = list(flir_folder.glob('*.csv'))
        data_files = [f for f in csv_files if 'metadata' not in f.name.lower()]
        frame_idx = len(data_files) // 2  # Middle frame
    
    frame, frame_name = load_flir_frame(flir_folder, frame_idx)
    
    # Create visualization
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
        
        ax2.text(center_x, center_y, comp_name, 
                color='white', fontsize=7, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], 
                         alpha=0.8, edgecolor='white', linewidth=1.5))
    
    plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
    plt.tight_layout()
    
    # Save figure
    output_dir = thermal_dir / "outputs" / session_id / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{board_name}_roi_verification.png"
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def visualize_hdf5_validation(board_name: str, session_id: str,
                              frame_idx: Optional[int] = None,
                              thermal_dir: Optional[Path] = None) -> Path:
    """
    Create HDF5 dataset validation visualization.
    
    Shows ROI masks from HDF5 dataset overlaid on FLIR frame to validate
    that dataset was built correctly.
    
    Args:
        board_name: Board name (HBridge, LoadShedding, etc.)
        session_id: Session ID for output organization
        frame_idx: Frame index to visualize (None = middle frame)
        thermal_dir: Thermal processing directory (None = auto-detect)
    
    Returns:
        Path to saved visualization
    """
    if thermal_dir is None:
        thermal_dir = Path(__file__).parent
    
    # Paths
    hdf5_file = thermal_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / f"{board_name}_cnn_dataset.h5"
    
    if not hdf5_file.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {hdf5_file}")
    
    # Load HDF5 data
    with h5py.File(hdf5_file, 'r') as f:
        flir_frames = f['flir_frames'][:]
        roi_masks = f['roi_masks'][:]
        component_names = [c.decode() for c in f['metadata']['component_names'][:]]
    
    # Use middle frame if not specified
    if frame_idx is None:
        frame_idx = len(flir_frames) // 2
    
    frame = flir_frames[frame_idx]
    
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
            ax2.contour(mask, levels=[0.5], colors=[colors[i]], linewidths=2, alpha=0.8, origin='upper')
            
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
    
    composite = np.clip(composite, 0, 1)
    
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
    
    # Save figure
    output_dir = thermal_dir / "outputs" / session_id / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{board_name}_hdf5_validation.png"
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_dataset_validation_report(board_name: str, session_id: str,
                                       frame_idx: Optional[int] = None,
                                       thermal_dir: Optional[Path] = None) -> dict:
    """
    Generate complete dataset validation report with all visualizations.
    
    Creates:
        1. ROI verification plot
        2. HDF5 validation plot
        3. Diagnostic statistics
    
    Args:
        board_name: Board name (HBridge, LoadShedding, etc.)
        session_id: Session ID for output organization
        frame_idx: Frame index to visualize (None = middle frame)
        thermal_dir: Thermal processing directory (None = auto-detect)
    
    Returns:
        Dictionary with paths to generated visualizations
    """
    if thermal_dir is None:
        thermal_dir = Path(__file__).parent
    
    print("\n" + "="*80)
    print(f"PHASE 8: DATASET VALIDATION VISUALIZATION")
    print("="*80)
    print(f"\nBoard: {board_name}")
    print(f"Session: {session_id}")
    
    results = {}
    
    # Generate ROI verification
    print("\n1. Generating ROI Pixel Map Verification...")
    try:
        roi_viz_path = visualize_roi_pixel_map(board_name, session_id, frame_idx, thermal_dir)
        results['roi_verification'] = roi_viz_path
        print(f"   ✓ Saved: {roi_viz_path.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['roi_verification'] = None
    
    # Generate HDF5 validation
    print("\n2. Generating HDF5 Dataset Validation...")
    try:
        hdf5_viz_path = visualize_hdf5_validation(board_name, session_id, frame_idx, thermal_dir)
        results['hdf5_validation'] = hdf5_viz_path
        print(f"   ✓ Saved: {hdf5_viz_path.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        results['hdf5_validation'] = None
    
    print("\n" + "="*80)
    print("DATASET VALIDATION COMPLETE")
    print("="*80)
    
    if results['roi_verification']:
        print(f"\n✓ ROI Verification: {results['roi_verification']}")
    if results['hdf5_validation']:
        print(f"✓ HDF5 Validation: {results['hdf5_validation']}")
    
    return results


if __name__ == "__main__":
    """Standalone execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 8: Dataset Validation Visualization")
    parser.add_argument('--board', type=str, default='HBridge', help='Board name')
    parser.add_argument('--session', type=str, default='test_session', help='Session ID')
    parser.add_argument('--frame', type=int, default=None, help='Frame index (None = middle)')
    parser.add_argument('--mode', type=str, choices=['roi', 'hdf5', 'all'], default='all',
                       help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        generate_dataset_validation_report(args.board, args.session, args.frame)
    elif args.mode == 'roi':
        visualize_roi_pixel_map(args.board, args.session, args.frame)
    elif args.mode == 'hdf5':
        visualize_hdf5_validation(args.board, args.session, args.frame)
