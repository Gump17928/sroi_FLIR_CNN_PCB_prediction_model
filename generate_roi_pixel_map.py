#!/usr/bin/env python3
"""
Generate ROI Pixel Map for CNN Training

Extracts pixel coordinates for each component from FLIR frames based on
component placement coordinates. Saves to CSV for Phase 8 to use without
needing to rerun the full pipeline.

Usage:
    python generate_roi_pixel_map.py

Outputs:
    outputs/<session>/<board>_roi_pixel_map.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_component_coords(csv_path):
    """Load component coordinates from pick-and-place CSV."""
    df = pd.read_csv(csv_path)
    
    # Map column names (handle different CSV formats)
    component_col = 'Component' if 'Component' in df.columns else 'Reference'
    x_col = 'X' if 'X' in df.columns else 'X(mm)'
    y_col = 'Y' if 'Y' in df.columns else 'Y(mm)'
    
    coords = {}
    for _, row in df.iterrows():
        comp_name = row[component_col]
        x_mm = row[x_col]
        y_mm = row[y_col]
        coords[comp_name] = (x_mm, y_mm)
    
    return coords


def pcb_to_pixel_coords(x_mm, y_mm, pcb_width_mm, pcb_height_mm,
                       pcb_bottom_left_px, pcb_top_right_px, image_shape):
    """
    Convert PCB mm coordinates to FLIR pixel coordinates using corner-based calibration.
    
    This method matches the sroi_generation_ResearchIR pipeline calibration.
    
    Args:
        x_mm, y_mm: PCB coordinates in mm (bottom-left origin)
        pcb_width_mm, pcb_height_mm: PCB dimensions in mm
        pcb_bottom_left_px: (x, y) pixel coordinates of PCB bottom-left corner in FLIR image
        pcb_top_right_px: (x, y) pixel coordinates of PCB top-right corner in FLIR image
        image_shape: (height, width) of FLIR image
    
    Returns:
        (pixel_x, pixel_y)
    """
    # Extract corner coordinates
    bl_x, bl_y = pcb_bottom_left_px
    tr_x, tr_y = pcb_top_right_px
    
    # Calculate PCB dimensions in FLIR pixels
    pcb_width_px = tr_x - bl_x
    pcb_height_px = bl_y - tr_y  # Y increases downward in image coordinates
    
    # Calculate scale factors (pixels per mm)
    scale_x = pcb_width_px / pcb_width_mm
    scale_y = pcb_height_px / pcb_height_mm
    
    # Use uniform scaling to maintain aspect ratio
    uniform_scale = min(scale_x, scale_y)
    
    # Transform: scale, flip Y, then offset
    # 1. Scale mm to pixels
    x_pixels = x_mm * uniform_scale
    y_pixels = y_mm * uniform_scale
    
    # 2. Y-flip (PCB uses bottom-left origin, image uses top-left origin)
    y_pixels = pcb_height_px - y_pixels
    
    # 3. Apply offset to position in image
    x_pixels += bl_x
    y_pixels += tr_y
    
    return int(x_pixels), int(y_pixels)


def extract_roi_pixels(comp_name, comp_coords, pcb_width_mm, pcb_height_mm,
                      pcb_bottom_left_px, pcb_top_right_px, image_shape, roi_radius=3):
    """
    Extract ROI pixel list for a component.
    
    Args:
        comp_name: Component name
        comp_coords: (x_mm, y_mm) in PCB coordinates
        pcb_width_mm: PCB width in mm
        pcb_height_mm: PCB height in mm
        pcb_bottom_left_px: Bottom-left corner in pixel coordinates
        pcb_top_right_px: Top-right corner in pixel coordinates
        image_shape: FLIR image shape (height, width)
        roi_radius: Radius in pixels around component center
    
    Returns:
        List of (pixel_x, pixel_y) tuples
    """
    x_mm, y_mm = comp_coords
    px, py = pcb_to_pixel_coords(
        x_mm, y_mm, pcb_width_mm, pcb_height_mm, 
        pcb_bottom_left_px, pcb_top_right_px, image_shape
    )
    
    # Generate circular ROI
    pixels = []
    for dy in range(-roi_radius, roi_radius + 1):
        for dx in range(-roi_radius, roi_radius + 1):
            if dx*dx + dy*dy <= roi_radius*roi_radius:  # Circle equation
                pixel_x = px + dx
                pixel_y = py + dy
                if 0 <= pixel_x < image_shape[1] and 0 <= pixel_y < image_shape[0]:
                    pixels.append((pixel_x, pixel_y))
    
    return pixels


def generate_roi_pixel_map(component_csv, output_csv, pcb_bounds=(0, 0, 100, 100), 
                           image_shape=(480, 640), roi_radius=3):
    """
    Generate ROI pixel map CSV from component coordinates.
    
    Args:
        component_csv: Path to component placement CSV
        output_csv: Path to save pixel map CSV
        pcb_bounds: PCB bounding box (x_min, y_min, x_max, y_max) in mm
        image_shape: FLIR image dimensions (height, width)
        roi_radius: ROI radius in pixels
    """
    print("="*80)
    print("GENERATING ROI PIXEL MAP")
    print("="*80)
    print(f"\nComponent CSV: {component_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"PCB bounds: {pcb_bounds} mm")
    print(f"Image shape: {image_shape}")
    print(f"ROI radius: {roi_radius} pixels\n")
    
    # Load component coordinates
    comp_coords = load_component_coords(component_csv)
    print(f"✅ Loaded {len(comp_coords)} components\n")
    
    # Extract PCB dimensions from bounds
    x_min, y_min, x_max, y_max = pcb_bounds
    pcb_width_mm = x_max - x_min
    pcb_height_mm = y_max - y_min
    
    # Define PCB corners in pixel coordinates (calibrated for H_Bridge)
    pcb_bottom_left_px = (66, 304)   # Bottom-left corner (calibrated)
    pcb_top_right_px = (453, 23)     # Top-right corner (calibrated)
    
    # Generate pixel maps for each component
    rows = []
    for comp_name, coords in comp_coords.items():
        pixels = extract_roi_pixels(
            comp_name, coords, pcb_width_mm, pcb_height_mm,
            pcb_bottom_left_px, pcb_top_right_px, image_shape, roi_radius
        )
        
        rows.append({
            'component_name': comp_name,
            'pixel_count': len(pixels),
            'pixel_list': str(pixels)  # Save as string representation
        })
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values('pixel_count', ascending=False)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Generated ROI pixel map: {output_csv}")
    print(f"   Components: {len(rows)}")
    print(f"   Total pixels: {df['pixel_count'].sum()}")
    print(f"   Coverage: {100 * df['pixel_count'].sum() / (image_shape[0] * image_shape[1]):.2f}%")
    
    return output_csv


if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    # Configuration
    component_csv = project_root / "inputs" / "hbridge_pcb_components_enhanced.csv"
    output_dir = project_root / "outputs" / "0115_1806_P1-7"
    output_csv = output_dir / "HBridge_15s_roi_pixel_map.csv"
    
    # PCB bounds for HBridge (from actual component coordinates)
    # These should match the actual PCB dimensions that the FLIR camera views
    pcb_bounds = (8.3566, 6.595, 223.6724, 161.9504)  # (x_min, y_min, x_max, y_max) in mm
    
    # FLIR image shape
    image_shape = (480, 640)  # Standard FLIR resolution
    
    # ROI radius (pixels around component center)
    roi_radius = 5  # Larger ROI for better thermal capture
    
    # Generate
    generate_roi_pixel_map(
        str(component_csv),
        str(output_csv),
        pcb_bounds=pcb_bounds,
        image_shape=image_shape,
        roi_radius=roi_radius
    )
    
    print("\n✅ Done! Use this CSV for Phase 8 CNN training.")
