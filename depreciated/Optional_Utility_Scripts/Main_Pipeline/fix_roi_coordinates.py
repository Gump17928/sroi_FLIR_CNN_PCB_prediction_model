"""
Interactive tool to fix ROI coordinate transformation.

This helps you test different coordinate transformations (flip X, flip Y, swap axes)
to find the correct mapping between PCB coordinates and FLIR pixel coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import sys

# Import FLIR loader
sys.path.append(str(Path(__file__).parent))
from flir_frame_loader import FLIRFrameLoader

def pcb_to_pixel_coords(x_mm, y_mm, pcb_bounds, image_shape, flip_x=False, flip_y=False, swap_xy=False):
    """
    Convert PCB mm coordinates to pixel coordinates with transformation options.
    
    Args:
        x_mm, y_mm: PCB coordinates in mm
        pcb_bounds: (x_min, y_min, x_max, y_max) in mm
        image_shape: (height, width) in pixels
        flip_x: Flip X axis
        flip_y: Flip Y axis (common for image coordinates)
        swap_xy: Swap X and Y axes
    """
    x_min, y_min, x_max, y_max = pcb_bounds
    height, width = image_shape
    
    # Normalize to [0, 1]
    x_norm = (x_mm - x_min) / (x_max - x_min)
    y_norm = (y_mm - y_min) / (y_max - y_min)
    
    # Apply flips
    if flip_x:
        x_norm = 1 - x_norm
    if flip_y:
        y_norm = 1 - y_norm
    
    # Scale to pixels
    if swap_xy:
        pixel_x = int(y_norm * width)
        pixel_y = int(x_norm * height)
    else:
        pixel_x = int(x_norm * width)
        pixel_y = int(y_norm * height)
    
    return pixel_x, pixel_y


def visualize_transformation(component_csv, flir_folder, 
                            pcb_bounds, image_shape,
                            flip_x=False, flip_y=False, swap_xy=False,
                            thermistor_components=None):
    """
    Visualize ROI locations with current transformation on a FLIR frame.
    """
    # Load component coordinates
    df = pd.read_csv(component_csv)
    comp_coords = {}
    for _, row in df.iterrows():
        comp_name = str(row['Component']).strip()
        x_mm = float(row['X'])
        y_mm = float(row['Y'])
        comp_coords[comp_name] = (x_mm, y_mm)
    
    # Load FLIR frames
    print(f"  Loading FLIR frames from {flir_folder}...")
    loader = FLIRFrameLoader(verbose=False)
    loader.load_sequence(flir_folder)
    
    # Pick middle frame
    frame_idx = len(loader.frames) // 2
    flir = loader.frames[frame_idx]
    flir_time = loader.timestamps[frame_idx]
    print(f"  Using frame {frame_idx}/{len(loader.frames)} at t={flir_time:.1f}s")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: FLIR frame only
    im1 = ax1.imshow(flir, cmap='hot', interpolation='nearest')
    ax1.set_title('FLIR Frame (Original)', fontsize=14)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
    
    # Right: FLIR + ROI overlay
    im2 = ax2.imshow(flir, cmap='hot', interpolation='nearest', alpha=0.6)
    transform_str = f"flip_x={flip_x}, flip_y={flip_y}, swap_xy={swap_xy}"
    ax2.set_title(f'FLIR + ROI Overlay ({transform_str})', fontsize=14)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
    
    # Plot all components in gray
    for comp_name, (x_mm, y_mm) in comp_coords.items():
        px, py = pcb_to_pixel_coords(x_mm, y_mm, pcb_bounds, image_shape, 
                                     flip_x, flip_y, swap_xy)
        if 0 <= px < image_shape[1] and 0 <= py < image_shape[0]:
            ax2.plot(px, py, 'o', color='gray', markersize=6, alpha=0.3)
            ax2.text(px, py, comp_name, fontsize=6, color='gray', alpha=0.5,
                    ha='center', va='bottom')
    
    # Highlight thermistor components in bright colors
    if thermistor_components:
        colors = plt.cm.Set1(np.linspace(0, 1, len(thermistor_components)))
        for i, comp in enumerate(thermistor_components):
            if comp in comp_coords:
                x_mm, y_mm = comp_coords[comp]
                px, py = pcb_to_pixel_coords(x_mm, y_mm, pcb_bounds, image_shape,
                                           flip_x, flip_y, swap_xy)
                if 0 <= px < image_shape[1] and 0 <= py < image_shape[0]:
                    # Draw circle around component
                    circle = plt.Circle((px, py), radius=10, fill=False, 
                                      color=colors[i], linewidth=3)
                    ax2.add_patch(circle)
                    
                    # Add label
                    ax2.text(px, py, comp, fontsize=10, color=colors[i], 
                           weight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', alpha=0.7, edgecolor=colors[i]))
    
    plt.tight_layout()
    output_path = f'outputs/0115_1806_P1-7/roi_check_fx{int(flip_x)}_fy{int(flip_y)}_swap{int(swap_xy)}.png'
    plt.savefig(output_path, dpi=200)
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    # Configuration
    component_csv = project_root / "inputs" / "hbridge_pcb_components_enhanced.csv"
    
    # Load FLIR frames folder
    flir_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s_filtered"
    
    # PCB bounds (from actual component coordinates)
    pcb_bounds = (8.3566, 6.595, 223.6724, 161.9504)
    
    # FLIR image shape
    image_shape = (480, 640)
    
    # Thermistor components to highlight
    thermistor_components = [
        'DL13', 'PS2', 'PS3', 'DL10', 'PS1', 'DL11', 'R88', 'U34',
        'U5', 'U40', 'U29', 'U26', 'U33', 'U4', 'U32', 'U6',
        'U23', 'U19', 'U8', 'U13', 'R5', 'U12'
    ]
    
    print("="*80)
    print("ROI COORDINATE TRANSFORMATION TEST")
    print("="*80)
    print(f"Component CSV: {component_csv}")
    print(f"FLIR folder: {flir_folder}")
    print(f"PCB bounds: {pcb_bounds} mm")
    print(f"Image shape: {image_shape}")
    print(f"Thermistor components: {len(thermistor_components)}")
    print()
    
    # Test different transformations
    transformations = [
        (False, False, False, "Original"),
        (False, True, False, "Flip Y only"),
        (True, False, False, "Flip X only"),
        (True, True, False, "Flip X and Y"),
        (False, False, True, "Swap XY"),
        (False, True, True, "Swap XY + Flip Y"),
    ]
    
    print("Testing transformations...")
    for flip_x, flip_y, swap_xy, desc in transformations:
        print(f"  {desc:20s}: flip_x={flip_x}, flip_y={flip_y}, swap_xy={swap_xy}")
        visualize_transformation(
            str(component_csv),
            str(flir_folder),
            pcb_bounds,
            image_shape,
            flip_x=flip_x,
            flip_y=flip_y,
            swap_xy=swap_xy,
            thermistor_components=thermistor_components
        )
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print("Check the generated images in outputs/0115_1806_P1-7/")
    print("Look for the one where the colored circles (thermistor components)")
    print("align with the HOT SPOTS on the FLIR image.")
    print()
    print("Once you identify the correct transformation:")
    print("1. Note which settings work (flip_x, flip_y, swap_xy)")
    print("2. I'll update generate_roi_pixel_map.py with the correct transformation")
    print("3. Rebuild the dataset and verify correlation improves!")
