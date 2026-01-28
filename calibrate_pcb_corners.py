"""
Interactive PCB corner calibration tool for FLIR coordinate transformation.

This tool helps you identify the PCB corners in the FLIR image and calculates
the correct transformation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import FLIR loader
sys.path.append(str(Path(__file__).parent))
from flir_frame_loader import FLIRFrameLoader

def calibrate_corners_interactive(flir_folder):
    """
    Interactive tool to calibrate PCB corners in FLIR image.
    
    User clicks on:
    1. Bottom-left corner of PCB
    2. Top-right corner of PCB
    
    Returns corner pixel coordinates.
    """
    # Load FLIR frames
    print("Loading FLIR frames...")
    loader = FLIRFrameLoader(verbose=False)
    loader.load_sequence(flir_folder)
    
    # Use middle frame
    frame_idx = len(loader.frames) // 2
    flir = loader.frames[frame_idx]
    flir_time = loader.timestamps[frame_idx]
    
    print(f"Using frame {frame_idx}/{len(loader.frames)} at t={flir_time:.1f}s")
    print(f"Temperature range: {np.min(flir):.1f} - {np.max(flir):.1f}°C")
    
    # Create interactive plot
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(flir, cmap='hot', interpolation='nearest')
    ax.set_title('Click PCB Corners:\n1. Bottom-Left, 2. Top-Right, then close window', 
                fontsize=14, weight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    # Store clicked points
    corners = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        corners.append((x, y))
        
        if len(corners) == 1:
            ax.plot(x, y, 'go', markersize=15, markeredgewidth=3, markeredgecolor='white')
            ax.text(x, y-15, 'Bottom-Left', color='white', fontsize=12, weight='bold',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7))
            print(f"✓ Bottom-Left corner: ({x}, {y})")
            ax.set_title('Now click Top-Right corner, then close window', 
                        fontsize=14, weight='bold')
        elif len(corners) == 2:
            ax.plot(x, y, 'mo', markersize=15, markeredgewidth=3, markeredgecolor='white')
            ax.text(x, y+15, 'Top-Right', color='white', fontsize=12, weight='bold',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='magenta', alpha=0.7))
            print(f"✓ Top-Right corner: ({x}, {y})")
            
            # Draw PCB bounding box
            bl_x, bl_y = corners[0]
            tr_x, tr_y = corners[1]
            rect = plt.Rectangle((bl_x, tr_y), tr_x - bl_x, bl_y - tr_y,
                                fill=False, edgecolor='cyan', linewidth=3)
            ax.add_patch(rect)
            
            ax.set_title('PCB calibrated! Close window to continue.', 
                        fontsize=14, weight='bold', color='green')
            print("\n✅ Calibration complete! Close the window to continue.")
        
        plt.draw()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("\n" + "="*80)
    print("INSTRUCTIONS:")
    print("="*80)
    print("1. Look at the FLIR thermal image")
    print("2. Click on the BOTTOM-LEFT corner of the PCB (where X=min, Y=max in image)")
    print("3. Click on the TOP-RIGHT corner of the PCB (where X=max, Y=min in image)")
    print("4. Close the window when done")
    print("="*80)
    
    plt.show()
    
    if len(corners) != 2:
        print("❌ Calibration cancelled or incomplete!")
        return None, None
    
    return corners[0], corners[1]


def test_calibration(flir_folder, component_csv, bottom_left_px, top_right_px):
    """Test the calibration by showing ROI overlay"""
    import pandas as pd
    
    # PCB dimensions for HBridge
    pcb_width_mm = 227.381
    pcb_height_mm = 164.937
    
    # Load FLIR
    loader = FLIRFrameLoader(verbose=False)
    loader.load_sequence(flir_folder)
    frame_idx = len(loader.frames) // 2
    flir = loader.frames[frame_idx]
    
    # Calculate transformation
    bl_x, bl_y = bottom_left_px
    tr_x, tr_y = top_right_px
    
    pcb_width_px = tr_x - bl_x
    pcb_height_px = bl_y - tr_y
    
    scale_x = pcb_width_px / pcb_width_mm
    scale_y = pcb_height_px / pcb_height_mm
    uniform_scale = min(scale_x, scale_y)
    
    offset_x = bl_x
    offset_y = tr_y
    image_height = pcb_height_px
    
    print(f"\nCalculated Transformation:")
    print(f"  PCB in FLIR: {pcb_width_px} x {pcb_height_px} pixels")
    print(f"  Scale: {uniform_scale:.4f} px/mm")
    print(f"  Offset: ({offset_x}, {offset_y}) px")
    print(f"  Image height for flip: {image_height} px")
    
    # Transform function
    def transform_coords(x_mm, y_mm):
        x_px = x_mm * uniform_scale
        y_px = y_mm * uniform_scale
        # Y-flip
        y_px = image_height - y_px
        # Apply offset
        x_px += offset_x
        y_px += offset_y
        return int(x_px), int(y_px)
    
    # Load components
    df = pd.read_csv(component_csv)
    
    # Thermistor components
    thermistor_components = [
        'DL13', 'PS2', 'PS3', 'DL10', 'PS1', 'DL11', 'R88', 'U34',
        'U5', 'U40', 'U29', 'U26', 'U33', 'U4', 'U32', 'U6',
        'U23', 'U19', 'U8', 'U13', 'R5', 'U12'
    ]
    
    # Visualize
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(flir, cmap='hot', interpolation='nearest', alpha=0.7)
    ax.set_title('ROI Calibration Test - Thermistor Components', fontsize=14, weight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    # Draw PCB boundary
    rect = plt.Rectangle((bl_x, tr_y), pcb_width_px, pcb_height_px,
                        fill=False, edgecolor='cyan', linewidth=3, linestyle='--')
    ax.add_patch(rect)
    ax.text(bl_x, bl_y+20, 'PCB Boundary', color='cyan', fontsize=10, weight='bold')
    
    # Plot thermistor components
    colors = plt.cm.Set1(np.linspace(0, 1, len(thermistor_components)))
    for i, comp in enumerate(thermistor_components):
        row = df[df['Component'] == comp]
        if not row.empty:
            x_mm = float(row['X'].values[0])
            y_mm = float(row['Y'].values[0])
            px, py = transform_coords(x_mm, y_mm)
            
            # Draw circle
            circle = plt.Circle((px, py), radius=10, fill=False, 
                              color=colors[i], linewidth=3)
            ax.add_patch(circle)
            
            # Add label
            ax.text(px, py, comp, fontsize=9, color=colors[i], 
                   weight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='black', alpha=0.7, edgecolor=colors[i]))
    
    plt.tight_layout()
    plt.savefig('outputs/0115_1806_P1-7/calibration_test.png', dpi=200)
    print(f"\n✅ Saved calibration test: outputs/0115_1806_P1-7/calibration_test.png")
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent
    flir_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s_filtered"
    component_csv = project_root / "inputs" / "hbridge_pcb_components_enhanced.csv"
    
    print("="*80)
    print("PCB CORNER CALIBRATION TOOL")
    print("="*80)
    
    # Get corners interactively
    bottom_left, top_right = calibrate_corners_interactive(str(flir_folder))
    
    if bottom_left and top_right:
        print(f"\n" + "="*80)
        print("CALIBRATION RESULTS")
        print("="*80)
        print(f"Bottom-Left corner: {bottom_left}")
        print(f"Top-Right corner: {top_right}")
        print()
        print("Add these to generate_roi_pixel_map.py:")
        print(f"  pcb_bottom_left_px = {bottom_left}")
        print(f"  pcb_top_right_px = {top_right}")
        
        # Test the calibration
        print(f"\n" + "="*80)
        print("TESTING CALIBRATION")
        print("="*80)
        test_calibration(str(flir_folder), str(component_csv), bottom_left, top_right)
