"""
Compare CSV pixel map vs HDF5 masks side-by-side
Show the actual mismatch

Usage:
    python viz_phase8_compare_csv_vs_hdf5.py HBridge
    python viz_phase8_compare_csv_vs_hdf5.py LoadShedding
"""
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys

# Board configurations
BOARD_CONFIGS = {
    "HBridge": {
        "dataset": "ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5",
        "pixel_map": "outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv",
        "output": "outputs/CSV_vs_HDF5_MISMATCH_HBridge.png"
    },
    "LoadShedding": {
        "dataset": "ml_model/cnn_thermal_modeling/datasets/LoadShedding_cnn_dataset.h5",
        "pixel_map": "../sroi_generation_ResearchIR/outputs/loadshedding_ac_switch_full_enhanced_roi_pixel_map.csv",
        "output": "outputs/CSV_vs_HDF5_MISMATCH_LoadShedding.png"
    }
}

# Get board name from command line (default: HBridge for backward compatibility)
board_name = sys.argv[1] if len(sys.argv) > 1 else "HBridge"

if board_name not in BOARD_CONFIGS:
    print(f"❌ Unknown board: {board_name}")
    print(f"Available boards: {list(BOARD_CONFIGS.keys())}")
    sys.exit(1)

config = BOARD_CONFIGS[board_name]
print(f"\n{'='*80}")
print(f"Comparing CSV Pixel Map vs HDF5 Masks for {board_name}")
print(f"{'='*80}\n")

# Load HDF5
print(f"Loading HDF5: {config['dataset']}")
with h5py.File(config['dataset'], 'r') as f:
    flir_frames = f['flir_frames'][:]
    roi_masks = f['roi_masks'][:]
    components = [n.decode() for n in f['metadata/component_names'][:]]

print(f"  ✓ Loaded {len(flir_frames)} frames, {len(components)} components")

# Load CSV
print(f"Loading CSV: {config['pixel_map']}")
pixel_map = pd.read_csv(config['pixel_map'])
pixel_map['pixel_list_parsed'] = pixel_map['pixel_list'].apply(ast.literal_eval)
print(f"  ✓ Loaded {len(pixel_map)} components from CSV\n")

# Use middle frame
frame_idx = len(flir_frames) // 2
frame = flir_frames[frame_idx]

# Create side-by-side comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

# Plot 1: CSV Pixel Map
ax1.imshow(frame, cmap='hot', alpha=0.6, origin='upper')
ax1.set_title(f'{board_name} CSV Pixel Map\n(Expected Positions)', fontsize=14, fontweight='bold')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

for idx, row in pixel_map.iterrows():
    if row['component_name'] in components[:10]:  # First 10 for clarity
        pixels = row['pixel_list_parsed']
        if pixels:
            x_coords = [p[0] for p in pixels]
            y_coords = [p[1] for p in pixels]
            center_x = row['center_x_px']
            center_y = row['center_y_px']
            
            ax1.scatter(x_coords, y_coords, c=[colors[components.index(row['component_name'])]], s=20, alpha=0.8)
            ax1.text(center_x, center_y, row['component_name'], 
                    color='white', fontsize=8, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[components.index(row['component_name'])], alpha=0.8))

# Plot 2: HDF5 Masks  
ax2.imshow(frame, cmap='hot', alpha=0.6, origin='upper')
ax2.set_title(f'{board_name} HDF5 Dataset Masks\n(Actual Positions Used by ML)', fontsize=14, fontweight='bold')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

for i in range(min(10, len(components))):
    mask = roi_masks[i]
    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask > 0)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        ax2.scatter(x_coords, y_coords, c=[colors[i]], s=20, alpha=0.8)
        ax2.text(center_x, center_y, components[i],
                color='white', fontsize=8, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.8))

# Plot 3: Overlay showing mismatch
ax3.imshow(frame, cmap='hot', alpha=0.4, origin='upper')
ax3.set_title(f'{board_name} Mismatch Visualization\nCSV (circles) vs HDF5 (X)', fontsize=14, fontweight='bold')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

for i in range(min(10, len(components))):
    # CSV position
    csv_row = pixel_map[pixel_map['component_name'] == components[i]]
    if len(csv_row) > 0:
        csv_x = csv_row.iloc[0]['center_x_px']
        csv_y = csv_row.iloc[0]['center_y_px']
        ax3.scatter(csv_x, csv_y, c=[colors[i]], s=200, marker='o', alpha=0.7, label=f'{components[i]} CSV')
    
    # HDF5 position
    mask = roi_masks[i]
    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask > 0)
        hdf5_x = np.mean(x_coords)
        hdf5_y = np.mean(y_coords)
        ax3.scatter(hdf5_x, hdf5_y, c=[colors[i]], s=200, marker='x', linewidths=3, alpha=0.7)
        
        # Draw arrow showing offset
        if len(csv_row) > 0:
            ax3.arrow(csv_x, csv_y, hdf5_x - csv_x, hdf5_y - csv_y,
                     head_width=10, head_length=10, fc=colors[i], ec=colors[i], alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig(config['output'], dpi=150, bbox_inches='tight')
print(f"\n✅ Saved to: {config['output']}")
plt.close()

print(f"\n{board_name} Mismatch Summary (first 10 components):")
for i in range(min(10, len(components))):
    csv_row = pixel_map[pixel_map['component_name'] == components[i]]
    mask = roi_masks[i]
    
    if len(csv_row) > 0 and np.sum(mask) > 0:
        csv_x = csv_row.iloc[0]['center_x_px']
        csv_y = csv_row.iloc[0]['center_y_px']
        
        y_coords, x_coords = np.where(mask > 0)
        hdf5_x = np.mean(x_coords)
        hdf5_y = np.mean(y_coords)
        
        dx = abs(hdf5_x - csv_x)
        dy = abs(hdf5_y - csv_y)
        
        print(f"{components[i]:8s}: CSV({csv_x:6.1f}, {csv_y:6.1f}) → HDF5({hdf5_x:6.1f}, {hdf5_y:6.1f}) | ΔX={dx:5.1f}px, ΔY={dy:5.1f}px")
