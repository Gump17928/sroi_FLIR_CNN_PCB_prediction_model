"""
Visualize ROI locations overlaid on a FLIR frame to verify pixel mapping.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load dataset
print("Loading dataset...")
with h5py.File('ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5', 'r') as f:
    flir = f['flir_frames'][:]
    masks = f['roi_masks'][:]
    comps = [c.decode() for c in f['metadata']['component_names'][:]]

# Pick a representative frame (middle of test)
frame_idx = len(flir) // 2
frame = flir[frame_idx]

print(f"Frame {frame_idx}: {np.min(frame):.1f} - {np.max(frame):.1f}°C")
print(f"Components: {len(comps)}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left plot: FLIR frame
im1 = ax1.imshow(frame, cmap='hot', interpolation='nearest')
ax1.set_title(f'FLIR Frame #{frame_idx} (t={frame_idx * 15}s)', fontsize=14)
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=ax1, label='Temperature (°C)')

# Right plot: ROI overlay
im2 = ax2.imshow(frame, cmap='hot', interpolation='nearest', alpha=0.5)
ax2.set_title('ROI Locations Overlay', fontsize=14)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

# Overlay ROI masks with labels
colors = plt.cm.tab20(np.linspace(0, 1, len(comps)))
for i, (comp, mask) in enumerate(zip(comps, masks)):
    if np.sum(mask) > 0:
        # Get ROI centroid
        y_coords, x_coords = np.where(mask > 0)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        # Draw ROI outline
        ax2.contour(mask, levels=[0.5], colors=[colors[i]], linewidths=2)
        
        # Add label at centroid
        ax2.text(centroid_x, centroid_y, comp, 
                color='white', fontsize=8, 
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7, edgecolor='white'))

plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
plt.tight_layout()
plt.savefig('outputs/0115_1806_P1-7/roi_visualization.png', dpi=200)
print("\n✅ Saved ROI visualization: outputs/0115_1806_P1-7/roi_visualization.png")

# Create a detailed list of ROI locations
print("\n" + "="*80)
print("ROI PIXEL LOCATIONS")
print("="*80)
for i, (comp, mask) in enumerate(zip(comps, masks)):
    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask > 0)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        temp = np.mean(frame[mask > 0])
        print(f"{comp:8s}: Center=({centroid_x:6.1f}, {centroid_y:6.1f}), Pixels={np.sum(mask):4d}, Temp={temp:5.1f}°C")
    else:
        print(f"{comp:8s}: NO PIXELS")

# Check if ROIs are clustered or spread out
print("\n" + "="*80)
print("SPATIAL DISTRIBUTION CHECK")
print("="*80)
all_x = []
all_y = []
for mask in masks:
    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask > 0)
        all_x.extend(x_coords)
        all_y.extend(y_coords)

if all_x:
    print(f"X range: {min(all_x):.0f} - {max(all_x):.0f} (span = {max(all_x) - min(all_x):.0f} pixels)")
    print(f"Y range: {min(all_y):.0f} - {max(all_y):.0f} (span = {max(all_y) - min(all_y):.0f} pixels)")
    print(f"Image size: {frame.shape[1]} x {frame.shape[0]} pixels")
    
    if max(all_x) - min(all_x) < frame.shape[1] * 0.3:
        print("⚠️  WARNING: All ROIs clustered in narrow X region!")
    if max(all_y) - min(all_y) < frame.shape[0] * 0.3:
        print("⚠️  WARNING: All ROIs clustered in narrow Y region!")
