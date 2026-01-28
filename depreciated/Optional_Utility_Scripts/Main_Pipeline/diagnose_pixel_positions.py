"""
DIAGNOSTIC 1: Compare Pixel Map CSV vs HDF5 Masks
Find if coordinate mismatch is causing poor correlation
"""
import h5py
import pandas as pd
import numpy as np

print('='*80)
print('DIAGNOSTIC 1: Pixel Map CSV vs HDF5 Mask Comparison')
print('='*80)

# Load HDF5
with h5py.File('ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5', 'r') as f:
    masks = f['roi_masks'][:]
    components = [n.decode() if isinstance(n, bytes) else n 
                  for n in f['metadata/component_names'][:]]
    print(f"\nHDF5 Components: {len(components)}")

# Load CSV
pixel_map = pd.read_csv('outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv')
print(f"CSV Components: {len(pixel_map)}")

print(f'\nChecking first 10 components:\n')

mismatches = []

for i in range(min(10, len(components))):
    mask_pixels = np.argwhere(masks[i] > 0)
    csv_rows = pixel_map[pixel_map['component_name'] == components[i]]
    
    print(f'{components[i]}:')
    
    csv_x, csv_y = None, None
    if len(csv_rows) > 0:
        csv_row = csv_rows.iloc[0]
        csv_x = csv_row["center_x_px"]
        csv_y = csv_row["center_y_px"]
        print(f'  CSV center: ({csv_x}, {csv_y})')
    else:
        print(f'  CSV: NOT FOUND in pixel map!')
    
    if len(mask_pixels) > 0:
        hdf5_x = mask_pixels[:,1].mean()
        hdf5_y = mask_pixels[:,0].mean()
        print(f'  HDF5 center: ({hdf5_x:.1f}, {hdf5_y:.1f})')
        print(f'  HDF5 pixels: {len(mask_pixels)}')
        print(f'  HDF5 Y range: {mask_pixels[:,0].min()}-{mask_pixels[:,0].max()}')
        print(f'  HDF5 X range: {mask_pixels[:,1].min()}-{mask_pixels[:,1].max()}')
        
        if csv_x is not None:
            dx = abs(hdf5_x - csv_x)
            dy = abs(hdf5_y - csv_y)
            if dx > 5 or dy > 5:
                mismatches.append((components[i], dx, dy))
                print(f'  ⚠️  MISMATCH: ΔX={dx:.1f}, ΔY={dy:.1f}')
    else:
        print(f'  HDF5: NO PIXELS IN MASK!')
    print()

print('='*80)
print('SUMMARY')
print('='*80)
if mismatches:
    print(f'\n⚠️  Found {len(mismatches)} components with position mismatch:')
    for comp, dx, dy in mismatches:
        print(f'  {comp}: ΔX={dx:.1f}px, ΔY={dy:.1f}px')
else:
    print('\n✅ All checked components match between CSV and HDF5')
