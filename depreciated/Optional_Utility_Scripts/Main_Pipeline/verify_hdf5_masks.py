"""
Verify HDF5 ROI masks match the pixel map CSV.
"""
import h5py
import numpy as np
import pandas as pd

# Load HDF5 dataset
with h5py.File('ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5', 'r') as f:
    masks = f['roi_masks'][:]
    components = [name.decode() if isinstance(name, bytes) else name for name in f['metadata/component_names'][:]]
    
print('HDF5 ROI Mask Pixel Locations:')
print('='*80)
for i in range(min(10, len(components))):
    mask = masks[i]
    pixels = np.argwhere(mask > 0)
    if len(pixels) > 0:
        y_center = pixels[:,0].mean()
        x_center = pixels[:,1].mean()
        print(f'{components[i]:8s}: {len(pixels):3d} pixels at center ({x_center:6.1f}, {y_center:6.1f})')
    else:
        print(f'{components[i]:8s}: NO PIXELS!')

# Load pixel map CSV
pixel_map = pd.read_csv('outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv')
print('\nPixel Map CSV Expected Locations:')
print('='*80)
for idx, row in pixel_map.head(10).iterrows():
    print(f"{row['component_name']:8s}: center ({row['center_x_px']:6.1f}, {row['center_y_px']:6.1f})")

print('\n' + '='*80)
print('COMPARISON (first 10 components):')
print('='*80)
print(f"{'Component':<10} {'HDF5 (x,y)':<20} {'CSV (x,y)':<20} {'Match?'}")
print('-'*80)

for i in range(min(10, len(components))):
    mask = masks[i]
    pixels = np.argwhere(mask > 0)
    
    if len(pixels) > 0:
        hdf5_y = pixels[:,0].mean()
        hdf5_x = pixels[:,1].mean()
        
        # Find matching component in CSV
        csv_row = pixel_map[pixel_map['component_name'] == components[i]]
        if not csv_row.empty:
            csv_x = csv_row.iloc[0]['center_x_px']
            csv_y = csv_row.iloc[0]['center_y_px']
            
            # Check if within 5 pixels
            dist = np.sqrt((hdf5_x - csv_x)**2 + (hdf5_y - csv_y)**2)
            match = '✓' if dist < 5 else f'✗ (off by {dist:.1f}px)'
            
            print(f"{components[i]:<10} ({hdf5_x:5.1f}, {hdf5_y:5.1f})   ({csv_x:5.1f}, {csv_y:5.1f})   {match}")
        else:
            print(f"{components[i]:<10} ({hdf5_x:5.1f}, {hdf5_y:5.1f})   NOT IN CSV         ✗")
    else:
        print(f"{components[i]:<10} NO PIXELS          -                  ✗")
