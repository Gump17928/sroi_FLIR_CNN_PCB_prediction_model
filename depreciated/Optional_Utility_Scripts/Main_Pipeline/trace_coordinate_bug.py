"""
Trace the coordinate transformation bug in HDF5 mask generation.
"""
import numpy as np
import pandas as pd
import h5py

print("="*80)
print("TRACING COORDINATE BUG")
print("="*80)

# 1. Load CSV pixel map
print("\n1. CSV PIXEL MAP")
print("-"*40)
df = pd.read_csv('outputs/roi_pixel_maps/HBridge_roi_pixel_map.csv')
df['pixel_list_parsed'] = df['pixel_list'].apply(eval)
dl13_row = df[df['component_name'] == 'DL13'].iloc[0]

print(f"DL13 from CSV:")
print(f"  center_x_px: {dl13_row['center_x_px']}")
print(f"  center_y_px: {dl13_row['center_y_px']}")
print(f"  pixel_list: {dl13_row['pixel_list_parsed']}")
print(f"  pixel_count: {len(dl13_row['pixel_list_parsed'])}")

# 2. Load HDF5 dataset
print("\n2. HDF5 DATASET")
print("-"*40)
with h5py.File('ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5', 'r') as f:
    masks = f['roi_masks'][:]
    comp_names = [n.decode() for n in f['metadata/component_names'][:]]
    
dl13_idx = comp_names.index('DL13')
dl13_mask = masks[dl13_idx]

print(f"DL13 from HDF5:")
print(f"  mask shape: {dl13_mask.shape}")
print(f"  non-zero pixels: {np.sum(dl13_mask)}")

positions = np.argwhere(dl13_mask == 1)
if len(positions) > 0:
    center_y, center_x = positions.mean(axis=0)
    print(f"  center (row, col): ({center_y:.1f}, {center_x:.1f})")
    print(f"  center (x, y): ({center_x:.1f}, {center_y:.1f})")
    print(f"\n  First 5 pixels (row, col):")
    for i in range(min(5, len(positions))):
        print(f"    {positions[i]}")

# 3. Simulate what generate_roi_masks() SHOULD do
print("\n3. SIMULATING generate_roi_masks() LOGIC")
print("-"*40)
print("Code logic:")
print("  for x, y in pixel_list:")
print("      roi_masks[idx, y, x] = 1")
print()

# Create a test mask
test_mask = np.zeros((480, 640), dtype=np.uint8)
pixel_list = dl13_row['pixel_list_parsed']

print(f"Applying pixel_list from CSV to test mask:")
for x, y in pixel_list:
    if 0 <= x < 640 and 0 <= y < 480:
        test_mask[y, x] = 1
        print(f"  Setting mask[{y}, {x}] = 1  (from pixel ({x}, {y}))")

# Check result
test_positions = np.argwhere(test_mask == 1)
if len(test_positions) > 0:
    test_center_y, test_center_x = test_positions.mean(axis=0)
    print(f"\nTest mask result:")
    print(f"  non-zero pixels: {np.sum(test_mask)}")
    print(f"  center (row, col): ({test_center_y:.1f}, {test_center_x:.1f})")
    print(f"  center (x, y): ({test_center_x:.1f}, {test_center_y:.1f})")

# 4. Compare
print("\n4. COMPARISON")
print("-"*40)
print(f"CSV center:      (x={dl13_row['center_x_px']:.1f}, y={dl13_row['center_y_px']:.1f})")
print(f"Test mask:       (x={test_center_x:.1f}, y={test_center_y:.1f})")
print(f"HDF5 mask:       (x={center_x:.1f}, y={center_y:.1f})")
print()
print(f"Δ (Test vs CSV):  ΔX={test_center_x - dl13_row['center_x_px']:.1f}px, ΔY={test_center_y - dl13_row['center_y_px']:.1f}px")
print(f"Δ (HDF5 vs CSV):  ΔX={center_x - dl13_row['center_x_px']:.1f}px, ΔY={center_y - dl13_row['center_y_px']:.1f}px")
print(f"Δ (HDF5 vs Test): ΔX={center_x - test_center_x:.1f}px, ΔY={center_y - test_center_y:.1f}px")

print("\n" + "="*80)
if np.allclose(test_center_x, center_x) and np.allclose(test_center_y, center_y):
    print("✓ Test mask matches HDF5 - code logic is working as written")
    print("  BUT coordinates don't match CSV - there must be a data format issue!")
elif np.allclose(test_center_x, dl13_row['center_x_px']) and np.allclose(test_center_y, dl13_row['center_y_px']):
    print("✓ Test mask matches CSV - pixel_list data is correct")
    print("  HDF5 mask is wrong - something else modified it after generation!")
else:
    print("? Neither matches - complex transformation happening")
print("="*80)
