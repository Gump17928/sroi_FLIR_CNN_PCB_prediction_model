"""
DIAGNOSTIC 2: Compare Raw vs Filtered FLIR Frames
Verify filtering actually happened
"""
import pandas as pd
import numpy as np
from pathlib import Path

print('='*80)
print('DIAGNOSTIC 2: Raw vs Filtered Frame Comparison')
print('='*80)

# Pick a middle frame
frame_name = 'Rec-000010_150.csv'

raw_path = Path('inputs/ResearchIR_Outputs_HBridge_15s') / frame_name
filtered_path = Path('inputs/ResearchIR_Outputs_HBridge_15s_filtered') / frame_name

if not raw_path.exists():
    print(f"⚠️  Raw frame not found: {raw_path}")
elif not filtered_path.exists():
    print(f"⚠️  Filtered frame not found: {filtered_path}")
else:
    # Load frames
    raw_frame = pd.read_csv(raw_path, header=None).values
    filtered_frame = pd.read_csv(filtered_path, header=None).values
    
    print(f"\nFrame: {frame_name}")
    print(f"Raw shape: {raw_frame.shape}")
    print(f"Filtered shape: {filtered_frame.shape}")
    
    # Compare stats
    print(f"\nRaw frame stats:")
    print(f"  Mean: {raw_frame.mean():.2f}°C")
    print(f"  Std: {raw_frame.std():.3f}°C")
    print(f"  Min: {raw_frame.min():.2f}°C")
    print(f"  Max: {raw_frame.max():.2f}°C")
    
    print(f"\nFiltered frame stats:")
    print(f"  Mean: {filtered_frame.mean():.2f}°C")
    print(f"  Std: {filtered_frame.std():.3f}°C")
    print(f"  Min: {filtered_frame.min():.2f}°C")
    print(f"  Max: {filtered_frame.max():.2f}°C")
    
    # Check differences
    diff = np.abs(raw_frame - filtered_frame)
    print(f"\nDifferences (raw - filtered):")
    print(f"  Max difference: {diff.max():.2f}°C")
    print(f"  Mean difference: {diff.mean():.3f}°C")
    print(f"  Pixels changed >0.1°C: {(diff > 0.1).sum()}/{diff.size} ({(diff > 0.1).sum()/diff.size*100:.1f}%)")
    
    if diff.max() < 0.01:
        print("\n❌ FRAMES ARE IDENTICAL - Filtering did NOT work!")
    elif diff.mean() < 0.1:
        print("\n⚠️  Very small differences - Filtering may not be effective")
    else:
        print("\n✅ Filtering applied successfully")
