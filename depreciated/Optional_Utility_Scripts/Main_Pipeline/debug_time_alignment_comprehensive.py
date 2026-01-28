"""
Comprehensive time alignment debugging for CNN dataset.

This script tests different time offsets to find the optimal alignment
between FLIR surface temperatures and sand-embedded thermistor temperatures.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

# Load dataset
print("Loading dataset...")
with h5py.File('ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5', 'r') as f:
    flir = f['flir_frames'][:]
    sand = f['sand_temps'][:]
    masks = f['roi_masks'][:]
    times = f['timestamps'][:]
    comps = [c.decode() for c in f['metadata']['component_names'][:]]
    current_offset = float(f['metadata'].attrs['time_offset'])

print(f"Current offset: {current_offset}s")
print(f"FLIR frames: {len(times)}")
print(f"Components: {len(comps)}")

# Extract FLIR ROI temps for all components
print("\nExtracting FLIR ROI temperatures...")
flir_rois = np.zeros((len(comps), len(times)))
for i, comp in enumerate(comps):
    mask = masks[i]
    for j, frame in enumerate(flir):
        if np.sum(mask) > 0:
            flir_rois[i, j] = np.mean(frame[mask > 0])
        else:
            flir_rois[i, j] = np.nan

# Test different offsets
print("\nTesting different time offsets...")
offsets_to_test = np.arange(-120, 121, 5)  # -120s to +120s in 5s steps
avg_correlations = []

for offset in offsets_to_test:
    # Adjust FLIR times
    flir_times_adj = times + offset
    
    # For each component, interpolate and calculate correlation
    corrs = []
    for i in range(len(comps)):
        if np.all(np.isnan(flir_rois[i])):
            continue
            
        # Interpolate FLIR to original sand times
        flir_interp = interp1d(flir_times_adj, flir_rois[i], 
                               bounds_error=False, fill_value='extrapolate')
        
        # We need to align to the original thermistor times (before the 20s offset)
        # The thermistor was recorded starting at t=0
        # If offset > 0, FLIR leads, so FLIR t=0 corresponds to thermistor t=offset
        
        # Get overlapping region
        overlap_start = max(flir_times_adj[0], times[0])
        overlap_end = min(flir_times_adj[-1], times[-1])
        
        if overlap_end <= overlap_start:
            continue
            
        # Sample at FLIR timestamps within overlap
        sample_times = times[(times >= overlap_start) & (times <= overlap_end)]
        
        if len(sample_times) < 10:
            continue
            
        flir_sampled = flir_interp(sample_times)
        
        # Get corresponding sand temps
        sand_idx = np.where((times >= overlap_start) & (times <= overlap_end))[0]
        sand_sampled = sand[sand_idx, i]
        
        # Calculate correlation
        valid = ~(np.isnan(flir_sampled) | np.isnan(sand_sampled))
        if np.sum(valid) > 10:
            corr, _ = pearsonr(flir_sampled[valid], sand_sampled[valid])
            corrs.append(corr)
    
    if corrs:
        avg_correlations.append(np.mean(corrs))
    else:
        avg_correlations.append(np.nan)

# Find optimal offset
valid_mask = ~np.isnan(avg_correlations)
valid_offsets = offsets_to_test[valid_mask]
valid_corrs = np.array(avg_correlations)[valid_mask]
optimal_idx = np.argmax(valid_corrs)
optimal_offset = valid_offsets[optimal_idx]
optimal_corr = valid_corrs[optimal_idx]

print(f"\n{'='*80}")
print(f"OPTIMAL TIME OFFSET SEARCH RESULTS")
print(f"{'='*80}")
current_idx = np.where(offsets_to_test == current_offset)[0]
if len(current_idx) > 0:
    current_corr = avg_correlations[current_idx[0]]
    print(f"Current offset: {current_offset:+.1f}s (avg corr = {current_corr:.3f})")
    print(f"Optimal offset: {optimal_offset:+.1f}s (avg corr = {optimal_corr:.3f})")
    print(f"Improvement: {optimal_corr - current_corr:+.3f}")
else:
    print(f"Current offset: {current_offset:+.1f}s (not in test range)")
    print(f"Optimal offset: {optimal_offset:+.1f}s (avg corr = {optimal_corr:.3f})")

# Plot offset sweep
plt.figure(figsize=(12, 6))
plt.plot(offsets_to_test, avg_correlations, 'b-', linewidth=2)
plt.axvline(current_offset, color='red', linestyle='--', linewidth=2, label=f'Current ({current_offset:+.1f}s)')
plt.axvline(optimal_offset, color='green', linestyle='--', linewidth=2, label=f'Optimal ({optimal_offset:+.1f}s)')
plt.xlabel('Time Offset (s)', fontsize=12)
plt.ylabel('Average Correlation', fontsize=12)
plt.title('FLIR-Thermistor Correlation vs Time Offset', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('outputs/0115_1806_P1-7/offset_sweep.png', dpi=150)
print(f"\n✅ Saved offset sweep plot: outputs/0115_1806_P1-7/offset_sweep.png")

# Show top 5 offsets
print(f"\nTop 5 offsets:")
top_indices = np.argsort(valid_corrs)[-5:][::-1]
for idx in top_indices:
    print(f"  {valid_offsets[idx]:+6.1f}s: {valid_corrs[idx]:.3f}")

print(f"\n{'='*80}")
print(f"RECOMMENDATION")
print(f"{'='*80}")
current_idx = np.where(offsets_to_test == current_offset)[0]
if len(current_idx) > 0 and optimal_corr > avg_correlations[current_idx[0]] + 0.05:
    print(f"⚠️  Try using offset = {optimal_offset:+.1f}s instead of {current_offset:+.1f}s")
    print(f"   Expected correlation improvement: {optimal_corr - avg_correlations[current_idx[0]]:+.3f}")
else:
    print(f"✅ Current offset ({current_offset:+.1f}s) is near-optimal")
    
if optimal_corr < 0.5:
    print(f"\n⚠️  WARNING: Even optimal correlation ({optimal_corr:.3f}) is low!")
    print("   Possible causes:")
    print("   1. FLIR surface temps don't correlate well with sand-embedded temps")
    print("   2. Thermal lag between surface and embedded sensors")
    print("   3. ROI pixel coordinates might be misaligned")
    print("   4. Thermistor channels might be mislabeled or swapped")
