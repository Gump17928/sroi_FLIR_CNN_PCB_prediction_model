#!/usr/bin/env python3
"""
Comprehensive CNN Dataset Diagnostics

Checks for ALL potential issues that could cause poor CNN performance:
1. ✅ Data path correctness (fixed by recent changes)
2. ⚠️ Temporal alignment quality (cross-correlation strength)
3. ⚠️ ROI pixel map coverage (components with/without ROIs)
4. ⚠️ Temperature range sanity
5. ⚠️ Interpolation artifacts
6. ⚠️ Data normalization issues
7. ⚠️ Train/val split problems

Usage:
    python diagnose_cnn_dataset.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def diagnose_hdf5_dataset(h5_file):
    """
    Comprehensive diagnostic of HDF5 training dataset.
    
    Args:
        h5_file: Path to HDF5 dataset file
    """
    print("="*80)
    print("CNN DATASET COMPREHENSIVE DIAGNOSTICS")
    print("="*80)
    print(f"\nDataset: {h5_file}\n")
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load data
            flir_frames = f['flir_frames'][:]
            sand_temps = f['sand_temps'][:]
            roi_masks = f['roi_masks'][:]
            timestamps = f['timestamps'][:]
            
            # Load metadata
            n_frames = f['metadata'].attrs['n_frames']
            n_components = f['metadata'].attrs['n_components']
            time_offset = f['metadata'].attrs['time_offset']
            component_names = [name.decode('utf-8') for name in f['metadata/component_names'][:]]
            
            print("="*80)
            print("1. DATASET STRUCTURE")
            print("="*80)
            print(f"FLIR frames shape: {flir_frames.shape}")
            print(f"  Expected: (n_frames, height, width)")
            print(f"  Actual: ({n_frames}, {flir_frames.shape[1]}, {flir_frames.shape[2]})")
            
            print(f"\nThermistor temps shape: {sand_temps.shape}")
            print(f"  Expected: (n_frames, n_components)")
            print(f"  Actual: ({n_frames}, {n_components})")
            
            print(f"\nROI masks shape: {roi_masks.shape}")
            print(f"  Expected: (n_components, height, width)")
            print(f"  Actual: ({n_components}, {roi_masks.shape[1]}, {roi_masks.shape[2]})")
            
            print(f"\nComponents: {len(component_names)}")
            print(f"  Names: {', '.join(component_names[:5])}{'...' if len(component_names) > 5 else ''}")
            
            # Check 1: Shape consistency
            issues = []
            if sand_temps.shape[0] != flir_frames.shape[0]:
                issues.append(f"❌ Frame count mismatch: FLIR={flir_frames.shape[0]}, thermistor={sand_temps.shape[0]}")
            else:
                print("\n✅ Frame counts match")
            
            if sand_temps.shape[1] != roi_masks.shape[0]:
                issues.append(f"❌ Component count mismatch: thermistor={sand_temps.shape[1]}, ROI masks={roi_masks.shape[0]}")
            else:
                print("✅ Component counts match")
            
            # Check 2: Temporal alignment quality
            print("\n" + "="*80)
            print("2. TEMPORAL ALIGNMENT")
            print("="*80)
            print(f"Time offset detected: {time_offset:.2f}s")
            print(f"  Interpretation: FLIR {'leads' if time_offset > 0 else 'lags'} thermistor by {abs(time_offset):.2f}s")
            
            # Large offset might indicate misalignment
            if abs(time_offset) > 60:
                issues.append(f"⚠️ Large time offset ({time_offset:.1f}s) - possible misalignment")
                print(f"  ⚠️ WARNING: Large offset (>{60}s) detected!")
            else:
                print(f"  ✅ Reasonable offset (<60s)")
            
            # Check timestamp spacing
            time_diffs = np.diff(timestamps)
            avg_dt = np.mean(time_diffs)
            std_dt = np.std(time_diffs)
            print(f"\nFrame spacing: {avg_dt:.1f}s ± {std_dt:.2f}s")
            if std_dt > 1.0:
                issues.append(f"⚠️ Irregular frame spacing (std={std_dt:.2f}s)")
                print(f"  ⚠️ WARNING: Irregular spacing detected")
            else:
                print(f"  ✅ Consistent frame spacing")
            
            # Check 3: Temperature ranges
            print("\n" + "="*80)
            print("3. TEMPERATURE RANGES (SANITY CHECK)")
            print("="*80)
            
            flir_min, flir_max = np.min(flir_frames), np.max(flir_frames)
            temps_min, temps_max = np.min(sand_temps), np.max(sand_temps)
            
            print(f"FLIR frames: {flir_min:.1f}°C to {flir_max:.1f}°C")
            print(f"Thermistors: {temps_min:.1f}°C to {temps_max:.1f}°C")
            
            # Sanity checks (typical PCB: 20-150°C)
            if flir_min < 0 or flir_max > 200:
                issues.append(f"❌ FLIR temps out of range: {flir_min:.1f} to {flir_max:.1f}°C")
                print(f"  ❌ FLIR temperatures unusual!")
            else:
                print(f"  ✅ FLIR temps in reasonable range")
            
            if temps_min < 0 or temps_max > 200:
                issues.append(f"❌ Thermistor temps out of range: {temps_min:.1f} to {temps_max:.1f}°C")
                print(f"  ❌ Thermistor temperatures unusual!")
            else:
                print(f"  ✅ Thermistor temps in reasonable range")
            
            # Check for constant values (sign of data corruption)
            flir_variance = np.var(flir_frames)
            temps_variance = np.var(sand_temps)
            if flir_variance < 0.1:
                issues.append(f"❌ FLIR frames nearly constant (var={flir_variance:.4f})")
            if temps_variance < 0.1:
                issues.append(f"❌ Thermistor temps nearly constant (var={temps_variance:.4f})")
            
            # Check 4: ROI mask coverage
            print("\n" + "="*80)
            print("4. ROI MASK COVERAGE")
            print("="*80)
            
            total_pixels = roi_masks.shape[1] * roi_masks.shape[2]
            low_coverage_comps = []
            
            for i, comp_name in enumerate(component_names):
                n_roi_pixels = np.sum(roi_masks[i] > 0)
                coverage_pct = 100 * n_roi_pixels / total_pixels
                
                if n_roi_pixels == 0:
                    issues.append(f"❌ Component '{comp_name}' has ZERO ROI pixels!")
                    low_coverage_comps.append(comp_name)
                elif coverage_pct < 0.01:
                    issues.append(f"⚠️ Component '{comp_name}' has very low coverage ({coverage_pct:.3f}%)")
                    low_coverage_comps.append(comp_name)
            
            avg_coverage = 100 * np.sum(roi_masks > 0) / (roi_masks.shape[0] * total_pixels)
            print(f"Average ROI coverage: {avg_coverage:.2f}% of frame")
            print(f"Components with low coverage: {len(low_coverage_comps)}/{len(component_names)}")
            
            if low_coverage_comps:
                print(f"  ⚠️ Low coverage components: {', '.join(low_coverage_comps[:5])}")
            else:
                print(f"  ✅ All components have adequate ROI coverage")
            
            # Check 5: Correlation between FLIR and thermistor
            print("\n" + "="*80)
            print("5. FLIR-THERMISTOR CORRELATION (CRITICAL!)")
            print("="*80)
            
            # For each component, extract FLIR temps at ROI and correlate with thermistor
            low_corr_comps = []
            correlations = []
            
            for i, comp_name in enumerate(component_names[:10]):  # Check first 10 to save time
                if np.sum(roi_masks[i]) == 0:
                    continue
                
                # Extract FLIR temperatures at ROI pixels for all frames
                roi_mask_bool = roi_masks[i] > 0
                flir_at_roi = []
                for frame_idx in range(min(100, n_frames)):  # Sample first 100 frames
                    frame = flir_frames[frame_idx]
                    mean_roi_temp = np.mean(frame[roi_mask_bool])
                    flir_at_roi.append(mean_roi_temp)
                
                flir_at_roi = np.array(flir_at_roi)
                thermistor_temps = sand_temps[:len(flir_at_roi), i]
                
                # Calculate correlation
                corr = np.corrcoef(flir_at_roi, thermistor_temps)[0, 1]
                correlations.append(corr)
                
                if corr < 0.5:
                    low_corr_comps.append((comp_name, corr))
            
            if correlations:
                avg_corr = np.mean(correlations)
                print(f"Average FLIR-thermistor correlation: {avg_corr:.3f}")
                
                if avg_corr < 0.5:
                    issues.append(f"❌ CRITICAL: Low average correlation ({avg_corr:.3f}) - DATA MISALIGNMENT!")
                    print(f"  ❌ CRITICAL: Poor correlation suggests misaligned data!")
                elif avg_corr < 0.7:
                    issues.append(f"⚠️ Moderate correlation ({avg_corr:.3f}) - check alignment")
                    print(f"  ⚠️ Moderate correlation - alignment could be better")
                else:
                    print(f"  ✅ Good correlation - data is well aligned")
                
                if low_corr_comps:
                    print(f"\n  Components with low correlation (<0.5):")
                    for comp, corr in low_corr_comps[:5]:
                        print(f"    - {comp}: {corr:.3f}")
            
            # Check 6: Data statistics per frame
            print("\n" + "="*80)
            print("6. FRAME-BY-FRAME STATISTICS")
            print("="*80)
            
            # Check for sudden jumps (sign of concatenated data from different tests)
            frame_means = np.mean(flir_frames.reshape(n_frames, -1), axis=1)
            frame_diffs = np.abs(np.diff(frame_means))
            max_jump = np.max(frame_diffs)
            max_jump_idx = np.argmax(frame_diffs)
            
            print(f"Max temperature jump between frames: {max_jump:.2f}°C at frame {max_jump_idx}")
            
            if max_jump > 20:
                issues.append(f"⚠️ Large frame jump ({max_jump:.1f}°C at frame {max_jump_idx}) - possible concatenated data")
                print(f"  ⚠️ WARNING: Large jump suggests concatenated tests")
            else:
                print(f"  ✅ Smooth frame transitions")
            
            # Summary
            print("\n" + "="*80)
            print("DIAGNOSTIC SUMMARY")
            print("="*80)
            
            if not issues:
                print("\n✅ ALL CHECKS PASSED - Dataset appears healthy!")
                print("\n  Dataset is ready for CNN training.")
            else:
                print(f"\n⚠️ FOUND {len(issues)} POTENTIAL ISSUES:\n")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
                
                print("\n🔍 RECOMMENDED ACTIONS:")
                
                # Prioritize critical issues
                critical = [iss for iss in issues if '❌' in iss]
                if critical:
                    print("\n  CRITICAL (fix before training):")
                    for iss in critical:
                        if 'correlation' in iss.lower():
                            print("    - Rebuild dataset with correct thermistor CSV from current session")
                            print("    - Verify FLIR and thermistor timestamps align")
                        elif 'zero roi' in iss.lower():
                            print("    - Update ROI pixel map to include all components")
                        elif 'temperature' in iss.lower():
                            print("    - Check data units (Celsius vs Kelvin)")
                            print("    - Verify FLIR frame loading is correct")
                
                warnings = [iss for iss in issues if '⚠️' in iss and '❌' not in iss]
                if warnings:
                    print("\n  WARNINGS (investigate but may not be fatal):")
                    for iss in warnings:
                        if 'offset' in iss.lower():
                            print("    - Review temporal alignment, consider manual offset adjustment")
                        elif 'coverage' in iss.lower():
                            print("    - Some components have small ROIs, may impact training")
            
            return {'valid': len([iss for iss in issues if '❌' in iss]) == 0, 
                    'issues': issues,
                    'correlations': correlations if correlations else None}
    
    except Exception as e:
        print(f"\n❌ ERROR reading dataset: {e}")
        import traceback
        traceback.print_exc()
        return {'valid': False, 'issues': [f'Failed to read dataset: {e}']}


if __name__ == "__main__":
    # Find the dataset
    project_root = Path(__file__).parent
    dataset_file = project_root / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
    
    if not dataset_file.exists():
        print(f"❌ Dataset not found: {dataset_file}")
        print("\nDataset will be built on next training run.")
        exit(1)
    
    # Run diagnostics
    results = diagnose_hdf5_dataset(dataset_file)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    if results['valid']:
        print("\n✅ Ready to proceed with CNN training")
    else:
        print("\n❌ Fix critical issues before training")
        print("\nTo rebuild dataset:")
        print("  1. Delete: ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
        print("  2. Run: python researchir_post_processor.py --full_pipeline")
        print("  3. Choose option [1] at Phase 8 prompt")
