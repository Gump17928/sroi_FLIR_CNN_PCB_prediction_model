#!/usr/bin/env python3
"""
Verify FLIR Frame Filtering Quality

Wrapper script that uses viz_phase2_filtering.py functions to verify
that FLIR frame filtering worked correctly.

Usage:
    python verify_flir_filtering.py
"""

from pathlib import Path
import viz_phase2_filtering as viz_phase2


if __name__ == "__main__":
    # Detect available FLIR folders
    project_root = Path(__file__).parent
    
    # Check for HBridge_15s (most common)
    raw_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s"
    filtered_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s_filtered"
    
    if not raw_folder.exists():
        # Try without _15s suffix
        raw_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge"
        filtered_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_filtered"
    
    if not raw_folder.exists() or not filtered_folder.exists():
        print("❌ ERROR: Could not find FLIR folders!")
        print(f"   Looked for: {raw_folder}")
        print(f"           and: {filtered_folder}")
        exit(1)
    
    # Run comprehensive verification
    results = viz_phase2.verify_flir_filtering_quality(
        raw_folder=raw_folder,
        filtered_folder=filtered_folder,
        num_samples=20,
        output_dir='outputs',
        verbose=True
    )
    
    # Print assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    
    if results['valid']:
        issues = []
        
        if abs(results['range_reduction_pct']) > 20:
            issues.append(f"Range reduction: {results['range_reduction_pct']:.1f}% (>20%)")
        
        if results['noise_reduction_pct'] < 5:
            issues.append(f"Low noise reduction: {results['noise_reduction_pct']:.1f}% (<5%)")
        elif results['noise_reduction_pct'] > 40:
            issues.append(f"High noise reduction: {results['noise_reduction_pct']:.1f}% (>40%, possible over-smoothing)")
        
        if results['mean_diff_degC'] > 5:
            issues.append(f"Large mean difference: {results['mean_diff_degC']:.2f}°C (>5°C)")
        
        if not issues:
            print("\n✓ FILTERING QUALITY: GOOD")
            print("  Frames are suitable for CNN training")
        else:
            print("\n⚠️  FILTERING QUALITY: ISSUES DETECTED")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print(f"\n❌ VALIDATION FAILED: {results.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    if results.get('plot_file'):
        print(f"\n✓ Check visualization: {results['plot_file']}")
    print("\nNext steps:")
    print("1. Review the metrics and plot above")
    print("2. If quality is good, proceed with CNN training")
    print("3. If issues detected, consider re-filtering with adjusted parameters")

