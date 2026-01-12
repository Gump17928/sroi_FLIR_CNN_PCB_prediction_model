"""
Quick Results Generator - Skip Training if Model Exists
========================================================

This script:
1. Checks if trained model exists
2. If YES: Skip training, generate predictions and visualizations
3. If NO: Run complete pipeline (train + predict + visualize)

Use this when you want results fast without retraining.

Author: CNN Pipeline
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))


def find_latest_model(results_dir):
    """Check if trained model exists."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return None
    
    # Find all .keras model files
    model_files = list(results_path.glob("unet_*.keras"))
    
    if not model_files:
        return None
    
    # Get most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def main():
    """Execute pipeline with smart model detection."""
    
    results_dir = current_dir / "results"
    
    print("\n" + "="*80)
    print("PHASE 8C SMART PIPELINE")
    print("="*80)
    
    # Check for existing model
    existing_model = find_latest_model(results_dir)
    
    if existing_model:
        print(f"\n✓ Found existing trained model: {existing_model.name}")
        print(f"  Last modified: {existing_model.stat().st_mtime}")
        print(f"\n⚡ SKIPPING TRAINING - Using existing model for fast results")
        print("\nThis pipeline will:")
        print("  1. ⏭️  Skip training (model exists)")
        print("  2. ✅ Generate predictions and visualizations")
        print("  3. ✅ Compare Phase 8 vs Phase 8c")
        print("  4. ✅ Export final results")
        print("\nEstimated time: ~5 minutes")
        
        skip_training = True
    else:
        print("\n⚠️  No trained model found")
        print("\nThis pipeline will:")
        print("  1. ✅ Train U-Net model (~30-60 min)")
        print("  2. ✅ Generate predictions and visualizations")
        print("  3. ✅ Compare Phase 8 vs Phase 8c")
        print("  4. ✅ Export final results")
        print("\nEstimated time: ~30-60 minutes for training + 5 minutes for analysis")
        
        skip_training = False
    
    print("="*80 + "\n")
    
    # Import modules
    import train_hbridge_model
    import generate_predictions
    import compare_phase8_vs_phase8c
    
    try:
        if not skip_training:
            # Step 1: Train model
            print("\n" + "▶"*40)
            print("STEP 1: TRAINING U-NET MODEL")
            print("▶"*40 + "\n")
            train_hbridge_model.main()
        else:
            print("\n" + "▶"*40)
            print("STEP 1: TRAINING SKIPPED (Model exists)")
            print("▶"*40 + "\n")
        
        # Step 2: Generate predictions
        print("\n" + "▶"*40)
        print("STEP 2: GENERATING PREDICTIONS")
        print("▶"*40 + "\n")
        generate_predictions.main()
        
        # Step 3: Compare with Phase 8
        print("\n" + "▶"*40)
        print("STEP 3: COMPARING PHASE 8 vs PHASE 8C")
        print("▶"*40 + "\n")
        compare_phase8_vs_phase8c.main()
        
        # Final summary
        print("\n" + "="*80)
        print("✓ PIPELINE FINISHED SUCCESSFULLY")
        print("="*80)
        print("\nAll results saved to:")
        print(f"  {results_dir.relative_to(parent_dir)}/")
        print("\nGenerated files:")
        if not skip_training:
            print("  - unet_hbridge_*.keras (trained model)")
            print("  - training_curves_*.png (loss/MAE plots)")
        print("  - thermal_comparison_*.png (field visualizations)")
        print("  - scatter_actual_vs_predicted_*.png")
        print("  - phase8_vs_phase8c_comparison_*.png")
        print("  - approach_comparison_*.png")
        print("  - predictions_*.csv")
        print("  - metrics_*.txt")
        print("  - comparison_report_*.txt")
        print("\nReview the results to assess Phase 8c performance!")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ PIPELINE FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
