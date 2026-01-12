"""
Complete Phase 8c CNN Pipeline - Full Workflow
===============================================

Executes the complete CNN workflow:
1. Train U-Net model
2. Generate predictions and visualizations
3. Compare Phase 8 vs Phase 8c
4. Export all results

Run this script to execute the entire pipeline end-to-end.

Author: CNN Pipeline
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


def main():
    """Execute complete CNN pipeline."""
    
    print("\n" + "="*80)
    print("PHASE 8C CNN COMPLETE PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Train U-Net model (Item #12)")
    print("  2. Generate predictions and visualizations (Item #13)")
    print("  3. Compare Phase 8 vs Phase 8c (Item #14)")
    print("  4. Export final results (Item #15)")
    print("\nEstimated time: 30-60 minutes for training + 5 minutes for analysis")
    print("="*80 + "\n")
    
    # Import after path is set
    import train_hbridge_model
    import generate_predictions
    import compare_phase8_vs_phase8c
    
    try:
        # Step 1: Train model
        print("\n" + "▶"*40)
        print("STEP 1: TRAINING U-NET MODEL")
        print("▶"*40 + "\n")
        train_hbridge_model.main()
        
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
        print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        print("="*80)
        print("\nAll results saved to:")
        print("  ml_model/cnn_thermal_modeling/results/")
        print("\nGenerated files:")
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
