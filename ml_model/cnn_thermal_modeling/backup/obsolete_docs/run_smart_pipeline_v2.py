"""
Updated Smart Pipeline with Timestamped Analysis Folders and 3D Visualization

Changes:
1. Creates analysis_TIMESTAMP folder for each run
2. All outputs go into that single folder
3. Optional 3D thermal visualizations
4. Cleaner result organization

Author: CNN Pipeline
Date: 2026-01-14
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))


def find_latest_analysis(results_dir):
    """Find the most recent analysis folder."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return None
    
    # Find all analysis_* directories
    analysis_dirs = list(results_path.glob("analysis_*"))
    
    if not analysis_dirs:
        return None
    
    # Get most recent
    latest = max(analysis_dirs, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    """Execute pipeline with smart model detection and organized output."""
    
    results_dir = current_dir / "results"
    
    print("\n" + "="*80)
    print("PHASE 8C SMART PIPELINE - Enhanced Edition")
    print("="*80)
    
    # Check for existing analysis
    existing_analysis = find_latest_analysis(results_dir)
    
    if existing_analysis:
        # Check if it has a trained model
        model_file = existing_analysis / "unet_hbridge.keras"
        if model_file.exists():
            print(f"\n✓ Found existing analysis: {existing_analysis.name}")
            print(f"  Created: {datetime.fromtimestamp(existing_analysis.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Ask user if they want to retrain
            print("\nOptions:")
            print("  [1] Use existing model (fast - ~5 minutes)")
            print("  [2] Retrain model (slow - ~30-60 minutes)")
            print("  [3] Generate 3D visualizations from existing analysis")
            
            while True:
                choice = input("\nYour choice (1, 2, or 3): ").strip()
                if choice == "1":
                    skip_training = True
                    create_3d = False
                    output_dir = existing_analysis
                    print("\n⚡ Using existing model for fast results")
                    break
                elif choice == "2":
                    skip_training = False
                    create_3d = False
                    # Create new timestamped folder
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_dir = results_dir / f"analysis_{timestamp}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n🔄 Retraining model - results will be saved to: {output_dir.name}")
                    break
                elif choice == "3":
                    print("\n🎨 Generating 3D visualizations from existing analysis...")
                    generate_3d_only(existing_analysis)
                    return
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
        else:
            # Analysis folder exists but no model - create new one
            skip_training = False
            create_3d = False
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = results_dir / f"analysis_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # No existing analysis - must train
        skip_training = False
        create_3d = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = results_dir / f"analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🆕 No existing analysis found")
        print(f"   Creating new analysis folder: {output_dir.name}")
    
    # Print pipeline summary
    if not skip_training:
        print("\nThis pipeline will:")
        print(f"  1. ✅ Train U-Net model (~30-60 min)")
        print(f"  2. ✅ Generate predictions and visualizations")
        print(f"  3. ✅ Compare Phase 8 vs Phase 8c")
        print(f"  4. ✅ Export final results to {output_dir.name}/")
        print("\nEstimated time: ~30-60 minutes for training + 5 minutes for analysis")
    else:
        print("\nThis pipeline will:")
        print(f"  1. ⏭️  Skip training (using existing model)")
        print(f"  2. ✅ Generate predictions and visualizations")
        print(f"  3. ✅ Compare Phase 8 vs Phase 8c")
        print(f"  4. ✅ Export final results to {output_dir.name}/")
        print("\nEstimated time: ~5 minutes")
    
    print("="*80)
    
    # Execute pipeline
    if not skip_training:
        print("\n\n" + "▶"*40)
        print("STEP 1: TRAINING U-NET MODEL")
        print("▶"*40 + "\n")
        
        import train_hbridge_model
        # Temporarily modify train_hbridge_model to use our output_dir
        original_main = train_hbridge_model.main
        
        def custom_train():
            dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
            if not dataset_file.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_file}")
            
            train_hbridge_model.load_dataset_info(dataset_file)
            trainer, history = train_hbridge_model.train_model(
                dataset_path=dataset_file,
                output_dir=output_dir,
                epochs=5,
                batch_size=8
            )
            train_hbridge_model.evaluate_model(trainer, dataset_file, output_dir)
        
        custom_train()
    
    # Step 2: Generate predictions
    print("\n\n" + "▶"*40)
    print("STEP 2: GENERATING PREDICTIONS")
    print("▶"*40 + "\n")
    
    import generate_predictions_updated as gen_pred
    gen_pred.run_predictions(output_dir)
    
    # Step 3: Compare with Phase 8
    print("\n\n" + "▶"*40)
    print("STEP 3: COMPARING PHASE 8 vs PHASE 8C")
    print("▶"*40 + "\n")
    
    import compare_phase8_vs_phase8c_updated as comp
    comp.run_comparison(output_dir)
    
    # Ask about 3D visualizations
    print("\n" + "="*80)
    print("Would you like to generate 3D visualizations? (slower but looks cool!)")
    choice_3d = input("Generate 3D plots? (y/n): ").strip().lower()
    
    if choice_3d == 'y':
        generate_3d_visualizations(output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("✓ PIPELINE FINISHED SUCCESSFULLY")
    print("="*80)
    
    print(f"\nAll results saved to:")
    print(f"  {output_dir}/")
    
    print(f"\nGenerated files in {output_dir.name}/:")
    print("  - unet_hbridge.keras (trained model)")
    print("  - training_curves.png (loss/MAE plots)")
    print("  - thermal_comparison_*.png (field visualizations)")
    print("  - scatter_actual_vs_predicted.png")
    print("  - phase8_vs_phase8c_comparison.png")
    print("  - approach_comparison.png")
    print("  - predictions.csv")
    print("  - metrics.txt")
    print("  - comparison_report.txt")
    if choice_3d == 'y':
        print("  - 3d_*.png (3D visualizations)")
    
    print(f"\nReview the results in: {output_dir.relative_to(parent_dir)}/")
    print("="*80)


def generate_3d_only(analysis_dir):
    """Generate only 3D visualizations from existing analysis."""
    print(f"\nGenerating 3D visualizations in: {analysis_dir.name}")
    generate_3d_visualizations(analysis_dir)
    print(f"\n✓ 3D visualizations complete!")
    print(f"  Check {analysis_dir}/ for 3d_*.png files")


def generate_3d_visualizations(output_dir):
    """Generate 3D thermal visualizations."""
    print("\n" + "▶"*40)
    print("GENERATING 3D VISUALIZATIONS")
    print("▶"*40 + "\n")
    
    try:
        from viz_3d_thermal import Thermal3DVisualizer, demo_3d_visualizations
        import h5py
        import tensorflow as tf
        import numpy as np
        
        dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
        model_file = output_dir / "unet_hbridge.keras"
        
        if not model_file.exists():
            print("⚠️  Model file not found - skipping 3D visualizations")
            return
        
        # Load model and generate predictions
        print("Loading model and generating predictions...")
        model = tf.keras.models.load_model(model_file, compile=False)
        
        with h5py.File(dataset_file, 'r') as f:
            flir_frames = f['flir_frames'][:]
        
        # Predict on middle frame
        frame_idx = len(flir_frames) // 2
        frame_input = flir_frames[frame_idx:frame_idx+1, ..., np.newaxis]
        prediction = model.predict(frame_input, verbose=0)[0, :, :, 0]
        
        # Generate 3D visualizations
        demo_3d_visualizations(dataset_file, prediction, output_dir)
        
        print("\n✓ 3D visualizations created successfully!")
        
    except ImportError as e:
        print(f"\n⚠️  Could not generate 3D visualizations: {e}")
        print("   Make sure matplotlib and numpy are installed")
    except Exception as e:
        print(f"\n⚠️  Error generating 3D visualizations: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
