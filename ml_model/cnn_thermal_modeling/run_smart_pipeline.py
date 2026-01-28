"""
Quick Results Generator - Skip Training if Model Exists
========================================================

This script:
1. Checks if trained model exists
2. If YES: Skip training, generate predictions and visualizations
3. If NO: Run complete pipeline (train + predict + visualize)

Features:
- Timestamped analysis folders for organized results
- Optional 3D thermal visualizations
- Smart model detection and reuse

Use this when you want results fast without retraining.

Author: CNN Pipeline
Date: 2025-01-12, Updated: 2026-01-14
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
    """Find the most recent analysis folder with a trained model."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return None
    
    # First, check for new-style analysis_* directories
    analysis_dirs = list(results_path.glob("analysis_*"))
    
    # Get most recent that has a model
    for analysis_dir in sorted(analysis_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        model_file = analysis_dir / "unet_hbridge.keras"
        if model_file.exists():
            return analysis_dir
    
    # Backwards compatibility: Check for old-style models in results root
    old_models = list(results_path.glob("unet_*.keras"))
    if old_models:
        # Return the results_dir itself to indicate old-style layout
        return results_path
    
    return None


def main():
    """Execute pipeline with smart model detection."""
    
    results_dir = current_dir / "results"
    
    print("\n" + "="*80)
    print("PHASE 8C SMART PIPELINE")
    print("="*80)
    
    # Check for existing analysis
    existing_analysis = find_latest_analysis(results_dir)
    
    if existing_analysis:
        # Check if it's old-style (results_dir itself) or new-style (analysis_* subfolder)
        is_old_style = existing_analysis == results_dir
        
        if is_old_style:
            print(f"\n✓ Found existing trained models (old format)")
            print(f"  Location: {results_dir.name}/")
            print(f"\n⚠️  Note: Old format detected. New results will use organized analysis_TIMESTAMP folders.")
        else:
            print(f"\n✓ Found existing analysis: {existing_analysis.name}")
            print(f"  Created: {datetime.fromtimestamp(existing_analysis.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ask user if they want to retrain
        print("\nOptions:")
        print("  [1] Use existing model (fast - ~5 minutes)")
        print("  [2] Retrain model (slow - ~30-60 minutes)")
        if not is_old_style:
            print("  [3] Generate 3D visualizations from existing analysis")
        
        while True:
            choice_prompt = "\nYour choice (1, 2, or 3): " if not is_old_style else "\nYour choice (1 or 2): "
            choice = input(choice_prompt).strip()
            if choice == "1":
                skip_training = True
                output_dir = existing_analysis
                print("\n⚡ Using existing model for fast results")
                print("\nThis pipeline will:")
                print("  1. ⏭️  Skip training (using existing model)")
                print("  2. ✅ Generate predictions and visualizations")
                print("  3. ✅ Compare Phase 8 vs Phase 8c")
                if is_old_style:
                    print(f"  4. ✅ Results stay in {results_dir.name}/")
                else:
                    print(f"  4. ✅ Export final results to {output_dir.name}/")
                print("\nEstimated time: ~5 minutes")
                break
            elif choice == "2":
                skip_training = False
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = results_dir / f"analysis_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n🔄 Retraining model - results will save to: {output_dir.name}")
                print("\nThis pipeline will:")
                print("  1. ✅ Train U-Net model (~30-60 min)")
                print("  2. ✅ Generate predictions and visualizations")
                print("  3. ✅ Compare Phase 8 vs Phase 8c")
                print(f"  4. ✅ Export final results to {output_dir.name}/")
                print("\nEstimated time: ~30-60 minutes for training + 5 minutes for analysis")
                break
            elif choice == "3" and not is_old_style:
                print("\n🎨 Generating 3D visualizations from existing analysis...")
                generate_3d_visualizations(existing_analysis)
                print("\n✓ Done! Check the analysis folder for 3d_*.png files")
                return
            else:
                if is_old_style:
                    print("Invalid choice. Please enter 1 or 2.")
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
    else:
        print("\n⚠️  No existing analysis found")
        skip_training = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = results_dir / f"analysis_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🆕 Creating new analysis folder: {output_dir.name}")
        print("\nThis pipeline will:")
        print("  1. ✅ Train U-Net model (~30-60 min)")
        print("  2. ✅ Generate predictions and visualizations")
        print("  3. ✅ Compare Phase 8 vs Phase 8c")
        print(f"  4. ✅ Export final results to {output_dir.name}/")
        print("\nEstimated time: ~30-60 minutes for training + 5 minutes for analysis")
    
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
            
            # Train with output to specific analysis folder
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
        else:
            print("\n" + "▶"*40)
            print("STEP 1: TRAINING SKIPPED (Using existing model)")
            print("▶"*40 + "\n")
        
        # Step 2: Generate predictions (will use output_dir for results)
        print("\n" + "▶"*40)
        print("STEP 2: GENERATING PREDICTIONS")
        print("▶"*40 + "\n")
        
        # Temporarily modify generate_predictions to use our output_dir
        original_main = generate_predictions.main
        def custom_predictions():
            dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
            
            # Handle both old-style and new-style model locations
            model_file = output_dir / "unet_hbridge.keras"
            if not model_file.exists():
                # Try old-style naming
                old_models = list(output_dir.glob("unet_hbridge_*.keras"))
                if old_models:
                    model_file = max(old_models, key=lambda p: p.stat().st_mtime)
                else:
                    raise FileNotFoundError(f"Model not found in {output_dir}")
            
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file, compile=False)
            
            results = generate_predictions.generate_predictions(model, dataset_file, n_frames=None)
            df = generate_predictions.extract_component_predictions(results)
            metrics = generate_predictions.calculate_metrics(df)
            generate_predictions.create_visualizations(results, output_dir, n_examples=10)
            generate_predictions.create_scatter_plot(df, output_dir)
            generate_predictions.create_timeseries_plot(results, df, output_dir)
            generate_predictions.export_results(df, metrics, output_dir, model_file)
            
            print("\n" + "="*80)
            print("PREDICTION GENERATION COMPLETE")
            print("="*80)
            print(f"\nResults saved to: {output_dir}")
            print(f"\nFinal Performance:")
            print(f"  R² Score: {metrics['overall']['r2']:.4f}")
            print(f"  RMSE: {metrics['overall']['rmse']:.2f}°C")
            print(f"  MAE: {metrics['overall']['mae']:.2f}°C")
            print()
        
        custom_predictions()
        
        # Step 3: Compare with Phase 8 (will use output_dir for results)
        print("\n" + "▶"*40)
        print("STEP 3: COMPARING PHASE 8 vs PHASE 8C")
        print("▶"*40 + "\n")
        
        def custom_comparison():
            phase8_metrics = compare_phase8_vs_phase8c.load_phase8_metrics()
            phase8c_metrics = compare_phase8_vs_phase8c.load_phase8c_metrics(output_dir)
            compare_phase8_vs_phase8c.create_comparison_plots(phase8_metrics, phase8c_metrics, output_dir)
            compare_phase8_vs_phase8c.create_approach_comparison(phase8_metrics, phase8c_metrics, output_dir)
            compare_phase8_vs_phase8c.generate_comparison_report(phase8_metrics, phase8c_metrics, output_dir)
            
            print("\n" + "="*80)
            print("COMPARISON COMPLETE")
            print("="*80)
            print(f"\nResults saved to: {output_dir}")
            print(f"\nKey Findings:")
            print(f"  Phase 8 R²: {phase8_metrics['r2']:.4f}")
            print(f"  Phase 8c R²: {phase8c_metrics['r2']:.4f}")
            improvement = ((phase8c_metrics['r2'] - phase8_metrics['r2']) / phase8_metrics['r2']) * 100
            print(f"  Improvement: {improvement:+.1f}%")
            print()
        
        custom_comparison()
        
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
        print("\nAll results saved to:")
        print(f"  {output_dir}/")
        print(f"\nGenerated files in {output_dir.name}/:")
        if not skip_training:
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
        
        # Find model file (backwards compatible with old format)
        model_file = output_dir / "unet_hbridge.keras"
        if not model_file.exists():
            # Try old format with timestamp
            old_models = list(output_dir.glob("unet_hbridge_*.keras"))
            if old_models:
                model_file = max(old_models, key=lambda p: p.stat().st_mtime)
                print(f"Using model: {model_file.name}")
            else:
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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
