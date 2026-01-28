"""
Re-evaluate an existing trained model with updated evaluation logic.

Loads a previously trained model, re-runs evaluation on ALL frames,
saves raw predictions, and optionally creates animation.
"""

import sys
from pathlib import Path
import tensorflow as tf
import subprocess

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from phase8c_spatial_cnn import SpatialCNNTrainer, MaskedMSELoss

# Import evaluation function from train_hbridge_model
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hbridge_model import evaluate_model, generate_spatial_comparisons


def main():
    # Paths
    model_dir = Path(__file__).resolve().parent / "results" / "analysis_20260127_154615"  # Most recent model
    model_file = model_dir / "unet_hbridge.keras"
    dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
    
    # Output to latest folder
    output_dir = Path(__file__).resolve().parent / "results" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RE-EVALUATING MODEL WITH ALL FRAMES + ANIMATION")
    print("="*80)
    print(f"\nModel: {model_file}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")
    
    # Create trainer instance
    trainer = SpatialCNNTrainer(verbose=True)
    
    # Load dataset (needed for prepare_training_data to extract normalization stats)
    trainer.load_dataset(str(dataset_file))
    
    # Prepare training data to populate normalization stats
    print("\nExtracting normalization stats from dataset...")
    _, _, _, _ = trainer.prepare_training_data(
        val_split=0.2,
        temporal_split=False,
        component_split=True
    )
    print(f"✓ Normalization stats loaded:")
    print(f"  FLIR: [{trainer.normalization_stats['flir_min']:.2f}, {trainer.normalization_stats['flir_max']:.2f}]°C")
    print(f"  Sand: [{trainer.normalization_stats['sand_min']:.2f}, {trainer.normalization_stats['sand_max']:.2f}]°C")
    
    # Load trained model
    print("\nLoading trained model...")
    trainer.model = tf.keras.models.load_model(
        model_file,
        custom_objects={'MaskedMSELoss': MaskedMSELoss}
    )
    print("✓ Model loaded successfully")
    
    # Sampling params (MUST match training configuration)
    # Dense: every 45s (every 3rd frame) → ~100 frames
    # Sparse: every 1500s → ~84 frames
    sampling_params = {
        'use_all_timesteps': False,  # Training used hybrid dense+sparse sampling
        'dense_limit_time': 4500,    # Dense sampling up to 4500s
        'dense_step_time': 45,       # Every 45s in dense region (every 3rd frame)
        'sparse_step_time': 1500     # Every 1500s in sparse region
    }
    
    # Re-run evaluation on ALL frames with updated code
    r2, rmse, mae = evaluate_model(
        trainer=trainer,
        dataset_path=str(dataset_file),
        output_dir=str(output_dir),
        sampling_params=sampling_params,
        generate_plots=True
    )
    
    if r2 is None:
        print("\n✗ Evaluation failed")
        return
    
    # Skip spatial comparison plots - GIF animation provides better temporal visualization
    # print("\nGenerating spatial comparison plots...")
    # generate_spatial_comparisons(
    #     trainer=trainer,
    #     dataset_path=str(dataset_file),
    #     output_dir=str(output_dir),
    #     sampling_params=sampling_params,
    #     n_samples=15
    # )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"\nResults saved to: {output_dir}")
    
    # Ask if user wants to create animation
    print("\n" + "="*80)
    print("CREATE ANIMATION?")
    print("="*80)
    print("\n  [1] Yes - create GIF animation (~10-50 MB)")
    print("  [2] Yes - create MP4 video (smaller)")
    print("  [3] No - skip animation")
    
    choice = input("\nSelect option [1-3] (default: 3): ").strip()
    
    if choice == '1' or choice == '2':
        format = 'gif' if choice == '1' else 'mp4'
        ext = '.gif' if choice == '1' else '.mp4'
        output_file = output_dir / f"thermal_animation{ext}"
        
        print(f"\nCreating {format.upper()} animation...")
        print("This may take a few minutes...")
        
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / 'create_thermal_animation.py'),
            '--npz', str(output_dir / 'predictions_thermal_maps.npz'),
            '--dataset', str(dataset_file),
            '--output', str(output_file),
            '--format', format,
            '--fps', '10'
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"\n✓ Animation complete!")
        else:
            print(f"\n✗ Animation creation failed")
    else:
        print("\n⊘ Skipping animation")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
