"""
Train U-Net model for HBridge thermal field reconstruction.

This script:
1. Loads HDF5 training dataset
2. Initializes U-Net model with custom masked MSE loss
3. Trains with validation split
4. Saves trained model and training history
5. Generates performance metrics and visualizations

Author: CNN Pipeline
Date: 2025-01-12
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from phase8c_spatial_cnn import SpatialCNNTrainer

# Add parent for viz imports
viz_parent = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(viz_parent))
from viz_phase8c_spatial import Phase8cVisualizer


def load_dataset_info(h5_file):
    """Load and display dataset information."""
    print("="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    with h5py.File(h5_file, 'r') as f:
        print(f"\nDataset: {h5_file}")
        print(f"\nShapes:")
        print(f"  FLIR frames: {f['flir_frames'].shape}")
        print(f"  Sand temps: {f['sand_temps'].shape}")
        print(f"  ROI masks: {f['roi_masks'].shape}")
        print(f"  Timestamps: {f['timestamps'].shape}")
        
        print(f"\nMetadata:")
        for key, val in f['metadata'].attrs.items():
            print(f"  {key}: {val}")
        
        # Get component names
        component_names = [name.decode('utf-8') for name in f['metadata']['component_names'][:]]
        print(f"\n  Components ({len(component_names)}): {', '.join(component_names[:10])}{'...' if len(component_names) > 10 else ''}")
        
        # Temperature statistics
        temps = f['sand_temps'][:]
        print(f"\nTemperature Statistics:")
        print(f"  Min: {np.min(temps):.2f}°C")
        print(f"  Max: {np.max(temps):.2f}°C")
        print(f"  Mean: {np.mean(temps):.2f}°C")
        print(f"  Std: {np.std(temps):.2f}°C")
        
        # FLIR statistics
        frames = f['flir_frames'][:]
        print(f"\nFLIR Frame Statistics:")
        print(f"  Min: {np.min(frames):.2f}°C")
        print(f"  Max: {np.max(frames):.2f}°C")
        print(f"  Mean: {np.mean(frames):.2f}°C")
        print(f"  Std: {np.std(frames):.2f}°C")
    
    print()


def train_model(dataset_path, output_dir, epochs=5, batch_size=16, validation_split=0.2, generate_plots=True):
    """
    Train U-Net model on HBridge dataset.
    
    Args:
        dataset_path: Path to HDF5 dataset
        output_dir: Directory to save model and results
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        generate_plots: Whether to generate training curve plots
    """
    print("\n" + "="*80)
    print("TRAINING U-NET MODEL")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SpatialCNNTrainer(verbose=True)
    
    # Ask user for memory mode FIRST (BEFORE loading any data!)
    print("\nTraining Options:")
    print("  [1] Memory-efficient generator (recommended for large datasets)")
    print("      └─ RAM usage: ~500 MB - 2 GB")
    print("  [2] In-memory training (faster but requires ~32 GB RAM)")
    print("      └─ RAM usage: ~31.5 GB")
    memory_choice = input("\nSelect option [1-2] (default: 1): ").strip()
    use_generator = (memory_choice != '2')
    print(f"\nData generator enabled: {use_generator}")
    
    # Load dataset with appropriate mode
    if use_generator:
        # Lightweight mode - only metadata, keep large arrays on disk
        print("\n📦 Loading dataset metadata (large arrays kept on disk)...")
        trainer.load_dataset(str(dataset_path), lightweight=True)
    else:
        # Full mode - load everything into memory
        print("\n⚠️  Loading entire dataset into RAM (~31.5 GB)...")
        print("⚠️  This may take several minutes and could cause system slowdown.")
        print("\n  [1] Switch to memory-efficient generator (recommended)")
        print("  [2] Continue with in-memory loading")
        proceed_choice = input("\nSelect option [1-2] (default: 1): ").strip()
        if proceed_choice != '2':
            print("Switching to memory-efficient generator instead.")
            use_generator = True
            trainer.load_dataset(str(dataset_path), lightweight=True)
        else:
            trainer.load_dataset(str(dataset_path), lightweight=False)
    
    # Build model AFTER loading dataset (needs metadata for dimensions)
    trainer.build_model(learning_rate=0.001)
    
    
    # Ask about timestep sampling strategy
    print("\nTimestep Sampling Strategy:")
    print("  [1] Hybrid sampling (dense transient + sparse steady-state) - Efficient")
    print("  [2] All timesteps (no sampling) - Maximum data, higher RAM/compute")
    sampling_choice = input("\nSelect option [1-2] (default: 1): ").strip()
    use_all_timesteps = (sampling_choice == '2')
    print(f"\nUsing all timesteps: {use_all_timesteps}")
    
    # Only prepare in-memory data if NOT using generator
    if not use_generator:
        # Prepare training data for in-memory mode
        X_train, y_train, X_val, y_val = trainer.prepare_training_data(
            val_split=validation_split,
            temporal_split=False,  # Use component-level split (not temporal)
            component_split=True   # NEW: Component-level validation
        )
    else:
        # Generator path - no need to prepare data
        X_train = y_train = X_val = y_val = None
    
    # If using generator, prompt for hybrid sampling parameters
    if use_generator:
        # Load dataset info for sampling preview
        with h5py.File(dataset_path, 'r') as f:
            timestamps = f['timestamps'][:]
            total_duration = timestamps[-1] - timestamps[0]
            time_step = timestamps[1] - timestamps[0]
            n_samples = len(timestamps)
        
        # Sampling configuration (only needed if NOT using all timesteps)
        if not use_all_timesteps:
            print("\n" + "="*80)
            print("  HYBRID BATCH SAMPLING CONFIGURATION")
            print("="*80)
            print(f"\nDataset Info:")
            print(f"  Total samples: {n_samples}")
            print(f"  Time step: {time_step:.2f}s")
            print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            
            # Ask if user wants to customize
            print("\nSampling Strategy:")
            print("  Dense region: Sample frequently during thermal transients (rapid changes)")
            print("  Sparse region: Sample less often during steady-state (stable temps)")
            print("\n  [1] Use default sampling (recommended)")
            print("  [2] Customize sampling parameters")
            customize_choice = input("\nSelect option [1-2] (default: 1): ").strip()
            
            if customize_choice == '2':
                print("\nEnter sampling parameters (in seconds):")
                print("  Tip: FLIR is 15s intervals. Dense should match FLIR rate, sparse can skip frames.")
                dense_limit_time_input = input(f"  Dense limit (cutoff time) [default: 4500s = 75 min]: ").strip()
                dense_limit_time = int(dense_limit_time_input) if dense_limit_time_input else 4500
                dense_step_time_input = input(f"  Dense step (sample every X seconds) [default: 15s = every FLIR frame]: ").strip()
                dense_step_time = int(dense_step_time_input) if dense_step_time_input else 15
                sparse_step_time_input = input(f"  Sparse step (sample every X seconds) [default: 300s = every 20 FLIR frames]: ").strip()
                sparse_step_time = int(sparse_step_time_input) if sparse_step_time_input else 300
            else:
                # Use defaults
                dense_limit_time = 4500
                dense_step_time = 15
                sparse_step_time = 300
            
            # Calculate and preview sampling
            dense_limit_idx = int(dense_limit_time / time_step)
            dense_step_idx = max(1, int(dense_step_time / time_step))
            sparse_step_idx = max(1, int(sparse_step_time / time_step))
            
            dense_samples = len(range(0, min(dense_limit_idx, n_samples), dense_step_idx))
            sparse_samples = len(range(dense_limit_idx, n_samples, sparse_step_idx))
            total_samples = dense_samples + sparse_samples
            dense_pct = (dense_samples / total_samples * 100) if total_samples > 0 else 0
            
            print("\n" + "-"*80)
            print("SAMPLING PREVIEW:")
            print("-"*80)
            print(f"Dense region (0 - {dense_limit_time}s):")
            print(f"  Sample every {dense_step_time}s → {dense_samples} samples")
            print(f"Sparse region ({dense_limit_time}s - {total_duration:.0f}s):")
            print(f"  Sample every {sparse_step_time}s → {sparse_samples} samples")
            print(f"\nTotal training samples: {total_samples}")
            print(f"  Dense: {dense_pct:.1f}% | Sparse: {100-dense_pct:.1f}%")
            print(f"  Component split: 80/20 (validation components excluded from input)")
            print(f"  Expected batches/epoch: ~{total_samples * 0.8 / batch_size:.0f} (batch_size={batch_size})")
            print("-"*80)
        else:
            # All timesteps mode
            dense_limit_time = 0
            dense_step_time = 1
            sparse_step_time = 1
            print("\n" + "-"*80)
            print("USING ALL TIMESTEPS:")
            print("-"*80)
            print(f"Total samples: {n_samples}")
            print(f"Component split: 80/20 (validation components excluded from input)")
            print(f"Expected batches/epoch: ~{n_samples * 0.8 / batch_size:.0f} (batch_size={batch_size})")
            print("-"*80)
        
        print("\n  [1] Proceed with these settings")
        print("  [2] Exit and restart to adjust")
        proceed_choice = input("\nSelect option [1-2] (default: 1): ").strip()
        if proceed_choice == '2':
            print("Exiting. Re-run to adjust parameters.")
            return None
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split:.1%}")
    print(f"  Output directory: {output_path}")
    print()
    
    if use_generator:
        trainer.train(
            None, None, None, None,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=15,
            use_generator=True,
            h5_path=str(dataset_path),
            dense_limit_time=dense_limit_time,
            dense_step_time=dense_step_time,
            sparse_step_time=sparse_step_time,
            use_all_timesteps=use_all_timesteps
        )
        # For evaluation, need to load val data from generator
        val_gen = trainer.HDF5DataGenerator(str(dataset_path), split='val', batch_size=trainer.HDF5DataGenerator(str(dataset_path), split='val').__len__(),
                                            dense_limit_time=dense_limit_time, dense_step_time=dense_step_time, sparse_step_time=sparse_step_time)
        X_val, y_val = val_gen[0][0], val_gen[0][1]
    else:
        trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=15,
            use_generator=False
        )
    
    # Save model
    model_file = output_path / "unet_hbridge.keras"
    trainer.model.save(model_file)
    print(f"\n✓ Model saved: {model_file}")
    
    # Save frame indices for future validation (if using generator)
    if use_generator:
        # Create temporary generators to extract frame indices
        train_gen = trainer.HDF5DataGenerator(
            str(dataset_path), split='train', batch_size=batch_size,
            dense_limit_time=dense_limit_time,
            dense_step_time=dense_step_time,
            sparse_step_time=sparse_step_time,
            use_all_timesteps=use_all_timesteps
        )
        val_gen = trainer.HDF5DataGenerator(
            str(dataset_path), split='val', batch_size=batch_size,
            dense_limit_time=dense_limit_time,
            dense_step_time=dense_step_time,
            sparse_step_time=sparse_step_time,
            use_all_timesteps=use_all_timesteps
        )
        
        # Save training and validation frame indices
        np.save(output_path / 'training_frame_indices.npy', train_gen.indices)
        np.save(output_path / 'validation_frame_indices.npy', val_gen.indices)
        print(f"✓ Saved training frame indices: {len(train_gen.indices)} frames → {output_path / 'training_frame_indices.npy'}")
        print(f"✓ Saved validation frame indices: {len(val_gen.indices)} frames → {output_path / 'validation_frame_indices.npy'}")
        
        # Close generators if they have a close method
        if hasattr(train_gen, 'close'):
            train_gen.close()
        if hasattr(val_gen, 'close'):
            val_gen.close()
    
    history = trainer.history
    
    # Save training history
    history_file = output_path / "training_history.npz"
    np.savez(history_file, **history.history)
    print(f"\n✓ Training history saved: {history_file}")
    
    # Plot training curves (conditional)
    if generate_plots:
        plot_training_curves(history, output_path)
    else:
        print("⊘ Skipping training curve plots")
    
    # Return sampling parameters for spatial comparison plots
    sampling_params = {
        'use_all_timesteps': use_all_timesteps if use_generator else False,
        'dense_limit_time': dense_limit_time if use_generator else 0,
        'dense_step_time': dense_step_time if use_generator else 1,
        'sparse_step_time': sparse_step_time if use_generator else 1
    }
    
    return trainer, history, sampling_params


def plot_training_curves(history, output_dir):
    """Plot and save training/validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Masked MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE curve
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE (°C)', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / "training_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved: {output_file}")
    plt.close()


def evaluate_model(trainer, dataset_path, output_dir, sampling_params, generate_plots=True):
    """
    Evaluate trained model on VALIDATION COMPONENTS using SAME TEMPORAL SAMPLING as training.
    
    This function evaluates the model by:
    1. Using the SAME frame sampling (dense+sparse) as training
    2. Only evaluating on held-out validation components (spatial generalization test)
    3. Using the same normalization as training
    
    This tests: "Can the model predict unseen components at the training time points?"
    
    Args:
        trainer: Trained SpatialCNNTrainer instance
        dataset_path: Path to HDF5 dataset
        output_dir: Directory to save results
        sampling_params: Dict with sampling parameters for reconstruction
        generate_plots: Whether to generate scatter plot
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION (VALIDATION COMPONENTS, TRAINING TEMPORAL SAMPLING)")
    print("="*80)
    
    # Load dataset and component split info
    with h5py.File(dataset_path, 'r') as f:
        timestamps = f['timestamps'][:]
        n_frames = f['flir_frames'].shape[0]
        frame_indices_map = f['frame_indices'][:]  # Maps timestamps to frames
        n_samples = len(timestamps)  # Total timestamps (8719)
        
        # Try to load saved validation timestamp indices
        saved_val_indices_file = Path(output_dir) / 'validation_frame_indices.npy'
        if saved_val_indices_file.exists():
            timestamp_indices = np.load(saved_val_indices_file)
            print(f"\n  ✓ Loaded saved validation timestamp indices: {len(timestamp_indices)} samples")
        else:
            # Reconstruct using same logic as HDF5DataGenerator (samples on TIMESTAMPS, not frames)
            print(f"\n  ⚠️  No saved indices found, reconstructing...")
            print(f"  Sampling on {n_samples} timestamps (not {n_frames} frames)")
            time_step = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 15.0
            
            if sampling_params['use_all_timesteps']:
                # All timesteps mode - use ALL timestamps (no sampling)
                timestamp_indices = np.arange(0, n_samples)
            else:
                # Hybrid sampling mode (same as HDF5DataGenerator)
                dense_limit_time = sampling_params['dense_limit_time']
                dense_step_time = sampling_params['dense_step_time']
                sparse_step_time = sampling_params['sparse_step_time']
                
                dense_limit_idx = int(dense_limit_time / time_step)
                dense_step_idx = max(1, int(dense_step_time / time_step))
                sparse_step_idx = max(1, int(sparse_step_time / time_step))
                
                # Sample dense and sparse regions on TIMESTAMPS
                all_indices = np.arange(0, n_samples)
                dense_indices = all_indices[all_indices < dense_limit_idx][::dense_step_idx]
                sparse_indices = all_indices[all_indices >= dense_limit_idx][::sparse_step_idx]
                
                # Use ALL sampled timestamps (not split 80/20)
                # The validation is on COMPONENTS, not time
                timestamp_indices = np.sort(np.concatenate([dense_indices, sparse_indices]))
        
        # Map timestamp indices to unique frame indices
        frame_indices = np.unique(frame_indices_map[timestamp_indices])
        
        print(f"\n  Evaluating on {len(timestamp_indices)} VALIDATION timestamps → {len(frame_indices)} unique frames")
        if not sampling_params['use_all_timesteps']:
            dense_limit = sampling_params.get('dense_limit_time', 4500)
            n_dense_ts = np.sum(timestamps[timestamp_indices] < dense_limit)
            n_sparse_ts = np.sum(timestamps[timestamp_indices] >= dense_limit)
            print(f"    Dense samples (<{dense_limit}s): {n_dense_ts} timestamps")
            print(f"    Sparse samples (>={dense_limit}s): {n_sparse_ts} timestamps")
        print(f"  Time range: {timestamps[timestamp_indices[0]]:.1f}s - {timestamps[timestamp_indices[-1]]:.1f}s")
        
        # Load data for these timestamps
        flir_frames = f['flir_frames'][frame_indices]  # Load unique frames
        sand_temps = f['sand_temps'][:][timestamp_indices]  # Load temps at sampled timestamps
        roi_masks = f['roi_masks'][:]
        timestamps_sampled = timestamps[timestamp_indices]  # Timestamps for evaluation
        
        # Get component split info
        if 'metadata/train_component_indices' in f:
            train_comp_indices = f['metadata/train_component_indices'][:]
            n_components = sand_temps.shape[1]
            all_indices = np.arange(n_components)
            val_comp_indices = np.array([i for i in all_indices if i not in train_comp_indices])
            print(f"\n  Training components: {len(train_comp_indices)}")
            print(f"  Validation components: {len(val_comp_indices)}")
            print(f"  Validation component IDs: {val_comp_indices.tolist()}")
        else:
            print("\n  ⚠️  WARNING: No component split found in dataset!")
            print("  Using last 20% of components as validation")
            n_components = sand_temps.shape[1]
            val_comp_indices = np.arange(split_idx, n_components)
        
        # Build dual inputs (FLIR + time) for model prediction
        # We have:
        #   - timestamp_indices: sampled timestamps to evaluate
        #   - frame_indices: unique FLIR frames corresponding to those timestamps
        #   - frame_indices_map: maps each timestamp to its FLIR frame
        H, W = flir_frames.shape[1], flir_frames.shape[2]
        n_eval_samples = len(timestamp_indices)
        
        # Build input arrays (one per sampled timestamp)
        X_eval_flir = np.zeros((n_eval_samples, H, W, 1), dtype=np.float32)
        X_eval_time = np.zeros((n_eval_samples, 1), dtype=np.float32)
        
        # Get normalization stats from trainer
        if hasattr(trainer, 'normalization_stats') and trainer.normalization_stats is not None:
            norm_stats = trainer.normalization_stats
            flir_min = norm_stats['flir_min']
            flir_max = norm_stats['flir_max']
            sand_min = norm_stats['sand_min']
            sand_max = norm_stats['sand_max']
            time_min = norm_stats['time_min']
            time_max = norm_stats['time_max']
            print(f"\n  Using training normalization:")
            print(f"    FLIR: [{flir_min:.2f}, {flir_max:.2f}]°C")
            print(f"    Time: [{time_min:.2f}, {time_max:.2f}]s")
            print(f"    Sand: [{sand_min:.2f}, {sand_max:.2f}]°C")
        else:
            # Fallback: compute from data
            flir_min = float(np.min(flir_frames))
            flir_max = float(np.max(flir_frames))
            sand_min = float(np.min(sand_temps))
            sand_max = float(np.max(sand_temps))
            time_min = float(timestamps[frame_indices].min())
            time_max = float(timestamps[frame_indices].max())
            print(f"\n  ⚠️  Computing normalization from evaluation data (may mismatch training!)")
        
        # Create frame index lookup for fast access
        frame_to_idx = {f: i for i, f in enumerate(frame_indices)}
        
        # Fill input arrays (normalized) for each sampled timestamp
        for i, ts_idx in enumerate(timestamp_indices):
            # Get the FLIR frame for this timestamp
            frame_for_ts = frame_indices_map[ts_idx]
            flir_idx = frame_to_idx[frame_for_ts]
            
            # Normalize FLIR frame
            X_eval_flir[i, :, :, 0] = (flir_frames[flir_idx] - flir_min) / (flir_max - flir_min + 1e-8)
            
            # Normalize timestamp
            timestamp = timestamps_sampled[i]
            X_eval_time[i, 0] = (timestamp - time_min) / (time_max - time_min + 1e-8)
    
    # Generate predictions (model expects FLIR + time dual inputs)
    print("\nGenerating predictions...")
    print(f"  Processing {n_eval_samples} timestamps (this may take several minutes on CPU)...")
    
    # Batch processing with progress updates
    batch_size = 8
    predictions_list = []
    for i in range(0, n_eval_samples, batch_size):
        batch_end = min(i + batch_size, n_eval_samples)
        batch_input = {
            'flir_input': X_eval_flir[i:batch_end],
            'time_input': X_eval_time[i:batch_end]
        }
        batch_preds = trainer.model.predict(batch_input, verbose=0)
        predictions_list.append(batch_preds)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Progress: {batch_end}/{n_eval_samples} samples ({100*batch_end/n_eval_samples:.0f}%)")
    
    predictions = np.concatenate(predictions_list, axis=0).squeeze()
    print("  ✓ Predictions complete")
    
    # Denormalize predictions back to °C
    predictions = predictions * (sand_max - sand_min) + sand_min
    
    # Save raw predictions to disk
    print("\nSaving raw prediction data...")
    
    # Save full thermal maps as NPZ (compressed NumPy format)
    predictions_npz = Path(output_dir) / "predictions_thermal_maps.npz"
    np.savez_compressed(
        predictions_npz,
        predictions=predictions,  # [n_timestamps, H, W] predicted thermal maps
        flir_frames=flir_frames,  # [n_unique_frames, H, W] unique input FLIR frames
        timestamps=timestamps_sampled,  # [n_timestamps] timestamp for each prediction
        timestamp_indices=timestamp_indices,  # [n_timestamps] timestamp indices
        frame_indices=frame_indices,  # [n_unique_frames] unique frame indices
        frame_indices_map=frame_indices_map[timestamp_indices]  # [n_timestamps] maps each prediction to its frame
    )
    print(f"  ✓ Thermal maps saved: {predictions_npz}")
    print(f"    Shape: {predictions.shape} (timestamps × height × width)")
    
    # Save component-level predictions as CSV (easier to analyze)
    predictions_csv = Path(output_dir) / "predictions_by_component.csv"
    csv_data = []
    
    # Extract predictions for ALL components (both train and val)
    all_comp_indices = np.arange(roi_masks.shape[0])
    
    for i, ts_idx in enumerate(timestamp_indices):
        timestamp = timestamps_sampled[i]
        for comp_idx in all_comp_indices:
            if comp_idx >= roi_masks.shape[0]:
                continue
            
            mask = roi_masks[comp_idx]
            if np.sum(mask) == 0:
                continue
            
            # Actual temperature
            actual_temp = sand_temps[i, comp_idx]
            
            # Predicted temperature (average over ROI)
            predicted_temp = predictions[i][mask > 0].mean()
            
            # Determine if this is a training or validation component
            is_validation = comp_idx in val_comp_indices
            
            # Get the frame index for this timestamp
            frame_for_ts = frame_indices_map[ts_idx]
            
            csv_data.append({
                'timestamp_index': int(ts_idx),
                'frame_index': int(frame_for_ts),
                'timestamp_s': float(timestamp),
                'component_id': int(comp_idx),
                'component_type': 'validation' if is_validation else 'training',
                'actual_temp_C': float(actual_temp),
                'predicted_temp_C': float(predicted_temp),
                'error_C': float(predicted_temp - actual_temp),
                'abs_error_C': float(abs(predicted_temp - actual_temp))
            })
    
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv(predictions_csv, index=False)
    print(f"  ✓ Component predictions saved: {predictions_csv}")
    print(f"    Rows: {len(df)} (timestamps × components)")
    
    # Calculate metrics at ROI locations FOR VALIDATION COMPONENTS ONLY
    print("\nCalculating metrics at VALIDATION component ROI locations...")
    
    all_actual = []
    all_predicted = []
    
    skipped_empty_masks = 0
    skipped_nan = 0
    
    # Only loop through VALIDATION components
    for i in range(len(timestamp_indices)):
        for comp_idx in val_comp_indices:
            if comp_idx >= roi_masks.shape[0]:
                continue
            
            mask = roi_masks[comp_idx]
            
            # Skip if mask is empty
            if np.sum(mask) == 0:
                skipped_empty_masks += 1
                continue
            
            # Extract values at ROI pixels
            actual_temp = sand_temps[i, comp_idx]
            predicted_temp = predictions[i][mask > 0].mean()
            
            # Skip if NaN
            if np.isnan(actual_temp) or np.isnan(predicted_temp):
                skipped_nan += 1
                continue
            
            all_actual.append(actual_temp)
            all_predicted.append(predicted_temp)
    
    print(f"  Valid predictions: {len(all_actual)}")
    print(f"  Skipped (empty ROI): {skipped_empty_masks}")
    print(f"  Skipped (NaN values): {skipped_nan}")
    print(f"  Frames evaluated: {len(flir_frames)}")
    print(f"  Components evaluated: {len(val_comp_indices)}")
    
    if len(all_actual) == 0:
        print("\n✗ ERROR: No valid predictions! All validation ROI masks are empty or contain NaN.")
        return None, None, None
    
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    
    # Calculate R², RMSE, MAE
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(all_actual, all_predicted)
    rmse = np.sqrt(mean_squared_error(all_actual, all_predicted))
    mae = mean_absolute_error(all_actual, all_predicted)
    
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"  Sample count: {len(all_actual)}")
    
    # Save metrics
    metrics_file = Path(output_dir) / "evaluation_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("HBridge U-Net Model Evaluation (VALIDATION COMPONENTS ONLY)\n")
        f.write("="*50 + "\n\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.2f}°C\n")
        f.write(f"MAE: {mae:.2f}°C\n")
        f.write(f"Sample count: {len(all_actual)}\n")
        f.write(f"Timestamps evaluated: {len(timestamp_indices)}\n")
        f.write(f"Unique frames used: {len(frame_indices)}\n")
        f.write(f"Validation components: {len(val_comp_indices)}\n")
        f.write(f"\nTemperature Range:\n")
        f.write(f"  Actual: {all_actual.min():.2f}°C - {all_actual.max():.2f}°C\n")
        f.write(f"  Predicted: {all_predicted.min():.2f}°C - {all_predicted.max():.2f}°C\n")
    
    print(f"\n✓ Metrics saved: {metrics_file}")
    
    # Save validation timestamp indices for future use
    np.save(Path(output_dir) / 'validation_frame_indices.npy', timestamp_indices)
    print(f"\n✓ Saved validation timestamp indices: {Path(output_dir) / 'validation_frame_indices.npy'}")
    
    # Create scatter plot (conditional)
    if generate_plots:
        plot_scatter(all_actual, all_predicted, r2, rmse, output_dir)
    else:
        print("⊘ Skipping scatter plot")
    
    return r2, rmse, mae


def plot_scatter(actual, predicted, r2, rmse, output_dir):
    """Plot actual vs predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    
    plt.scatter(actual, predicted, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title(f'U-Net Predictions vs Actual Thermistor Readings\nR² = {r2:.4f}, RMSE = {rmse:.2f}°C', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save figure
    output_file = Path(output_dir) / "scatter_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Scatter plot saved: {output_file}")
    plt.close()


def generate_spatial_comparisons(trainer, dataset_path, output_dir, sampling_params, n_samples=15):
    """
    Generate spatial thermal field comparison plots from training frames.
    
    Creates thermal_comparison_t{time}s.png visualizations showing:
    - FLIR input
    - CNN prediction
    - Ground truth thermistor data
    - Absolute error map
    
    Samples frames from actual training set (both dense and sparse regions).
    
    Args:
        trainer: Trained SpatialCNNTrainer instance
        dataset_path: Path to HDF5 dataset
        output_dir: Directory to save plots
        sampling_params: Dict with 'use_all_timesteps', 'dense_limit_time', 'dense_step_time', 'sparse_step_time'
        n_samples: Number of frames to visualize (split between dense/sparse regions)
    """
    print("\n" + "="*80)
    print("GENERATING SPATIAL COMPARISON PLOTS")
    print("="*80)
    
    output_path = Path(output_dir)
    
    # Load full dataset for visualization
    with h5py.File(dataset_path, 'r') as f:
        flir_frames = f['flir_frames'][:]
        sand_temps = f['sand_temps'][:]
        roi_masks = f['roi_masks'][:]
        timestamps = f['timestamps'][:]
        
        # Get normalization stats
        if hasattr(trainer, 'normalization_stats') and trainer.normalization_stats is not None:
            norm_stats = trainer.normalization_stats
            flir_min = norm_stats['flir_min']
            flir_max = norm_stats['flir_max']
            sand_min = norm_stats['sand_min']
            sand_max = norm_stats['sand_max']
            time_min = norm_stats['time_min']
            time_max = norm_stats['time_max']
        else:
            flir_min = float(np.min(flir_frames))
            flir_max = float(np.max(flir_frames))
            sand_min = float(np.min(sand_temps))
            sand_max = float(np.max(sand_temps))
            time_min = float(timestamps.min())
            time_max = float(timestamps.max())
    
    # Reconstruct training indices using same logic as HDF5DataGenerator
    n_frames = len(flir_frames)
    time_step = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 15.0
    
    if sampling_params['use_all_timesteps']:
        # All timesteps mode - sample evenly from all frames
        all_indices = np.arange(0, n_frames)
        np.random.seed(42)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:int(len(all_indices) * 0.8)]  # 80% for training
        frame_indices = np.sort(train_indices[::max(1, len(train_indices) // n_samples)])[:n_samples]
    else:
        # Hybrid sampling mode - reconstruct dense and sparse regions
        dense_limit_time = sampling_params['dense_limit_time']
        dense_step_time = sampling_params['dense_step_time']
        sparse_step_time = sampling_params['sparse_step_time']
        
        dense_limit_idx = int(dense_limit_time / time_step)
        dense_step_idx = max(1, int(dense_step_time / time_step))
        sparse_step_idx = max(1, int(sparse_step_time / time_step))
        
        # Get dense and sparse indices (same as HDF5DataGenerator)
        all_indices = np.arange(0, n_frames)
        dense_indices = all_indices[all_indices < dense_limit_idx][::dense_step_idx]
        sparse_indices = all_indices[all_indices >= dense_limit_idx][::sparse_step_idx]
        
        # Shuffle and split 80/20 (same as training)
        np.random.seed(42)
        np.random.shuffle(dense_indices)
        np.random.shuffle(sparse_indices)
        dense_train = dense_indices[:int(len(dense_indices) * 0.8)]
        sparse_train = sparse_indices[:int(len(sparse_indices) * 0.8)]
        
        # Sample evenly from both regions
        n_dense_samples = n_samples // 2
        n_sparse_samples = n_samples - n_dense_samples
        
        dense_sample = np.sort(dense_train[::max(1, len(dense_train) // n_dense_samples)])[:n_dense_samples]
        sparse_sample = np.sort(sparse_train[::max(1, len(sparse_train) // n_sparse_samples)])[:n_sparse_samples]
        
        frame_indices = np.sort(np.concatenate([dense_sample, sparse_sample]))
    
    print(f"\nGenerating {len(frame_indices)} spatial comparison plots from TRAINING frames...")
    print(f"  Frame indices: {frame_indices.tolist()}")
    print(f"  Time range: {timestamps[frame_indices[0]]:.1f}s - {timestamps[frame_indices[-1]]:.1f}s")
    if not sampling_params['use_all_timesteps']:
        print(f"  Dense region samples: {np.sum(timestamps[frame_indices] < sampling_params['dense_limit_time'])}")
        print(f"  Sparse region samples: {np.sum(timestamps[frame_indices] >= sampling_params['dense_limit_time'])}")
    
    # Create visualizer
    visualizer = Phase8cVisualizer(output_dir=output_path)
    
    # Combined ROI mask
    combined_mask = np.max(roi_masks, axis=0)
    n_components = sand_temps.shape[1]
    
    for idx in frame_indices:
        # Prepare dual inputs (FLIR + time, normalized)
        flir_raw = flir_frames[idx]
        flir_input = np.zeros((1, flir_raw.shape[0], flir_raw.shape[1], 1), dtype=np.float32)
        time_input = np.zeros((1, 1), dtype=np.float32)
        
        flir_input[0, :, :, 0] = (flir_raw - flir_min) / (flir_max - flir_min + 1e-8)
        timestamp = timestamps[idx]
        time_input[0, 0] = (timestamp - time_min) / (time_max - time_min + 1e-8)
        
        # Generate prediction with dual inputs (dict format)
        model_input = {'flir_input': flir_input, 'time_input': time_input}
        pred_normalized = trainer.model.predict(model_input, verbose=0)
        pred_temp = pred_normalized[0, :, :, 0] * (sand_max - sand_min) + sand_min
        
        # Build ground truth map from thermistor data
        H, W = flir_raw.shape
        ground_truth_map = np.zeros((H, W), dtype=np.float32)
        for comp_idx in range(min(n_components, roi_masks.shape[0])):
            mask = roi_masks[comp_idx]
            ground_truth_map[mask > 0] = sand_temps[idx, comp_idx]
        
        # Create comparison plot
        timestamp = timestamps[idx]
        save_name = f"thermal_comparison_t{timestamp:.1f}s.png"
        
        visualizer.plot_thermal_field_comparison(
            flir_frame=flir_raw,
            predicted_frame=pred_temp,
            ground_truth_frame=ground_truth_map,
            roi_mask=combined_mask,
            frame_index=idx,
            save_name=save_name
        )
    
    print(f"\n✓ Generated {len(frame_indices)} spatial comparison plots")
    print(f"  Saved to: {output_path}")


def main(board_name=None, validation_mode=None, build_dataset_flag=False, component_val_split=0.2):
    """
    Main training pipeline.
    
    Args:
        board_name: "HBridge" or "LoadShedding" for training dataset (None = prompt user)
        validation_mode: "component_holdout" or "cross_pcb" (None = prompt user)
        build_dataset_flag: Whether to build dataset before training
        component_val_split: Validation split for component holdout mode
    """
    # Interactive prompts if options not provided
    if validation_mode is None:
        print("\n" + "="*80)
        print("VALIDATION MODE SELECTION")
        print("="*80)
        print("\n  [1] Component Holdout - Train/validate on SAME PCB")
        print("      • Splits components 80/20 (train/val)")
        print("      • Tests if model learns thermal coupling")
        print("      • Expected: R² ≈ -3 (model copies FLIR)")
        print("\n  [2] Cross-PCB - Train on ONE PCB, validate on ANOTHER")
        print("      • Trains on HBridge, validates on LoadShedding")
        print("      • Tests if model generalizes to new PCB designs")
        print("      • Expected: R² < -10 (proves location-specific learning)")
        print("      • Use this to establish baseline for architectural improvements")
        
        val_choice = input("\nSelect validation mode [1-2] (default: 1): ").strip()
        validation_mode = "cross_pcb" if val_choice == "2" else "component_holdout"
    
    if validation_mode == "component_holdout" and board_name is None:
        print("\n" + "="*80)
        print("BOARD SELECTION")
        print("="*80)
        print("\n  [1] HBridge (22 components, 36 hours)")
        print("  [2] LoadShedding (20 components, 18 hours)")
        
        board_choice = input("\nSelect board [1-2] (default: 1): ").strip()
        board_name = "LoadShedding" if board_choice == "2" else "HBridge"
    elif validation_mode == "cross_pcb":
        board_name = "HBridge"  # Always use HBridge for training in cross-PCB mode
    
    # Ask about building dataset
    if not build_dataset_flag:
        print("\n" + "="*80)
        print("DATASET BUILDING")
        print("="*80)
        
        if validation_mode == "cross_pcb":
            print("\n  Cross-PCB mode requires both HBridge and LoadShedding datasets.")
            print("  Do you want to rebuild them before training?")
        else:
            print(f"\n  Do you want to rebuild the {board_name} dataset before training?")
        
        print("\n  [1] No - Use existing dataset(s)")
        print("  [2] Yes - Rebuild dataset(s) (ensures latest data)")
        
        build_choice = input("\nSelect option [1-2] (default: 1): ").strip()
        build_dataset_flag = (build_choice == "2")
    
    # Build datasets if requested
    if build_dataset_flag:
        print("\n" + "="*80)
        print("BUILDING DATASET(S)")
        print("="*80)
        
        # Import build_dataset module
        build_module_path = Path(__file__).parent / "build_dataset.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_dataset", build_module_path)
        build_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_module)
        
        if validation_mode == "cross_pcb":
            print("\nBuilding HBridge dataset (training)...")
            build_module.build_dataset("HBridge", component_val_split=0.0)
            print("\nBuilding LoadShedding dataset (validation)...")
            build_module.build_dataset("LoadShedding", component_val_split=0.0)
        else:
            print(f"\nBuilding {board_name} dataset...")
            build_module.build_dataset(board_name, component_val_split=component_val_split)
    
    # Display selected configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    
    # Paths
    if validation_mode == "cross_pcb":
        # Cross-PCB: Train on HBridge, validate on LoadShedding
        train_dataset = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "HBridge_cnn_dataset.h5"
        val_dataset = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / "LoadShedding_cnn_dataset.h5"
        dataset_file = train_dataset  # For now, use train dataset (val handled separately)
        print(f"\n🔀 CROSS-PCB VALIDATION MODE")
        print(f"   Training: HBridge")
        print(f"   Validation: LoadShedding")
    else:
        # Component holdout: Train and validate on same board
        dataset_file = parent_dir / "ml_model" / "cnn_thermal_modeling" / "datasets" / f"{board_name}_cnn_dataset.h5"
        print(f"\n📊 COMPONENT HOLDOUT VALIDATION MODE")
        print(f"   Board: {board_name}")
    
    # Create output directories
    results_base = parent_dir / "ml_model" / "cnn_thermal_modeling" / "results"
    
    # Always use 'latest' folder (overwrites previous run)
    latest_dir = results_base / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    output_dir = latest_dir
    
    print(f"\n  Results will be saved to: {output_dir}")
    print(f"  (This will overwrite previous 'latest' results)")
    
    # Ask if user wants to also archive with timestamp
    print("\n" + "="*80)
    print("ARCHIVE RESULTS?")
    print("="*80)
    print("\n  [1] No - only save to 'latest' folder (overwrites each run)")
    print("  [2] Yes - also create timestamped archive copy (keeps all runs)")
    
    archive_choice = input("\nSelect option [1-2] (default: 1): ").strip()
    create_archive = (archive_choice == '2')
    
    if create_archive:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = results_base / f"analysis_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  ✓ Timestamped archive will be created: {archive_dir}")
    else:
        archive_dir = None
        print(f"\n  ⊘ No archive - results only in 'latest' folder")
    
    # Verify dataset exists
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    # Check if dataset matches validation mode
    if validation_mode == "cross_pcb" and not build_dataset_flag:
        import h5py
        with h5py.File(dataset_file, 'r') as f:
            val_components = len(f['metadata/val_component_indices'][:])
            if val_components > 0:
                print(f"\n⚠️  WARNING: HBridge dataset has {val_components} validation components")
                print(f"   For cross-PCB mode, HBridge should have 0 validation components")
                print(f"   (all components used for training, validation on LoadShedding)")
                print(f"\n   Current dataset needs to be rebuilt with --component_val_split 0.0")
                print(f"\n   [1] Continue anyway (will ignore validation components)")
                print(f"   [2] Exit and rebuild dataset")
                
                choice = input("\nSelect option [1-2] (default: 1): ").strip()
                if choice == '2':
                    print("\n   Please run: python build_dataset.py HBridge --component_val_split 0.0")
                    print("   Then re-run training with --validation_mode cross_pcb")
                    return
                else:
                    print(f"\n   ⚠️  Continuing with existing dataset (not ideal for cross-PCB)")
    
    # Load and display dataset info
    load_dataset_info(dataset_file)
    
    # Training hyperparameter prompts
    print("\n" + "="*80)
    print("TRAINING HYPERPARAMETERS")
    print("="*80)
    
    # Epochs
    print("\nNumber of training epochs:")
    print("  Tip: More epochs = better learning but longer training time")
    print("  Recommendation: 5-10 epochs for initial testing, 20-50 for production")
    epochs_input = input("\nEnter number of epochs (default: 5): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() and int(epochs_input) > 0 else 5
    print(f"  Using {epochs} epochs")
    
    # Batch size (with confirmation loop)
    batch_confirmed = False
    while not batch_confirmed:
        print("\nBatch size (samples processed together):")
        print("  ⚠️  CPU WARNING: Batch sizes > 8 may cause memory exhaustion and hanging")
        print("  💡 RECOMMENDATION: Use batch_size=8 for CPU training (safest)")
        print("  [1] 8  (lowest memory, slower) ← RECOMMENDED FOR CPU")
        print("  [2] 16 (balanced, may hang on CPU)")
        print("  [3] 32 (higher memory, faster, likely to hang on CPU)")
        print("  [4] 64 (highest memory, fastest, very likely to hang on CPU)")
        batch_choice = input("\nSelect option [1-4] (default: 1): ").strip()
        batch_size_map = {'1': 8, '2': 16, '3': 32, '4': 64}
        batch_size = batch_size_map.get(batch_choice, 8)
        
        # Secondary warning for batch_size > 8
        if batch_size > 8:
            print(f"\n  ⚠️  WARNING: You selected batch_size={batch_size}")
            print(f"  This may exhaust CPU memory and cause the training to hang indefinitely.")
            print(f"  Previous tests showed batch_size=16 and 32 both caused hanging.")
            print(f"  Recommendation: Use batch_size=8 instead.")
        
        # Confirm selection
        print(f"\n  ✓ You selected: batch size {batch_size}")
        confirm = input("  Is this correct? [1] Yes, [2] No, let me try again (default: 1): ").strip()
        
        if confirm == '2':
            print("\n  ↻ Restarting batch size selection...\n")
            batch_confirmed = False
        else:
            print(f"  ✓ Confirmed: batch size {batch_size}")
            batch_confirmed = True
    
    # Ask if user wants to generate plots (UNIFIED TOGGLE)
    print("\n" + "="*80)
    print("PLOT GENERATION OPTIONS")
    print("="*80)
    print("\n  [1] Generate all visualization plots (training curves, scatter plot, spatial comparisons)")
    print("  [2] Skip plots (saves ~1-2 minutes for quick test runs)")
    print("\n  Note: Metrics files will be saved regardless of choice")
    
    plot_choice = input("\nSelect option [1-2] (default: 1): ").strip()
    generate_plots = (plot_choice != '2')
    
    if not generate_plots:
        print("\n⊘ Plot generation disabled - metrics will still be saved\n")
    
    # Train model
    result = train_model(
        dataset_path=dataset_file,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        generate_plots=generate_plots
    )
    
    # Handle case where training was cancelled
    if result is None or len(result) != 3:
        print("\nTraining cancelled or failed.")
        return
    
    trainer, history, sampling_params = result
    
    # Evaluate model (pass sampling_params to use same frames as training)
    r2, rmse, mae = evaluate_model(trainer, dataset_file, output_dir, sampling_params, generate_plots=generate_plots)
    
    # Check if evaluation succeeded
    if r2 is None:
        print("\n" + "="*80)
        print("⚠️  TRAINING COMPLETE (EVALUATION FAILED)")
        print("="*80)
        print("\nModel trained successfully but evaluation failed.")
        print("This likely means ROI masks are empty or misaligned.")
        print(f"\nModel saved to: {output_dir}")
        print()
        return
    
    # Generate spatial comparison plots (conditional)
    if generate_plots:
        generate_spatial_comparisons(trainer, dataset_file, output_dir, sampling_params, n_samples=15)
    else:
        print("\n⊘ Skipping spatial comparison plots (already disabled)")
    
    # Copy results to archive if requested
    if create_archive:
        print(f"\nCopying results to timestamped archive...")
        import shutil
        shutil.copytree(latest_dir, archive_dir, dirs_exist_ok=True)
        print(f"  ✓ Archive created: {archive_dir}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nFinal Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}°C")
    print(f"  MAE: {mae:.2f}°C")
    print(f"\nResults saved to: {output_dir}")
    if create_archive:
        print(f"Archive saved to: {archive_dir}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train U-Net thermal prediction model")
    parser.add_argument('--board', type=str, choices=['HBridge', 'LoadShedding'], 
                        default='HBridge',
                        help='Board to train on (default: HBridge)')
    parser.add_argument('--validation_mode', type=str, 
                        choices=['component_holdout', 'cross_pcb'],
                        default='component_holdout',
                        help='Validation mode: component_holdout (validate on held-out components) or cross_pcb (train HBridge, validate LoadShedding)')
    parser.add_argument('--build_dataset', action='store_true',
                        help='Build dataset before training')
    parser.add_argument('--component_val_split', type=float, default=0.2,
                        help='Fraction of components for validation (component_holdout mode only, default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        # Run training with command-line arguments (or None to trigger interactive prompts)
        main(
            board_name=args.board if args.board != 'HBridge' else None,  # None triggers prompt
            validation_mode=args.validation_mode if args.validation_mode != 'component_holdout' else None,  # None triggers prompt
            build_dataset_flag=args.build_dataset,
            component_val_split=args.component_val_split
        )
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
