#!/usr/bin/env python3
"""
Create animated GIF/video from CNN thermal predictions.

Generates temporal animation showing:
- FLIR input frames
- CNN predictions
- Ground truth
- Prediction errors

Over time to visualize model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import argparse

def create_thermal_animation(npz_path, dataset_path, output_path, 
                            format='gif', fps=10, max_frames=None,
                            component_indices=None):
    """
    Create animated visualization of thermal predictions.
    
    Args:
        npz_path: Path to predictions_thermal_maps.npz
        dataset_path: Path to HDF5 dataset (for ground truth and ROI masks)
        output_path: Output file path (.gif or .mp4)
        format: 'gif' or 'mp4'
        fps: Frames per second
        max_frames: Maximum number of frames to include (None = all)
        component_indices: List of component IDs to highlight (None = all validation components)
    """
    print("="*80)
    print("CREATING THERMAL ANIMATION")
    print("="*80)
    
    # Load predictions
    print(f"\nLoading predictions from: {npz_path}")
    data = np.load(npz_path)
    predictions = data['predictions']
    timestamps = data['timestamps']
    
    # Handle both old format (frame_indices) and new format (timestamp_indices)
    if 'timestamp_indices' in data:
        timestamp_indices = data['timestamp_indices']
        frame_indices_map = data.get('frame_indices_map', None)
        print(f"  Loaded {len(predictions)} timestamp predictions")
    else:
        # Old format - frame-based
        frame_indices = data['frame_indices']
        timestamp_indices = frame_indices  # Fallback
        print(f"  Loaded {len(predictions)} frame predictions (old format)")
    
    n_frames = len(predictions)
    print(f"  Time range: {timestamps[0]:.1f}s - {timestamps[-1]:.1f}s")
    
    # Limit frames if requested
    if max_frames is not None and n_frames > max_frames:
        step = n_frames // max_frames
        indices = np.arange(0, n_frames, step)[:max_frames]
        predictions = predictions[indices]
        timestamps = timestamps[indices]
        timestamp_indices = timestamp_indices[indices]
        n_frames = len(predictions)
        print(f"  Reduced to {n_frames} frames (step={step})")
    
    # Load ground truth and ROI masks
    print(f"\nLoading ground truth from: {dataset_path}")
    with h5py.File(dataset_path, 'r') as f:
        # Load sand temps using timestamp indices
        sand_temps = f['sand_temps'][:][timestamp_indices]
        roi_masks = f['roi_masks'][:]
        
        # Load FLIR frames
        if 'timestamp_indices' in data and 'frame_indices_map' in data:
            # New format - load frames corresponding to timestamps
            frame_map = data['frame_indices_map']
            all_flir = f['flir_frames'][:]
            flir_frames = np.array([all_flir[frame_map[i]] for i in range(n_frames)])
        elif 'flir_frames' in data:
            # Stored in NPZ
            flir_frames = data['flir_frames']
        else:
            # Fallback - load from HDF5
            flir_frames = f['flir_frames'][:][timestamp_indices]
        
        # Get validation component indices
        if component_indices is None:
            if 'metadata/val_component_indices' in f:
                component_indices = f['metadata/val_component_indices'][:]
                print(f"  Using {len(component_indices)} validation components")
            else:
                component_indices = np.arange(roi_masks.shape[0])
                print(f"  Using all {len(component_indices)} components")
        else:
            print(f"  Using {len(component_indices)} specified components")
    
    # Build ground truth maps
    print("\nBuilding ground truth thermal maps...")
    H, W = predictions.shape[1], predictions.shape[2]
    ground_truth = np.zeros((n_frames, H, W), dtype=np.float32)
    
    for i in range(n_frames):
        for comp_idx in component_indices:
            if comp_idx < roi_masks.shape[0]:
                mask = roi_masks[comp_idx]
                ground_truth[i][mask > 0] = sand_temps[i, comp_idx]
    
    # Compute errors
    errors = predictions - ground_truth
    errors_masked = errors.copy()
    combined_mask = np.any(roi_masks[component_indices], axis=0)
    errors_masked[:, combined_mask == 0] = 0  # Only show errors at ROI locations
    
    # Set up figure
    print("\nCreating animation frames...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CNN Thermal Prediction Animation', fontsize=16, fontweight='bold')
    
    # Global colormap limits
    vmin_temp = min(flir_frames.min(), ground_truth[ground_truth > 0].min())
    vmax_temp = max(flir_frames.max(), ground_truth.max())
    vmax_error = np.abs(errors_masked[errors_masked != 0]).max()
    
    # Initialize plots
    im1 = axes[0, 0].imshow(flir_frames[0], cmap='hot', vmin=vmin_temp, vmax=vmax_temp)
    axes[0, 0].set_title('FLIR Input (Surface)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Temperature (°C)')
    
    im2 = axes[0, 1].imshow(predictions[0], cmap='hot', vmin=vmin_temp, vmax=vmax_temp)
    axes[0, 1].set_title('CNN Prediction')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='Temperature (°C)')
    
    im3 = axes[1, 0].imshow(ground_truth[0], cmap='hot', vmin=vmin_temp, vmax=vmax_temp)
    axes[1, 0].set_title('Ground Truth (Embedded)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], label='Temperature (°C)')
    
    im4 = axes[1, 1].imshow(errors_masked[0], cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error)
    axes[1, 1].set_title('Prediction Error')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], label='Error (°C)')
    
    # Time label
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        """Update animation frame."""
        im1.set_array(flir_frames[frame_idx])
        im2.set_array(predictions[frame_idx])
        im3.set_array(ground_truth[frame_idx])
        im4.set_array(errors_masked[frame_idx])
        
        time_text.set_text(f'Time: {timestamps[frame_idx]:.1f}s | Frame: {frame_idx+1}/{n_frames}')
        
        return im1, im2, im3, im4, time_text
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save animation
    print(f"\nSaving animation to: {output_path}")
    print(f"  Format: {format.upper()}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {n_frames/fps:.1f}s")
    
    if format == 'gif':
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    elif format == 'mp4':
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    plt.close()
    
    print(f"\n✓ Animation saved: {output_path}")
    file_size_mb = Path(output_path).stat().st_size / (1024*1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print()


def main():
    parser = argparse.ArgumentParser(description='Create thermal prediction animation')
    parser.add_argument('--npz', type=str, default='results/latest/predictions_thermal_maps.npz',
                       help='Path to predictions NPZ file')
    parser.add_argument('--dataset', type=str, 
                       default='datasets/HBridge_cnn_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='results/latest/thermal_animation.gif',
                       help='Output file path (.gif or .mp4)')
    parser.add_argument('--format', type=str, choices=['gif', 'mp4'], default='gif',
                       help='Output format (gif or mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to include (None = all)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    npz_path = script_dir / args.npz
    dataset_path = script_dir / args.dataset
    output_path = script_dir / args.output
    
    # Verify inputs exist
    if not npz_path.exists():
        print(f"✗ ERROR: NPZ file not found: {npz_path}")
        print(f"  Run training first to generate predictions.")
        return
    
    if not dataset_path.exists():
        print(f"✗ ERROR: Dataset not found: {dataset_path}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Detect format from extension if not specified
    if args.format is None:
        ext = output_path.suffix.lower()
        if ext == '.gif':
            format = 'gif'
        elif ext == '.mp4':
            format = 'mp4'
        else:
            format = 'gif'
    else:
        format = args.format
    
    # Create animation
    create_thermal_animation(
        npz_path=npz_path,
        dataset_path=dataset_path,
        output_path=output_path,
        format=format,
        fps=args.fps,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
