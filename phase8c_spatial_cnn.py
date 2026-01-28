#!/usr/bin/env python3
import h5py
import numpy as np
"""
===============================================================================
PHASE 8C SPATIAL CNN - U-Net Thermal Field Reconstruction
===============================================================================
U-Net CNN for spatial thermal field prediction using FLIR input images.

Purpose:
    - Train U-Net to predict embedded component temperatures from FLIR+thermistor
    - Use sparse thermistor ground truth at ROI locations
    - Custom masked loss: penalize only at component ROI pixels
    - Component-level validation: held-out components test thermal learning

Model Architecture (U-Net):
    Encoder:
        - Conv(32) -> Conv(32) -> MaxPool
        - Conv(64) -> Conv(64) -> MaxPool
        - Conv(128) -> Conv(128) -> MaxPool
        - Conv(256) -> Conv(256) -> MaxPool
    
    Bottleneck:
        - Conv(512) -> Conv(512)
    
    Decoder:
        - UpConv(256) + Concat(encoder4) -> Conv(256) -> Conv(256)
        - UpConv(128) + Concat(encoder3) -> Conv(128) -> Conv(128)
        - UpConv(64) + Concat(encoder2) -> Conv(64) -> Conv(64)
        - UpConv(32) + Concat(encoder1) -> Conv(32) -> Conv(32)
    
    Output:
        - Conv(1, activation='linear') -> Predicted thermal map

Training Strategy:
    - Input Channel 0: FLIR thermal image [H, W] (surface temps for all pixels)
    - Input Channel 1: Thermistor temps [H, W, sparse] (embedded temps at ROI pixels)
    - Output: Predicted embedded temperature field [H, W, 1]
    - Loss: Masked MSE (penalize only at ROI locations)
    - Validation: Component-level holdout (20% of components excluded from input)

Component-Level Validation (NEW):
    - Training: Uses 80% of components in input Channel 1
    - Validation: Uses ONLY training components in input, predicts held-out components
    - Tests if model learns thermal coupling vs copying thermistor inputs
    - All timesteps used (no temporal split) - validation is on different components

Created: January 12, 2026
Updated: January 23, 2026 - Component-level validation
===============================================================================
"""

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import gc

# Configure TensorFlow for CPU memory efficiency
# Prevent TensorFlow from pre-allocating all available memory
tf.config.threading.set_intra_op_parallelism_threads(4)  # Limit parallel ops within single operation
tf.config.threading.set_inter_op_parallelism_threads(4)  # Limit parallel independent operations


class MaskedMSELoss(keras.losses.Loss):
    """
    Custom masked MSE loss for sparse ground truth.
    
    Penalizes only at ROI locations where thermistor data exists.
    """
    
    def __init__(self, name='masked_mse', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        """
        Compute masked MSE.
        
        Args:
            y_true: Ground truth [batch, H, W, 1]
            y_pred: Predictions [batch, H, W, 1]
        
        Returns:
            Scalar loss value
        """
        # Mask: non-zero values in y_true indicate ROI locations
        mask = tf.cast(tf.not_equal(y_true, 0.0), dtype=tf.float32)
        
        # Compute squared error only at masked locations
        squared_error = tf.square(y_true - y_pred) * mask
        
        # Average over valid pixels
        n_valid = tf.reduce_sum(mask) + 1e-8  # Avoid division by zero
        loss = tf.reduce_sum(squared_error) / n_valid
        
        return loss


def build_unet(input_shape: Tuple[int, int, int] = (480, 640, 1)) -> keras.Model:
    """
    Build U-Net architecture for temporal-spatial thermal field reconstruction.
    
    Args:
        input_shape: FLIR input shape (height, width, channels)
                    Channel 0: FLIR surface temperatures (all pixels)
    
    Returns:
        Keras U-Net model with dual inputs (FLIR image + time scalar)
    """
    # Dual inputs: FLIR image and time scalar
    flir_input = keras.Input(shape=input_shape, name='flir_input')
    time_input = keras.Input(shape=(1,), name='time_input')
    
    # Encoder (Contracting Path) - processes FLIR only
    # Block 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(flir_input)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    # Block 4
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    # Bottleneck - spatial feature extraction
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    # Time injection via FC layers
    # Flatten spatial features, concatenate with time, apply FC transformation, reshape
    bottleneck_shape = conv5.shape[1:]  # e.g., (30, 40, 512)
    flat = layers.Flatten()(conv5)
    combined = layers.Concatenate()([flat, time_input])
    fc1 = layers.Dense(512, activation='relu', name='fc_time_1')(combined)
    fc2 = layers.Dense(int(np.prod(bottleneck_shape)), activation='relu', name='fc_time_2')(fc1)
    reshaped = layers.Reshape(bottleneck_shape)(fc2)
    
    # Decoder (Expanding Path) - uses time-adjusted features
    # Block 6
    up6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(reshaped)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    # Block 8
    up8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    # Block 9
    up9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    # Output layer (linear activation for temperature regression)
    outputs = layers.Conv2D(1, (1, 1), activation='linear', padding='same')(conv9)
    
    model = keras.Model(inputs=[flir_input, time_input], outputs=outputs, name='UNet_Temporal')
    
    return model


class SpatialCNNTrainer:

    class HDF5DataGenerator(keras.utils.Sequence):
        """
        Keras data generator for memory-efficient batch loading from HDF5.
        Supports hybrid sampling: dense for early data, sparse for steady-state.
        Supports component-level train/val split: model predicts validation components from FLIR only.
        """
        def __init__(self, h5_path, split='train', batch_size=8, 
                     dense_limit_time=4500, dense_step_time=15, sparse_step_time=300, 
                     shuffle=True, seed=42, use_all_timesteps=False, **kwargs):
            """
            Args:
                h5_path: Path to HDF5 file
                split: 'train' or 'val'
                batch_size: Batch size
                dense_limit_time: Time cutoff for dense sampling in seconds (e.g., 4500s = first 300 FLIR frames)
                dense_step_time: Sampling interval for dense region in seconds (e.g., 15s = match FLIR rate)
                sparse_step_time: Sampling interval for sparse region in seconds (e.g., 300s = every 20 FLIR frames)
                shuffle: Whether to shuffle batches each epoch (recommended for training)
                seed: Random seed for reproducibility
                use_all_timesteps: If True, use ALL timesteps (no hybrid sampling). Overrides dense/sparse parameters.
                **kwargs: Additional arguments for Keras Sequence (workers, use_multiprocessing, etc.)
            """
            super().__init__(**kwargs)
            self.h5_path = h5_path
            self.split = split
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed
            self.use_all_timesteps = use_all_timesteps
            self.rng = np.random.default_rng(seed)
            self.h5 = h5py.File(h5_path, 'r')
            self.n_samples = self.h5['timestamps'].shape[0]
            
            # Load component train/val split if available (read on-demand to avoid multiplying across workers)
            if 'metadata/train_component_indices' in self.h5:
                # Store references, don't load arrays (each worker would duplicate)
                self._train_component_indices = None
                self._val_component_indices = None
            else:
                # No split defined - use all components for both train/val
                n_components = self.h5['sand_temps'].shape[1]
                self._train_component_indices = np.arange(n_components)
                self._val_component_indices = np.array([], dtype=int)
            
            # Get timestamp step size WITHOUT loading full array (use first 2 values only)
            timestamp_0 = float(self.h5['timestamps'][0])
            timestamp_1 = float(self.h5['timestamps'][1])
            time_step = timestamp_1 - timestamp_0
            
            # Compute normalization statistics from training data for consistent scaling
            # Use simple min-max normalization to [0, 1] range
            if split == 'train':
                # MEMORY FIX: Check if stats are cached in HDF5 metadata first
                if 'metadata/flir_min' in self.h5:
                    # Use cached stats to avoid loading entire dataset
                    self.flir_min = float(self.h5['metadata/flir_min'][()])
                    self.flir_max = float(self.h5['metadata/flir_max'][()])
                    self.sand_min = float(self.h5['metadata/sand_min'][()])
                    self.sand_max = float(self.h5['metadata/sand_max'][()])
                    self.time_min = float(self.h5['metadata/time_min'][()])
                    self.time_max = float(self.h5['metadata/time_max'][()])
                else:
                    # Fallback: Compute incrementally to avoid OOM (chunk-wise processing)
                    # Process FLIR frames in chunks
                    n_frames = self.h5['flir_frames'].shape[0]
                    chunk_size = min(100, n_frames)
                    flir_min, flir_max = float('inf'), float('-inf')
                    for i in range(0, n_frames, chunk_size):
                        chunk = self.h5['flir_frames'][i:i+chunk_size]
                        flir_min = min(flir_min, float(np.min(chunk)))
                        flir_max = max(flir_max, float(np.max(chunk)))
                    
                    # Process sand temps in chunks
                    sand_min, sand_max = float('inf'), float('-inf')
                    for i in range(0, self.n_samples, chunk_size):
                        chunk = self.h5['sand_temps'][i:i+chunk_size]
                        sand_min = min(sand_min, float(np.min(chunk)))
                        sand_max = max(sand_max, float(np.max(chunk)))
                    
                    self.flir_min = flir_min
                    self.flir_max = flir_max
                    self.sand_min = sand_min
                    self.sand_max = sand_max
                    # Get time min/max from HDF5 attributes without loading array
                    self.time_min = float(self.h5['timestamps'][0])
                    self.time_max = float(self.h5['timestamps'][self.n_samples - 1])
            else:
                # Validation uses same stats as training (prevent data leakage)
                # These will be overridden by set_normalization_stats() called from trainer
                self.flir_min = 20.0  # Placeholder
                self.flir_max = 80.0
                self.sand_min = 20.0
                self.sand_max = 150.0
                self.time_min = 0.0  # Placeholder
                self.time_max = 10000.0
            
            # Convert time-based parameters to timestamp indices
            dense_limit = int(dense_limit_time / time_step)
            dense_step = max(1, int(dense_step_time / time_step))
            sparse_step = max(1, int(sparse_step_time / time_step))
            
            # Choose sampling strategy
            if use_all_timesteps:
                # Use ALL timesteps for both train and val
                all_indices = np.arange(0, self.n_samples)
                
                # Shuffle and split 80/20
                rng_split = np.random.default_rng(seed)
                rng_split.shuffle(all_indices)
                split_idx = int(len(all_indices) * 0.8)
                
                if split == 'train':
                    self.indices = all_indices[:split_idx]
                else:
                    self.indices = all_indices[split_idx:]
            else:
                # IMPORTANT: Apply hybrid sampling to ALL data, then stratified 80/20 split
                # This ensures both train and val have PROPORTIONAL transient (dense) AND steady-state (sparse) samples
                # Stratified split maintains the same dense/sparse ratio in train and val
                all_indices = np.arange(0, self.n_samples)
                dense_indices = all_indices[all_indices < dense_limit][::dense_step]
                sparse_indices = all_indices[all_indices >= dense_limit][::sparse_step]
                
                # CRITICAL: Shuffle indices BEFORE splitting to avoid temporal bias
                # Use deterministic shuffle based on seed for reproducibility
                rng_split = np.random.default_rng(seed)
                rng_split.shuffle(dense_indices)
                rng_split.shuffle(sparse_indices)
                
                # Split dense samples 80/20 (now randomized, not sequential blocks)
                dense_split_idx = int(len(dense_indices) * 0.8)
                # Split sparse samples 80/20 (now randomized, not sequential blocks)
                sparse_split_idx = int(len(sparse_indices) * 0.8)
                
                if split == 'train':
                    train_dense = dense_indices[:dense_split_idx]
                    train_sparse = sparse_indices[:sparse_split_idx]
                    self.indices = np.concatenate([train_dense, train_sparse])
                else:
                    val_dense = dense_indices[dense_split_idx:]
                    val_sparse = sparse_indices[sparse_split_idx:]
                    self.indices = np.concatenate([val_dense, val_sparse])
            
            # Get dataset dimensions (shape queries don't load data)
            self.H = self.h5['flir_frames'].shape[1]
            self.W = self.h5['flir_frames'].shape[2]
            self.n_components = self.h5['sand_temps'].shape[1]
            
            # MEMORY FIX: Don't load entire arrays - access on-demand from HDF5
            # self.frame_indices and self.roi_masks will be accessed directly from h5 file
            # This saves ~27MB per generator instance (critical with multiprocessing)
        
        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))
        def __getitem__(self, idx):
            batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
            X_flir = np.zeros((len(batch_indices), self.H, self.W, 1), dtype=np.float32)
            X_time = np.zeros((len(batch_indices), 1), dtype=np.float32)
            y = np.zeros((len(batch_indices), self.H, self.W, 1), dtype=np.float32)
            
            for i, t in enumerate(batch_indices):
                # Access frame_indices from HDF5 on-demand (single scalar read)
                frame_idx = self.h5['frame_indices'][t]
                
                # FLIR channel (normalized to [0, 1])
                flir_frame = self.h5['flir_frames'][frame_idx]
                X_flir[i, :, :, 0] = (flir_frame - self.flir_min) / (self.flir_max - self.flir_min + 1e-8)
                
                # Time channel (normalized to [0, 1])
                timestamp = self.h5['timestamps'][t]
                X_time[i, 0] = (timestamp - self.time_min) / (self.time_max - self.time_min + 1e-8)
                
                # Target: For train use train components, for val use val components
                if self.split == 'train':
                    target_components = self.train_component_indices
                else:
                    target_components = self.val_component_indices
                
                for comp_idx in target_components:
                    # Access ROI mask from HDF5 on-demand (single mask read per component)
                    roi_mask = self.h5['roi_masks'][comp_idx]
                    comp_temp = self.h5['sand_temps'][t, comp_idx]
                    comp_temp_norm = (comp_temp - self.sand_min) / (self.sand_max - self.sand_min + 1e-8)
                    y[i, :, :, 0][roi_mask == 1] = comp_temp_norm
            
            return {'flir_input': X_flir, 'time_input': X_time}, y
        
        def on_epoch_end(self):
            """Shuffle indices at the end of each epoch for better generalization."""
            if self.shuffle:
                self.rng.shuffle(self.indices)
            
            # MEMORY CLEANUP: Force garbage collection between epochs to prevent accumulation
            import gc
            gc.collect()
        
        @property
        def train_component_indices(self):
            """Lazy load train component indices on first access."""
            if self._train_component_indices is None:
                self._train_component_indices = self.h5['metadata/train_component_indices'][:]
            return self._train_component_indices
        
        @property
        def val_component_indices(self):
            """Lazy load validation component indices on first access."""
            if self._val_component_indices is None:
                self._val_component_indices = self.h5['metadata/val_component_indices'][:]
            return self._val_component_indices
        
        def set_normalization_stats(self, flir_min, flir_max, sand_min, sand_max, time_min, time_max):
            """Set normalization statistics (for validation generator)."""
            self.flir_min = flir_min
            self.flir_max = flir_max
            self.sand_min = sand_min
            self.sand_max = sand_max
            self.time_min = time_min
            self.time_max = time_max
        
        def get_normalization_stats(self):
            """Get normalization statistics."""
            return {
                'flir_min': float(self.flir_min),
                'flir_max': float(self.flir_max),
                'sand_min': float(self.sand_min),
                'sand_max': float(self.sand_max),
                'time_min': float(self.time_min),
                'time_max': float(self.time_max)
            }
        
        def __del__(self):
            """Cleanup: Close HDF5 file when generator is deleted."""
            try:
                if hasattr(self, 'h5') and self.h5 is not None:
                    self.h5.close()
            except Exception:
                pass
    """
    Trainer for spatial thermal CNN (U-Net).
    
    Handles data loading, training, evaluation, and prediction.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize CNN trainer.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.model = None
        self.history = None
        self.dataset = None
        self.normalization_stats = None  # Store for denormalization
    
    def load_dataset(self, h5_file: str, lightweight: bool = False):
        """
        Load HDF5 training dataset.
        
        Args:
            h5_file: Path to HDF5 file from cnn_data_preprocessor
            lightweight: If True, only load metadata (for generator mode)
        """
        if self.verbose:
            print("\n" + "="*80)
            if lightweight:
                print("  LOADING DATASET METADATA (lightweight mode)")
            else:
                print("  LOADING TRAINING DATASET")
            print("="*80)
        
        # Store HDF5 path for later access
        self.h5_path = h5_file
        
        with h5py.File(h5_file, 'r') as f:
            if lightweight:
                # Only load metadata (small arrays) - keep large data on disk for generator
                self.dataset = {
                    'flir_frames': None,  # Will be accessed on-demand by generator
                    'sand_temps': None,   # Will be accessed on-demand by generator
                    'roi_masks': None,    # Will be accessed on-demand by generator
                    'timestamps': f['timestamps'][:],  # Small array, safe to load
                    'frame_indices': f['frame_indices'][:] if 'frame_indices' in f else None,  # Small array
                    'metadata': {
                        'n_frames': f['metadata'].attrs['n_frames'],
                        'n_components': f['metadata'].attrs['n_components'],
                        'image_width': f['metadata'].attrs['image_width'],
                        'image_height': f['metadata'].attrs['image_height'],
                        'component_names': [name.decode('utf-8') for name in f['metadata/component_names'][:]],
                        'uses_frame_indices': f['metadata'].attrs.get('uses_frame_indices', False),
                        'train_component_indices': f['metadata/train_component_indices'][:] if 'metadata/train_component_indices' in f else None,
                        'val_component_indices': f['metadata/val_component_indices'][:] if 'metadata/val_component_indices' in f else None,
                        'component_val_split': f['metadata'].attrs.get('component_val_split', None),
                    }
                }
                
                if self.verbose:
                    print(f"  ✓ Metadata loaded (large arrays kept on disk)")
                    print(f"  Frames: {f['flir_frames'].shape[0]} frames × {f['flir_frames'].shape[1]}×{f['flir_frames'].shape[2]} (NOT loaded into RAM)")
                    print(f"  Sand temps: {f['sand_temps'].shape} (NOT loaded into RAM)")
                    print(f"  ROI masks: {f['roi_masks'].shape} (NOT loaded into RAM)")
            else:
                # Load full dataset into memory (for non-generator mode)
                self.dataset = {
                    'flir_frames': f['flir_frames'][:],
                    'sand_temps': f['sand_temps'][:],
                    'roi_masks': f['roi_masks'][:],
                    'timestamps': f['timestamps'][:],
                    'frame_indices': f['frame_indices'][:] if 'frame_indices' in f else None,
                    'metadata': {
                        'n_frames': f['metadata'].attrs['n_frames'],
                        'n_components': f['metadata'].attrs['n_components'],
                        'image_width': f['metadata'].attrs['image_width'],
                        'image_height': f['metadata'].attrs['image_height'],
                        'component_names': [name.decode('utf-8') for name in f['metadata/component_names'][:]],
                        'uses_frame_indices': f['metadata'].attrs.get('uses_frame_indices', False),
                        'train_component_indices': f['metadata/train_component_indices'][:] if 'metadata/train_component_indices' in f else None,
                        'val_component_indices': f['metadata/val_component_indices'][:] if 'metadata/val_component_indices' in f else None,
                        'component_val_split': f['metadata'].attrs.get('component_val_split', None),
                    }
                }
                
                if self.verbose:
                    print(f"  ✓ Dataset loaded into memory")
                    print(f"  Frames: {self.dataset['flir_frames'].shape}")
                    print(f"  Sand temps: {self.dataset['sand_temps'].shape}")
                    print(f"  ROI masks: {self.dataset['roi_masks'].shape}")
            
            # Display component split info if available
            if self.dataset['metadata']['train_component_indices'] is not None:
                n_train = len(self.dataset['metadata']['train_component_indices'])
                n_val = len(self.dataset['metadata']['val_component_indices'])
                print(f"\n  Component Split:")
                print(f"    Training components: {n_train} ({n_train/(n_train+n_val)*100:.0f}%)")
                train_names = [self.dataset['metadata']['component_names'][i] for i in self.dataset['metadata']['train_component_indices']]
                print(f"      {', '.join(train_names[:5])}{'...' if len(train_names) > 5 else ''}")
                print(f"    Validation components: {n_val} ({n_val/(n_train+n_val)*100:.0f}%)")
                val_names = [self.dataset['metadata']['component_names'][i] for i in self.dataset['metadata']['val_component_indices']]
                print(f"      {', '.join(val_names)}")
            else:
                print(f"\n  ⚠️  No component split found in dataset - will use all components")
            print()
    
    def prepare_training_data(self, val_split: float = 0.2, 
                             temporal_split: bool = False,
                             component_split: bool = True) -> Tuple:
        """
        Prepare training and validation data from FLIR thermal images.
        
        Args:
            val_split: Validation split fraction
            temporal_split: If True, use temporal holdout (last frames for validation) - LEGACY, not recommended
            component_split: If True, use component-level holdout (validation components excluded from input)
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if self.verbose:
            print("="*80)
            print("  PREPARING TRAINING DATA (FLIR Input Only)")
            print("="*80)
        
        # Check if dataset uses frame_indices mapping (for extended datasets)
        uses_frame_indices = self.dataset['metadata'].get('uses_frame_indices', False)
        
        # Prepare 2-channel input
        n_timestamps = len(self.dataset['timestamps'])  # Number of temporal samples
        n_frames = self.dataset['flir_frames'].shape[0]  # Actual stored frames
        H = self.dataset['metadata']['image_height']
        W = self.dataset['metadata']['image_width']
        
        if uses_frame_indices and self.verbose:
            print(f"  Using frame_indices mapping: {n_timestamps} timestamps → {n_frames} stored frames")
        
        # Initialize dual inputs: FLIR [n_timestamps, H, W, 1] and time [n_timestamps, 1]
        X_flir = np.zeros((n_timestamps, H, W, 1), dtype=np.float32)
        X_time = np.zeros((n_timestamps, 1), dtype=np.float32)
        
        # FLIR surface temperatures (only spatial input channel)
        if uses_frame_indices:
            # Use frame_indices to map timestamps to frames, using custom indices sampling the main dataset 
            frame_indices = self.dataset['frame_indices']
            X_flir[:, :, :, 0] = self.dataset['flir_frames'][frame_indices]
        else:
            # Direct mapping (one frame per timestamp), or uses them all (uses complete data set, no sampling)
            X_flir[:, :, :, 0] = self.dataset['flir_frames']
        
        # Time values (temporal input)
        timestamps = self.dataset['timestamps']
        X_time[:, 0] = timestamps
        
        # Load component split if available
        if component_split and 'train_component_indices' in self.dataset['metadata']:
            train_comp_indices = self.dataset['metadata']['train_component_indices']
            val_comp_indices = self.dataset['metadata']['val_component_indices']
            if self.verbose:
                print(f"  Using component-level split: {len(train_comp_indices)} train, {len(val_comp_indices)} val")
        else:
            # No component split - use all components
            train_comp_indices = range(self.dataset['metadata']['n_components'])
            val_comp_indices = []
            if self.verbose and component_split:
                print("  Warning: Component split requested but no split defined in dataset")
        
        # Output: Full thermal map target
        # Build target: [n_timestamps, H, W] with thermistor temps at ROI pixels
        if self.verbose:
            print("  Building ground truth target (sparse ROI)...")
        
        y_train = np.zeros((n_timestamps, H, W), dtype=np.float32)
        y_val = np.zeros((n_timestamps, H, W), dtype=np.float32)
        
        # Fill in ground truth at ROI locations (vectorized)
        # Training target: use TRAIN components
        for comp_idx in train_comp_indices:
            roi_mask = self.dataset['roi_masks'][comp_idx]  # [H, W]
            comp_temps = self.dataset['sand_temps'][:, comp_idx]  # [n_timestamps]
            
            # Broadcast temperatures to all pixels in ROI (vectorized)
            y_train[:, roi_mask == 1] = comp_temps[:, np.newaxis]
        
        # Validation target: use VAL components (if component_split enabled)
        if component_split and len(val_comp_indices) > 0:
            for comp_idx in val_comp_indices:
                roi_mask = self.dataset['roi_masks'][comp_idx]  # [H, W]
                comp_temps = self.dataset['sand_temps'][:, comp_idx]  # [n_timestamps]
                
                # Broadcast temperatures to all pixels in ROI (vectorized)
                y_val[:, roi_mask == 1] = comp_temps[:, np.newaxis]
        else:
            # No component split - use all components for validation target
            for comp_idx in range(self.dataset['metadata']['n_components']):
                roi_mask = self.dataset['roi_masks'][comp_idx]  # [H, W]
                comp_temps = self.dataset['sand_temps'][:, comp_idx]  # [n_timestamps]
                
                # Broadcast temperatures to all pixels in ROI (vectorized)
                y_val[:, roi_mask == 1] = comp_temps[:, np.newaxis]
        
        # Add channel dimension: [n_timestamps, H, W, 1]
        y_train = y_train[..., np.newaxis]
        y_val = y_val[..., np.newaxis]
        
        # Compute normalization statistics from training data
        # Use all data to get global min/max for consistent scaling
        flir_min = float(np.min(self.dataset['flir_frames']))
        flir_max = float(np.max(self.dataset['flir_frames']))
        sand_min = float(np.min(self.dataset['sand_temps']))
        sand_max = float(np.max(self.dataset['sand_temps']))
        timestamps = self.dataset['timestamps']
        time_min = float(timestamps.min())
        time_max = float(timestamps.max())
        
        # Store normalization stats for evaluation/inference
        self.normalization_stats = {
            'flir_min': flir_min,
            'flir_max': flir_max,
            'sand_min': sand_min,
            'sand_max': sand_max,
            'time_min': time_min,
            'time_max': time_max
        }
        
        # Normalize inputs to [0, 1]
        if self.verbose:
            print(f"  Normalizing data to [0, 1]:")
            print(f"    FLIR: [{flir_min:.2f}, {flir_max:.2f}]°C → [0, 1]")
            print(f"    Time: [{time_min:.2f}, {time_max:.2f}]s → [0, 1]")
            print(f"    Targets: [{sand_min:.2f}, {sand_max:.2f}]°C → [0, 1]")
        
        X_flir[:, :, :, 0] = (X_flir[:, :, :, 0] - flir_min) / (flir_max - flir_min + 1e-8)
        X_time[:, 0] = (X_time[:, 0] - time_min) / (time_max - time_min + 1e-8)
        y_train = (y_train - sand_min) / (sand_max - sand_min + 1e-8)
        y_val = (y_val - sand_min) / (sand_max - sand_min + 1e-8)
        
        if self.verbose:
            print(f"  ✓ Data preparation complete")
        
        # Split data
        if component_split and len(val_comp_indices) > 0:
            # Component-level split: ALL timestamps for both train/val
            # Validation is on DIFFERENT COMPONENTS, not different times
            if self.verbose:
                print(f"  Using component-level validation split")
                print(f"  Train: All {n_timestamps} timesteps, {len(train_comp_indices)} components")
                print(f"  Val: All {n_timestamps} timesteps, {len(val_comp_indices)} components")
            X_train_out = {'flir_input': X_flir, 'time_input': X_time}
            X_val_out = {'flir_input': X_flir, 'time_input': X_time}  # Same inputs for both (but different targets!)
            y_train_out = y_train
            y_val_out = y_val
        elif temporal_split:
            # Temporal holdout: last 20% of timestamps for validation (LEGACY)
            if self.verbose:
                print(f"  Using temporal validation split (LEGACY)")
            split_idx = int(n_timestamps * (1 - val_split))
            X_train_out = {'flir_input': X_flir[:split_idx], 'time_input': X_time[:split_idx]}
            X_val_out = {'flir_input': X_flir[split_idx:], 'time_input': X_time[split_idx:]}
            y_train_out, y_val_out = y_train[:split_idx], y_val[split_idx:]
        else:
            # Random split (LEGACY)
            if self.verbose:
                print(f"  Using random validation split (LEGACY)")
            indices = np.arange(n_timestamps)
            np.random.shuffle(indices)
            split_idx = int(n_timestamps * (1 - val_split))
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            X_train_out = {'flir_input': X_flir[train_idx], 'time_input': X_time[train_idx]}
            X_val_out = {'flir_input': X_flir[val_idx], 'time_input': X_time[val_idx]}
            y_train_out = y_train[train_idx]
            y_val_out = y_val[val_idx]
        
        if self.verbose:
            print(f"  ✓ Data prepared")
            if isinstance(X_train_out, dict):
                print(f"  Training set: FLIR {X_train_out['flir_input'].shape}, Time {X_train_out['time_input'].shape}")
                print(f"  Validation set: FLIR {X_val_out['flir_input'].shape}, Time {X_val_out['time_input'].shape}")
            print(f"  Input: FLIR surface thermal image [H, W, 1] + time scalar [1]")
            print(f"  Output: Component temperatures at ROI locations [H, W, 1]")
            if component_split and len(val_comp_indices) > 0:
                print(f"  Split type: Component-level ({len(train_comp_indices)} train, {len(val_comp_indices)} val)\n")
            elif temporal_split:
                print(f"  Split type: Temporal (LEGACY)\n")
            else:
                print(f"  Split type: Random (LEGACY)\n")
        
        return X_train_out, y_train_out, X_val_out, y_val_out
    
    def build_model(self, learning_rate: float = 0.001):
        """
        Build and compile U-Net model.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.verbose:
            print("="*80)
            print("  BUILDING U-NET MODEL")
            print("="*80)
        
        H = self.dataset['metadata']['image_height']
        W = self.dataset['metadata']['image_width']
        
        self.model = build_unet(input_shape=(H, W, 1))
        
        # Compile with custom masked loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MaskedMSELoss(),
            metrics=['mae']
        )
        
        if self.verbose:
            print(f"  ✓ Model built")
            print(f"  Input shape: ({H}, {W}, 1) - FLIR only")
            print(f"  Total parameters: {self.model.count_params():,}")
            print(f"  Optimizer: Adam (lr={learning_rate})")
            print(f"  Loss: Masked MSE\n")
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs: int = 100, batch_size: int = 8,
             early_stopping_patience: int = 15,
             use_generator: bool = False,
             h5_path: str = None,
             dense_limit_time: int = 4500, dense_step_time: int = 15, sparse_step_time: int = 300,
             use_all_timesteps: bool = False):
        """
        Train U-Net model from FLIR thermal images only.
        
        Args:
            X_train: Training inputs [n_train, H, W, 1] - FLIR only
            y_train: Training targets [n_train, H, W, 1]
            X_val: Validation inputs [n_val, H, W, 1] - FLIR only
            y_val: Validation targets [n_val, H, W, 1]
            epochs: Maximum training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            use_generator: If True, use memory-efficient generator
            h5_path: Path to HDF5 file (required if use_generator)
            dense_limit_time: Time cutoff for dense sampling in seconds (4500s = first 300 FLIR frames)
            dense_step_time: Sampling interval for dense region in seconds (15s = match FLIR rate)
            sparse_step_time: Sampling interval for sparse region in seconds (300s = every 20 FLIR frames)
            use_all_timesteps: If True, use ALL timesteps (no hybrid sampling)
        """
        if self.verbose:
            print("="*80)
            print("  TRAINING U-NET MODEL")
            print("="*80)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        if use_generator:
            if h5_path is None:
                raise ValueError("h5_path must be provided when use_generator=True")
            if use_all_timesteps:
                print("\n⚡ Using memory-efficient data generator with ALL timesteps!")
            else:
                print("\n⚡ Using memory-efficient data generator for training!")
                print(f"  Hybrid sampling: dense_limit={dense_limit_time}s, dense_step={dense_step_time}s, sparse_step={sparse_step_time}s")
            train_gen = self.HDF5DataGenerator(h5_path, split='train', batch_size=batch_size,
                                                dense_limit_time=dense_limit_time, dense_step_time=dense_step_time, 
                                                sparse_step_time=sparse_step_time, shuffle=True, seed=42,
                                                use_all_timesteps=use_all_timesteps)
            val_gen = self.HDF5DataGenerator(h5_path, split='val', batch_size=batch_size,
                                              dense_limit_time=dense_limit_time, dense_step_time=dense_step_time, 
                                              sparse_step_time=sparse_step_time, shuffle=False, seed=42,
                                              use_all_timesteps=use_all_timesteps)
            
            # Share normalization stats from train to val (prevent data leakage)
            norm_stats = train_gen.get_normalization_stats()
            val_gen.set_normalization_stats(**norm_stats)
            
            # Store normalization stats for denormalization during evaluation/inference
            self.normalization_stats = norm_stats
            
            if self.verbose:
                print(f"  Normalization: FLIR [{norm_stats['flir_min']:.1f}, {norm_stats['flir_max']:.1f}]°C → [0, 1]")
                print(f"  Normalization: Time [{norm_stats['time_min']:.1f}, {norm_stats['time_max']:.1f}]s → [0, 1]")
                print(f"  Normalization: Thermistor [{norm_stats['sand_min']:.1f}, {norm_stats['sand_max']:.1f}]°C → [0, 1]")
            
            self.history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1 if self.verbose else 0
            )
        else:
            print("\n⚡ Using in-memory arrays for training (requires large RAM)!")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1 if self.verbose else 0
            )
        
        if self.verbose:
            print(f"\n  ✓ Training complete")
            print(f"  Final training loss: {self.history.history['loss'][-1]:.4f}")
            print(f"  Final validation loss: {self.history.history['val_loss'][-1]:.4f}\n")
    
    def denormalize_predictions(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions [0, 1] back to temperature in °C.
        
        Args:
            y_normalized: Normalized predictions [0, 1]
        
        Returns:
            Denormalized predictions in °C
        """
        if self.normalization_stats is None:
            # No normalization was applied (in-memory training)
            return y_normalized
        
        sand_min = self.normalization_stats['sand_min']
        sand_max = self.normalization_stats['sand_max']
        
        return y_normalized * (sand_max - sand_min) + sand_min
    
    def evaluate(self, X_val, y_val) -> Dict:
        """
        Evaluate model on validation set.
        
        Args:
            X_val: Validation inputs
            y_val: Validation targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.verbose:
            print("="*80)
            print("  EVALUATING MODEL")
            print("="*80)
        
        # Predict
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Denormalize predictions if normalization was used
        if self.normalization_stats is not None:
            y_pred = self.denormalize_predictions(y_pred)
            y_val = self.denormalize_predictions(y_val)
            if self.verbose:
                print(f"  ✓ Predictions denormalized to °C\n")
        
        # Extract ROI predictions vs ground truth
        roi_mask_combined = np.any(self.dataset['roi_masks'], axis=0)  # Combined ROI mask
        
        y_true_roi = []
        y_pred_roi = []
        
        for frame_idx in range(len(y_val)):
            # Get ROI pixels for this frame
            roi_pixels_true = y_val[frame_idx][roi_mask_combined == 1]
            roi_pixels_pred = y_pred[frame_idx][roi_mask_combined == 1]
            
            # Filter out zeros (non-ROI locations in sparse ground truth)
            valid_mask = roi_pixels_true != 0
            
            y_true_roi.extend(roi_pixels_true[valid_mask])
            y_pred_roi.extend(roi_pixels_pred[valid_mask])
        
        y_true_roi = np.array(y_true_roi)
        y_pred_roi = np.array(y_pred_roi)
        
        # Calculate metrics
        r2 = r2_score(y_true_roi, y_pred_roi)
        rmse = np.sqrt(mean_squared_error(y_true_roi, y_pred_roi))
        mae = mean_absolute_error(y_true_roi, y_pred_roi)
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true_roi)
        }
        
        if self.verbose:
            print(f"  ✓ Evaluation complete")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}°C")
            print(f"  MAE: {mae:.2f}°C")
            print(f"  ROI samples evaluated: {len(y_true_roi)}\n")
        
        return metrics
    
    def save_model(self, model_path: str):
        """
        Save trained model and normalization stats.
        
        Args:
            model_path: Path to save model (.keras format)
        """
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_path)
        
        # Save normalization stats alongside model
        if self.normalization_stats is not None:
            import json
            stats_path = str(output_path).replace('.keras', '_normalization.json')
            with open(stats_path, 'w') as f:
                json.dump(self.normalization_stats, f, indent=2)
            if self.verbose:
                print(f"  ✓ Model saved: {model_path}")
                print(f"  ✓ Normalization stats saved: {stats_path}\n")
        else:
            if self.verbose:
                print(f"  ✓ Model saved: {model_path}\n")
    
    def load_model(self, model_path: str):
        """
        Load trained model and normalization stats.
        
        Args:
            model_path: Path to model file (.keras)
        """
        import json
        from tensorflow import keras
        
        self.model = keras.models.load_model(model_path, custom_objects={'MaskedMSELoss': MaskedMSELoss})
        
        # Try to load normalization stats
        stats_path = model_path.replace('.keras', '_normalization.json')
        try:
            with open(stats_path, 'r') as f:
                self.normalization_stats = json.load(f)
            if self.verbose:
                print(f"  ✓ Model loaded: {model_path}")
                print(f"  ✓ Normalization stats loaded: {stats_path}\n")
        except FileNotFoundError:
            self.normalization_stats = None
            if self.verbose:
                print(f"  ✓ Model loaded: {model_path}")
                print(f"  ⚠ No normalization stats found (model may use raw temperatures)\n")
    
    def save_training_history(self, history_path: str):
        """
        Save training history.
        
        Args:
            history_path: Path to save history JSON
        """
        output_path = Path(history_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        if self.verbose:
            print(f"  ✓ Training history saved: {history_path}\n")


def main():
    """
    Example usage: Train U-Net on HBridge dataset.
    """
    print("\n" + "="*80)
    print("  PHASE 8C: SPATIAL CNN THERMAL MODELING")
    print("="*80 + "\n")
    
    # Initialize trainer
    trainer = SpatialCNNTrainer(verbose=True)
    
    # Load dataset
    trainer.load_dataset("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
    
    # Ask user FIRST about generator (before loading 31.5 GB into RAM!)
    print("\nTraining Options:")
    print("  [1] Memory-efficient generator (recommended for large datasets)")
    print("      └─ RAM usage: ~300-500 MB")
    print("  [2] In-memory training (faster but requires ~32 GB RAM)")
    print("      └─ RAM usage: ~31.5 GB")
    use_generator = input("\nUse memory-efficient data generator? (y/n) [default: y]: ").strip().lower()
    use_generator = (use_generator != 'n')
    print(f"\nData generator enabled: {use_generator}")
    
    # Ask about timestep sampling strategy
    print("\nTimestep Sampling Strategy:")
    print("  [1] Hybrid sampling (dense transient + sparse steady-state) - Efficient")
    print("  [2] All timesteps (no sampling) - Maximum data, higher RAM/compute")
    use_all_timesteps = input("\nUse all timesteps? (y/n) [default: n]: ").strip().lower()
    use_all_timesteps = (use_all_timesteps == 'y')
    print(f"\nUsing all timesteps: {use_all_timesteps}")
    
    # Only prepare in-memory data if NOT using generator
    if not use_generator:
        print("\n⚠️  Loading entire dataset into RAM (~31.5 GB)...")
        print("⚠️  This may take several minutes and could cause system slowdown.")
        proceed = input("Continue? (y/n) [default: n]: ").strip().lower()
        if proceed != 'y':
            print("Switching to memory-efficient generator instead.")
            use_generator = True
        else:
            # Prepare data for in-memory training with component split
            X_train, y_train, X_val, y_val = trainer.prepare_training_data(
                val_split=0.2,
                temporal_split=False,  # Use component split instead
                component_split=True
            )
    else:
        # Generator path - no need to prepare data
        X_train = y_train = X_val = y_val = None
    
    # Build model
    trainer.build_model(learning_rate=0.001)
    
    print("\n⚠️  RUNNING QUICK TEST WITH 5 EPOCHS")
    print("For full training, change epochs=5 to epochs=100 in main()\n")
    
    if use_generator:
        h5_path = "ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5"
        
        # Load dataset info for sampling preview
        with h5py.File(h5_path, 'r') as f:
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
            customize = input("\nCustomize sampling parameters? (y/n) [default: n]: ").strip().lower()
            
            if customize == 'y':
                print("\nEnter sampling parameters (in seconds):")
                print("  Tip: FLIR is 15s intervals. Dense should match FLIR rate, sparse can skip frames.")
                dense_limit_time = input(f"  Dense limit (cutoff time) [default: 4500s = 75 min]: ").strip()
                dense_limit_time = int(dense_limit_time) if dense_limit_time else 4500
                dense_step_time = input(f"  Dense step (sample every X seconds) [default: 15s = every FLIR frame]: ").strip()
                dense_step_time = int(dense_step_time) if dense_step_time else 15
                sparse_step_time = input(f"  Sparse step (sample every X seconds) [default: 300s = every 20 FLIR frames]: ").strip()
                sparse_step_time = int(sparse_step_time) if sparse_step_time else 300
            else:
                # Use defaults
                dense_limit_time = 4500
                dense_step_time = 15
                sparse_step_time = 300
        else:
            # All timesteps - no need for sampling parameters
            dense_limit_time = 0
            dense_step_time = 1
            sparse_step_time = 1
        
        # Calculate and preview sampling (only if NOT using all timesteps)
        if not use_all_timesteps:
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
            print(f"  Train/Val split: 80/20 within each region (randomized)")
            print(f"  Expected batches/epoch: ~{total_samples * 0.8 / 8:.0f} (batch_size=8)")
            print("-"*80)
        else:
            print("\n" + "-"*80)
            print("USING ALL TIMESTEPS:")
            print("-"*80)
            print(f"Total samples: {n_samples}")
            print(f"Train/Val split: 80/20 (randomized)")
            print(f"Expected batches/epoch: ~{n_samples * 0.8 / 8:.0f} (batch_size=8)")
            print("-"*80)
        
        proceed = input("\nProceed with these settings? (y/n) [default: y]: ").strip().lower()
        if proceed == 'n':
            print("Exiting. Re-run to adjust parameters.")
            return
        trainer.train(
            None, None, None, None,
            epochs=5,
            batch_size=8,
            early_stopping_patience=15,
            use_generator=True,
            h5_path=h5_path,
            dense_limit_time=dense_limit_time,
            dense_step_time=dense_step_time,
            sparse_step_time=sparse_step_time,
            use_all_timesteps=use_all_timesteps
        )
        # For evaluation, need to load val data from generator
        val_gen = trainer.HDF5DataGenerator(h5_path, split='val', batch_size=trainer.HDF5DataGenerator(h5_path, split='val').__len__(),
                                            dense_limit_time=dense_limit_time, dense_step_time=dense_step_time, sparse_step_time=sparse_step_time)
        X_val, y_val = val_gen[0][0], val_gen[0][1]
    else:
        # In-memory training
        trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=5,
            batch_size=8,
            early_stopping_patience=15,
            use_generator=False
        )
    
    # Evaluate
    metrics = trainer.evaluate(X_val, y_val)
    
    # Save
    trainer.save_model("ml_model/cnn_thermal_modeling/models/unet_thermal_hbridge.keras")
    trainer.save_training_history("ml_model/cnn_thermal_modeling/models/training_history_hbridge.json")
    
    print("="*80)
    print("  PHASE 8C TRAINING COMPLETE ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
