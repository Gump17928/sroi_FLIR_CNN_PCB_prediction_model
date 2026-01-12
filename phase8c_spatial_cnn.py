"""
===============================================================================
PHASE 8C SPATIAL CNN - U-Net Thermal Field Reconstruction
===============================================================================
U-Net CNN for spatial thermal field prediction using FLIR input images.

Purpose:
    - Train U-Net to reconstruct sand thermal field from FLIR input
    - Use sparse thermistor ground truth at ROI locations
    - Custom masked loss: penalize only at component ROI pixels
    - Generate full-field thermal predictions

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
    - Input: FLIR thermal image [H, W, 1]
    - Output: Predicted sand temperature field [H, W, 1]
    - Loss: Masked MSE (penalize only at ROI locations)
    - Ground truth: Thermistor temps at ROI pixels
    - Validation: 20% temporal holdout split

Created: January 12, 2026
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


class MaskedMSELoss(keras.losses.Loss):
    """
    Custom masked MSE loss for sparse ground truth.
    
    Penalizes only at ROI locations where thermistor data exists.
    """
    
    def __init__(self, name='masked_mse'):
        super().__init__(name=name)
    
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
    Build U-Net architecture for thermal field reconstruction.
    
    Args:
        input_shape: Input shape (height, width, channels)
    
    Returns:
        Compiled Keras U-Net model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
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
    
    # Bottleneck
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    # Decoder (Expanding Path)
    # Block 6
    up6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
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
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_Thermal')
    
    return model


class SpatialCNNTrainer:
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
    
    def load_dataset(self, h5_file: str):
        """
        Load HDF5 training dataset.
        
        Args:
            h5_file: Path to HDF5 file from cnn_data_preprocessor
        """
        if self.verbose:
            print("\n" + "="*80)
            print("  LOADING TRAINING DATASET")
            print("="*80)
        
        with h5py.File(h5_file, 'r') as f:
            self.dataset = {
                'flir_frames': f['flir_frames'][:],
                'sand_temps': f['sand_temps'][:],
                'roi_masks': f['roi_masks'][:],
                'timestamps': f['timestamps'][:],
                'metadata': {
                    'n_frames': f['metadata'].attrs['n_frames'],
                    'n_components': f['metadata'].attrs['n_components'],
                    'image_width': f['metadata'].attrs['image_width'],
                    'image_height': f['metadata'].attrs['image_height'],
                    'component_names': [name.decode('utf-8') for name in f['metadata/component_names'][:]],
                }
            }
        
        if self.verbose:
            print(f"  ✓ Dataset loaded")
            print(f"  Frames: {self.dataset['flir_frames'].shape}")
            print(f"  Sand temps: {self.dataset['sand_temps'].shape}")
            print(f"  ROI masks: {self.dataset['roi_masks'].shape}\n")
    
    def prepare_training_data(self, val_split: float = 0.2, 
                             temporal_split: bool = True) -> Tuple:
        """
        Prepare training and validation data.
        
        Args:
            val_split: Validation split fraction
            temporal_split: If True, use temporal holdout (last frames for validation)
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if self.verbose:
            print("="*80)
            print("  PREPARING TRAINING DATA")
            print("="*80)
        
        # Input: FLIR frames [n_frames, H, W] -> [n_frames, H, W, 1]
        X = self.dataset['flir_frames'][..., np.newaxis]
        
        # Output: Sparse ground truth thermal map at ROI locations
        # Build target: [n_frames, H, W] with thermistor temps at ROI pixels
        n_frames = self.dataset['flir_frames'].shape[0]
        H = self.dataset['metadata']['image_height']
        W = self.dataset['metadata']['image_width']
        
        y = np.zeros((n_frames, H, W), dtype=np.float32)
        
        # Fill in ground truth at ROI locations
        for comp_idx in range(self.dataset['metadata']['n_components']):
            roi_mask = self.dataset['roi_masks'][comp_idx]  # [H, W]
            
            for frame_idx in range(n_frames):
                comp_temp = self.dataset['sand_temps'][frame_idx, comp_idx]
                # Set all ROI pixels to component temperature
                y[frame_idx][roi_mask == 1] = comp_temp
        
        # Add channel dimension: [n_frames, H, W, 1]
        y = y[..., np.newaxis]
        
        # Split data
        if temporal_split:
            # Temporal holdout: last 20% of frames for validation
            split_idx = int(n_frames * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            # Random split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42
            )
        
        if self.verbose:
            print(f"  ✓ Data prepared")
            print(f"  Training set: {X_train.shape}")
            print(f"  Validation set: {X_val.shape}")
            print(f"  Split type: {'Temporal' if temporal_split else 'Random'}\n")
        
        return X_train, y_train, X_val, y_val
    
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
            print(f"  Input shape: ({H}, {W}, 1)")
            print(f"  Total parameters: {self.model.count_params():,}")
            print(f"  Optimizer: Adam (lr={learning_rate})")
            print(f"  Loss: Masked MSE\n")
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs: int = 100, batch_size: int = 8,
             early_stopping_patience: int = 15):
        """
        Train U-Net model.
        
        Args:
            X_train: Training inputs [n_train, H, W, 1]
            y_train: Training targets [n_train, H, W, 1]
            X_val: Validation inputs [n_val, H, W, 1]
            y_val: Validation targets [n_val, H, W, 1]
            epochs: Maximum training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
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
        
        # Train
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
        Save trained model.
        
        Args:
            model_path: Path to save model (.keras format)
        """
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_path)
        
        if self.verbose:
            print(f"  ✓ Model saved: {model_path}\n")
    
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
    
    # Prepare data
    X_train, y_train, X_val, y_val = trainer.prepare_training_data(
        val_split=0.2,
        temporal_split=True
    )
    
    # Build model
    trainer.build_model(learning_rate=0.001)
    
    # Train
    trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=8,
        early_stopping_patience=15
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
