"""
PHASE 8E: SENSOR FUSION FOR EMBEDDED TEMPERATURE PREDICTION

Use FLIR surface temps + PARTIAL embedded thermistor data to predict
embedded temps for components WITHOUT thermistors.

Architecture:
    Input Features:
        - FLIR surface temps for all 22 components (mean, std, max, min)
        - Embedded thermistor temps for subset of components (e.g., 10 components)
        - Spatial position (x, y)
    
    Output:
        - Embedded temps for components WITHOUT thermistors
    
    Training Strategy:
        - During training: Randomly mask 40-60% of thermistor readings
        - Model learns to predict masked temps from unmasked temps + FLIR
        - At inference: Use actual available thermistors to predict unavailable ones

Key Insight:
    Components are thermally coupled (r=0.95-0.99). Knowing temps of nearby
    components + their surface temps allows accurate prediction of unmeasured temps.

Created: January 21, 2026
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, List
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class SensorFusionPredictor:
    """
    Sensor fusion model: FLIR + partial thermistors → full embedded temp field
    """
    
    def __init__(self, n_components: int = 22, mask_ratio: float = 0.5, verbose: bool = True):
        """
        Args:
            n_components: Total number of components
            mask_ratio: Fraction of thermistors to mask during training (simulate unavailable)
            verbose: Print progress
        """
        self.n_components = n_components
        self.mask_ratio = mask_ratio
        self.verbose = verbose
        
        self.model = None
        self.history = None
        self.dataset = None
        self.roi_masks = None
        
    def build_model(self, learning_rate: float = 0.001):
        """
        Build sensor fusion network.
        
        Input: [batch, n_components * 7]
            - 4 FLIR features per component (mean, std, max, min)
            - 1 embedded temp (or 0 if masked)
            - 1 mask indicator (1 = available, 0 = masked)
            - 2 spatial coords (x, y)
        
        Output: [batch, n_components]
            - Predicted embedded temp for each component
        """
        
        # Input: flattened features
        inputs = keras.Input(shape=(self.n_components * 7,), name='fused_features')
        
        # Dense network to learn thermal coupling
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output: embedded temp for each component
        outputs = layers.Dense(self.n_components, activation='linear', name='embedded_temps')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='SensorFusion')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        if self.verbose:
            print("\n" + "="*80)
            print("  SENSOR FUSION MODEL")
            print("="*80)
            print(f"  Input: FLIR (22×4) + Thermistors (22×1) + Masks (22×1) + Position (22×2)")
            print(f"  Output: Embedded temps (22×1)")
            print(f"  Masking ratio: {self.mask_ratio:.1%} (simulates missing sensors)")
            print(f"  Parameters: {self.model.count_params():,}")
            print("="*80 + "\n")
        
        return self.model
    
    def load_dataset(self, h5_file: str):
        """Load HDF5 dataset"""
        if self.verbose:
            print("="*80)
            print("  LOADING DATASET")
            print("="*80)
        
        with h5py.File(h5_file, 'r') as f:
            self.dataset = {
                'flir_frames': f['flir_frames'][:],
                'sand_temps': f['sand_temps'][:],
                'roi_masks': f['roi_masks'][:],
                'metadata': {
                    'n_components': f['metadata'].attrs['n_components'],
                    'component_names': [name.decode('utf-8') if isinstance(name, bytes) else name 
                                       for name in f['metadata']['component_names'][:]],
                    'image_height': f['metadata'].attrs['image_height'],
                    'image_width': f['metadata'].attrs['image_width']
                }
            }
        
        self.roi_masks = self.dataset['roi_masks']
        
        if self.verbose:
            print(f"  Dataset: {h5_file}")
            print(f"  Frames: {self.dataset['flir_frames'].shape}")
            print(f"  Components: {self.n_components}")
            print("="*80 + "\n")
    
    def extract_component_positions(self):
        """Extract component centroids from ROI masks"""
        positions = np.zeros((self.n_components, 2))
        
        for comp_idx in range(self.n_components):
            mask = self.roi_masks[comp_idx] > 0
            if np.any(mask):
                y_coords, x_coords = np.where(mask)
                positions[comp_idx, 0] = np.mean(x_coords)
                positions[comp_idx, 1] = np.mean(y_coords)
        
        # Normalize to [0, 1]
        positions[:, 0] /= self.dataset['metadata']['image_width']
        positions[:, 1] /= self.dataset['metadata']['image_height']
        
        return positions
    
    def prepare_training_data(self, val_split: float = 0.2, temporal_split: bool = True):
        """
        Prepare sensor fusion training data.
        
        For each sample, randomly mask some thermistor readings to simulate
        unavailable sensors. Model learns to predict masked values from
        unmasked values + FLIR data.
        """
        if self.verbose:
            print("="*80)
            print("  PREPARING SENSOR FUSION DATA")
            print("="*80)
        
        flir_frames = self.dataset['flir_frames']
        sand_temps = self.dataset['sand_temps']
        n_frames = flir_frames.shape[0]
        
        # Extract FLIR features
        flir_features = np.zeros((n_frames, self.n_components, 4))
        
        for comp_idx in range(self.n_components):
            mask = self.roi_masks[comp_idx] > 0
            if np.any(mask):
                for frame_idx in range(n_frames):
                    roi_pixels = flir_frames[frame_idx][mask]
                    flir_features[frame_idx, comp_idx, 0] = np.mean(roi_pixels)
                    flir_features[frame_idx, comp_idx, 1] = np.std(roi_pixels)
                    flir_features[frame_idx, comp_idx, 2] = np.max(roi_pixels)
                    flir_features[frame_idx, comp_idx, 3] = np.min(roi_pixels)
        
        # Get component positions
        positions = self.extract_component_positions()
        
        # Build feature matrix
        # For each frame: [comp0_flir(4), comp0_temp(1), comp0_mask(1), comp0_pos(2), comp1_...]
        X = np.zeros((n_frames, self.n_components * 7))
        y = sand_temps.copy()
        
        for frame_idx in range(n_frames):
            # Randomly mask thermistors
            mask = np.random.rand(self.n_components) < self.mask_ratio
            
            for comp_idx in range(self.n_components):
                base_idx = comp_idx * 7
                
                # FLIR features (4)
                X[frame_idx, base_idx:base_idx+4] = flir_features[frame_idx, comp_idx, :]
                
                # Embedded temp (1) - zero if masked
                if not mask[comp_idx]:
                    X[frame_idx, base_idx+4] = sand_temps[frame_idx, comp_idx]
                else:
                    X[frame_idx, base_idx+4] = 0.0
                
                # Mask indicator (1) - 1 if available, 0 if masked
                X[frame_idx, base_idx+5] = 0.0 if mask[comp_idx] else 1.0
                
                # Position (2)
                X[frame_idx, base_idx+6:base_idx+8] = positions[comp_idx, :]
        
        # Split data
        if temporal_split:
            split_idx = int(n_frames * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            indices = np.random.permutation(n_frames)
            split_idx = int(n_frames * (1 - val_split))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        
        if self.verbose:
            print(f"  Training: {X_train.shape} → {y_train.shape}")
            print(f"  Validation: {X_val.shape} → {y_val.shape}")
            print(f"  Masking: {self.mask_ratio:.1%} of thermistors hidden per sample")
            print(f"  Features: FLIR(4) + Temp(1) + Mask(1) + Position(2) × {self.n_components}")
            print("="*80 + "\n")
        
        return X_train, y_train, X_val, y_val
    
    def train(self, X_train, y_train, X_val, y_val,
              epochs: int = 100, batch_size: int = 16, early_stopping_patience: int = 20):
        """Train sensor fusion model"""
        if self.verbose:
            print("="*80)
            print("  TRAINING SENSOR FUSION MODEL")
            print("="*80 + "\n")
        
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
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        if self.verbose:
            print(f"\n  ✓ Training complete")
            print(f"  Final training loss: {self.history.history['loss'][-1]:.4f}")
            print(f"  Final validation loss: {self.history.history['val_loss'][-1]:.4f}\n")
        
        return self.history
    
    def evaluate(self, X_val, y_val) -> Dict:
        """Evaluate model"""
        if self.verbose:
            print("="*80)
            print("  EVALUATING SENSOR FUSION")
            print("="*80 + "\n")
        
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Overall metrics
        r2 = r2_score(y_val.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_val.flatten(), y_pred.flatten()))
        mae = mean_absolute_error(y_val.flatten(), y_pred.flatten())
        
        # Per-component metrics
        component_r2 = [r2_score(y_val[:, i], y_pred[:, i]) for i in range(self.n_components)]
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'component_r2_mean': np.mean(component_r2),
            'component_r2_median': np.median(component_r2),
            'per_component_r2': component_r2
        }
        
        if self.verbose:
            print(f"  Overall R²: {r2:.4f}")
            print(f"  Overall RMSE: {rmse:.2f}°C")
            print(f"  Overall MAE: {mae:.2f}°C")
            print(f"\n  Per-component R² (mean): {metrics['component_r2_mean']:.4f}")
            print("="*80 + "\n")
        
        return metrics
    
    def save_model(self, model_path: str):
        """Save model"""
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        if self.verbose:
            print(f"  ✓ Model saved: {model_path}\n")
    
    def save_training_history(self, history_path: str):
        """Save training history"""
        output_path = Path(history_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        if self.verbose:
            print(f"  ✓ Training history saved: {history_path}\n")


def main():
    print("\n" + "="*80)
    print("  PHASE 8E: SENSOR FUSION THERMAL PREDICTION")
    print("  FLIR + Partial Thermistors → Full Embedded Temp Field")
    print("="*80 + "\n")
    
    # Initialize
    predictor = SensorFusionPredictor(
        n_components=22,
        mask_ratio=0.5,  # Hide 50% of thermistors during training
        verbose=True
    )
    
    # Load dataset
    predictor.load_dataset("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
    
    # Prepare data
    X_train, y_train, X_val, y_val = predictor.prepare_training_data(
        val_split=0.2,
        temporal_split=True
    )
    
    # Build model
    predictor.build_model(learning_rate=0.001)
    
    # Train
    predictor.train(
        X_train, y_train, X_val, y_val,
        epochs=150,
        batch_size=16,
        early_stopping_patience=25
    )
    
    # Evaluate
    metrics = predictor.evaluate(X_val, y_val)
    
    # Save
    predictor.save_model("ml_model/cnn_thermal_modeling/models/sensor_fusion_hbridge.keras")
    predictor.save_training_history("ml_model/cnn_thermal_modeling/models/sensor_fusion_training_history.json")
    
    print("="*80)
    print("  PHASE 8E SENSOR FUSION TRAINING COMPLETE ✓")
    print(f"  Final R²: {metrics['r2']:.4f}")
    print(f"  Final MAE: {metrics['mae']:.2f}°C")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
