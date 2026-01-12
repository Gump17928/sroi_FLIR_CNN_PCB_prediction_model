"""
Training Script for PCB Thermal Prediction Model

This script trains the CNN model on thermal images of PCBs.
"""

import os
import argparse
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import PCBThermalCNN
from data_loader import ThermalDataLoader
from sklearn.model_selection import train_test_split


def train_model(data_dir, epochs=50, batch_size=32, learning_rate=0.001, 
                validation_split=0.2, model_save_path='pcb_thermal_model.h5'):
    """
    Train the PCB thermal prediction model.
    
    Args:
        data_dir (str): Directory containing training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        validation_split (float): Fraction of data to use for validation
        model_save_path (str): Path to save the trained model
    """
    print("Loading thermal image data...")
    data_loader = ThermalDataLoader(data_dir)
    images = data_loader.create_training_dataset(target_size=(224, 224))
    
    if len(images) == 0:
        print("No training data found. Please add .sroi files to the data directory.")
        return
    
    print(f"Loaded {len(images)} thermal images")
    
    # NOTE: This is placeholder code for demonstration purposes.
    # In a real scenario, you must replace this with actual temperature labels
    # loaded from your annotations/ground truth data (e.g., CSV file, JSON, or database).
    # Using random labels will result in a model that cannot learn meaningful patterns.
    print("\nWARNING: Using random placeholder labels for demonstration.")
    print("Replace this with your actual labeled data for real training!")
    labels = np.random.rand(len(images)) * 100  # Random temps between 0-100°C
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=validation_split, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create and compile model
    print("Building CNN model...")
    pcb_cnn = PCBThermalCNN(input_shape=(224, 224, 1))
    pcb_cnn.build_model()
    pcb_cnn.compile_model(learning_rate=learning_rate)
    pcb_cnn.get_model_summary()
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = pcb_cnn.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nTraining completed. Model saved to {model_save_path}")
    
    # Evaluate on validation set
    val_loss, val_mae, val_mse = pcb_cnn.model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  MAE: {val_mae:.4f}°C")
    print(f"  MSE: {val_mse:.4f}")
    
    return history


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train PCB Thermal Prediction Model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--output', type=str, default='pcb_thermal_model.h5',
                        help='Output model file path')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        model_save_path=args.output
    )


if __name__ == '__main__':
    main()
