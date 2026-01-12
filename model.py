"""
CNN Model for PCB Thermal Temperature Prediction

This module defines the CNN architecture for predicting embedded thermal
temperatures of PCBs from FLIR thermal images.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class PCBThermalCNN:
    """CNN model for PCB thermal prediction."""
    
    def __init__(self, input_shape=(224, 224, 1), num_outputs=1):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_outputs (int): Number of output values (temperature predictions)
        """
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.model = None
        
    def build_model(self):
        """
        Build the CNN architecture.
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_outputs, activation='linear')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
    
    def get_model_summary(self):
        """Print model architecture summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


def create_pcb_thermal_model(input_shape=(224, 224, 1), learning_rate=0.001):
    """
    Convenience function to create and compile a PCB thermal CNN model.
    
    Args:
        input_shape (tuple): Shape of input images
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled CNN model
    """
    pcb_cnn = PCBThermalCNN(input_shape=input_shape)
    pcb_cnn.build_model()
    pcb_cnn.compile_model(learning_rate=learning_rate)
    
    return pcb_cnn.model
