"""
Data Loading Utilities for PCB Thermal Prediction

This module provides utilities for loading and preprocessing thermal images
for the CNN model.
"""

import os
import numpy as np
from sroi_parser import load_sroi_file


class ThermalDataLoader:
    """Data loader for thermal images."""
    
    def __init__(self, data_dir):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing thermal image files
        """
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        
    def load_dataset(self, file_extension='.sroi'):
        """
        Load all thermal images from the data directory.
        
        Args:
            file_extension (str): File extension to filter (default: .sroi)
            
        Returns:
            tuple: (images, labels) numpy arrays
        """
        images = []
        
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return np.array([]), np.array([])
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(file_extension):
                    filepath = os.path.join(root, file)
                    try:
                        temp_array = load_sroi_file(filepath)
                        images.append(temp_array)
                        self.file_list.append(filepath)
                    except Exception as e:
                        print(f"Error loading {filepath}: {str(e)}")
        
        if len(images) == 0:
            print("No valid thermal images found")
            return np.array([]), np.array([])
        
        return np.array(images), np.array(self.labels)
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess thermal image for CNN input.
        
        Args:
            image (numpy.ndarray): Input thermal image
            target_size (tuple): Target size (height, width)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        import cv2
        
        # Resize image
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        
        # Add channel dimension
        preprocessed = np.expand_dims(normalized, axis=-1)
        
        return preprocessed
    
    def create_training_dataset(self, target_size=(224, 224)):
        """
        Create preprocessed training dataset.
        
        Args:
            target_size (tuple): Target image size
            
        Returns:
            numpy.ndarray: Preprocessed image dataset
        """
        images, labels = self.load_dataset()
        
        if len(images) == 0:
            return np.array([])
        
        preprocessed = []
        for img in images:
            preprocessed.append(self.preprocess_image(img, target_size))
        
        return np.array(preprocessed)
