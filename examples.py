"""
Example usage of the PCB Thermal Prediction Model

This script demonstrates how to use the various components of the project.
"""

import numpy as np
from model import PCBThermalCNN, create_pcb_thermal_model
from sroi_parser import SROIParser
from data_loader import ThermalDataLoader


def example_model_creation():
    """Example: Create and inspect the CNN model."""
    print("=" * 60)
    print("Example 1: Creating the CNN Model")
    print("=" * 60)
    
    # Create model
    model = create_pcb_thermal_model(input_shape=(224, 224, 1))
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    print("\nModel created successfully!")


def example_sroi_parser():
    """Example: Using the SROI parser."""
    print("\n" + "=" * 60)
    print("Example 2: Parsing SROI Files")
    print("=" * 60)
    
    print("\nTo parse an SROI file:")
    print("```python")
    print("from sroi_parser import SROIParser, load_sroi_file")
    print("")
    print("# Method 1: Using the parser class")
    print("parser = SROIParser('path/to/file.sroi')")
    print("data = parser.read_file()")
    print("temp_array = parser.get_temperature_array()")
    print("")
    print("# Method 2: Using the convenience function")
    print("temp_array = load_sroi_file('path/to/file.sroi')")
    print("```")


def example_data_loading():
    """Example: Loading and preprocessing data."""
    print("\n" + "=" * 60)
    print("Example 3: Data Loading and Preprocessing")
    print("=" * 60)
    
    print("\nTo load thermal image data:")
    print("```python")
    print("from data_loader import ThermalDataLoader")
    print("")
    print("# Create data loader")
    print("loader = ThermalDataLoader('./data')")
    print("")
    print("# Load all images from directory")
    print("images, labels = loader.load_dataset()")
    print("")
    print("# Create preprocessed training dataset")
    print("preprocessed = loader.create_training_dataset(target_size=(224, 224))")
    print("```")


def example_training():
    """Example: Training the model."""
    print("\n" + "=" * 60)
    print("Example 4: Training the Model")
    print("=" * 60)
    
    print("\nCommand line training:")
    print("```bash")
    print("python train.py --data_dir ./data --epochs 50 --batch_size 32")
    print("```")
    
    print("\nProgrammatic training:")
    print("```python")
    print("from train import train_model")
    print("")
    print("history = train_model(")
    print("    data_dir='./data',")
    print("    epochs=50,")
    print("    batch_size=32,")
    print("    learning_rate=0.001,")
    print("    validation_split=0.2,")
    print("    model_save_path='my_model.h5'")
    print(")")
    print("```")


def example_prediction():
    """Example: Making predictions."""
    print("\n" + "=" * 60)
    print("Example 5: Making Predictions")
    print("=" * 60)
    
    print("\nSingle image prediction:")
    print("```bash")
    print("python predict.py --model my_model.h5 --image test_image.sroi")
    print("```")
    
    print("\nBatch prediction:")
    print("```bash")
    print("python predict.py --model my_model.h5 --data_dir ./test_images --output results.csv")
    print("```")
    
    print("\nProgrammatic prediction:")
    print("```python")
    print("from predict import predict_temperature")
    print("")
    print("temp = predict_temperature(")
    print("    model_path='my_model.h5',")
    print("    image_path='test_image.sroi',")
    print("    visualize=True")
    print(")")
    print("print(f'Predicted temperature: {temp:.2f}°C')")
    print("```")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# PCB Thermal Prediction Model - Usage Examples")
    print("#" * 60)
    
    example_model_creation()
    example_sroi_parser()
    example_data_loading()
    example_training()
    example_prediction()
    
    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
