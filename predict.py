"""
Prediction/Inference Script for PCB Thermal Model

This script uses a trained model to predict thermal temperatures from new images.
"""

import os
import argparse
import numpy as np
from model import PCBThermalCNN
from data_loader import ThermalDataLoader
import matplotlib.pyplot as plt


def predict_temperature(model_path, image_path, visualize=True):
    """
    Predict temperature from a thermal image.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the thermal image file
        visualize (bool): Whether to visualize the prediction
        
    Returns:
        float: Predicted temperature in Celsius
    """
    # Load the model
    pcb_cnn = PCBThermalCNN()
    pcb_cnn.load_model(model_path)
    
    # Load and preprocess the image
    data_loader = ThermalDataLoader(os.path.dirname(image_path))
    
    if image_path.endswith('.sroi'):
        from sroi_parser import load_sroi_file
        thermal_image = load_sroi_file(image_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .sroi file.")
    
    # Preprocess
    preprocessed = data_loader.preprocess_image(thermal_image, target_size=(224, 224))
    input_batch = np.expand_dims(preprocessed, axis=0)
    
    # Predict
    prediction = pcb_cnn.model.predict(input_batch, verbose=0)
    predicted_temp = prediction[0][0]
    
    print(f"Predicted Temperature: {predicted_temp:.2f}°C")
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original thermal image
        im1 = axes[0].imshow(thermal_image, cmap='hot')
        axes[0].set_title('Thermal Image')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')
        
        # Preprocessed image
        im2 = axes[1].imshow(preprocessed[:, :, 0], cmap='hot')
        axes[1].set_title(f'Preprocessed\nPredicted: {predicted_temp:.2f}°C')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        output_path = image_path.replace('.sroi', '_prediction.png')
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()
    
    return predicted_temp


def batch_predict(model_path, data_dir, output_csv='predictions.csv'):
    """
    Perform batch predictions on multiple images.
    
    Args:
        model_path (str): Path to the trained model
        data_dir (str): Directory containing images to predict
        output_csv (str): Path to save prediction results
    """
    # Load the model
    pcb_cnn = PCBThermalCNN()
    pcb_cnn.load_model(model_path)
    
    # Load all images
    data_loader = ThermalDataLoader(data_dir)
    images = data_loader.create_training_dataset(target_size=(224, 224))
    
    if len(images) == 0:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Predicting temperatures for {len(images)} images...")
    
    # Predict
    predictions = pcb_cnn.model.predict(images, verbose=1)
    
    # Save results
    results = []
    for i, (filepath, pred) in enumerate(zip(data_loader.file_list, predictions)):
        results.append({
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'predicted_temperature': pred[0]
        })
        print(f"{os.path.basename(filepath)}: {pred[0]:.2f}°C")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to {output_csv}")


def main():
    """Main function for predictions."""
    parser = argparse.ArgumentParser(description='Predict PCB Temperatures')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--image', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--data_dir', type=str,
                        help='Directory for batch prediction')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file for batch predictions')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image prediction
        predict_temperature(args.model, args.image, visualize=not args.no_viz)
    elif args.data_dir:
        # Batch prediction
        batch_predict(args.model, args.data_dir, args.output)
    else:
        print("Please specify either --image or --data_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
