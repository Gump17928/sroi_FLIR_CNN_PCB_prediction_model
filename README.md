# sroi_FLIR_CNN_PCB_prediction_model

The sroi_FLIR_CNN_PCB_prediction_model is an attempt at learning about, and reverse engineering the .sroi files that exist for FLIR, and to allow for a better understanding of the implication using an ML model to predict embedded thermal temperatures of PCB's.

## Overview

This project implements a Convolutional Neural Network (CNN) for predicting thermal temperatures of Printed Circuit Boards (PCBs) from FLIR thermal images stored in .sroi format.

## Features

- **SROI File Parser**: Custom parser for FLIR .sroi thermal image files
- **CNN Architecture**: Deep learning model optimized for thermal prediction
- **Training Pipeline**: Complete training workflow with validation and callbacks
- **Prediction System**: Both single-image and batch prediction capabilities
- **Visualization**: Thermal image visualization with prediction overlays

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gump17928/sroi_FLIR_CNN_PCB_prediction_model.git
cd sroi_FLIR_CNN_PCB_prediction_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── sroi_parser.py           # SROI file parser
├── data_loader.py           # Data loading and preprocessing
├── model.py                 # CNN model architecture
├── train.py                 # Training script
└── predict.py               # Prediction/inference script
```

## Usage

### Training a Model

To train the CNN model on your thermal image data:

```bash
python train.py --data_dir ./data --epochs 50 --batch_size 32 --output my_model.h5
```

Parameters:
- `--data_dir`: Directory containing .sroi training files
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--validation_split`: Fraction for validation (default: 0.2)
- `--output`: Output model file path

### Making Predictions

#### Single Image Prediction

```bash
python predict.py --model my_model.h5 --image thermal_image.sroi
```

#### Batch Prediction

```bash
python predict.py --model my_model.h5 --data_dir ./test_images --output results.csv
```

Parameters:
- `--model`: Path to trained model file (required)
- `--image`: Single image for prediction
- `--data_dir`: Directory for batch prediction
- `--output`: Output CSV file for batch results
- `--no-viz`: Disable visualization

### Using the SROI Parser

```python
from sroi_parser import load_sroi_file

# Load a thermal image
temperature_array = load_sroi_file('path/to/image.sroi')
print(f"Temperature range: {temperature_array.min():.2f}°C to {temperature_array.max():.2f}°C")
```

### Custom Model Training

```python
from model import PCBThermalCNN
from data_loader import ThermalDataLoader

# Create and compile model
cnn = PCBThermalCNN(input_shape=(224, 224, 1))
cnn.build_model()
cnn.compile_model(learning_rate=0.001)

# Load data
loader = ThermalDataLoader('./data')
images = loader.create_training_dataset()

# Train
# (add your labels and training code here)
```

## Model Architecture

The CNN model consists of:
- 4 convolutional blocks with batch normalization and max pooling
- Dense layers with dropout for regularization
- Linear output activation for temperature regression

## Data Format

The model expects thermal images in FLIR .sroi format. The SROI parser extracts:
- Thermal data arrays
- Image dimensions
- Metadata from file headers

## Requirements

- Python >= 3.7
- TensorFlow >= 2.6.0
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.4.0
- scikit-learn >= 0.24.0

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

This project involves reverse engineering FLIR .sroi file formats for thermal imaging analysis on PCBs.
