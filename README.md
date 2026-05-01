# MONAI UNet Training & Evaluation

This repository contains code for training and evaluating U-Net models for medical image segmentation using the MONAI framework. The implementation includes both basic and advanced U-Net variants along with LRP (Local Rule-based Perturbation) analysis.

## Overview

The project consists of three main components:

1. **Model Training (`unet_training_array.py`)**:
   - Trains various U-Net architectures for medical image segmentation.
   - Supports different normalization modes: instance and batch normalization.
   - Implements data augmentation techniques including random rotations, flips, and padding/cropping to fixed sizes.
   - Provides sliding window inference for efficient validation.

2. **Data Preparation (`DataPrepair.py`)**:
   - Processes both medical and XCT (X-ray Computed Tomography) data into suitable formats.
   - Converts 3D medical images and other supported formats into 2D slices for training.
   - Handles DICOM files and other medical image formats, outputting prepared images in PNG format.

3. **LRP Analysis & Evaluation (`TestMonai.py`)**:
   - Implements LRP (Local Rule-based Perturbation) analysis for model interpretability.
   - Provides visualization of attention maps to understand model decisions.
   - Includes evaluation metrics and visualizations for both positive and negative attributions, with overlay plots for better interpretation.

## Prerequisites

### Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MONAI-UNet.git
cd MONAI-UNet
```

2. Install Python and required packages:
```bash
python --version  # Ensure Python 3.8+
pip install -r requirements.txt
```

### Required Packages:
- **MONAI**: `monai>=0.6.0`
- **PyTorch**: `torch>=1.9.0`
- **Captum**: `captum>=0.2.1` (For LRP analysis)
- **Numpy & Scipy**: For numerical operations.
- **Matplotlib**: For visualization tasks.
- **SimpleITK**: For medical image processing.

3. Install additional dependencies for medical image processing:
```bash
pip install pydicom dask-image scikit-image
```

## Usage

### Training Models

To train a model on your dataset:

1. Prepare your dataset in the following structure:
   ```
   /your_dataset/
       ├── train/
           ├── img/         # Input images
           └── mask/       # Corresponding segmentation masks
       └── val/
           ├── img/
           └── mask/
   ```

2. Run the training script:
```bash
python unet_training_array.py \
    --train_imgs /path/to/train/img/ \
    --train_labels /path/to/train/mask/ \
    --val_imgs /path/to/val/img/ \
    --val_labels /path/to/val/mask/
```

### Performing LRP Analysis

1. Evaluate a trained model:
```bash
python TestMonai.py \
    --model_path /path/to/trained/model.pth \
    --input_image /path/to/test/image.png \
    --output_folder ./evaluation_results/
```

2. For full dataset analysis:
```bash
python evaluate.py \
    --models_dir /path/to/models/ \
    --input_images_dir /path/to/input/images/
```

### Data Preparation

To prepare medical and XCT data:

1. Place your input images in the following structure:
   ```
   /your_input_data/
       ├── DICOM/           # DICOM format files
       └── Other_Format/    # Other supported image formats
   ```

2. Run the data preparation script:
```bash
python DataPrepair.py
```

## Project Structure

```
MONAI-UNet/
├── unet_training_array.py         # Model training implementation
├── TestMonai.py                   # LRP analysis and evaluation
├── DataPrepair.py                # Data preparation script
└── README.md                      # This documentation
```

## Notes

1. **Supported Models**:
   - BasicUNet
   - ResidualUNet (ResiduelUnet)
   - UNet++ (BasicUNetPlusPlus)

2. **Normalization Modes**:
   - Instance normalization: Maintains statistics per sample.
   - Batch normalization: Applies the same transformation across a batch of samples.

3. **Data Augmentation**:
   - Random rotations
   - Random flips
   - Normalization
   - Padding/cropping to fixed size

4. **LRP Analysis**:
   - Visualizes attention maps for model decision-making.
   - Provides detailed attributions (positive and negative) for each prediction.
   - Includes overlay plots on original images for better interpretability.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to:
- Open an issue on GitHub
- Fork the repository and submit a pull request


## License

MIT License

Copyright (c) [Year] [Your Name/Username]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



We hope this README provides a clear understanding of the project and its functionalities. Happy coding!