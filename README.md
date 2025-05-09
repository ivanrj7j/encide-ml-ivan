# Cat vs Dog Classifier - MACE ENCIDE Competition Entry

Entry for the Cat vs Dog Classifier machine learning competition organized by MACE ENCIDE.

## Overview

This project classifies images of cats and dogs using deep learning models built with PyTorch. Multiple architectures were experimented with, and the best-performing model was selected based on training accuracy and general performance.

* 🔍 **Accuracy**: Achieved 95.6% on training data
* 🧠 **Framework**: PyTorch
* 🧪 **Experimentation**: Tried various models before final selection
* 🏆 **Competition**: Entry for MACE ENCIDE ML Competition

---

## Approach and Architecture

### Data Augmentation
To improve model generalization and robustness, extensive data augmentation techniques were applied using the `albumentations` library. These include:
- **Resizing**: All images were resized to 128x128 pixels.
- **Random Horizontal Flip**: Applied with a probability of 50%.
- **Random Rotation**: Random 90-degree rotations were applied with a probability of 50%.
- **Shift, Scale, and Rotate**: Small shifts, scaling, and rotations were applied with controlled limits.
- **Brightness and Contrast Adjustment**: Randomly adjusted brightness and contrast.
- **Hue, Saturation, and Value Adjustment**: Randomly shifted hue, saturation, and value.
- **Gaussian Blur**: Applied with a probability of 30%.
- **Normalization**: Images were normalized using ImageNet statistics (mean and standard deviation).

### Data Loading
The dataset was loaded using a custom PyTorch `Dataset` class:
- **Training Data**: Images were organized into `cats` and `dogs` directories, with labels assigned accordingly.
- **Inference Data**: A separate dataset loader was used for inference, applying only resizing and normalization.
- **Batch Processing**: Data was loaded in batches of 20 for inference and 32 for training using PyTorch's `DataLoader`.

### Model Architectures
Two deep learning models were implemented and evaluated:
1. **CatDogClassifierV1**:
   - Based on a ResNet-like architecture with 5 residual blocks.
   - Includes global average pooling and a fully connected layer for binary classification.
   - Designed for robust feature extraction and efficient training.

2. **CatsVsDogsV3**:
   - A custom convolutional neural network with three convolutional blocks.
   - Includes dropout layers for regularization and fully connected layers for classification.
   - Optimized for smaller datasets with fewer parameters.

Both models were trained using the binary cross-entropy loss function and evaluated using sigmoid activation for binary classification.

---

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ivanrj7j/encide-ml-ivan.git
   cd encide-ml-ivan
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**:
   Place the cat and dog images in the appropriate directories as expected by the training script.

5. **Run training**:

   ```bash
   python train.py
   ```

6. **Run inference**:

   ```bash
   python inference.py
   ```

---

## License

This project is licensed under the MIT License.
