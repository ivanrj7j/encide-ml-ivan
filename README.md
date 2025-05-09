# Cat vs Dog Classifier - MACE ENCIDE Competition Entry

Entry for the Cat vs Dog Classifier machine learning competition organized by MACE ENCIDE.

## Overview

This project classifies images of cats and dogs using deep learning models built with PyTorch. Multiple architectures were experimented with, and the best-performing model was selected based on training accuracy and general performance.

* 🔍 **Accuracy**: Achieved 95.6% on training data
* 🧠 **Framework**: PyTorch
* 🧪 **Experimentation**: Tried various models before final selection
* 🏆 **Competition**: Entry for MACE ENCIDE ML Competition

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

## License

This project is licensed under the MIT License.
