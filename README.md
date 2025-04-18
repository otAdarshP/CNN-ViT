# CNN-ViT

A comparative study exploring Convolutional Neural Networks (CNNs) versus Vision Transformers (ViTs) for computer vision tasks, with a focus on defect detection in manufacturing.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Model Architectures](#model-architectures)
  - [Training & Evaluation](#training--evaluation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a Jupyter Notebook (`ViT.ipynb`) that:

1. Implements one or more CNN architectures (e.g., ResNet, custom CNN).  
2. Implements a Vision Transformer (ViT) backbone.  
3. Compares their performance on an image classification or defect-detection dataset relevant to manufacturing.  
4. Visualizes training/validation metrics and sample predictions.

The goal is to understand trade-offs between convolutional and transformer-based models in terms of accuracy, robustness, and computational cost.

## Dataset

The notebook loads and preprocesses a dataset of images representing defects and non-defects in manufactured parts.  

> _Note: If you wish to use your own dataset, replace the data-loading cells with your dataset path and adjust preprocessing as needed._

## Methodology

### Model Architectures

- **CNN**: A standard convolutional network (e.g., ResNet-18) trained from scratch or fine-tuned.  
- **Vision Transformer (ViT)**: A transformer-based model (e.g., ViT-B/16) adapted for image classification.

### Training & Evaluation

1. **Data Split**: Train/validation/test split (e.g., 70/15/15).  
2. **Hyperparameters**: Learning rate, batch size, number of epochs, optimizer settings.  
3. **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix.  
4. **Visualization**: Plot loss and metric curves; display sample predictions.

## Project Structure

```bash
CNN-ViT/
├── ViT.ipynb       # Main Jupyter Notebook with experiments
└── README.md        # Project documentation
```

## Dependencies

- Python 3.7+  
- Jupyter Notebook or JupyterLab  
- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- torch  
- torchvision  
- timm (for Vision Transformer implementations)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/otAdarshP/CNN-ViT.git
   cd CNN-ViT
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt  # if provided, else install manually
   ```

3. **Install dependencies** (if no `requirements.txt`):
   ```bash
   pip install jupyter numpy pandas matplotlib scikit-learn torch torchvision timm
   ```

## Running the Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook ViT.ipynb
   ```
2. Execute cells sequentially to reproduce data loading, model training, and evaluation.

## Results

After running, you will obtain:

- **Training Curves**: Loss and accuracy for both CNN and ViT.  
- **Evaluation Metrics**: Tabulated scores and confusion matrices.  
- **Sample Outputs**: Predicted labels on test images.

These results help assess which model is better suited for manufacturing defect detection under varying data and compute constraints.

## Contributing

Contributions and suggestions are welcome:

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/your-feature`.  
3. Commit your changes: `git commit -m 'Add feature'`.  
4. Push to your fork and open a Pull Request.

## License

_No license specified. Please add a `LICENSE` file if you wish to license this project._

