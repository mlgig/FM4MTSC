# Traditional ML Methods for Time Series Classification

This package provides a framework for evaluating traditional machine learning models on time series classification tasks.

## Features

- Support for multiple time series datasets in NumPy format
- Implementation of various classical ML models for time series classification
- Standardized evaluation metrics and reporting
- Performance comparison between different models

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/FM4MTSC.git # Original repo will be provided in camera ready version 
cd traditional_ml_methods

# Create and activate the conda environment
conda env create -f environment.yml
conda activate traditional-ml-methods
```


## Usage

### Running the Classification Pipeline

```bash
# Run the classification on all datasets
python scripts/run_classification.py --datasets path/to/dataset1.npy path/to/dataset2.npy

# Run with specific models
python scripts/run_classification.py --datasets path/to/dataset.npy --models "Random Forest" "SVM" "KNN"
```

### Example Code

```python
from src.data.loader import load_npy_dataset
from src.models.classifiers import get_classifier_models
from src.evaluation.metrics import evaluate_model
from src.utils.preprocessing import prepare_data

# Load data
X_train, y_train, X_test, y_test = load_npy_dataset("path/to/dataset.npy")

# Preprocess data
X_train_scaled, X_test_scaled, _ = prepare_data(X_train, X_test)

# Get models
models = get_classifier_models()

# Evaluate a specific model
model = models["Random Forest"]
results = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

## Supported Models

- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Logistic Regression
- Ridge Classifier
- Stochastic Gradient Descent (SGD) Classifier

## Dataset Format

The code expects NumPy datasets in the following format:
- `.npy` file containing a dictionary with 'train' and 'test' keys
- Each containing 'X' and 'y' keys for features and labels
- Features will be reshaped automatically if needed

## Results

Results are saved in both JSON and TXT formats in the 'results' directory, including:
- Model accuracy (train and test)
- Training and prediction times
- Samples processed per second
- Model size
- Detailed classification reports