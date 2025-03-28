# Time Series Classification Methods

This package provides a framework for evaluating specialized time series classification models from the Aeon library.

## Features

- Support for multiple time series datasets in NumPy format
- Implementation of various specialized time series classification algorithms
- Standardized evaluation metrics and reporting
- Performance comparison between different models

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/FM4MTSC.git # Original repo will be provided in camera ready version 
cd time_series_methods

# Create and activate the conda environment
conda env create -f environment.yml
conda activate time-series-methods
```


## Supported Models

The package includes the following time series classification models from the Aeon library:

- **ROCKET**: Random Convolutional Kernel Transform
- **MiniRocket**: Faster version of ROCKET with smaller kernel set
- **QUANT**: Quantile transform-based interval features
- **Hydra**: Convolutional neural network classifier
- **Catch22**: 22 CAnonical Time-series CHaracteristics

## Usage

### Running the Classification Pipeline

```bash
# Run the classification on all datasets
python scripts/run_classification.py --datasets path/to/dataset1.npy path/to/dataset2.npy

# Run with specific models
python scripts/run_classification.py --datasets path/to/dataset.npy --models "Rocket" "MiniRocket"

# Run on all datasets in a directory
python scripts/run_classification.py --data-dir path/to/datasets
```

### Example Code

```python
from src.data.loader import load_npy_dataset
from src.models.classifiers import get_ts_classifiers
from src.evaluation.metrics import evaluate_model

# Load data
X_train, y_train, X_test, y_test = load_npy_dataset("path/to/dataset.npy")

# Get models
models = get_ts_classifiers()

# Evaluate a specific model
model = models["Rocket"]
results = evaluate_model(model, X_train, X_test, y_train, y_test, "Rocket")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

## Dataset Format

The code expects NumPy datasets in the following format:
- `.npy` file containing a dictionary with 'train' and 'test' keys
- Each containing 'X' and 'y' keys for features and labels

## Results

Results are saved in both JSON and TXT formats in the 'results' directory, including:
- Model accuracy (train and test)
- Training and prediction times
- Samples processed per second
- Model size
- Detailed classification reports