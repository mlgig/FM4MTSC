# Deep Learning Methods for Time Series Classification

This package provides implementations of various deep learning models for time series classification.

## Models Included

- Aeon CNN: CNN classifier from the Aeon library
- Custom CNN: Custom implementation of a CNN for time series
- TimesNet: Implementation of the TimesNet architecture
- InceptionTime
- DisjointCNN
- LITETime

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/mlgig/FM4MTSC 
cd deep_learning_methods

# Create and activate the conda environment
conda env create -f environment.yml
conda activate deep_learning_methods
```

## Usage

> **Important:** All deep learning scripts must be run from within the `deep_learning_methods` directory for correct imports and relative paths.

### Example (from project root):

```bash
cd deep_learning_methods
conda activate deep_learning_methods

# InceptionTime
python scripts/train_inceptiontime.py --dataset ../Datasets/CMJ.npy --epochs 10

# DisjointCNN
python scripts/train_disjointcnn.py --dataset ../Datasets/CMJ.npy --epochs 10

# LITETime
python scripts/train_litetime.py --dataset ../Datasets/CMJ.npy --epochs 10

# Aeon CNN
python scripts/train_aeon_cnn.py --dataset ../Datasets/CMJ.npy

# Custom CNN
python scripts/train_cnn.py --dataset ../Datasets/CMJ.npy

# TimesNet
python scripts/train_timesnet.py --dataset ../Datasets/CMJ.npy
```

Each script will print training and test accuracy and save results to the `results/` directory.

## Results

Model evaluation results are saved in both JSON and TXT formats in the `results` directory, including:

- Accuracy metrics (train and test)
- Training and prediction times
- F1, precision, and recall scores
- Detailed classification reports

## Project Structure

```
deep_learning_methods/
├── src/                       # Source code
│   ├── data/                  # Data loading utilities
│   ├── models/                # Model implementations
│   │   ├── aeon_cnn/          # Aeon CNN model
│   │   ├── cnn/               # Custom CNN model
│   │   ├── timesnet/          # TimesNet model
│   │   └── ...                # Other models
│   ├── evaluation/            # Evaluation
│   ├── training/              # Training utilities
│   └── notebooks/             # Notebooks
├── scripts/                   # Training scripts
├── results/                   # Output results
...
```