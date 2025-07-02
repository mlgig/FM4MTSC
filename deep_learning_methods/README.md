# Deep Learning Methods for Time Series Classification

This package provides implementations of various deep learning models for time series classification.

## Models Included

- **Aeon CNN**: CNN classifier from the Aeon library
- **Custom CNN**: Custom implementation of a CNN for time series
- **TimesNet**: Implementation of the TimesNet architecture

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

### Training the Aeon CNN Model

```bash
# Train with default parameters
python scripts/train_aeon_cnn.py --dataset path/to/dataset.npy

# Train with custom parameters
python scripts/train_aeon_cnn.py --dataset path/to/dataset.npy --epochs 1000 --batch-size 32
```

### Training the Custom CNN Model

```bash
# Train with default parameters
python scripts/train_cnn.py --dataset path/to/dataset.npy

# Train with custom parameters
python scripts/train_cnn.py --dataset path/to/dataset.npy --lr 0.001 --epochs 100
```

### Training the TimesNet Model

```bash
# Train with default parameters
python scripts/train_timesnet.py --dataset path/to/dataset.npy

# Train with custom parameters
python scripts/train_timesnet.py --dataset path/to/dataset.npy --depth 3 --num-heads 8
```

## Model Architecture Details

### Aeon CNN

The CNN model from the Aeon library is a simple yet effective architecture for time series classification. It consists of:

- 1D convolutional layers
- Global average pooling
- Dense layers for classification

### Custom CNN

Our custom CNN implementation includes:

- Multiple convolutional layers with batch normalization
- Residual connections for better gradient flow
- Dropout for regularization

### TimesNet

TimesNet is a specialized architecture that uses:

- Time-frequency decomposition
- Multi-scale feature extraction
- Self-attention mechanisms

## Results

Model evaluation results are saved in both JSON and TXT formats in the `results` directory, including:

- Accuracy metrics (train and test)
- Training and prediction times
- F1, precision, and recall scores
- Model size
- Detailed classification reports

## Project Structure

```

deep_learning_methods/
├── src/                       # Source code
│   ├── data/                  # Data loading utilities
│   ├── models/                # Model implementations
│   │   ├── aeon_cnn/          # Aeon CNN model
│   │   ├── cnn/               # Custom CNN model
│   │   └── timesnet/          # TimesNet model
│   │       └── layers/        # TimesNet specific layers
│   ├── evaluation/            # Evaluation
│   └── training/              # Training utilities
│   └── notebooks/             # Notebooks
├── scripts/                   # Training scripts

```