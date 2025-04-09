# Transformer-Based Methods for Time Series Classification

This package provides implementations of transformer-based models for time series classification tasks.

## Models Included

- **TSLANet**: Time Series Spectral-Attention Network with adaptive frequency filtering
- **ConvTran**: Convolutional Transformer Network

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/FM4MTSC.git # Original repo will be provided in camera ready version 
cd transformer_based_methods

# Create and activate the conda environment
conda env create -f environment.yml
conda activate transformer_based_methods
```

## Usage

### Training TSLANet

```bash
# Train with default parameters
python scripts/train_tslanet.py --dataset path/to/dataset.npy

# Train with custom parameters
python scripts/train_tslanet.py --dataset path/to/dataset.npy --emb_dim 128 --depth 2 --batch_size 32
```

### Training ConvTran

```bash
# Train with default parameters
python scripts/train_convtran.py --dataset path/to/dataset.npy

# Train with custom parameters
python scripts/train_convtran.py --dataset path/to/dataset.npy --d_model 128 --n_heads 8
```

## Model Architecture Details

### TSLANet

TSLANet is a novel transformer-based architecture designed specifically for time series classification, featuring:

- **Adaptive Spectral Block (ASB)**: Applies adaptive frequency filtering in the spectral domain
- **Interactive Convolutional Block (ICB)**: Captures complex local patterns using interactive convolutions
- **Patch Embedding**: Transforms time series segments into embeddings
- **Position Encoding**: Adds positional information to the embeddings
- **Self-Attention Mechanisms**: Captures dependencies across the time series

### ConvTran

ConvTran combines convolutional layers with transformer architectures to handle time series data by:

- Using convolutional layers to capture local patterns
- Employing self-attention mechanisms to model global dependencies
- Processing data at multiple temporal resolutions

## Results

Models are evaluated based on:
- Accuracy
- F1 Score
- Classification Reports
- Computation Time

## Dataset Compatibility

The models are compatible with:
- UCR/UEA Time Series Classification Archive
- Custom time series datasets in NumPy format

## Project Structure

```
transformer_based_methods/
├── src/                       # Source code
│   ├── data/                  # Data loading utilities
│   ├── models/                # Model implementations
│   │   ├── tslanet/           # TSLANet model
│   │   └── convtran/          # ConvTran model
│   └── training/              # Training utilities
├── scripts/                   # Scripts to run experiments
└── notebooks/                 # Jupyter notebooks for demos
```

## References

- [\[Paper Reference for TSLANet\]](https://arxiv.org/pdf/2404.08472)
- [\[Paper Reference for ConvTran\]](https://arxiv.org/pdf/2305.16642)


