import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def normalize_time_series(data):
    """
    Normalize time series data by z-score normalization

    Args:
        data: Input time series data

    Returns:
        Normalized data with zero mean and unit variance
    """
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / (
        std + 1e-8
    )  # Add epsilon to avoid division by zero
    return normalized_data


def zero_pad_sequence(input_tensor, pad_length):
    """
    Zero-pad a sequence to desired length

    Args:
        input_tensor: Input tensor to pad
        pad_length: Amount of padding to add

    Returns:
        Padded tensor
    """
    return torch.nn.functional.pad(input_tensor, (0, pad_length))


def calculate_padding(seq_len, patch_size):
    """
    Calculate padding needed to make sequence length divisible by patch size

    Args:
        seq_len: Original sequence length
        patch_size: Size of patches

    Returns:
        Amount of padding to add
    """
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Generic dataset for time series data

    Supports loading data from PyTorch tensors or NumPy arrays
    """

    def __init__(self, X, y=None, normalize=False):
        """
        Initialize dataset

        Args:
            X: Input features [N, C, L] or [N, L]
               N = number of samples
               C = number of channels (optional)
               L = sequence length
            y: Class labels (optional)
            normalize: Whether to normalize data (default: False)
        """
        # Convert to torch tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        # Add channel dimension if not present
        if len(X.shape) == 2:
            X = X.unsqueeze(1)

        # Normalize if requested
        if normalize:
            X = normalize_time_series(X)

        self.X = X.float()

        # Handle labels if provided
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
            self.y = y.long().squeeze()
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return x, y, idx  # Include index for compatibility with both models
        return x, idx


class UCRDataset(TimeSeriesDataset):
    """
    Dataset class for UCR/UEA time series datasets

    Specifically handles the format of the UCR archive datasets
    """

    def __init__(self, X, y=None, normalize=True):
        super().__init__(X, y, normalize)


def load_ucr_dataset(data_path, dataset_name):
    """
    Load a UCR/UEA time series dataset

    Args:
        data_path: Path to data directory
        dataset_name: Name of the dataset

    Returns:
        train_data, test_data: Training and test datasets
    """
    train_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN.npz")
    test_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST.npz")

    train_data = np.load(train_file)
    test_data = np.load(test_file)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    # UCR datasets typically have [N, L] format, add channel dimension
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    train_dataset = UCRDataset(X_train, y_train)
    test_dataset = UCRDataset(X_test, y_test)

    return train_dataset, test_dataset


def load_npy_dataset(data_path, normalize=True):
    """
    Load dataset from .npy files

    Args:
        data_path: Path to directory containing X_train.npy, y_train.npy, etc.
        normalize: Whether to normalize the data

    Returns:
        train_dataset, test_dataset: Training and test datasets
    """
    # Define file paths
    X_train_path = os.path.join(data_path, "X_train.npy")
    y_train_path = os.path.join(data_path, "y_train.npy")
    X_test_path = os.path.join(data_path, "X_test.npy")
    y_test_path = os.path.join(data_path, "y_test.npy")

    # Load data
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # Ensure X has shape [samples, channels, length]
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, normalize=normalize)
    test_dataset = TimeSeriesDataset(X_test, y_test, normalize=normalize)

    return train_dataset, test_dataset


def load_dataset(data_path, format_type="auto", normalize=True, dataset_name=None):
    """
    Unified function to load dataset from various formats

    Args:
        data_path: Path to data directory
        format_type: Type of data format - "npy", "ucr", "pt", or "auto" (detect automatically)
        normalize: Whether to normalize the data
        dataset_name: Name of the dataset (required for UCR format)

    Returns:
        train_dataset, test_dataset: Training and test datasets
    """
    # Auto-detect format based on file extensions in directory
    if format_type == "auto":
        files = os.listdir(data_path)
        if any(f.endswith(".npy") for f in files):
            format_type = "npy"
        elif any(f.endswith(".pt") for f in files):
            format_type = "pt"
        elif dataset_name and any(f.endswith(".npz") for f in files):
            format_type = "ucr"
        else:
            raise ValueError(
                "Could not automatically detect file format. Please specify format_type."
            )

    # Load based on format
    if format_type == "npy":
        return load_npy_dataset(data_path, normalize)
    elif format_type == "ucr":
        if not dataset_name:
            dataset_name = os.path.basename(data_path)
        return load_ucr_dataset(os.path.dirname(data_path), dataset_name)
    elif format_type == "pt":
        # For PT format, we need a patch_size value for potential padding
        # Default to 8 if not specified, can be overridden later
        train_loader, val_loader, test_loader = load_pt_dataset(data_path, patch_size=8)
        # Extract datasets from loaders
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=32,
    val_split=0.1,
    num_workers=4,
    pin_memory=True,
    shuffle_train=True,
):
    """
    Create train, validation, and test dataloaders

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for dataloaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Split training data for validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def load_pt_dataset(data_path, patch_size=8):
    """
    Load dataset from .pt files with specific preprocessing for TSLANet

    Args:
        data_path: Path to the directory containing train.pt, val.pt, test.pt
        patch_size: Patch size for the model (used for padding calculation)

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for the datasets
    """
    # Load .pt files
    train_file = torch.load(os.path.join(data_path, "train.pt"))
    val_file = torch.load(os.path.join(data_path, "val.pt"))
    test_file = torch.load(os.path.join(data_path, "test.pt"))

    # Get sequence length and calculate required padding
    seq_len = train_file["samples"].shape[-1]
    required_padding = calculate_padding(seq_len, patch_size)

    # Apply padding if needed
    if required_padding != 0:
        train_file["samples"] = zero_pad_sequence(
            train_file["samples"], required_padding
        )
        val_file["samples"] = zero_pad_sequence(val_file["samples"], required_padding)
        test_file["samples"] = zero_pad_sequence(test_file["samples"], required_padding)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_file["samples"], train_file.get("labels"))
    val_dataset = TimeSeriesDataset(val_file["samples"], val_file.get("labels"))
    test_dataset = TimeSeriesDataset(test_file["samples"], test_file.get("labels"))

    # Determine batch size based on dataset size
    batch_size = min(32, len(train_dataset) // 4)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
