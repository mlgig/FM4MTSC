"""
Data loading utilities for time series classification.
"""

import os
from typing import Dict, List, Tuple, Union

import numpy as np


def load_npy_dataset(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare numpy dataset for time series classification.

    Args:
        file_path: Path to the .npy file containing the dataset

    Returns:
        Tuple containing (X_train, y_train, X_test, y_test)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the dataset format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=True).item()
    except (ValueError, AttributeError):
        raise ValueError(
            "Invalid dataset format. Expected a .npy file containing a dictionary."
        )

    # Validate dataset structure
    if not isinstance(data, dict) or "train" not in data or "test" not in data:
        raise ValueError("Dataset must contain 'train' and 'test' keys.")

    for split in ["train", "test"]:
        if "X" not in data[split] or "y" not in data[split]:
            raise ValueError(f"Dataset '{split}' split must contain 'X' and 'y' keys.")

    # Extract and reshape data
    X_train = data["train"]["X"]
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array([int(x) for x in data["train"]["y"]])

    X_test = data["test"]["X"]
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = np.array([int(x) for x in data["test"]["y"]])

    print(f"Dataset loaded: {os.path.basename(file_path)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


def list_available_datasets(data_dir: str) -> List[str]:
    """
    List all available .npy datasets in the given directory.

    Args:
        data_dir: Directory containing the datasets

    Returns:
        List of file paths to available datasets
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dataset_paths = []
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            dataset_paths.append(os.path.join(data_dir, file))

    return dataset_paths


def get_dataset_info(file_path: str) -> Dict[str, Union[str, int, tuple]]:
    """
    Get information about a dataset without loading the full data.

    Args:
        file_path: Path to the .npy file

    Returns:
        Dictionary with dataset information
    """
    data = np.load(file_path, allow_pickle=True).item()

    info = {
        "name": os.path.basename(file_path).split(".")[0],
        "train_samples": data["train"]["X"].shape[0],
        "test_samples": data["test"]["X"].shape[0],
        "feature_shape": data["train"]["X"].shape[1:],
        "num_classes": len(np.unique(data["train"]["y"])),
        "classes": sorted(list(np.unique(data["train"]["y"]))),
    }

    return info
