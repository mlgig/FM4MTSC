"""
Preprocessing utilities for time series data.
"""

import time
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_data(
    X_train: np.ndarray, X_test: np.ndarray, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply standard scaling to the training and test data.

    Args:
        X_train: Training data
        X_test: Test data
        scaler: Optional pre-fitted scaler to use

    Returns:
        Tuple containing (X_train_scaled, X_test_scaled, preprocessing_time)
    """
    start_time = time.time()

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    preprocessing_time = time.time() - start_time

    return X_train_scaled, X_test_scaled, preprocessing_time


def handle_missing_values(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """
    Handle missing values in the data.

    Args:
        X: Input data
        strategy: Strategy for imputation ('mean', 'median', 'zero')

    Returns:
        Data with missing values handled
    """
    if strategy not in ["mean", "median", "zero"]:
        raise ValueError("Strategy must be one of 'mean', 'median', or 'zero'")

    X_copy = X.copy()
    mask = np.isnan(X_copy)

    if not np.any(mask):
        return X_copy

    if strategy == "mean":
        col_mean = np.nanmean(X_copy, axis=0)
        X_copy[:, np.any(mask, axis=0)] = col_mean
    elif strategy == "median":
        col_median = np.nanmedian(X_copy, axis=0)
        X_copy[:, np.any(mask, axis=0)] = col_median
    elif strategy == "zero":
        X_copy[mask] = 0

    return X_copy


def normalize_time_series(X: np.ndarray, method: str = "z-score") -> np.ndarray:
    """
    Normalize time series data.

    Args:
        X: Input time series data
        method: Normalization method ('z-score', 'min-max')

    Returns:
        Normalized time series data
    """
    if method not in ["z-score", "min-max"]:
        raise ValueError("Method must be one of 'z-score' or 'min-max'")

    X_normalized = np.zeros_like(X)

    if method == "z-score":
        for i in range(X.shape[0]):
            mean = np.mean(X[i])
            std = np.std(X[i])
            if std == 0:
                X_normalized[i] = 0
            else:
                X_normalized[i] = (X[i] - mean) / std

    elif method == "min-max":
        for i in range(X.shape[0]):
            min_val = np.min(X[i])
            max_val = np.max(X[i])
            if max_val == min_val:
                X_normalized[i] = 0
            else:
                X_normalized[i] = (X[i] - min_val) / (max_val - min_val)

    return X_normalized
