"""
AEON CNN Model implementation for time series classification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from aeon.classification.deep_learning import CNNClassifier


class AeonCNNModel:
    """
    Wrapper for the Aeon CNN Classifier with additional functionality.
    """

    def __init__(
        self,
        n_epochs: int = 2000,
        batch_size: int = 16,
        kernel_size: int = 7,
        n_filters: int = 16,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the CNN classifier with specified parameters.

        Args:
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            kernel_size: Size of convolutional kernels
            n_filters: Number of convolutional filters
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters to pass to CNNClassifier
        """
        self.params = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "kernel_size": kernel_size,
            "n_filters": n_filters,
            "random_state": random_state,
            **kwargs,
        }

        self.model = CNNClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            kernel_size=kernel_size,
            n_filters=n_filters,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AeonCNNModel":
        """
        Fit the model to the training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input data.

        Args:
            X: Input features

        Returns:
            Array of predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.

        Args:
            X: Input features

        Returns:
            Array of class probabilities
        """
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy score on the given data.

        Args:
            X: Input features
            y: Ground truth labels

        Returns:
            Accuracy score
        """
        return self.model.score(X, y)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.params

    def set_params(self, **params) -> "AeonCNNModel":
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            self.params[key] = value

        # Recreate the model with updated parameters
        self.model = CNNClassifier(**self.params)
        return self

    def summary(self) -> str:
        """
        Get a summary of the model.

        Returns:
            String summary of the model
        """
        try:
            # Try to access model summary if available
            return self.model.summary()
        except (AttributeError, NotImplementedError):
            # Fall back to parameter description
            params_str = "\n".join([f"{k}: {v}" for k, v in self.params.items()])
            return f"Aeon CNN Classifier\n\nParameters:\n{params_str}"
