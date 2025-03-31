"""
Generic trainer class for deep learning models.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.metrics import evaluate_model, save_results


class Trainer:
    """
    Generic trainer for deep learning models.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        output_dir: str = "results",
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            model_name: Name of the model for logging
            output_dir: Directory to save results
        """
        self.model = model
        self.model_name = model_name
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results
        """
        # Record start time
        start_time = time.time()

        # Train the model
        print(f"Training {self.model_name}...")

        try:
            # Try to fit with validation data if provided
            if X_val is not None and y_val is not None:
                history = self.model.fit(
                    X_train, y_train, validation_data=(X_val, y_val), **kwargs
                )
            else:
                history = self.model.fit(X_train, y_train, **kwargs)

        except (TypeError, ValueError):
            # If model doesn't support validation data or history
            try:
                self.model.fit(X_train, y_train)
                history = None
            except Exception as e:
                print(f"Error during training: {str(e)}")
                raise

        # Calculate training time
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")

        # Return training information
        return {
            "model_name": self.model_name,
            "train_time": train_time,
            "history": history,
        }

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fit_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            fit_time: Time taken to fit the model

        Returns:
            Dictionary with evaluation results
        """
        # Evaluate the model
        print(f"Evaluating {self.model_name}...")

        results = evaluate_model(
            model=self.model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=self.model_name,
            fit_time=fit_time,
        )

        return results

    def save_results(
        self, results: Dict[str, Any], dataset_name: str
    ) -> Tuple[str, str]:
        """
        Save evaluation results.

        Args:
            results: Evaluation results
            dataset_name: Name of the dataset

        Returns:
            Tuple of (json_filename, txt_filename)
        """
        # Save results
        return save_results(
            results=results,
            output_dir=self.output_dir,
            prefix=f"{self.model_name.lower()}_{dataset_name}",
        )

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train and evaluate the model in one step.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with evaluation results
        """
        # Train the model
        training_info = self.train(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **kwargs
        )

        # Evaluate the model
        results = self.evaluate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_time=training_info["train_time"],
        )

        # Save the results
        self.save_results(results, dataset_name)

        return results
