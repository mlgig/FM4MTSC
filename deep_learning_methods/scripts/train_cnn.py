#!/usr/bin/env python
"""
Script to train and evaluate the custom CNN model for time series classification.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_npy_dataset
from src.models.cnn.model import CNNModel
from src.evaluation.metrics import evaluate_model, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the custom CNN for time series classification"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the .npy dataset file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimization (default: 5e-5)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation (default: 0.2)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training (if not specified, uses best available)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: 'results')",
    )

    return parser.parse_args()


def prepare_data_for_cnn(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data specifically for the CNN model.
    Ensures correct format for PyTorch CNN (batch, channels, sequence).

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_prepared, X_test_prepared)
    """
    # Check if data needs reshaping
    if len(X_train.shape) == 2:
        # For univariate time series: reshape from (batch, time) to (batch, 1, time)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # If data has the shape (batch, time, channels), transpose to (batch, channels, time)
    elif len(X_train.shape) == 3 and X_train.shape[1] > X_train.shape[2]:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test


def main():
    """Main function to train and evaluate the CNN model."""
    args = parse_arguments()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    X_train, y_train, X_test, y_test = load_npy_dataset(args.dataset)

    # Prepare data for CNN (ensuring channels-first format)
    X_train, X_test = prepare_data_for_cnn(X_train, X_test)
    print(f"Prepared data shape - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(
        args.output_dir, f"cnn_model_{os.path.basename(args.dataset).split('.')[0]}.pth"
    )

    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42
    )

    print(f"Training set size: {X_train_split.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Initialize model
    print("\nInitializing CNN model...")
    model = CNNModel(
        num_channels=X_train.shape[1],
        seq_length=X_train.shape[2],
        num_classes=len(np.unique(y_train)),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        model_path=model_path,
    )

    print(model.summary())

    # Train model
    print("\nTraining CNN model...")
    start_time = time.time()
    history = model.fit(X_train_split, y_train_split, validation_data=(X_val, y_val))
    fit_time = time.time() - start_time
    print(f"Training completed in {fit_time:.2f} seconds")

    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_name="CNN",
        fit_time=fit_time,
    )

    # Print key metrics
    print("\nResults:")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

    # Save results
    dataset_name = os.path.basename(args.dataset).split(".")[0]
    save_results(
        results=results, output_dir=args.output_dir, prefix=f"cnn_{dataset_name}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
