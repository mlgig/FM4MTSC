#!/usr/bin/env python
"""
Script to train and evaluate the TimesNet model for time series classification.
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
from src.models.timesnet.model import TimesNetWrapper
from src.utils.metrics import evaluate_model, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate TimesNet for time series classification"
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
        default=0.001,
        help="Learning rate for optimization (default: 0.001)",
    )

    parser.add_argument(
        "--d-model", type=int, default=64, help="Model dimension (default: 64)"
    )

    parser.add_argument(
        "--d-ff", type=int, default=64, help="Feed-forward dimension (default: 64)"
    )

    parser.add_argument(
        "--e-layers", type=int, default=2, help="Number of encoder layers (default: 2)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top frequencies to use (default: 3)",
    )

    parser.add_argument(
        "--num-kernels",
        type=int,
        default=6,
        help="Number of kernels in inception blocks (default: 6)",
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


def prepare_data_for_timesnet(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data specifically for the TimesNet model.
    Ensures (batch, length, channels) format for TimesNet.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_prepared, X_test_prepared)
    """
    # Check if data needs reshaping
    if len(X_train.shape) == 2:
        # For univariate time series: reshape from (batch, time) to (batch, time, 1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # If data has the shape (batch, channels, time), transpose to (batch, time, channels)
    elif len(X_train.shape) == 3 and X_train.shape[1] < X_train.shape[2]:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test


def main():
    """Main function to train and evaluate the TimesNet model."""
    args = parse_arguments()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    X_train, y_train, X_test, y_test = load_npy_dataset(args.dataset)

    # Prepare data for TimesNet (ensuring length,channels format)
    X_train, X_test = prepare_data_for_timesnet(X_train, X_test)
    print(f"Prepared data shape - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(
        args.output_dir,
        f"timesnet_model_{os.path.basename(args.dataset).split('.')[0]}.pth",
    )

    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42
    )

    print(f"Training set size: {X_train_split.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Initialize model
    print("\nInitializing TimesNet model...")
    model = TimesNetWrapper(
        seq_len=X_train.shape[1],
        enc_in=X_train.shape[2],
        num_class=len(np.unique(y_train)),
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        top_k=args.top_k,
        num_kernels=args.num_kernels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        device=args.device,
        model_path=model_path,
    )

    print(model.summary())

    # Train model
    print("\nTraining TimesNet model...")
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
        model_name="TimesNet",
        fit_time=fit_time,
    )

    # Add model size information
    model_size = model.get_model_size()
    results["model_size_bytes"] = (
        model_size["num_params"] * 4
    )  # Approximate size in bytes
    results["model_size_mb"] = model_size["size_in_mb"]

    # Print key metrics
    print("\nResults:")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Model Size: {model_size['size_in_mb']:.2f} MB")
    print(f"Number of Parameters: {model_size['num_params']:,}")

    # Save results
    dataset_name = os.path.basename(args.dataset).split(".")[0]
    save_results(
        results=results, output_dir=args.output_dir, prefix=f"timesnet_{dataset_name}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
