#!/usr/bin/env python
"""
Script to train and evaluate the Aeon CNN model for time series classification.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_npy_dataset, prepare_data_for_dl
from src.models.aeon_cnn.model import AeonCNNModel
from src.utils.metrics import evaluate_model, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Aeon CNN for time series classification"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the .npy dataset file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs (default: 2000)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=7,
        help="Size of convolutional kernels (default: 7)",
    )

    parser.add_argument(
        "--n-filters",
        type=int,
        default=16,
        help="Number of convolutional filters (default: 16)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: 'results')",
    )

    return parser.parse_args()


def main():
    """Main function to train and evaluate the Aeon CNN model."""
    args = parse_arguments()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    X_train, y_train, X_test, y_test = load_npy_dataset(args.dataset)

    # Prepare data for deep learning if needed
    X_train, X_test = prepare_data_for_dl(X_train, X_test)

    # Initialize model
    print("\nInitializing Aeon CNN model...")
    model = AeonCNNModel(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        n_filters=args.n_filters,
        random_state=args.random_state,
    )

    # Train model
    print("\nTraining Aeon CNN model...")
    start_time = time.time()
    model.fit(X_train, y_train)
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
        model_name="AeonCNN",
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
        results=results, output_dir=args.output_dir, prefix=f"aeon_cnn_{dataset_name}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
