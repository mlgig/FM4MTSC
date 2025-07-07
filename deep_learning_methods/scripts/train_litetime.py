#!/usr/bin/env python
"""
Script to train and evaluate the LITETime model for time series classification.
"""
import argparse
import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_npy_dataset
from src.evaluation.metrics import evaluate_model, save_results
from aeon.classification.deep_learning import LITETimeClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate LITETime for time series classification")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .npy dataset file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results (default: 'results')")
    return parser.parse_args()

def main():
    args = parse_arguments()
    X_train, y_train, X_test, y_test = load_npy_dataset(args.dataset)
    model = LITETimeClassifier(
        use_litemv=True,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        verbose=False
    )
    model.fit(X_train, y_train)
    results = evaluate_model(model, X_train, X_test, y_train, y_test, "LITETime")
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, output_dir=args.output_dir, prefix="litetime")
    print(f"Train acc: {results['train_accuracy']}")
    print(f"Test acc: {results['test_accuracy']}")

if __name__ == "__main__":
    main() 