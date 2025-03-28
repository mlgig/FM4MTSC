"""
Evaluation metrics and reporting for time series classification.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.classifiers import get_model_size


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Train and evaluate a model, returning detailed metrics including timing.

    Args:
        model: Machine learning model to evaluate
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model for identification

    Returns:
        Dictionary of evaluation metrics
    """
    # Measure fit time
    start_fit_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit_time

    # Measure prediction time for training data
    start_pred_train_time = time.time()
    y_pred_train = model.predict(X_train)
    pred_train_time = time.time() - start_pred_train_time

    # Measure prediction time for test data
    start_pred_test_time = time.time()
    y_pred_test = model.predict(X_test)
    pred_test_time = time.time() - start_pred_test_time

    # Calculate performance metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred_test, average="weighted")
    recall = recall_score(y_test, y_pred_test, average="weighted")
    f1 = f1_score(y_test, y_pred_test, average="weighted")

    # Calculate samples per second
    train_samples_per_second = (
        X_train.shape[0] / pred_train_time if pred_train_time > 0 else float("inf")
    )
    test_samples_per_second = (
        X_test.shape[0] / pred_test_time if pred_test_time > 0 else float("inf")
    )

    # Get classification report
    class_report = classification_report(y_test, y_pred_test)

    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test).tolist()

    # Store all metrics in a dictionary
    metrics = {
        "model_name": model_name,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fit_time": float(fit_time),
        "pred_train_time": float(pred_train_time),
        "pred_test_time": float(pred_test_time),
        "train_samples_per_second": float(train_samples_per_second),
        "test_samples_per_second": float(test_samples_per_second),
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "model_size_bytes": get_model_size(model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return metrics


def save_results(
    results: Dict[str, List[Dict[str, Any]]], output_dir: str = "results"
) -> tuple:
    """
    Save results in both JSON and TXT formats.

    Args:
        results: Dictionary mapping dataset names to lists of model results
        output_dir: Directory to save results

    Returns:
        Tuple of (json_filename, txt_filename)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(output_dir, f"classification_results_{timestamp}")

    # Save as JSON
    json_filename = f"{base_filename}.json"
    with open(json_filename, "w") as f:
        # Convert results to JSON-serializable format
        json_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        json.dump(json_results, f, indent=4)

    # Save as TXT
    txt_filename = f"{base_filename}.txt"
    with open(txt_filename, "w") as f:
        f.write("Time Series Classification Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

        for dataset_name, dataset_results in results.items():
            f.write(f"\nResults for dataset: {dataset_name}\n")
            f.write("-" * 50 + "\n")

            for result in dataset_results:
                f.write(f"\nModel: {result['model_name']}\n")
                f.write("=" * (len(result["model_name"]) + 7) + "\n")
                f.write(f"Training Accuracy: {result['train_accuracy']:.4f}\n")
                f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"F1 Score: {result['f1_score']:.4f}\n")
                f.write(f"Fit Time: {result['fit_time']:.4f} seconds\n")
                f.write(
                    f"Training Prediction Time: {result['pred_train_time']:.4f} seconds\n"
                )
                f.write(
                    f"Test Prediction Time: {result['pred_test_time']:.4f} seconds\n"
                )
                f.write(
                    f"Training Samples/Second: {result['train_samples_per_second']:.2f}\n"
                )
                f.write(
                    f"Test Samples/Second: {result['test_samples_per_second']:.2f}\n"
                )
                f.write(f"Model Size: {result['model_size_bytes'] / 1024:.2f} KB\n")
                f.write("\nClassification Report:\n")
                f.write(result["classification_report"])
                f.write("\n" + "-" * 50 + "\n")

    print(f"\nResults have been saved to:")
    print(f"JSON: {json_filename}")
    print(f"TXT: {txt_filename}")

    return json_filename, txt_filename


def compare_models(
    results: Dict[str, List[Dict[str, Any]]], metric: str = "test_accuracy"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare models across datasets based on a specific metric.

    Args:
        results: Dictionary mapping dataset names to lists of model results
        metric: Metric to use for comparison

    Returns:
        Dictionary with model comparison results
    """
    if not results:
        return {}

    # Get all model names
    all_models = set()
    for dataset_results in results.values():
        for result in dataset_results:
            all_models.add(result["model_name"])

    all_models = sorted(list(all_models))

    # Initialize comparison dictionary
    comparison = {model: {"avg_" + metric: 0.0, "datasets": {}} for model in all_models}

    # Populate comparison data
    for dataset_name, dataset_results in results.items():
        for result in dataset_results:
            model_name = result["model_name"]
            if model_name in comparison:
                comparison[model_name]["datasets"][dataset_name] = result[metric]

    # Calculate average metric values
    for model_name, model_data in comparison.items():
        if model_data["datasets"]:
            model_data["avg_" + metric] = sum(model_data["datasets"].values()) / len(
                model_data["datasets"]
            )

    # Sort by average metric
    sorted_comparison = dict(
        sorted(comparison.items(), key=lambda x: x[1]["avg_" + metric], reverse=True)
    )

    return sorted_comparison
