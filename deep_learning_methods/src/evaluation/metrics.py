"""
Evaluation metrics and reporting for deep learning time series models.
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_model_size(model: Any) -> int:
    """
    Estimate the size of the model in bytes.

    Args:
        model: The trained model

    Returns:
        Size of the model in bytes
    """
    return sys.getsizeof(pickle.dumps(model))


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    fit_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate a trained model, return detailed metrics including timing.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model for identification
        fit_time: Time taken to fit the model (if already measured)

    Returns:
        Dictionary of evaluation metrics
    """
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
    results: Dict[str, Any], output_dir: str = "results", prefix: str = "dl_model"
) -> Tuple[str, str]:
    """
    Save model evaluation results in both JSON and TXT formats.

    Args:
        results: Dictionary of model evaluation results
        output_dir: Directory to save results
        prefix: Prefix for output filenames

    Returns:
        Tuple of (json_filename, txt_filename)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(output_dir, f"{prefix}_{timestamp}")

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
        f.write(f"Deep Learning Model Evaluation: {results['model_name']}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {results['model_name']}\n")
        f.write("=" * (len(results["model_name"]) + 7) + "\n")
        f.write(f"Training Accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Fit Time: {results['fit_time']:.4f} seconds\n")
        f.write(f"Training Prediction Time: {results['pred_train_time']:.4f} seconds\n")
        f.write(f"Test Prediction Time: {results['pred_test_time']:.4f} seconds\n")
        f.write(f"Training Samples/Second: {results['train_samples_per_second']:.2f}\n")
        f.write(f"Test Samples/Second: {results['test_samples_per_second']:.2f}\n")
        f.write(f"Model Size: {results['model_size_bytes'] / 1024:.2f} KB\n")

        f.write("\nClassification Report:\n")
        f.write(results["classification_report"])

        f.write("\nConfusion Matrix:\n")
        for row in results["confusion_matrix"]:
            f.write(f"{row}\n")

    print(f"\nResults have been saved to:")
    print(f"JSON: {json_filename}")
    print(f"TXT: {txt_filename}")

    return json_filename, txt_filename


def compare_models(
    results_list: List[Dict[str, Any]], metric: str = "test_accuracy"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models based on a specific metric.

    Args:
        results_list: List of model evaluation results
        metric: Metric to use for comparison

    Returns:
        Dictionary with model comparison results
    """
    if not results_list:
        return {}

    # Initialize comparison dictionary
    comparison = {}

    # Populate comparison data
    for result in results_list:
        model_name = result["model_name"]
        comparison[model_name] = {
            metric: result[metric],
            "fit_time": result["fit_time"],
            "pred_test_time": result["pred_test_time"],
            "model_size_kb": result["model_size_bytes"] / 1024,
        }

    # Sort by metric
    sorted_comparison = dict(
        sorted(comparison.items(), key=lambda x: x[1][metric], reverse=True)
    )

    return sorted_comparison
