#!/usr/bin/env python
"""
Script to run time series classification with Aeon time series models.
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import list_available_datasets, load_npy_dataset
from src.evaluation.metrics import compare_models, evaluate_model, save_results
from src.models.classifiers import get_model_info, get_ts_classifiers


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run time series classification with Aeon models"
    )

    parser.add_argument(
        "--datasets", nargs="+", help="Paths to one or more .npy dataset files"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing .npy dataset files (alternative to --datasets)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (if not specified, all models will be used)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: 'results')",
    )

    parser.add_argument(
        "--compare-metric",
        type=str,
        default="test_accuracy",
        help="Metric for model comparison (default: 'test_accuracy')",
    )

    parser.add_argument(
        "--model-info",
        type=str,
        help="Display information about a specific model and exit",
    )

    return parser.parse_args()


def run_classification(
    dataset_paths: List[str],
    selected_models: Optional[List[str]] = None,
    output_dir: str = "results",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run time series classification on multiple datasets with different models.

    Args:
        dataset_paths: List of paths to dataset files
        selected_models: List of model names to evaluate (if None, all models are used)
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each dataset
    """
    # Get all available models
    all_models = get_ts_classifiers()

    # Filter models if needed
    if selected_models:
        models = {
            name: model for name, model in all_models.items() if name in selected_models
        }
        if not models:
            print(
                f"No valid models selected. Available models: {list(all_models.keys())}"
            )
            return {}
    else:
        models = all_models

    all_results = defaultdict(list)

    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 50)

        try:
            # Load data
            X_train, y_train, X_test, y_test = load_npy_dataset(dataset_path)

            dataset_results = []

            for model_name, model in models.items():
                print(f"\nTraining {model_name}...")
                try:
                    result = evaluate_model(
                        model, X_train, X_test, y_train, y_test, model_name
                    )
                    dataset_results.append(result)

                    # Print results to console
                    print(f"Model: {model_name}")
                    print(f"Training Accuracy: {result['train_accuracy']:.4f}")
                    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
                    print(f"F1 Score: {result['f1_score']:.4f}")
                    print(f"Fit Time: {result['fit_time']:.4f} seconds")
                    print("-" * 50)

                except Exception as e:
                    print(f"Error with model {model_name}: {str(e)}")
                    continue

            all_results[dataset_name] = dataset_results

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue

    return all_results


def print_model_info(model_name: str) -> None:
    """
    Print detailed information about a specific model.

    Args:
        model_name: Name of the model
    """
    try:
        info = get_model_info(model_name)

        print(f"\nModel: {model_name}")
        print("=" * (len(model_name) + 7))
        print(f"Description: {info['description']}")
        print(f"Paper: {info['paper']}")
        print("\nStrengths:")
        print(f"- {info['strengths']}")
        print("\nLimitations:")
        print(f"- {info['limitations']}")

    except ValueError as e:
        print(f"Error: {str(e)}")
        print(f"Available models: {', '.join(get_ts_classifiers().keys())}")


def main():
    """Main function to run the classification pipeline."""
    args = parse_arguments()

    # Check if user wants model info
    if args.model_info:
        print_model_info(args.model_info)
        return 0

    # Get dataset paths
    dataset_paths = []
    if args.datasets:
        dataset_paths.extend(args.datasets)

    if args.data_dir:
        try:
            additional_datasets = list_available_datasets(args.data_dir)
            dataset_paths.extend(additional_datasets)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")

    if not dataset_paths:
        print("Error: No datasets provided. Use --datasets or --data-dir")
        return 1

    # Remove duplicates while preserving order
    dataset_paths = list(dict.fromkeys(dataset_paths))

    print(
        f"Running time series classification on {len(dataset_paths)} datasets with "
        f"{len(args.models) if args.models else 'all'} models"
    )

    # Run classification
    results = run_classification(
        dataset_paths=dataset_paths,
        selected_models=args.models,
        output_dir=args.output_dir,
    )

    if results:
        # Save results
        json_file, txt_file = save_results(results, args.output_dir)

        # Compare models
        comparison = compare_models(results, args.compare_metric)

        print("\nModel Comparison:")
        print(f"Based on {args.compare_metric}:")
        for i, (model_name, data) in enumerate(comparison.items(), 1):
            print(f"{i}. {model_name}: {data['avg_' + args.compare_metric]:.4f}")

        print("\nTime Series Classification completed successfully!")
        print(f"Results saved to:")
        print(f"- {json_file}")
        print(f"- {txt_file}")
    else:
        print("No results were generated.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
