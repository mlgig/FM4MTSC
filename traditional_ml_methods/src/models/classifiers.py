"""
Classification models for time series data.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_classifier_models(
    random_state: int = 42, custom_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Get a dictionary of classifier models for time series classification.

    Args:
        random_state: Random seed for reproducibility
        custom_params: Optional dictionary of custom parameters for models

    Returns:
        Dictionary mapping model names to initialized model objects
    """
    default_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "SVM": SVC(kernel="rbf", random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Ridge Classifier": RidgeClassifier(random_state=random_state),
        "SGD Classifier": SGDClassifier(
            max_iter=1000, random_state=random_state, loss="modified_huber"
        ),
    }

    # Update with custom parameters if provided
    if custom_params:
        for model_name, params in custom_params.items():
            if model_name in default_models:
                # Create a new model with updated parameters
                model_class = default_models[model_name].__class__
                default_models[model_name] = model_class(**params)

    return default_models


def get_model_size(model: Any) -> int:
    """
    Estimate the size of the model in bytes.

    Args:
        model: The trained model

    Returns:
        Size of the model in bytes
    """
    import pickle
    import sys

    return sys.getsizeof(pickle.dumps(model))


def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """
    Get recommended hyperparameters for grid search.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary of hyperparameters for grid search
    """
    hyperparameters = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.1, 0.01],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet", None],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        },
        "Ridge Classifier": {
            "alpha": [0.1, 1.0, 10.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        },
        "SGD Classifier": {
            "loss": ["hinge", "log", "modified_huber"],
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
            "max_iter": [500, 1000, 2000],
        },
    }

    if model_name not in hyperparameters:
        raise ValueError(f"Model {model_name} not found in hyperparameters dictionary")

    return hyperparameters[model_name]
