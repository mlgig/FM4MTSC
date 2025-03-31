"""
Time series classification models from the Aeon library.
"""

import pickle
import sys
import warnings
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")

from aeon.classification.convolution_based import (
    HydraClassifier,
    RocketClassifier,
)
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import QUANTClassifier


def get_ts_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Get a dictionary of time series classifier models from Aeon.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping model names to initialized model objects
    """
    classifiers = {
        "Rocket": RocketClassifier(random_state=random_state),
        "MiniRocket": RocketClassifier(
            rocket_transform="minirocket", random_state=random_state
        ),
        "QUANT": QUANTClassifier(random_state=random_state),
        "Hydra": HydraClassifier(random_state=random_state),
        "Catch22": Catch22Classifier(random_state=random_state),
    }

    return classifiers


def get_model_size(model: Any) -> int:
    """
    Estimate the size of the model in bytes.

    Args:
        model: The trained model

    Returns:
        Size of the model in bytes
    """
    return sys.getsizeof(pickle.dumps(model))


def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Get information about a specific time series classifier.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information
    """
    model_info = {
        "Rocket": {
            "description": "Random Convolutional Kernel Transform classifier",
            "paper": "Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34(5), 1454-1495.",
            "strengths": "Fast and accurate for large datasets",
            "limitations": "Black-box model with limited interpretability",
        },
        "MiniRocket": {
            "description": "Faster variant of ROCKET with smaller kernel set",
            "paper": "Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). MiniRocket: A very fast (almost) deterministic transform for time series classification. Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.",
            "strengths": "Very fast, almost deterministic, less memory usage than ROCKET",
            "limitations": "Similar interpretability limitations as ROCKET",
        },
        "QUANT": {
            "description": "Quantile transform-based interval features",
            "paper": "QUANT: Quantile Interval-Based Time Series Classifier",
            "strengths": "Works well with complex time-dependent patterns",
            "limitations": "Can be slower than transform-based methods",
        },
        "Hydra": {
            "description": "Convolutional neural network classifier for time series",
            "paper": "Dempster, A., Schmidt, D. F., & Webb, G. I. (2021). Hydra: Competing convolutional kernels for fast and accurate time series classification.",
            "strengths": "Competitive accuracy with deep learning approaches",
            "limitations": "More computationally intensive than other transform-based methods",
        },
        "Catch22": {
            "description": "22 CAnonical Time-series CHaracteristics classifier",
            "paper": "Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019). catch22: CAnonical Time-series CHaracteristics. Data Mining and Knowledge Discovery, 33(6), 1821-1852.",
            "strengths": "Uses only 22 time series features, interpretable",
            "limitations": "May not capture all complex patterns in time series data",
        },
    }

    if model_name not in model_info:
        raise ValueError(f"Model information for '{model_name}' not available")

    return model_info[model_name]
