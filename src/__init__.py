"""
Numpy for Data Science - Credit Card Fraud Detection
======================================================

A comprehensive machine learning package for credit card fraud detection
built entirely with NumPy, demonstrating ML fundamentals from scratch.

Modules:
--------
- data_processing: Data loading, preprocessing, and feature engineering
- models: Machine learning model implementations (Logistic Regression, Neural Networks, KNN)
- visualization: Advanced plotting and analysis tools

Example Usage:
--------------
>>> from src import NumpyDataProcessor, LogisticRegression, AdvancedVisualizer
>>> processor = NumpyDataProcessor()
>>> X, y = processor.load_data("data/raw/creditcard.csv")
>>> model = LogisticRegression()
>>> model.fit(X, y)

Author: Lang Phu Quy
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Lang Phu Quy"
__email__ = "phuquy.lang@gmail.com"
__all__ = [
    # Data Processing
    "NumpyDataProcessor",
    "AdvancedPreprocessor",
    "PreprocessingConfig",
    "DataStats",
    "load_credit_card_data",
    "compute_correlation_matrix",
    "analyze_class_imbalance",
    
    # Models
    "BaseModel",
    "LogisticRegression",
    "NeuralNetwork",
    "KNN",
    "ModelConfig",
    "TrainingHistory",
    "train_test_split",
    "k_fold_cross_validation",
    "compute_metrics",
    
    # Visualization
    "AdvancedVisualizer",
    "PlotConfig",
    "save_plot",
    "plot_statistical_tests",
]

# Import main classes for convenient access
try:
    from .data_processing import (
        NumpyDataProcessor,
        AdvancedPreprocessor,
        PreprocessingConfig,
        DataStats,
        load_credit_card_data,
        compute_correlation_matrix,
        analyze_class_imbalance,
    )
    
    from .models import (
        BaseModel,
        LogisticRegression,
        NeuralNetwork,
        KNN,
        ModelConfig,
        TrainingHistory,
        train_test_split,
        k_fold_cross_validation,
        compute_metrics,
    )
    
    from .visualization import (
        AdvancedVisualizer,
        PlotConfig,
        save_plot,
        plot_statistical_tests,
    )
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some modules could not be imported: {e}. "
        "Please ensure all dependencies are installed: pip install -r requirements.txt",
        ImportWarning
    )


def get_version():
    """Return the current version of the package."""
    return __version__


def get_info():
    """Return package information."""
    return {
        "name": "Numpy for Data Science",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Credit Card Fraud Detection using pure NumPy",
    }


# Package-level configurations
import warnings
import numpy as np

# Configure NumPy print options for better readability
np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=100,
    threshold=1000,
)

# Filter specific warnings that are common in ML workflows
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
