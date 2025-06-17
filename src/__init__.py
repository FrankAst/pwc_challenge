"""
Salary Prediction Package

This package provides tools for predicting salaries based on employee characteristics
including demographics, education, experience, and job descriptions.

Usage:
    from src import load_dataset, BaselineModel, DecisionTree, LinearModel
    from src import calculate_metrics, compare_models
    from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, SEED, TEST_SIZE, FILE_PATH
"""

# Import configuration
from .config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES, 
    TARGET_COLUMN,
    SEED,
    TEST_SIZE,
    FILE_PATH
)

# Import data utilities data_loader
try:
    from .data_preparation_workflow import (
        load_dataset,
        get_features_and_target,
        create_train_test_split
    )
except ImportError:
    print(ImportError)
    pass

# Import models
try:
    from .models import dummy_model
except ImportError:
    pass

try:
    from .models import DecisionTree
except ImportError:
    pass

try:
    from .models import LinearModel
except ImportError:
    pass

# Import evaluation tools
try:
    from .evaluation import (
        calculate_metrics,
        compare_models
    )
except ImportError:
    pass

# Import utilities
try:
    from .utils import ExperimentTracker
except ImportError:
    pass

# Define what gets imported with "from src import *"
__all__ = [
    # Configuration
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES", 
    "TARGET_COLUMN",
    "SEED",
    "TEST_SIZE",
    "FILE_PATH"
    
    # Data utilities
    "load_dataset",
    "get_features_and_target", 
    "create_train_test_split",
    
    # Models
    "dummy_model",
    "DecisionTree",
    "LinearModel",
    
    # Evaluation
    "calculate_metrics",
    "compare_models",
    
    # Utils
    "ExperimentTracker"
]

# Clean up namespace - remove items that failed to import
__all__ = [item for item in __all__ if item in globals()]