"""Data loading utilities for model training."""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, SEED,TARGET_COLUMN, FILE_PATH


def load_dataset(file_path = None):
    """Load final dataset with FE and processing.."""
    
    if not file_path:
        file_path = FILE_PATH
    
    return pd.read_csv(file_path)

def get_features_and_target(df):
    
    """Split into features and target."""
    
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=SEED, stratify_by=None):
    """Create train/test split."""
    
    if stratify_by is not None:
        stratify = X[stratify_by]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)







