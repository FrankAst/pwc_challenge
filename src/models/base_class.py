"""Base model class with common functionality."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, **kwargs):
        """
        Initialize base model.
        
        Args:
            **kwargs: Model-specific parameters
        """
        self.is_fitted = False
        self.model_params = kwargs
    
    @abstractmethod
    def _fit_model(self, X, y):
        """
        Fit the actual model with feature engineering.
        
        This method must:
        1. Apply model-specific feature engineering (encoding, scaling, etc.)
        2. Save any encoders/scalers as instance attributes
        3. Train the actual model
        
        Args:
            X: Training features (after feature selection)
            y: Training target
        """
        pass
    
    @abstractmethod
    def _predict_model(self, X):
        """
        Make predictions with the model.
        
        This method must:
        1. Apply the same feature engineering as _fit_model
        2. Use saved encoders/scalers from training
        3. Make predictions with the trained model
        
        Args:
            X: Features for prediction (after feature selection)
            
        Returns:
            Predictions array
        """
        pass
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        # Call model-specific fitting (each model handles its own preprocessing)
        self._fit_model(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Call model-specific prediction (each model handles its own preprocessing)
        return self._predict_model(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dict with performance metrics
        """
        y_pred = self.predict(X)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
    
    def get_model_info(self):
        """
        Get model information.
        
        Returns:
            Dict with model information
        """
        info = {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params
        }
        
        return info
    
    def plot_predictions(self, X, y, dataset_name='Dataset', figsize=(8, 6)):
        """
        Plot predictions vs actual values.
        
        Args:
            X: Features
            y: True target values
            dataset_name: Name for the plot title ('Train', 'Test', etc.)
            figsize: Figure size tuple
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting predictions")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Get metrics using existing evaluate method
        metrics = self.evaluate(X, y)
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.__class__.__name__} - {dataset_name} Predictions vs Actual')
        
        # Add metrics to plot
        textstr = f"RMSE: {metrics['rmse']:.0f}\nMAE: {metrics['mae']:.0f}\nR²: {metrics['r2']:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()