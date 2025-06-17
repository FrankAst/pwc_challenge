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
    
    def _bootstrap(self, X, y, alpha=0.05, n_bootstrap=100, sample_proportion=0.8):
        """
        Calculate bootstrap confidence intervals for metrics.
        
        Args:
            X: Features
            y: True target values
            alpha: Significance level (default 0.05 for 95% CI)
            n_bootstrap: Number of bootstrap iterations (default 100)
            sample_proportion: Proportion of data to be bootstrapped (default 0.8)
            
        Returns:
            Dict with confidence intervals for each metric
        """
        n_samples = len(X)
        bootstrap_size = int(n_samples * sample_proportion)
        
        # Store bootstrap results for each metric
        bootstrap_results = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # Perform bootstrap sampling
        for _ in range(n_bootstrap):
            # Generate bootstrap sample indices with replacement
            bootstrap_indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
            
            # Get bootstrap sample
            X_bootstrap = X.iloc[bootstrap_indices] if hasattr(X, 'iloc') else X[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices] if hasattr(y, 'iloc') else y[bootstrap_indices]
            
            # Make predictions on bootstrap sample
            y_pred_bootstrap = self.predict(X_bootstrap)
            
            # Calculate metrics for this bootstrap sample
            bootstrap_results['rmse'].append(np.sqrt(mean_squared_error(y_bootstrap, y_pred_bootstrap)))
            bootstrap_results['mae'].append(mean_absolute_error(y_bootstrap, y_pred_bootstrap))
            bootstrap_results['r2'].append(r2_score(y_bootstrap, y_pred_bootstrap))
            
        
        # Calculate confidence intervals
        confidence_intervals = {}
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        for metric, values in bootstrap_results.items():
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)
            confidence_intervals[metric] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def evaluate(self, X, y, alpha=0.05, n_bootstrap=100, sample_proportion=0.8, return_ci=True):
        """
        Evaluate model performance with optional confidence intervals.
        Bootstrap should be used on Test data.
        
        Args:
            X: Features
            y: True target values
            alpha: Significance level for confidence intervals (default 0.05 for 95% CI)
            n_bootstrap: Number of bootstrap iterations (default 100)
            sample_proportion: Proportion of data to be bootstrapped (default 0.8)
            return_ci: Whether to calculate and return confidence intervals (default True)
            
        Returns:
            If return_ci=True: DataFrame with metrics and confidence intervals
            If return_ci=False: Dict with performance metrics
        """
        # Calculate point estimates
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        if not return_ci:
            return metrics
        
        # Calculate confidence intervals using bootstrap
        confidence_intervals = self._bootstrap(X, y, alpha, n_bootstrap, sample_proportion)
        
        # Create DataFrame with metrics and confidence intervals
        results_data = []
        confidence_level = int((1 - alpha) * 100)
        
        for metric_name, point_estimate in metrics.items():
            lower_bound, upper_bound = confidence_intervals[metric_name]
            ci_string = f"({lower_bound:.3f}, {upper_bound:.3f})"
            
            results_data.append({
                'metric': metric_name.upper(),
                'point_estimate': point_estimate,
                f'{confidence_level}%_CI': ci_string
            })
        
        results_df = pd.DataFrame(results_data)
        return results_df
    
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
    
    def list_methods(self):
        """
        List all public methods available in this class.
        
        Returns:
            List of public method names (excluding private methods that start with _)
        """
        methods = [method for method in dir(self) 
                  if callable(getattr(self, method)) and not method.startswith('_')]
        
        print("Available public methods:")
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method}()")
        
        return methods
    
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
        metrics = self.evaluate(X, y, return_ci=False)
        
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
