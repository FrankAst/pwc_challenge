from sklearn.dummy import DummyRegressor
from .base_class import BaseModel


class dummy_model(BaseModel):
    """Dummy baseline model using DummyRegressor."""
    
    def __init__(self, strategy='mean', **kwargs):
        """
        Initialize dummy model.
        
        Args:
            strategy: DummyRegressor strategy ('mean', 'median', 'quantile', 'constant')
            **kwargs: Additional parameters passed to BaseModel
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.model = DummyRegressor(strategy=strategy)
        
        # Add strategy to model_params for model info
        self.model_params['strategy'] = strategy
    
    def _fit_model(self, X, y):
        """
        Fit the dummy regressor.
        
        Note: DummyRegressor doesn't need feature engineering.
        It just learns simple statistics (mean, median, etc.) from y.
        
        Args:
            X: Training features (after feature selection)
            y: Training target
        """
        # No feature engineering needed for dummy model
        # DummyRegressor ignores X and just learns from y
        self.model.fit(X, y)
    
    def _predict_model(self, X):
        """
        Make predictions with dummy regressor.
        
        Args:
            X: Features for prediction (after feature selection)
            
        Returns:
            Predictions array
        """
        # No feature engineering needed
        # DummyRegressor ignores X and returns learned statistic
        return self.model.predict(X)
    
    