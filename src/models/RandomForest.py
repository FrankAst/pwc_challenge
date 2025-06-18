from .base_class import BaseModel
from .DecisionTree import DecisionTree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

"""
Workflow:
1. Initialize model with parameters
2. Apply ordinal encoding for specified columns if provided
3. Apply one hot encoding for remaining categorical columns
4. Apply normalization if specified (for numeric and ordinal features)
5. Fit the model using the preprocessed data
6. Predict using the fitted model
7. Evaluate the model performance
"""

class RandomForest(DecisionTree):
    
    def __init__(self, normalize=False, ordinal_mappings=None, **kwargs):
        """
        Initialize decision tree model.
        
        Args:
            normalize: Whether to normalize features (default: False)
            ordinal_mappings: Dictionary of {column_name: [ordered_categories]} for ordinal encoding
                            Example: {'Education Level': ['High School', "Bachelor's", "Master's", 'PhD'],
                                     'Seniority': ['Junior', 'Senior', 'Director', 'Principal', 'C-level']}
            **kwargs: DecisionTreeRegressor parameters
        """
        super().__init__(**kwargs)
        
        # Set default parameters for decision tree
        default_params = {}
        # User overrides take precedence
        default_params.update(kwargs)
        
        self.model = RandomForestRegressor()
        
        # Store all parameters for reference
        self.model_params.update({
            'normalize': normalize,
            'ordinal_mappings': self.ordinal_mappings
        })   


    def cross_validate(self, X, y, cv=5, scoring='neg_root_mean_squared_error'):
        """
        Perform cross-validation on the decision tree model.
        
        Args:
            X: Features dataframe
            y: Target variable
            cv: Number of CV folds or CV splitter (default: 5)
            scoring: Scoring metric (default: 'neg_root_mean_squared_error')
            
        Returns:
            Dictionary with CV results including mean and std of test scores
        """
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', self._create_pipeline(X)),
            ('regressor', RandomForestRegressor(**self._get_model_params_for_sklearn()))
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            pipeline, X, y, 
            cv=cv, 
            scoring=scoring
        )
        
        # Convert scores to positive values and determine score name
        score_name, cv_scores = self._process_cv_scores(scoring, cv_scores)
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'score_name': score_name,
            'cv_folds': cv if isinstance(cv, int) else len(cv_scores)
        }
        
        print(f"Cross-Validation Results ({results['cv_folds']}-fold):")
        print(f"  {score_name}: {results['mean_score']:.2f} (±{results['std_score']:.2f})")
        print(f"  Individual fold scores: {[f'{score:.2f}' for score in cv_scores]}")
        
        return results
    
  
    
    def optimize(self, X, y, param_grid=None, cv=5, scoring='neg_root_mean_squared_error', update_params=True):
        
        return 
    
   




