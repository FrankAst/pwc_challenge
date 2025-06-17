from .base_class import BaseModel
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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

class DecisionTree(BaseModel):
    
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
        self.normalize = normalize
        self.ordinal_mappings = ordinal_mappings or {}
        self.preprocessor = None
        self.feature_names = None
        
        # Set default parameters for decision tree
        default_params = {
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        # User overrides take precedence
        default_params.update(kwargs)
        
        self.model = DecisionTreeRegressor(**default_params)
        
        # Store all parameters for reference
        self.model_params.update({
            'normalize': normalize,
            'ordinal_mappings': self.ordinal_mappings,
            **default_params
        })
        
    def _get_column_types(self, X):
        """
        Identify column types for preprocessing.
        
        Args:
            X: Features dataframe
            
        Returns:
            Tuple of (ordinal_cols, categorical_cols, numerical_cols)
        """
        ordinal_cols = [col for col in self.ordinal_mappings.keys() if col in X.columns]
        categorical_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns 
                           if col not in ordinal_cols]
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        return ordinal_cols, categorical_cols, numerical_cols

    def _create_pipeline(self, X):
        """
        Create complete preprocessing pipeline with ordinal and one-hot encoding.
        
        Args:
            X: Features dataframe
            
        Returns:
            ColumnTransformer for complete preprocessing
        """
        ordinal_cols, categorical_cols, numerical_cols = self._get_column_types(X)
        transformers = []
        
        # Add ordinal encoding transformers
        for col in ordinal_cols:
            order = self.ordinal_mappings[col]
            ordinal_encoder = OrdinalEncoder(
                categories=[order],
                handle_unknown='use_encoded_value',
                unknown_value=len(order)  # Use a value outside the range of encoded values
            )
            transformers.append((f'ord_{col}', ordinal_encoder, [col]))
        
        # Add numerical transformer (with or without scaling)
        if numerical_cols:
            if self.normalize:
                transformers.append(('num', MinMaxScaler(), numerical_cols))
            else:
                transformers.append(('num', 'passthrough', numerical_cols))
        
        # Add categorical transformer (one-hot encoding for remaining categoricals)
        if categorical_cols:
            transformers.append((
                'cat', 
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                categorical_cols
            ))
        
        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _fit_model(self, X, y):
        """
        Fit decision tree model with complete preprocessing pipeline.
        """
        # Create and fit preprocessing pipeline
        self.preprocessor = self._create_pipeline(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Log preprocessing info
        ordinal_cols, categorical_cols, numerical_cols = self._get_column_types(X)
        print(f"After preprocessing: {X_processed.shape[1]} features")
        print(f"  - Ordinal encoded: {ordinal_cols}")
        print(f"  - One-hot encoded: {categorical_cols}")
        print(f"  - Numerical features: {len(numerical_cols)}")
        if self.normalize:
            print("  - Features normalized")
        
        # Train decision tree
        self.model.fit(X_processed, y)
        
        # Store feature names for later use
        self.feature_names = (
            self.preprocessor.get_feature_names_out() 
            if hasattr(self.preprocessor, 'get_feature_names_out')
            else [f"feature_{i}" for i in range(X_processed.shape[1])]
        )

    def _predict_model(self, X):
        """Make predictions with same preprocessing pipeline."""
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def get_feature_importance(self, feature_names=None):
        """
        Get decision tree feature importance.
        
        Args:
            feature_names: Optional list of feature names
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Use provided names, stored names, or generate default names
        if feature_names is not None:
            names = feature_names
        elif self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        
        return pd.DataFrame({
            'feature': names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def _get_model_params_for_sklearn(self):
        """Get model parameters suitable for sklearn components."""
        return {k: v for k, v in self.model_params.items() 
                if k not in ['normalize', 'ordinal_mappings']}

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
            ('regressor', DecisionTreeRegressor(**self._get_model_params_for_sklearn()))
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
    
    def _process_cv_scores(self, scoring, cv_scores):
        """Process cross-validation scores based on scoring metric."""
        if scoring == 'neg_mean_squared_error':
            return 'RMSE', np.sqrt(-cv_scores)
        elif scoring == 'neg_root_mean_squared_error':
            return 'RMSE', -cv_scores
        elif scoring == 'neg_mean_absolute_error':
            return 'MAE', -cv_scores
        elif scoring == 'r2':
            return 'R²', cv_scores
        else:
            return scoring, cv_scores
    
    def grid_search(self, X, y, param_grid=None, cv=5, scoring='neg_root_mean_squared_error', update_params=True):
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Features dataframe
            y: Target variable
            param_grid: Dictionary of parameters to search over
            cv: Number of CV folds (default: 5)
            scoring: Scoring metric (default: 'neg_root_mean_squared_error')
            update_params: Whether to update model parameters with best found params (default: True)
            
        Returns:
            GridSearchCV object with results
        """
        if param_grid is None:
            param_grid = {
                'regressor__max_depth': [3, 5, 10, None],
                'regressor__min_samples_split': [2, 5, 10,20,30,40],
                'regressor__min_samples_leaf': [1, 2, 4,10,15,20],
                'regressor__max_features': ['sqrt', 'log2', None],
                'regressor__ccp_alpha': [0.0, 0.01, 0.05, 0.1, 0.2,0.5,1,1.5]
            }
        
        pipeline = Pipeline([
            ('preprocessor', self._create_pipeline(X)),
            ('regressor', DecisionTreeRegressor(**self._get_model_params_for_sklearn()))
        ])
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)
        
        print(f"Best score: {-grid_search.best_score_:.2f}")
        print(f"Best params: {grid_search.best_params_}")
        
        if update_params:
            self._update_params_from_grid_search(grid_search)
            print("Model parameters updated with best found parameters.")
            print("You can now use fit() or cross_validate() with the optimized parameters.")
        
        return grid_search
    
    def _update_params_from_grid_search(self, grid_search_result):
        """Update model parameters with best parameters from grid search."""
        best_params = grid_search_result.best_params_
        
        # Extract regressor parameters (remove 'regressor__' prefix)
        regressor_params = {
            key.replace('regressor__', ''): value 
            for key, value in best_params.items() 
            if key.startswith('regressor__')
        }
        
        # Update stored parameters
        self.model_params.update(regressor_params)
        
        # Create new model instance with updated parameters
        self.model = DecisionTreeRegressor(**self._get_model_params_for_sklearn())
        
        # Reset fitted status since we have a new model
        self.is_fitted = False
        
        print(f"Updated parameters: {regressor_params}")




