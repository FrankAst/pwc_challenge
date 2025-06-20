from .DecisionTree import DecisionTree
from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from src.config import SEED

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
        
        self.model = RandomForestRegressor(random_state= SEED)
        
        # Store all parameters for reference
        self.model_params.update({
            'normalize': normalize,
            'ordinal_mappings': self.ordinal_mappings
        })   


    def cross_validate(self, X, y, cv=5, scoring='neg_root_mean_squared_error', return_ci = True):
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
        self._get_column_types(X)
        
        pipeline = Pipeline([
            ('preprocessor', self._create_pipeline()),
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
        
        if return_ci:
            return results
        else:
            return results['mean_score']
    
  
    
    def optimize(self, X, y, cv=5, trials = 50, scoring='neg_root_mean_squared_error'):
        
        self._get_column_types(X)
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
           
           
            # Given the small amount of data, I will use CV to reduce parameters overfitting.
        
            pipeline = Pipeline([
                ('preprocessor', self._create_pipeline()),
                ('regressor', RandomForestRegressor(n_estimators= n_estimators,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    max_features=max_features,
                                                    random_state=SEED))
        ])
            score = -cross_val_score(pipeline, X, y, cv=cv, scoring = scoring,n_jobs=-1).mean()

            return score
   
        study = optuna.create_study()  # Create a new study.
        study.optimize(objective, n_trials = trials)  # Invoke optimization of the objective function.
    
    
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value:.2f}")
    
        # Update model parameters with the best found parameters 
        self.model_params.update(study.best_params)
        
        # Create a new model instance with the best parameters
        best_params_with_seed = study.best_params.copy()
        best_params_with_seed['random_state'] = SEED
        self.model = RandomForestRegressor(**best_params_with_seed)
        
        self.fit(X, y)  # Fit the model with the best parameters    
        
        return study
    
   




