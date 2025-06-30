import numpy as np
import pandas as pd

# Shap imports
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .base_class import BaseModel

"""
Processing steps:
1- Selects model
2- Applies min-max scaling to numerical features
3- Applies one-hot encoding to categorical features
4- Applies feature selection using f_regression
5- Applies VIF feature selection to remove multicollinearity
6- Fits the model

"""

class LinearModel(BaseModel):
    """Linear model with comprehensive preprocessing for categorical and numerical features."""
    
    def __init__(self, algorithm='ols', vif_threshold=5.0, k_best=None, **kwargs):
        """
        Initialize linear model.
        
        Args:
            algorithm: Linear algorithm ('ols', 'ridge', 'lasso', 'sgd')
            vif_threshold: VIF threshold for multicollinearity check (default: 5.0)
            k_best: Number of features to select with k_best (None = no selection)
            **kwargs: Additional parameters (alpha for ridge/lasso, etc.)
        """
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.vif_threshold = vif_threshold
        self.k_best = k_best
        self.preprocessor = None
        self.vif_selected_features = None
        self.kbest_selector = None
        
        self.background_data = None # For SHAP background data
        self._shap_explainer = None
        
        # Create appropriate sklearn model
        if algorithm == 'ols':
            self.model = LinearRegression()
        elif algorithm == 'ridge':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha)
        elif algorithm == 'lasso':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Lasso(alpha=alpha)
        elif algorithm == 'sgd':
            self.model = SGDRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Add parameters to model_params
        self.model_params.update({
            'algorithm': algorithm,
            'vif_threshold': vif_threshold,
            'k_best': k_best,
            **kwargs
        })
    
    def _apply_kbest_selection(self, X_encoded, y=None, fit=False):
        """
        Apply k_best feature selection after encoding.
        
        Args:
            X_encoded: Encoded features (numerical matrix)
            y: Target (only needed when fit=True)
            fit: If True, learn selection. If False, apply learned selection.
            
        Returns:
            X_encoded with selected features
        """
        if self.k_best is None:
            return X_encoded
        
        if fit:
            # Learn k_best selection on encoded features
            self.kbest_selector = SelectKBest(f_regression, k=self.k_best)
            X_selected = self.kbest_selector.fit_transform(X_encoded, y)
            print(f"Selected {self.k_best} best features from {X_encoded.shape[1]} features")
            return X_selected
        else:
            # Apply learned selection
            if self.kbest_selector is not None:
                return self.kbest_selector.transform(X_encoded)
            return X_encoded
    
    def _check_multicollinearity_vif(self, X_encoded, fit=False):
        """
        Check and remove features with high multicollinearity using VIF.
        This should be applied AFTER encoding and feature selection.
        
        Args:
            X_encoded: Encoded features (numerical matrix)
            fit: If True, learn VIF selection. If False, apply learned selection.
            
        Returns:
            X_encoded with high VIF features removed
        """
        if X_encoded.shape[1] <= 1:
            # Can't check VIF with <= 1 feature
            return X_encoded
        
        if fit:
            # Learn VIF selection from training data
            # Convert to DataFrame with consistent column names
            if isinstance(X_encoded, pd.DataFrame):
                X_vif = X_encoded.copy()
                remaining_features = list(range(X_encoded.shape[1]))  # Use indices instead of names
            else:
                X_vif = pd.DataFrame(X_encoded)
                remaining_features = list(range(X_encoded.shape[1]))
            
            # Convert DataFrame to numpy array for VIF calculation
            X_values = X_vif.values
            
            # Iteratively remove features with high VIF
            while len(remaining_features) > 1:
                # Get current feature subset
                X_current = X_values[:, remaining_features]
                
                # Calculate VIF for remaining features
                vif_scores = []
                for i in range(len(remaining_features)):
                    try:
                        vif = variance_inflation_factor(X_current, i)
                        vif_scores.append(vif)
                    except:
                        # Handle numerical issues
                        vif_scores.append(np.inf)
                
                # Find feature with highest VIF
                max_vif_idx = np.argmax(vif_scores)
                max_vif = vif_scores[max_vif_idx]
                
                if max_vif > self.vif_threshold:
                    # Remove feature with highest VIF
                    feature_to_remove = remaining_features[max_vif_idx]
                    remaining_features.remove(feature_to_remove)
                    print(f"Removed feature_{feature_to_remove} (VIF: {max_vif:.2f})")
                else:
                    break
            
            # Store indices of selected features
            self.vif_selected_features = remaining_features
        
        # Apply VIF selection
        if self.vif_selected_features is not None:
            if isinstance(X_encoded, pd.DataFrame):
                return X_encoded.iloc[:, self.vif_selected_features]
            else:
                return X_encoded[:, self.vif_selected_features]
        
        return X_encoded
    
    def _fit_model(self, X, y):
        """
        Fit linear model with comprehensive preprocessing.
        
        Preprocessing pipeline:
        1. One-hot encoding for categorical features (NO scaling)
        2. Min-max scaling for ONLY numerical features  
        3. K-best feature selection (selects most predictive features)
        4. VIF multicollinearity check (removes high VIF features from selected set)
        5. Train linear model
        """
        # Step 1 & 2: One-hot encoding (no scaling) + Min-max scaling (numerical only)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipeline - DON'T scale one-hot features!
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            # Both categorical and numerical features
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numerical_cols),  # Scale numerical only
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)  # No scaling for one-hot
                ],
                remainder='drop'
            )
        elif len(categorical_cols) > 0:
            # Only categorical features - no scaling needed
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
                ],
                remainder='drop'
            )
        elif len(numerical_cols) > 0:
            # Only numerical features - scale them
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numerical_cols)
                ],
                remainder='drop'
            )
        else:
            raise ValueError("No features available for preprocessing")
        
        # Step 1-2: Apply encoding and scaling (scaling only applied to numerical features)
        X_encoded = self.preprocessor.fit_transform(X)
        print(f"After encoding: {X_encoded.shape[1]} features")
        print(f"  - Numerical features (scaled): {len(numerical_cols)}")
        print(f"  - One-hot features (not scaled): {X_encoded.shape[1] - len(numerical_cols)}")
        
        # Step 3: K-best feature selection first (select most predictive)
        X_after_kbest = self._apply_kbest_selection(X_encoded, y, fit=True)
        print(f"After k-best selection: {X_after_kbest.shape[1]} features")
        
        # Step 4: VIF multicollinearity check on selected features
        X_final = self._check_multicollinearity_vif(X_after_kbest, fit=True)
        print(f"After VIF check: {X_final.shape[1]} features")
        
        # Step 5: Train linear model
        self.model.fit(X_final, y)
        
        # Save X_train sample for SHAP background data
        np.random.seed(37)
        sample_indices = np.random.choice(len(X_final), size=200, replace=False)
        background_sample = X_final[sample_indices]
        self._set_background_data(background_sample)
        
        # Store feature info for debugging
        self.n_features_after_encoding = X_encoded.shape[1]
        self.n_features_after_kbest = X_after_kbest.shape[1]
        self.n_features_final = X_final.shape[1]
        self.n_numerical_features = len(numerical_cols)
        self.n_onehot_features = X_encoded.shape[1] - len(numerical_cols)
    
    def _predict_model(self, X):
        """
        Make predictions with same preprocessing pipeline.
        """
        # Use the same preprocessing method as explain_prediction
        X_final = self._preprocess_for_prediction(X)
        
        # Step 5: Predict
        return self.model.predict(X_final)
    
    def _preprocess_for_prediction(self, X):
        """
        Transform input data through the same preprocessing pipeline used during training.
        
        Args:
            X: Raw input data (DataFrame)
            
        Returns:
            X_transformed: Preprocessed data ready for model prediction
        """
        # Step 1 & 2: Apply encoding and scaling
        X_encoded = self.preprocessor.transform(X)
        
        # Step 3: Apply k-best selection
        X_after_kbest = self._apply_kbest_selection(X_encoded, fit=False)
        
        # Step 4: Apply VIF selection
        X_final = self._check_multicollinearity_vif(X_after_kbest, fit=False)
        
        return X_final
    
    def get_feature_importance(self, feature_names=None):
        """
        Get linear model coefficients as feature importance.
        
        Args:
            feature_names: Optional feature names (will be inferred if None)
        
        Returns:
            DataFrame with features, coefficients, and absolute coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # The model coefficients correspond to features after all preprocessing and selection
        # So we need to create feature names that match the final feature count
        n_final_features = len(self.model.coef_)
        
        # Try to get meaningful feature names, but ensure count matches
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            all_feature_names = self.preprocessor.get_feature_names_out()
            
            # If we have K-best selection, we need to map to selected features
            if self.kbest_selector is not None:
                selected_indices = self.kbest_selector.get_support(indices=True)
                selected_feature_names = all_feature_names[selected_indices]
                
                # If we have VIF selection, further filter the names
                if self.vif_selected_features is not None and isinstance(self.vif_selected_features, list):
                    if len(self.vif_selected_features) <= len(selected_feature_names):
                        # VIF selected features are indices into the K-best selected features
                        if all(isinstance(x, int) for x in self.vif_selected_features):
                            final_feature_names = selected_feature_names[self.vif_selected_features]
                        else:
                            # VIF selected features are feature names
                            final_feature_names = self.vif_selected_features
                    else:
                        final_feature_names = [f"feature_{i}" for i in range(n_final_features)]
                else:
                    final_feature_names = selected_feature_names
            else:
                # No K-best, but might have VIF selection
                if self.vif_selected_features is not None and len(self.vif_selected_features) == n_final_features:
                    if all(isinstance(x, str) for x in self.vif_selected_features):
                        final_feature_names = self.vif_selected_features
                    else:
                        final_feature_names = all_feature_names[self.vif_selected_features]
                else:
                    final_feature_names = all_feature_names[:n_final_features]
        else:
            final_feature_names = [f"feature_{i}" for i in range(n_final_features)]
        
        # Ensure we have the right number of feature names
        if len(final_feature_names) != n_final_features:
            print(f"Warning: Feature name count mismatch. Expected {n_final_features}, got {len(final_feature_names)}")
            final_feature_names = [f"feature_{i}" for i in range(n_final_features)]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': final_feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        return importance_df.sort_values('abs_coefficient', ascending=False)
    
    def get_model_info(self):
        """Enhanced model info for linear models."""
        info = super().get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_features_after_encoding': getattr(self, 'n_features_after_encoding', None),
                'n_features_after_kbest': getattr(self, 'n_features_after_kbest', None),
                'n_features_final': getattr(self, 'n_features_final', None),
                'n_numerical_features_scaled': getattr(self, 'n_numerical_features', None),
                'n_onehot_features_not_scaled': getattr(self, 'n_onehot_features', None),
                'features_removed_by_kbest': getattr(self, 'n_features_after_encoding', 0) - getattr(self, 'n_features_after_kbest', 0),
                'features_removed_by_vif': getattr(self, 'n_features_after_kbest', 0) - getattr(self, 'n_features_final', 0)
            })
        
        return info
    
    ######################################## SHAP explanation methods ########################################
    
    def _set_background_data(self, X_train_sample):
        """Store background data for LinearExplainer"""
        self.background_data = X_train_sample
    
    def _get_final_feature_names(self):
        """
        Get the meaningful feature names that correspond to the final model features.
        This tracks feature names through the entire preprocessing pipeline.
        """
        if not self.is_fitted:
            return None
            
        # Get feature names after encoding
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            all_feature_names = self.preprocessor.get_feature_names_out()
            
            # Apply K-best selection if used
            if self.kbest_selector is not None:
                selected_indices = self.kbest_selector.get_support(indices=True)
                selected_feature_names = all_feature_names[selected_indices]
            else:
                selected_feature_names = all_feature_names
            
            # Apply VIF selection if used
            if self.vif_selected_features is not None:
                if all(isinstance(x, int) for x in self.vif_selected_features):
                    # VIF selected features are indices
                    final_feature_names = selected_feature_names[self.vif_selected_features]
                else:
                    # VIF selected features are already names
                    final_feature_names = self.vif_selected_features
            else:
                final_feature_names = selected_feature_names
                
            return final_feature_names
        
        return None

    def explain_prediction(self, X):
        """Generate SHAP explanation for predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating explanations")
            
        # Ensure SHAP explainer is initialized
        if not hasattr(self, '_shap_explainer') or self._shap_explainer is None:
            if not hasattr(self, 'background_data') or self.background_data is None:
                raise ValueError("Linear model needs background data for SHAP. Model may need to be retrained.")
            # LinearExplainer needs background data for baseline
            import shap
            self._shap_explainer = shap.LinearExplainer(self.model, self.background_data)
        
        # Transform input data through the same preprocessing pipeline
        X_transformed = self._preprocess_for_prediction(X)
        
        # Get meaningful feature names
        feature_names = self._get_final_feature_names()
        
        return self._generate_shap_plot(X_transformed, X, feature_names)
    
    def _generate_shap_plot(self, X_transformed, X_original, feature_names=None):
        # Use the transformed data for SHAP calculation (same shape as background_data)
        shap_values = self._shap_explainer(X_transformed)
        
        # ⚠️ CRITICAL FIX: Use transformed data for prediction to match SHAP calculation
        # This ensures the prediction matches the same data space as SHAP values
        model_prediction = self.model.predict(X_transformed)[0]
        
        # Validate SHAP math: base_value + sum(shap_values) should ≈ prediction
        shap_sum = sum(shap_values.values[0])
        expected_prediction = float(shap_values.base_values[0]) + shap_sum
        prediction_diff = abs(expected_prediction - model_prediction)
        
        if prediction_diff > 1.0:  # Allow small numerical differences
            print(f"⚠️ SHAP Math Warning:")
            print(f"  Model prediction: ${model_prediction:,.2f}")
            print(f"  Base value: ${float(shap_values.base_values[0]):,.2f}")
            print(f"  SHAP sum: ${shap_sum:,.2f}")
            print(f"  Expected (base + SHAP): ${expected_prediction:,.2f}")
            print(f"  Difference: ${prediction_diff:,.2f}")
        
        # If we have meaningful feature names, create a new Explanation object with them
        if feature_names is not None and len(feature_names) == len(shap_values.values[0]):
            # Create a new SHAP Explanation with meaningful feature names
            shap_values_with_names = shap.Explanation(
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=shap_values.data,
                feature_names=feature_names
            )
            # Generate waterfall plot with meaningful names
            shap.plots.waterfall(shap_values_with_names[0], show=False)
        else:
            # Fallback to original plot
            shap.plots.waterfall(shap_values[0], show=False)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "prediction": model_prediction,  # ← Now uses consistent transformed data
            "shap_plot": f"data:image/png;base64,{img_base64}",
            "shap_values": shap_values.values[0].tolist(),
            "base_value": float(shap_values.base_values[0]),
            "feature_names": feature_names.tolist() if feature_names is not None else None
        }