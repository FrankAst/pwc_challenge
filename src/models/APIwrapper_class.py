"""
API functionality mixin that can be inherited by your base class.
Keeps API concerns separate from core ML functionality.
"""

import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List



class APIwrapper:
    """
    Mixin class that adds API compatibility, save/load, and metadata management.
    Designed to be inherited alongside ABC in your BaseModel.
    """
    
    def __init_api_attributes__(self):
        """
        Initialize API-related attributes.
        Call this in your BaseModel.__init__()
        """
        
        # API compatibility attributes
        self.feature_names_ = None
        self.target_name_ = 'Salary'
        self.model_metrics_ = pd.DataFrame()  # Changed to model_metrics_ and DataFrame
        self.model_info_ = {}
        self.created_at_ = datetime.now().isoformat()
        
        
    def save(self, filename: str):
        """
        Save the model to the models/ directory with all metadata.
        
        Args:
            filename: Name of the file (e.g., 'my_model.pkl')
                     Will be saved to project_root/models/filename
        """
        # Find project root
        project_root = self._find_project_root(Path.cwd())
        
        # Create models directory path
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Full file path
        filepath = models_dir / filename
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        
        # Update model info before saving
        self.model_info_.update({
            'model_class': self.__class__.__name__,
            'saved_at': datetime.now().isoformat(),
            'feature_count': len(self.feature_names_) if self.feature_names_ else 0,
            'target_name': self.target_name_,
            'is_fitted': getattr(self, 'is_fitted', False),
            'file_path': str(filepath.relative_to(project_root))
        })
        
        try:
            joblib.dump(self, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise
        
    @classmethod
    def load(cls, filename: str):
        """
        Load a model from the models/ directory.
        
        Args:
            filename: Name of the file (e.g., 'my_model.pkl')
                     Will look in project_root/models/filename
            
        Returns:
            Loaded model instance
        """
        # Find project root
        project_root = cls._find_project_root_static(Path.cwd())
        
        # Look for the file in models directory
        models_dir = project_root / "models"
        filepath = models_dir / filename
        
        # Try with .pkl extension if no extension provided
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
            
        except Exception as e:
            print(f"Failed to load model from {filepath}: {e}")
            raise
    
    
    # API compatibility methods
    def predict_api_input(self, input_dict: Dict[str, Any]) -> float:
        """
        Make prediction from API-style input dictionary.
        
        Args:
            input_dict: Dictionary with API input format
                       e.g., {'age': 30, 'gender': 'Male', ...}
        
        Returns:
            Single prediction value
        """
        if not getattr(self, 'is_fitted', False):
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert API input to DataFrame
        df_input = self._convert_api_input_to_dataframe(input_dict)
        
        # Make prediction using the model's predict method
        prediction = self.predict(df_input)
        
        # Return single value
        return float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
    
    def _convert_api_input_to_dataframe(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert API input dictionary to DataFrame format.
        
        Args:
            input_dict: API input dictionary
            
        Returns:
            DataFrame ready for model prediction
        """
        # Standard API to model column mapping
        column_mapping = {
            'age': 'Age',
            'gender': 'Gender',
            'education_level': 'Education Level',
            'years_of_experience': 'Years of Experience',
            'seniority': 'Seniority',
            'area': 'Area',
            'role': 'Role'
        }
        
        # Convert API input to model format
        model_data = {}
        for api_key, value in input_dict.items():
            model_key = column_mapping.get(api_key, api_key)
            model_data[model_key] = value
        
        # Create DataFrame
        df = pd.DataFrame([model_data])
        
        # If we have stored feature names, ensure all are present
        if self.feature_names_:
            missing_features = set(self.feature_names_) - set(df.columns)
            if missing_features:
                print(f"Warning: Missing features in input: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    df[feature] = 0  # or some other default
            
            # Reorder columns to match training order
            df = df[self.feature_names_]
        
        return df
    
    # Metadata management methods
    def set_model_metrics(self, metrics: pd.DataFrame):
        """
        Set training metrics for the model.
        
        Args:
            metrics: DataFrame with metrics
        """
        self.model_metrics_ = metrics.copy()
        print(f"Model metrics updated with {len(self.model_metrics_)} metrics")
    
    def get_model_metrics(self) -> pd.DataFrame:
        """Get training metrics as DataFrame."""
        return self.model_metrics_.copy()
    
    
    def set_feature_names(self, feature_names: List[str]):
        """
        Set feature names for the model.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names_ = feature_names.copy()
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names used during training."""
        return self.feature_names_.copy() if self.feature_names_ else None
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get information for API integration.
        
        Returns:
            Dictionary with model information for API
        """
        # Convert DataFrame metrics to dictionary format for API
        metrics_dict = {}
        if hasattr(self, 'model_metrics_') and not self.model_metrics_.empty:
            # Convert DataFrame to a more API-friendly format
            for _, row in self.model_metrics_.iterrows():
                metric_name = row['metric']
                point_estimate = row['point_estimate']
                ci_columns = [col for col in row.index if 'CI' in col or 'ci' in col]
                
                metrics_dict[metric_name] = {
                    'value': float(point_estimate),
                    'confidence_interval': row[ci_columns[0]] if ci_columns else None
                }
        
        info = {
            'model_class': self.__class__.__name__,
            'is_fitted': getattr(self, 'is_fitted', False),
            'feature_names': self.feature_names_,
            'feature_count': len(self.feature_names_) if self.feature_names_ else 0,
            'target_name': self.target_name_,
            'model_metrics': metrics_dict,  # Now a proper dictionary instead of DataFrame
            'model_params': getattr(self, 'model_params', {}),
            'created_at': self.created_at_,
            'supports_predict': hasattr(self, 'predict'),
            'supports_api': True
        }
        info.update(self.model_info_)
        return info
    
    # Helper methods for path resolution
    def _find_project_root(self, start_path: Path) -> Path:
        """Find the project root directory."""
        return self._find_project_root_static(start_path)
    
    @staticmethod
    def _find_project_root_static(start_path: Path) -> Path:
        """
        Find the project root directory by looking for common indicators.
        
        Args:
            start_path: Path to start searching from
            
        Returns:
            Path to project root
        """
        current = start_path.resolve()
        
        # Look for common project root indicators
        indicators = [
            'requirements.txt',
            'setup.py',
            'pyproject.toml',
            '.git',
            'src',
            'models',
            'data'
        ]
        
        # Walk up the directory tree
        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # If no indicators found, use current directory
        print(f"Warning: Could not find project root, using: {current}")
        return current
