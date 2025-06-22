import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Add src to path so we can import your model classes
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

class ModelLoader:
    """
    Model loader for models that use the enhanced base class with API mixin.
    Much simpler since models are self-contained and API-compatible.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
        logger.info(f"Enhanced model loader initialized, looking in: {self.models_dir}")
    
    def load_all_models(self):
        """
        Find and load all model files from the models directory.
        Works with both enhanced and legacy models.
        """
        logger.info("Loading models from directory...")
        
        # Clear existing models
        self.loaded_models.clear()
        self.model_info.clear()
        
        # Find model files
        model_files = list(self.models_dir.glob("*.pkl"))
        
        logger.info(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")
        
        if not model_files:
            logger.warning("No model files found! Train and save models using the enhanced base class")
            return
        
        # Load each model file
        for model_file in model_files:
            try:
                model_name = model_file.stem  # Use file name without extension
                model = self._load_single_model(model_file)
                
                if model is not None:
                    self.loaded_models[model_name] = model
                    
                    # Get model info (enhanced models provide this automatically)
                    if hasattr(model, 'get_api_info'):
                        self.model_info[model_name] = model.get_api_info()
                        logger.info(f"✅ Loaded enhanced model: {model_name}")
                    else:
                        logger.warning(f"⚠️(limited API support)")
                else:
                    logger.error(f"❌ Failed to load: {model_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {model_file.name}: {e}")
        
        total_count = len(self.loaded_models)
        
        logger.info(f"Successfully loaded {total_count} models.")
    
    
    def _load_single_model(self, file_path: Path) -> Optional[Any]:
        """Load a single model file using the enhanced base class loader."""
        try:
            # Try to use the enhanced base class load method
            from models.base_class import BaseModel
            return BaseModel.load(file_path.name)  # Use filename only, load() handles path resolution
            
        except ImportError as e:
            logger.error(f"Cannot import enhanced BaseModel: {e}")
            logger.info("Make sure you've updated your base class with API mixin inheritance")
            return None
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            return None
    
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.loaded_models.keys())
    
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a specific model by name."""
        return self.loaded_models.get(model_name)
    
    def get_model_info(self, model_name: str = None) -> Dict:
        """Get information about models."""
        if model_name:
            return self.model_info.get(model_name, {})
        return self.model_info
    
    def predict(self, model_name: str, input_data: Dict) -> Dict:
        """
        Make a prediction using a specific model.
        
        Args:
            model_name: Name of the model to use
            input_data: Dictionary with prediction inputs
            
        Returns:
            Dictionary with prediction results
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.get_model_info(model_name)
        
        try:
            # Check if model has API compatibility (enhanced model)
            if hasattr(model, 'predict_api_input') and model_info.get('supports_api', False):
                # Use the enhanced API method
                predicted_salary = model.predict_api_input(input_data)
                
                # Get additional info if available
                extra_info = {}
                if hasattr(model, 'get_training_metrics'):
                    extra_info['training_metrics'] = model.get_training_metrics()
                
                return {
                    "predicted_salary": predicted_salary,
                    "model_used": model_name,
                    "model_type": model.__class__.__name__,
                    "input_format": "enhanced_api_compatible",
                    "success": True,
                    **extra_info
                }
            else:
                logger.warning(f"Broken method.")
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            raise ValueError(f"Prediction failed: {e}")
    


# Global model loader instance
_model_loader = None

def get_model_loader() -> EnhancedModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = EnhancedModelLoader()
        _model_loader.load_all_models()
    return _model_loader

def reload_models():
    """Reload all models from disk."""
    global _model_loader
    if _model_loader is not None:
        _model_loader.load_all_models()
    return get_model_loader()


# Example usage / testing
if __name__ == "__main__":
    # Test the enhanced model loader
    loader = ModelLoader()
    loader.load_all_models()
    
    print(f"Available models: {loader.get_available_models()}")
    print(f"Model info: {loader.get_model_info()}")
    
    # Test prediction if models are available
    available = loader.get_available_models()
    if available:
        test_input = {
            'age': 30,
            'gender': 'Male',
            'education_level': "Bachelor's",
            'years_of_experience': 5,
            'seniority': 'Mid',
            'area': 'Engineering',
            'role': 'Engineer'
        }
        
        try:
            result = loader.predict(available[0], test_input)
            print(f"Test prediction: {result}")
        except Exception as e:
            print(f"Test prediction failed: {e}")
    else:
        print("No models available for testing")
        print("Train and save a model using the enhanced base class!")