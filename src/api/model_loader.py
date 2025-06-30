import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Add project root to path so we can import src.models.* 
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))  # Add project root, not src

class ModelLoader:
    """
    Model loader for models that use the enhanced base class with API mixin.
    Much simpler since models are self-contained and API-compatible.
    """
    
    def __init__(self, models_dir: str = None):
        # Calculate models directory relative to project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        if models_dir is None:
            self.models_dir = project_root / "models"  # Use absolute path to project's models dir
        else:
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
            from src.models.base_class import BaseModel
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
        Make a prediction using a specific model with automatic SHAP explanation.
        
        Args:
            model_name: Name of the model to use
            input_data: Dictionary with prediction inputs
            
        Returns:
            Dictionary with prediction results and SHAP explanation
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_info = self.get_model_info(model_name)
        
        try:
            if hasattr(model, 'predict_api_input') and model_info.get('supports_api', False):
                predicted_salary = model.predict_api_input(input_data)
                
                # Get model metrics in proper dictionary format
                model_metrics = {}
                if hasattr(model, 'get_api_info'):
                    api_info = model.get_api_info()
                    model_metrics = api_info.get('model_metrics', {})
                
                # Generate SHAP explanation
                shap_explanation = None
                explanation_available = False
                explanation_error = None
                
                try:
                    logger.info(f"🔍 Generating SHAP explanation for {model_name}...")
                    
                    # FIXED: Use the same preprocessing as predict_api_input()
                    # Convert input using the same method, then call explain_prediction
                    if hasattr(model, '_convert_api_input_to_dataframe'):
                        # Use the exact same conversion as predict_api_input
                        input_df = model._convert_api_input_to_dataframe(input_data)
                        
                        # Call explain_prediction with preprocessed data and model prediction
                        if hasattr(model, 'explain_prediction'):
                            explanation_result = model.explain_prediction(input_df, model_prediction=predicted_salary)
                            
                            # Validate SHAP explanation consistency with model prediction
                            shap_prediction = explanation_result.get('prediction')
                            if shap_prediction is not None:
                                difference = abs(float(shap_prediction) - predicted_salary)
                                if difference > 0.01:  # Allow small numerical differences
                                    logger.warning(f"⚠️ SHAP-model discrepancy detected: ${difference:,.2f}")
                                    logger.warning(f"Model: ${predicted_salary:,.2f}, SHAP: ${shap_prediction:,.2f}")
                                    explanation_error = f"SHAP explanation inconsistent with model prediction (diff: ${difference:,.2f})"
                                else:
                                    logger.info(f"✅ SHAP explanation validated: consistent with model prediction")
                            
                            # Structure the SHAP explanation data
                            shap_explanation = {
                                "shap_plot": explanation_result.get('shap_plot'),
                                "shap_values": explanation_result.get('shap_values', []),
                                "base_value": explanation_result.get('base_value'),
                                "feature_names": explanation_result.get('feature_names', []),
                                "prediction": explanation_result.get('prediction'),
                                "model_prediction": predicted_salary,  # Show both for validation
                                "difference": explanation_result.get('difference', 0.0),
                                "validation_passed": explanation_result.get('validation_passed', None),
                                "validation_info": explanation_result.get('validation_info', None)
                            }
                            explanation_available = True
                            logger.info(f"✅ SHAP explanation generated successfully for {model_name}")
                        else:
                            explanation_error = f"Model {model_name} does not support SHAP explanations"
                            logger.warning(f"⚠️ {explanation_error}")
                    else:
                        explanation_error = f"Model {model_name} does not support API input conversion"
                        logger.warning(f"⚠️ {explanation_error}")
                        
                except Exception as e:
                    explanation_error = f"SHAP generation failed: {str(e)}"
                    logger.error(f"❌ SHAP explanation failed for {model_name}: {e}")
                
                return {
                    "predicted_salary": predicted_salary,  # Always use model prediction as authoritative
                    "model_used": model_name,
                    "model_type": model.__class__.__name__,
                    "model_metrics": model_metrics,
                    "input_format": "enhanced_api_compatible",
                    "success": True,
                    # SHAP explanation fields
                    "shap_explanation": shap_explanation,
                    "explanation_available": explanation_available,
                    "explanation_error": explanation_error
                }
            else:
                logger.warning(f"Broken method.")
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            raise ValueError(f"Prediction failed: {e}")
    


# Global model loader instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all_models()
    return _model_loader

def reload_models():
    """Reload all models from disk."""
    global _model_loader
    if _model_loader is not None:
        _model_loader.load_all_models()
    return get_model_loader()

