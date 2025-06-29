import sys
from pathlib import Path

# Add project root to Python path for imports
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Go up from src/api/ to root
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from datetime import datetime
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Import from your existing model_loader and config
from model_loader import get_model_loader, reload_models
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from src.data_preparation_workflow.FE_text import aggregate_categories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This replaces the deprecated @app.on_event("startup") decorator.
    """
    # Startup
    logger.info("🚀 Starting Salary Prediction API - Phase 4")
    logger.info(f"📁 Project root: {project_root}")
    
    # Initialize model loader (this loads all models)
    model_loader = get_model_loader()
    available_models = model_loader.get_available_models()
    
    logger.info(f"📊 Loaded {len(available_models)} models:")
    for model_name in available_models:
        logger.info(f"  - {model_name} ")
    
    yield  # Application runs here
    
    # Shutdown (if needed)
    logger.info("🛑 Shutting down Salary Prediction API")

def validate_and_standardize_input(input_data: dict) -> dict:
    """
    Validate and standardize input data using your aggregate_categories function.
    This ensures the input matches the categories your models were trained on.
    """
    validated_data = input_data.copy()
    
    # Standardize area and role using your FE_text function
    if 'area' in validated_data:
        original_area = validated_data['area']
        validated_data['area'] = aggregate_categories(original_area, 'area')
        if original_area != validated_data['area']:
            logger.info(f"📝 Standardized area: '{original_area}' → '{validated_data['area']}'")
    
    if 'role' in validated_data:
        original_role = validated_data['role']
        validated_data['role'] = aggregate_categories(original_role, 'role')
        if original_role != validated_data['role']:
            logger.info(f"📝 Standardized role: '{original_role}' → '{validated_data['role']}'")
    
    return validated_data

# Pydantic models for API
class PredictionInput(BaseModel):
    """Input model with validation matching your config. 
    Note: 'area' and 'role' values will be automatically standardized using aggregate_categories."""
    age: float = Field(..., ge=16, le=100, description="Age in years (16-100)")
    gender: str = Field(..., description="Gender")
    education_level: str = Field(..., description="Education level")
    years_of_experience: float = Field(..., ge=0, le=60, description="Years of experience (0-50)")
    seniority: str = Field(..., description="Seniority level")
    area: str = Field(..., description="Business area (will be standardized to trained categories)")
    role: str = Field(..., description="Job role (will be standardized to trained categories)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 30.0,
                "gender": "Male",
                "education_level": "Bachelor's",
                "years_of_experience": 5.0,
                "seniority": "Junior",
                "area": "Engineering",  # Will be standardized to "Software/data"
                "role": "Software Engineer"  # Will be standardized to "Engineer"
            }
        }

class PredictionOutput(BaseModel):
    """Output model for salary predictions."""
    predicted_salary: float = Field(..., description="Predicted salary in USD")
    model_used: str = Field(..., description="Name of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")
    confidence_info: Optional[Dict[str, Any]] = Field(None, description="Additional model info")


############################################ API Setup ############################################

# Create FastAPI app with lifespan
app = FastAPI(
    title="Salary Prediction API",
    description="Phase 4: Real model endpoints with trained ML models",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    model_loader = get_model_loader()
    available_models = model_loader.get_available_models()
    
    # Get default model (first available)
    default_model = available_models[0] if available_models else None
    
    return {
        "message": "Salary Prediction API - Phase 4", 
        "version": "4.0.0",
        "description": "Real model endpoints using your trained models",
        "project": "Data Science Challenge - Salary Prediction",
        "models_available": len(available_models),
        "available_models": available_models,
        "default_model": default_model,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "target": TARGET_COLUMN
        },
        "endpoints": {
            "health": "/health",
            "models": "/models", 
            "model_info": "/models/{model_name}",
            "predict": "/predict",
            "predict_specific": "/predict/{model_name}",
            "features": "/models/{model_name}/features",
            "categories": "/categories",
            "reload": "/models/reload",
            "docs": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    """Enhanced health check with model status."""
    model_loader = get_model_loader()
    available_models = model_loader.get_available_models()
    
    return {
        "status": "healthy",
        "version": "4.0.0",
        "models_loaded": len(available_models),
        "available_models": available_models,
        "default_model": available_models[0] if available_models else None,
        "project_root": str(project_root),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models")
def list_models():
    """List all available models with detailed information."""
    try:
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        
        # Build detailed model list
        models_detail = []
        for model_name in available_models:
            model_info = model_loader.get_model_info(model_name)
            models_detail.append({
                "name": model_name,
                "type": model_info.get('model_class', 'Unknown'),
                "status": "available" if model_info.get('is_fitted', False) else "not_fitted",
                "description": f"{model_info.get('model_class', 'Model')} for salary prediction",
                "feature_count": model_info.get('feature_count', 0),
                "model_metrics": model_info.get('model_metrics', {}),
                "created_at": model_info.get('created_at', 'Unknown'),
                "supports_api": model_info.get('supports_api', True)
            })
        
        return {
            "available_models": models_detail,
            "total_models": len(available_models),
            "default_model": available_models[0] if available_models else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model list: {e}")

@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Get detailed info from model loader
        model_info = model_loader.get_model_info(model_name)
        
        return {
            "name": model_name,
            "type": model_info.get('model_class', 'Unknown'),
            "status": "available" if model_info.get('is_fitted', False) else "not_fitted",
            "created_at": model_info.get('created_at', 'Unknown'),
            "feature_names": model_info.get('feature_names', []),
            "feature_count": model_info.get('feature_count', 0),
            "target_name": model_info.get('target_name', TARGET_COLUMN),
            "model_metrics": model_info.get('model_metrics', {}),
            "model_params": model_info.get('model_params', {}),
            "supports_predict": model_info.get('supports_predict', False),
            "supports_api": model_info.get('supports_api', False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")

@app.post("/predict", response_model=PredictionOutput)
def predict_salary(input_data: PredictionInput):
    """Make salary prediction using the default model (first available)."""
    model_loader = get_model_loader()
    available_models = model_loader.get_available_models()
    
    if not available_models:
        raise HTTPException(
            status_code=503,
            detail="No models available for prediction. Please train and save models first."
        )
    
    # Use first available model as default
    default_model = available_models[0]
    return predict_with_model(default_model, input_data)

@app.post("/predict/{model_name}", response_model=PredictionOutput)  
def predict_with_model(model_name: str, input_data: PredictionInput):
    """Make salary prediction using a specific model."""
    logger.info(f"🔮 Prediction request: {input_data.role} in {input_data.area} using {model_name}")
    
    try:
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        
        # Check if model exists
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Convert Pydantic input to dictionary and validate/standardize
        input_dict = input_data.dict()
        validated_input = validate_and_standardize_input(input_dict)
        
        # Use your model loader's predict method
        result = model_loader.predict(model_name, validated_input)
        
        # Extract prediction from result
        predicted_salary = result.get('predicted_salary')
        if predicted_salary is None:
            raise ValueError("Model prediction returned no salary value")
        
        # Get model metrics directly from the result
        model_metrics = result.get('model_metrics', {})
        
        # Create response with metrics prominently displayed
        response = PredictionOutput(
            predicted_salary=round(float(predicted_salary), 2),
            model_used=model_name,
            timestamp=datetime.utcnow().isoformat(),
            confidence_info={
                "model_type": result.get('model_type', 'Unknown'),
                "model_metrics": model_metrics,  # Include metrics here for visibility
                "performance": {
                    "RMSE": model_metrics.get('RMSE', {}),
                    "MAE": model_metrics.get('MAE', {}), 
                    "R2": model_metrics.get('R2', {})
                },
                "input_format": result.get('input_format', 'enhanced_api_compatible'),
                "success": result.get('success', True),
                "input_validated": True,
                "note": "Prediction from trained model with performance metrics"
            }
        )
        
        logger.info(f"✅ Prediction successful: ${predicted_salary:,.2f} using {model_name}")
        logger.info(f"📊 Model metrics: RMSE={model_metrics.get('RMSE', {}).get('value', 'N/A')}, R2={model_metrics.get('R2', {}).get('value', 'N/A')}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Prediction validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/models/reload")
def reload_models_endpoint():
    """Reload all models from the models directory."""
    try:
        logger.info("🔄 Reloading models...")
        model_loader = reload_models()  # Use your reload function
        
        available_models = model_loader.get_available_models()
        
        # Build response with model details
        models_detail = []
        for model_name in available_models:
            model_info = model_loader.get_model_info(model_name)
            models_detail.append({
                "name": model_name,
                "type": model_info.get('model_class', 'Unknown'),
                "status": "available" if model_info.get('is_fitted', False) else "not_fitted"
            })
        
        return {
            "message": "Models reloaded successfully",
            "models_loaded": len(available_models),
            "models": models_detail,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")

@app.get("/models/{model_name}/features")
def get_model_features(model_name: str):
    """Get feature information for a specific model."""
    try:
        model_loader = get_model_loader()
        available_models = model_loader.get_available_models()
        
        if model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available: {available_models}"
            )
        
        model_info = model_loader.get_model_info(model_name)
        
        return {
            "model_name": model_name,
            "feature_names": model_info.get('feature_names', []),
            "feature_count": model_info.get('feature_count', 0),
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "target": TARGET_COLUMN,
            "input_format": {
                "age": "float (16-100)",
                "gender": "string",
                "education_level": "string",
                "years_of_experience": "float (0-50)",
                "seniority": "string", 
                "area": "string",
                "role": "string"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting features for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving features: {e}")

@app.get("/debug/info")
def debug_info():
    """Debug information about the API setup."""
    model_loader = get_model_loader()
    available_models = model_loader.get_available_models()
    
    return {
        "project_root": str(project_root),
        "models_directory": str(model_loader.models_dir),
        "models_dir_exists": model_loader.models_dir.exists(),
        "pkl_files": [f.name for f in model_loader.models_dir.glob("*.pkl")] if model_loader.models_dir.exists() else [],
        "loaded_models": available_models,
        "model_loader_type": type(model_loader).__name__,
        "python_path": sys.path[:3],  # First 3 entries
        "config": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "target": TARGET_COLUMN
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Salary Prediction API - Phase 4")
    logger.info(f"📁 Working from: {project_root}")
    logger.info("🔧 Using existing model_loader.py for real model endpoints!")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )