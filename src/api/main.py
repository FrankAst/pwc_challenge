from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple models - we'll refine these later
class PredictionInput(BaseModel):
    """Simple input model - we'll improve validation later"""
    age: float
    gender: str
    education_level: str
    years_of_experience: float
    seniority: str
    area: str
    role: str

class PredictionOutput(BaseModel):
    """Simple output model"""
    predicted_salary: float
    model_used: str
    timestamp: str

# Create FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="Focus on API structure first",
    version="0.2.0"
)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Salary Prediction API",
        "version": "0.2.0",
        "focus": "API structure over input validation",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "models": "/models"
        }
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.2.0"
    }

@app.get("/models")
def list_models():
    """List available models (mock for now)"""
    return {
        "available_models": [
            {
                "name": "mock_model",
                "type": "MockModel",
                "status": "available",
                "description": "Simple mock model for testing API structure"
            }
        ],
        "total_models": 1,
        "note": "Real model loading will be added in Phase 3"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_salary(input_data: PredictionInput):
    """
    Core prediction endpoint - focus on API flow, not prediction accuracy
    """
    logger.info(f"Prediction request for {input_data.role} in {input_data.area}")
    
    try:
        # Very simple mock calculation - focus is on API working, not accuracy
        base_salary = 60000
        
        # Simple multipliers
        education_boost = {"Bachelor's": 1.3, "Master's": 1.6, "PhD": 1.8}.get(input_data.education_level, 1.0)
        seniority_boost = {"Junior": 1.0, "Mid": 1.4, "Senior": 1.8, "Executive": 2.5}.get(input_data.seniority, 1.0)
        
        predicted_salary = base_salary * education_boost * seniority_boost * (1 + input_data.years_of_experience * 0.03)
        
        # Add some randomness
        predicted_salary *= (0.9 + random.random() * 0.2)
        
        return PredictionOutput(
            predicted_salary=round(predicted_salary, 2),
            model_used="mock_model_v1",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_name}")
def predict_with_model(model_name: str, input_data: PredictionInput):
    """
    Prediction with specific model - structure for when we add real models
    """
    logger.info(f"Prediction request using model: {model_name}")
    
    if model_name != "mock_model":
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # For now, just call the same logic
    result = predict_salary(input_data)
    result.model_used = model_name
    return result

@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """Get info about a specific model"""
    if model_name != "mock_model":
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return {
        "name": model_name,
        "type": "MockModel",
        "status": "available",
        "created": "2024-11-15",
        "metrics": {
            "note": "Mock metrics - real metrics will come from trained models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting simplified Phase 2 API...")
    logger.info("Focus: API structure working correctly")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )