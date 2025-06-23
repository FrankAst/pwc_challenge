"""
Comprehensive API Test Suite for Salary Prediction API
Uses pytest and requests for robust testing
"""

import pytest
import requests
import json
import time
from typing import Dict, List, Any
import logging

# Test configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# Test data
VALID_PREDICTION_DATA = {
    "age": 30.0,
    "gender": "Male",
    "education_level": "Bachelor's",
    "years_of_experience": 5.0,
    "seniority": "Junior",
    "area": "Engineering",
    "role": "Software Engineer"
}

EDGE_CASE_DATA = [
    {
        "age": 16.0,
        "gender": "Female",
        "education_level": "High School",
        "years_of_experience": 0.0,
        "seniority": "Entry",
        "area": "Retail",
        "role": "Sales Associate"
    },
    {
        "age": 65.0,
        "gender": "Male",
        "education_level": "PhD",
        "years_of_experience": 40.0,
        "seniority": "Executive",
        "area": "Technology",
        "role": "CTO"
    },
    {
        "age": 25.5,
        "gender": "Non-binary",
        "education_level": "Bachelor's",
        "years_of_experience": 2.5,
        "seniority": "Junior",
        "area": "Design",
        "role": "UX Designer"
    }
]

INVALID_DATA_CASES = [
    # Missing required fields
    {"gender": "Male", "education_level": "Bachelor's"},
    # Invalid age
    {"age": "thirty", "gender": "Male", "education_level": "Bachelor's", 
     "years_of_experience": 5, "seniority": "Junior", "area": "Engineering", "role": "Engineer"},
    # Negative experience
    {"age": 30, "gender": "Male", "education_level": "Bachelor's", 
     "years_of_experience": -5, "seniority": "Junior", "area": "Engineering", "role": "Engineer"},
    # Age too high
    {"age": 150, "gender": "Male", "education_level": "Bachelor's", 
     "years_of_experience": 5, "seniority": "Junior", "area": "Engineering", "role": "Engineer"},
    # Empty object
    {},
    # Invalid field names
    {"invalid_field": "value"}
]

class APITestClient:
    """Test client for API interactions"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make GET request"""
        return self.session.get(f"{self.base_url}{endpoint}", timeout=TIMEOUT, **kwargs)
    
    def post(self, endpoint: str, data: Dict = None, **kwargs) -> requests.Response:
        """Make POST request"""
        headers = kwargs.pop('headers', {})
        headers['Content-Type'] = 'application/json'
        
        return self.session.post(
            f"{self.base_url}{endpoint}",
            data=json.dumps(data) if data else None,
            headers=headers,
            timeout=TIMEOUT,
            **kwargs
        )
    
    def is_available(self) -> bool:
        """Check if API is available"""
        try:
            response = self.get("/health")
            return response.status_code == 200
        except:
            return False

@pytest.fixture
def api_client():
    """Fixture for API client"""
    client = APITestClient()
    if not client.is_available():
        pytest.skip("API is not available. Start the API server first.")
    return client

class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API information"""
        response = api_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["version"] == "4.0.0"
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models_loaded" in data
    
    def test_debug_info(self, api_client):
        """Test debug information endpoint"""
        response = api_client.get("/debug/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "project_root" in data
        assert "models_directory" in data
        assert "config" in data

class TestModelEndpoints:
    """Test model-related endpoints"""
    
    def test_list_models(self, api_client):
        """Test listing available models"""
        response = api_client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "total_models" in data
        assert isinstance(data["available_models"], list)
    
    def test_model_reload(self, api_client):
        """Test model reload endpoint"""
        response = api_client.post("/models/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "models_loaded" in data
    
    @pytest.mark.parametrize("model_name", [
        "DecisionTree_GSopt", "RandomForest_BOopt", 
        "Lasso_Regression", "OLS_basic", "SGD_Regression"
    ])
    def test_get_model_info(self, api_client, model_name):
        """Test getting specific model information"""
        response = api_client.get(f"/models/{model_name}")
        
        # Model might not exist, so accept both 200 and 404
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["name"] == model_name
            assert "type" in data
            assert "status" in data
    
    def test_nonexistent_model_info(self, api_client):
        """Test getting info for non-existent model"""
        response = api_client.get("/models/NonExistentModel")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_model_features(self, api_client):
        """Test getting model features"""
        # First get available models
        models_response = api_client.get("/models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_models = [m["name"] if isinstance(m, dict) else m 
                              for m in models_data["available_models"]]
            
            if available_models:
                model_name = available_models[0]
                response = api_client.get(f"/models/{model_name}/features")
                assert response.status_code == 200
                
                data = response.json()
                assert "feature_names" in data
                assert "categorical_features" in data
                assert "numerical_features" in data

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_valid_prediction_default_model(self, api_client):
        """Test valid prediction with default model"""
        response = api_client.post("/predict", data=VALID_PREDICTION_DATA)
        
        # Might fail if no models are available
        if response.status_code == 200:
            data = response.json()
            assert "predicted_salary" in data
            assert "model_used" in data
            assert "timestamp" in data
            assert isinstance(data["predicted_salary"], (int, float))
            assert data["predicted_salary"] > 0
        elif response.status_code == 503:
            # No models available - acceptable in test environment
            assert "No models available" in response.json()["detail"]
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    @pytest.mark.parametrize("test_data", EDGE_CASE_DATA)
    def test_edge_case_predictions(self, api_client, test_data):
        """Test predictions with edge case data"""
        response = api_client.post("/predict", data=test_data)
        
        # Accept both success and service unavailable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_salary" in data
            assert isinstance(data["predicted_salary"], (int, float))
    
    @pytest.mark.parametrize("invalid_data", INVALID_DATA_CASES)
    def test_invalid_prediction_data(self, api_client, invalid_data):
        """Test predictions with invalid data"""
        response = api_client.post("/predict", data=invalid_data)
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
    
    def test_prediction_with_specific_model(self, api_client):
        """Test prediction with specific model"""
        # First get available models
        models_response = api_client.get("/models")
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_models = [m["name"] if isinstance(m, dict) else m 
                              for m in models_data["available_models"]]
            
            if available_models:
                model_name = available_models[0]
                response = api_client.post(f"/predict/{model_name}", data=VALID_PREDICTION_DATA)
                
                assert response.status_code in [200, 404]  # Model might not support API
                
                if response.status_code == 200:
                    data = response.json()
                    assert data["model_used"] == model_name
    
    def test_prediction_with_nonexistent_model(self, api_client):
        """Test prediction with non-existent model"""
        response = api_client.post("/predict/NonExistentModel", data=VALID_PREDICTION_DATA)
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"].lower()

class TestDocumentationEndpoints:
    """Test API documentation endpoints"""
    
    def test_openapi_docs(self, api_client):
        """Test OpenAPI documentation"""
        response = api_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_docs(self, api_client):
        """Test ReDoc documentation"""
        response = api_client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_json(self, api_client):
        """Test OpenAPI JSON schema"""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_nonexistent_endpoint(self, api_client):
        """Test request to non-existent endpoint"""
        response = api_client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, api_client):
        """Test method not allowed"""
        response = api_client.post("/health")  # Health endpoint only accepts GET
        assert response.status_code == 405
    
    def test_malformed_json(self, api_client):
        """Test malformed JSON in POST request"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422

class TestPerformance:
    """Test performance characteristics"""
    
    def test_response_time_health_check(self, api_client):
        """Test health check response time"""
        start_time = time.time()
        response = api_client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return api_client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200
    
    @pytest.mark.skipif(True, reason="Skip load test by default")
    def test_load_test_predictions(self, api_client):
        """Load test for predictions (skipped by default)"""
        successful_requests = 0
        total_requests = 100
        
        for _ in range(total_requests):
            response = api_client.post("/predict", data=VALID_PREDICTION_DATA)
            if response.status_code in [200, 503]:  # Accept both success and no models
                successful_requests += 1
        
        # At least 90% should succeed
        assert successful_requests >= total_requests * 0.9

# Utility functions for test setup
def pytest_configure(config):
    """Configure pytest"""
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers for slow tests
    for item in items:
        if "load_test" in item.name or "concurrent" in item.name:
            item.add_marker(pytest.mark.slow)

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])