# PwC Salary Prediction Challenge

A comprehensive machine learning solution for predicting salaries based on personal and professional attributes. This project includes data preprocessing, multiple ML models, a REST API, and an interactive web interface with SHAP explainability.

## 🎯 Project Overview

This solution predicts salaries using:
- **Personal attributes**: Age, Gender, Education Level, Years of Experience
- **Job information**: Job Title (automatically extracted into Area, Role, and Seniority)
- **Text analysis**: NLP features from personal descriptions

**Best performing model**: Random Forest with **R² = 0.936** and **RMSE = $12,941**

## 📁 Project Structure

```
pwc_challenge/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── data/                        # Raw and processed data
│   ├── people.csv
│   ├── salary.csv
│   ├── descriptions.csv
│   └── cleaned_data/
│       └── final_dataset.csv
├── models/                      # Trained model files (.pkl)
│   ├── RandomForest_BOopt.pkl   # Best model
│   ├── DecisionTree_GSopt.pkl
│   └── ...
├── notebooks/                   # Jupyter notebooks for exploration
├── report/                      # Detailed analysis report
└── src/                         # Source code
    ├── api/                     # FastAPI application
    │   ├── main.py              # API server
    │   ├── model_loader.py      # Model management
    │   └── UI/
    │       └── user_interface.html  # Web interface
    ├── data_preparation_workflow/   # ETL pipeline
    └── models/                      # Model classes
```

## 🚀 Quick Start

### 1. Environment Setup

**Python Requirements**: Python 3.8+

Create and activate a virtual environment:
```bash
python -m venv pwcc
source pwcc/bin/activate  
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- FastAPI & Uvicorn (API framework)
- Scikit-learn (ML models)
- SHAP (Model explainability)
- OpenAI (Text processing - requires API key)
- Pandas, NumPy, Matplotlib

After all dependencies have been installed, do:

```bash
python -m spacy download en_core_web_sm
```

### 3. Environment Variables

Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note**: Even if you don't have an API, still create the .env with a random string. The above 'your_openai_api_key_here' works.
> **Note**: The OpenAI API is used for extracting job information from titles. If you don't have an API key, the system will use fallback logic.

### 4. Run the API

From the project root directory:
```bash
python src/api/main.py
```

The API will start on `http://localhost:8000`

**Available endpoints:**
- `/` - API documentation and available models
- `/docs` - Interactive Swagger UI
- `/predict` - Make salary predictions
- `/models` - View loaded models and metrics

### 5. Access the Web Interface

1. Ensure the API is running (step 4)
2. Open `src/api/UI/user_interface.html` in your web browser
3. The interface will automatically connect to the API at `localhost:8000`

## 💻 Usage Examples

### API Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 30,
       "gender": "Male",
       "education_level": "Bachelor'\''s",
       "years_of_experience": 5,
       "seniority": "Junior",
       "job_title": "Data Engineer"
     }'
```

### Web Interface
1. Select a model from the dropdown
2. Fill in your personal and professional information
3. Click "Predict Salary"
4. View prediction with SHAP explanation showing feature importance

## 🤖 Available Models

| Model | RMSE | R² | Description |
|-------|------|----|-----------| 
| **RandomForest_BOopt** | **$12,941** | **0.936** | Optimized Random Forest (Best) |
| DecisionTree_GSopt | $18,198 | 0.873 | Optimized Decision Tree |
| Lasso_Regression | $20,593 | 0.837 | Linear model with L1 regularization |
| OLS_basic | $21,814 | 0.817 | Basic linear regression |

## 🔧 Development

### Data Preprocessing
The data pipeline handles:
- Missing data imputation using OpenAI GPT-4o-mini
- Job title standardization (Area, Role, Seniority extraction)
- Text feature engineering (POS tagging)
- Feature encoding and scaling

### Model Training
Models are trained with:
- Cross-validation for robust evaluation
- Hyperparameter optimization (GridSearch/Optuna)
- Bootstrap confidence intervals
- SHAP integration for explainability

### API Features
- Model management and loading
- Input validation and preprocessing
- SHAP explanations with visualizations
- Comprehensive error handling
- Health checks and monitoring

## 📊 Model Performance

All models include 95% bootstrap confidence intervals:

**Random Forest (Best Model)**
- RMSE: $12,941 (CI: $9,781 - $16,099)
- R²: 0.936 (CI: 0.895 - 0.965)
- MAE: $9,227 (CI: $7,154 - $11,721)

## 🔍 Explainability

Every prediction includes SHAP (SHapley Additive exPlanations) values showing:
- Feature importance for the specific prediction
- Positive/negative contribution of each feature
- Visual waterfall plots in the web interface




---

