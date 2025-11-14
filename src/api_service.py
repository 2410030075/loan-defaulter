"""
Loan Default Prediction API

A FastAPI service for loan default prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan defaults based on various features",
    version="1.0.0"
)

# Define request data model
class LoanApplication(BaseModel):
    age: int = Field(..., example=35, description="Age of applicant in years")
    annual_income: float = Field(..., example=50000, description="Annual income in dollars")
    loan_amount: float = Field(..., example=10000, description="Loan amount in dollars")
    loan_term: int = Field(..., example=36, description="Loan term in months")
    credit_score: int = Field(..., example=700, description="Credit score (300-850)")
    employment_years: float = Field(..., example=5, description="Years of employment")
    home_ownership: str = Field(..., example="RENT", description="Home ownership status: RENT, OWN, MORTGAGE, OTHER")
    loan_purpose: str = Field(..., example="DEBT_CONSOLIDATION", description="Purpose of the loan")
    debt_to_income: float = Field(..., example=20.5, description="Debt to income ratio in percentage")
    loan_grade: str = Field(..., example="B", description="Loan grade: A, B, C, D, E, F, G")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "annual_income": 50000,
                "loan_amount": 10000,
                "loan_term": 36,
                "credit_score": 700,
                "employment_years": 5,
                "home_ownership": "RENT",
                "loan_purpose": "DEBT_CONSOLIDATION",
                "debt_to_income": 20.5,
                "loan_grade": "B"
            }
        }

# Define response data model
class PredictionResponse(BaseModel):
    default_prediction: bool = Field(..., description="True if loan is predicted to default, False otherwise")
    default_probability: float = Field(..., description="Probability of loan default")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    
    class Config:
        schema_extra = {
            "example": {
                "default_prediction": False,
                "default_probability": 0.15,
                "risk_level": "Low"
            }
        }

# Load model and preprocessor
@app.on_event("startup")
async def startup_event():
    global model, preprocessor
    
    try:
        # Paths to model and preprocessor
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(project_dir, 'models')
        
        model_path = os.path.join(models_dir, 'best_model.joblib')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor not found. Please train the model first.")
            
        # Load model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        print("Model and preprocessor loaded successfully.")
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        # In a production environment, this should properly handle the error
        # or prevent the service from starting
        model = None
        preprocessor = None

# Define prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(loan_application: LoanApplication):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please contact the administrator.")
    
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame({
            'age': [loan_application.age],
            'annual_income': [loan_application.annual_income],
            'loan_amount': [loan_application.loan_amount],
            'loan_term': [loan_application.loan_term],
            'credit_score': [loan_application.credit_score],
            'employment_years': [loan_application.employment_years],
            'home_ownership': [loan_application.home_ownership],
            'loan_purpose': [loan_application.loan_purpose],
            'debt_to_income': [loan_application.debt_to_income],
            'loan_grade': [loan_application.loan_grade]
        })
        
        # Preprocess the input data
        X = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            default_prediction=bool(prediction),
            default_probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or preprocessor is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is running"}

# API Documentation info
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Loan Default Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "prediction": "/predict (POST)"
    }

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)