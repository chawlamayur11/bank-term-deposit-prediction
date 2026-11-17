#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py - Web Service API for Bank Marketing Term Deposit Prediction
Course: DataTalksClub Machine Learning Zoomcamp (Midterm Project)
Author: Mayur Chawla

This web service exposes REST API endpoints:
- GET  /         : Service status & metadata
- GET  /health   : Health check endpoint
- POST /predict  : Predicts probability of term deposit subscription for a prospective client
"""

import os
import pickle
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import xgboost as xgb

MODEL_PATH = os.environ.get("MODEL_PATH", "model.bin")

# Resolve model path relative to this file
if not os.path.isabs(MODEL_PATH):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(base_dir, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python train.py' first.")

# Load serialized DictVectorizer and XGBoost model
with open(MODEL_PATH, "rb") as f_in:
    dv, model = pickle.load(f_in)

# Pre-sanitize feature names for XGBoost DMatrix
sanitized_feature_names = [
    f.replace("[", "_").replace("]", "_").replace("<", "_").replace("=", "_")
    for f in dv.get_feature_names_out().tolist()
]

app = FastAPI(
    title="Bank Marketing Term Deposit Prediction API",
    description="API for scoring prospective bank clients on likelihood of term deposit subscription.",
    version="1.0.0"
)


class ClientProfile(BaseModel):
    """Schema for bank client data input."""
    age: int = Field(..., example=35, description="Client age in years")
    job: str = Field(..., example="technician", description="Type of job")
    marital: str = Field(..., example="single", description="Marital status")
    education: str = Field(..., example="university.degree", description="Education level")
    default: str = Field(default="no", example="no", description="Credit in default? ('no', 'yes', 'unknown')")
    housing: str = Field(default="no", example="yes", description="Housing loan? ('no', 'yes', 'unknown')")
    loan: str = Field(default="no", example="no", description="Personal loan? ('no', 'yes', 'unknown')")
    contact: str = Field(default="cellular", example="cellular", description="Contact communication type ('cellular', 'telephone')")
    month: str = Field(default="may", example="nov", description="Last contact month of year")
    day_of_week: str = Field(default="mon", example="fri", description="Last contact day of the week")
    campaign: int = Field(default=1, example=1, description="Number of contacts performed during this campaign")
    pdays: int = Field(default=999, example=999, description="Days passed after client was last contacted from a previous campaign (999 means not previously contacted)")
    previous: int = Field(default=0, example=0, description="Number of contacts performed before this campaign")
    poutcome: str = Field(default="nonexistent", example="nonexistent", description="Outcome of the previous marketing campaign ('failure', 'nonexistent', 'success')")
    emp_var_rate: float = Field(default=-0.1, example=-0.1, description="Employment variation rate - quarterly indicator")
    cons_price_idx: float = Field(default=93.2, example=93.2, description="Consumer price index - monthly indicator")
    cons_conf_idx: float = Field(default=-42.0, example=-42.0, description="Consumer confidence index - monthly indicator")
    euribor3m: float = Field(default=4.0, example=4.021, description="Euribor 3 month rate - daily indicator")
    nr_employed: float = Field(default=5195.8, example=5195.8, description="Number of employees - quarterly indicator")


class PredictionResponse(BaseModel):
    """Schema for model prediction response."""
    subscription_probability: float
    subscribe: bool
    threshold: float
    recommendation: str


def score_client(client_dict: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """Helper to transform and predict for a client dictionary."""
    X_vec = dv.transform([client_dict])
    dmatrix = xgb.DMatrix(X_vec, feature_names=sanitized_feature_names)
    prob = float(model.predict(dmatrix)[0])
    subscribe = bool(prob >= threshold)

    recommendation = (
        "High propensity lead: Prioritize outbound telemarketing call."
        if subscribe
        else "Low propensity lead: Do not prioritize for telemarketing."
    )

    return {
        "subscription_probability": round(prob, 4),
        "subscribe": subscribe,
        "threshold": threshold,
        "recommendation": recommendation
    }


@app.get("/")
def root():
    """Service metadata endpoint."""
    return {
        "service": "Bank Marketing Term Deposit Prediction Service",
        "status": "active",
        "docs_url": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": dv is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientProfile, threshold: Optional[float] = 0.5):
    """
    Score a client profile and return deposit subscription probability.
    """
    try:
        client_dict = client.model_dump()
        result = score_client(client_dict, threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Bank Marketing API server on http://0.0.0.0:{port}...")
    uvicorn.run("predict:app", host="0.0.0.0", port=port, reload=False)
