
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.joblib"

app = FastAPI(
    title="Fraud Risk Intelligence API",
    description="Credit card transaction fraud risk prediction",
    version="1.0.0",
)

model = None
feature_names = None


@app.on_event("startup")
def load_trained_model():
    global model, feature_names

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)


class Transaction(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def home():
    return {
        "message": "Fraud Risk Intelligence API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input_data = transaction.model_dump()
        input_df = pd.DataFrame([input_data], columns=feature_names)

        probability = float(model.predict_proba(input_df)[0, 1])

        return {
            "fraud_probability": round(probability, 6),
            "predicted_class": int(probability >= 0.5),
            "risk_level": (
                "HIGH" if probability >= 0.5
                else "LOW"
            ),
            "threshold": 0.5,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
