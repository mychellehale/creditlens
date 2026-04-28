from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path
import pandas as pd

app = FastAPI(
    title="CreditLens",
    description="Credit card default risk scoring API"
)

BASE_DIR = Path(__file__).parent.parent.parent
model_path = max(BASE_DIR.glob("models/best_model_*.pkl"))

with open(model_path, "rb") as f:
    model = pickle.load(f)


class CreditFeatures(BaseModel):
    """Input features for credit default prediction.

    Demographic, payment history, bill amount, and payment amount fields
    from the UCI Credit Card Default dataset. SEX, EDUCATION, and MARRIAGE
    default to their 'other/unknown' category if not provided.
    """

    LIMIT_BAL: float
    SEX: int = 1
    EDUCATION: int = 4
    MARRIAGE: int = 3
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float


class PredictionResponse(BaseModel):
    """Prediction output returned by the /predict endpoint."""

    default_probability: float
    prediction: int
    risk_level: str


@app.post("/predict")
def predict(features: CreditFeatures) -> PredictionResponse:
    """Score a single credit card holder for default risk.

    Returns the probability of default, a binary prediction (threshold 0.5),
    and a risk tier (low/medium/high). North-star metric is recall on class 1
    — missing a default is worse than a false alarm.
    """
    data = pd.DataFrame([features.model_dump()])
    prob = float(model.predict_proba(data)[0][1])
    return PredictionResponse(
        default_probability=round(prob, 4),
        prediction=1 if prob >= 0.5 else 0,
        risk_level="High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low",
    )


@app.get("/health")
def health():
    """Health check endpoint for load balancers and container orchestration."""
    return {"status": "ok"}
