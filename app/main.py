from pathlib import Path
import pickle

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("model/credit_risk_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


app = FastAPI(title="Credit Risk Model API")


class LoanApplication(BaseModel):
    income: float
    debt_ratio: float
    credit_score: int
    missed_payments: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(application: LoanApplication):
    features = [[
        application.income,
        application.debt_ratio,
        application.credit_score,
        application.missed_payments
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "risk": "HIGH" if prediction == 1 else "LOW"
    }
