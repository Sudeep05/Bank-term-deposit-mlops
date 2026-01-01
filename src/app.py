# app.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = "models/gb_bank_pipeline.joblib"

app = FastAPI(title="Bank Term Deposit API")

# Pydantic schema for one customer
class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: Optional[str] = None
    housing: Optional[str] = None
    loan: Optional[str] = None
    contact: str
    month: str
    dayofweek: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

# load model once at startup
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Bank term deposit prediction API is up"}

@app.post("/predict")
def predict(customer: Customer):
    # convert to DataFrame shaped like training data
    data = pd.DataFrame([{
        "age": customer.age,
        "job": customer.job,
        "marital": customer.marital,
        "education": customer.education,
        "default": customer.default,
        "housing": customer.housing,
        "loan": customer.loan,
        "contact": customer.contact,
        "month": customer.month,

        # provide BOTH names
        "dayofweek": customer.dayofweek,
        "day_of_week": customer.dayofweek,

        "duration": customer.duration,
        "campaign": customer.campaign,
        "pdays": customer.pdays,
        "previous": customer.previous,
        "poutcome": customer.poutcome,
        "emp.var.rate": customer.emp_var_rate,
        "cons.price.idx": customer.cons_price_idx,
        "cons.conf.idx": customer.cons_conf_idx,
        "euribor3m": customer.euribor3m,
        "nr.employed": customer.nr_employed
    }])

    proba = model.predict_proba(data)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "prediction": pred,
        "probability_yes": float(proba)
    }
