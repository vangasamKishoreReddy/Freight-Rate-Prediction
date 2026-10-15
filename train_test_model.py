
from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_of_year",
    "day_of_week",
    "month",
    "day",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "posted_rate"

app = FastAPI(
    title="Freight Rate Prediction API",
    description="ExtraTrees freight-rate model and assessment workflow",
    version="1.0.0",
)

model = None
training_data = None
holdout_metrics = None
model_lock = Lock()


# =========================================================
# FILE DISCOVERY
# =========================================================

def find_file(*names: str) -> Path:
    candidates = []

    for name in names:
        candidates.extend([
            ROOT / name,
            DATA_DIR / name,
        ])

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Could not find any of these files: {names}. "
        f"Check your project and data folders."
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "date" not in result.columns:
        raise ValueError("Required column 'date' is missing.")

    parsed_dates = pd.to_datetime(
        result["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    missing_dates = parsed_dates.isna()

    if missing_dates.any():
        parsed_dates.loc[missing_dates] = pd.to_datetime(
            result.loc[missing_dates, "date"],
            dayfirst=True,
            errors="coerce",
        )

    if parsed_dates.isna().any():
        raise ValueError("Invalid dates found in dataset.")

    result["date"] = parsed_dates
    result["day_of_year"] = parsed_dates.dt.dayofyear
    result["day_of_week"] = parsed_dates.dt.dayofweek
    result["month"] = parsed_dates.dt.month
    result["day"] = parsed_dates.dt.day

    for feature in FEATURES:
        if feature not in result.columns:
            result[feature] = np.nan

    return result


# =========================================================
# MODEL
# =========================================================

def make_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown",
                    ),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    estimator = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=17,
    )

    return make_pipeline(preprocessor, estimator)


# =========================================================
# TRAIN AND EVALUATE
# =========================================================

def train_model_and_evaluate():
    global model, training_data, holdout_metrics

    train_path = find_file(
        "train_test.csv",
        "train-test.csv",
    )

    train = add_date_features(pd.read_csv(train_path))

    if TARGET not in train.columns:
        raise ValueError(
            f"Training data must contain '{TARGET}'."
        )

    train[TARGET] = pd.to_numeric(
        train[TARGET], errors="coerce"
    )

    train = train.dropna(subset=[TARGET]).copy()
    train = train[
        np.isfinite(train[TARGET]) & (train[TARGET] > 0)
    ].copy()

    if train.empty:
        raise ValueError("No valid training target values found.")

    split_date = pd.Timestamp("2025-10-01")

    fit_rows = train[train["date"] < split_date]
    holdout_rows = train[train["date"] >= split_date]

    if fit_rows.empty or holdout_rows.empty:
        raise ValueError(
            "Chronological holdout requires training rows before "
            "and on/after October 1, 2025."
        )

    candidate = make_model()
    candidate.fit(fit_rows[FEATURES], fit_rows[TARGET])

    predictions = candidate.predict(holdout_rows[FEATURES])
    baseline = np.full(
        len(holdout_rows),
        fit_rows[TARGET].median(),
    )

    metrics = {
        "holdout_rows": int(len(holdout_rows)),
        "training_rows_for_holdout": int(len(fit_rows)),
        "mae": round(
            float(mean_absolute_error(
                holdout_rows[TARGET], predictions
            )), 2
        ),
        "rmse": round(
            float(np.sqrt(mean_squared_error(
                holdout_rows[TARGET], predictions
            ))), 2
        ),
        "r2": round(
            float(r2_score(
                holdout_rows[TARGET], predictions
            )), 4
        ),
        "median_baseline_mae": round(
            float(mean_absolute_error(
                holdout_rows[TARGET], baseline
            )), 2
        ),
    }

    # Fit the final model using all available labeled rows.
    final_model = make_model()
    final_model.fit(train[FEATURES], train[TARGET])

    model = final_model
    training_data = train
    holdout_metrics = metrics

    return metrics


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

def generate_validation_predictions():
    global model, training_data

    if model is None:
        train_model_and_evaluate()

    validation_path = find_file("validation.csv")

    template_path = find_file(
        "validation_predictions_template.csv",
        "validation-predictions-template.csv",
    )

    validation = add_date_features(
        pd.read_csv(validation_path)
    )
    template = pd.read_csv(template_path)

    if "load_id" not in validation.columns:
        raise ValueError("Validation file is missing load_id.")

    if "load_id" not in template.columns:
        raise ValueError("Prediction template is missing load_id.")

    if validation["load_id"].isna().any():
        raise ValueError("Validation contains missing load IDs.")

    if validation["load_id"].duplicated().any():
        raise ValueError("Validation contains duplicate load IDs.")

    if template["load_id"].isna().any():
        raise ValueError("Template contains missing load IDs.")

    if template["load_id"].duplicated().any():
        raise ValueError("Template contains duplicate load IDs.")

    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError(
            "Validation IDs do not match template IDs."
        )

    predictions = model.predict(validation[FEATURES])

    if (
        not np.isfinite(predictions).all()
        or (predictions <= 0).any()
    ):
        raise ValueError(
            "Model produced invalid or non-positive rates."
        )

    predicted = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": predictions,
    })

    completed = template[["load_id"]].merge(
        predicted,
        on="load_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if completed["predicted_rate"].isna().any():
        raise ValueError("Some template loads have no predictions.")

    completed["predicted_rate"] = (
        completed["predicted_rate"].round(2)
    )

    output_path = ROOT / "validation_predictions.csv"
    completed.to_csv(output_path, index=False)

    return {
        "rows": len(completed),
        "output_file": str(output_path),
        "columns": list(completed.columns),
    }


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Freight Rate Prediction API",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_trained": model is not None,
    }


@app.post("/train")
def train_endpoint():
    try:
        with model_lock:
            metrics = train_model_and_evaluate()

        return {
            "message": "Model training and evaluation completed.",
            "model": "ExtraTreesRegressor",
            "metrics": metrics,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/metrics")
def metrics_endpoint():
    if holdout_metrics is None:
        raise HTTPException(
            status_code=409,
            detail="Model not trained yet. Call POST /train first.",
        )

    return holdout_metrics


class LoadInput(BaseModel):
    pickup: str
    delivery: str
    equipment: str
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    delivery_lat: float | None = None
    delivery_lon: float | None = None
    distance: float = Field(gt=0)
    weight: float = Field(gt=0)
    market_index: float | None = None
    quote_signal: float | None = None
    date: str


@app.post("/predict")
def predict_endpoint(data: LoadInput):
    try:
        with model_lock:
            if model is None:
                train_model_and_evaluate()

            frame = add_date_features(
                pd.DataFrame([data.model_dump()])
            )
            prediction = float(model.predict(frame[FEATURES])[0])

        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("Model produced an invalid rate.")

        return {
            "predicted_rate": round(prediction, 2),
            "currency": "USD",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/predict-validation")
def validation_endpoint():
    try:
        with model_lock:
            result = generate_validation_predictions()

        return {
            "message": "Validation predictions generated.",
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/run-assessment")
def run_assessment_endpoint():
    try:
        with model_lock:
            metrics = train_model_and_evaluate()
            result = generate_validation_predictions()

        return {
            "message": "Assessment model run completed.",
            "metrics": metrics,
            "predictions": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# =========================================================
# RUN DIRECTLY OR WITH UVICORN
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
