from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
VALIDATION_PATH = ROOT / "validation.csv"
TEMPLATE_PATH = ROOT / "validation-predictions-template.csv"
DECEMBER_PATH = ROOT / "december-chart-inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

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


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date_values = result["date"]
    parsed_dates = pd.to_datetime(date_values, format="%Y-%m-%d", errors="coerce")
    unparsed = parsed_dates.isna()
    if unparsed.any():
        parsed_dates.loc[unparsed] = pd.to_datetime(
            date_values.loc[unparsed], format="%d-%m-%Y", errors="raise"
        )
    result["date"] = parsed_dates
    result["day_of_year"] = result["date"].dt.dayofyear
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    for feature in FEATURES:
        if feature not in result.columns:
            result[feature] = np.nan
    return result


def make_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    estimator = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=17,
    )
    return make_pipeline(preprocessor, estimator)


def report_holdout_metrics(train: pd.DataFrame) -> None:
    split_date = pd.Timestamp("2025-10-01")
    fit_rows = train[train["date"] < split_date]
    holdout_rows = train[train["date"] >= split_date]
    if fit_rows.empty or holdout_rows.empty:
        raise ValueError("Chronological holdout requires data before and during October 2025")

    model = make_model()
    model.fit(fit_rows[FEATURES], fit_rows[TARGET])
    predictions = model.predict(holdout_rows[FEATURES])
    baseline = np.full(len(holdout_rows), fit_rows[TARGET].median())

    print(
        f"October 2025 holdout ({len(holdout_rows):,} loads; "
        f"trained on {len(fit_rows):,} Jan-Sep loads):"
    )
    print(f"  ExtraTrees MAE: ${mean_absolute_error(holdout_rows[TARGET], predictions):,.2f}")
    print(
        "  ExtraTrees RMSE: "
        f"${mean_squared_error(holdout_rows[TARGET], predictions) ** 0.5:,.2f}"
    )
    print(f"  ExtraTrees R2: {r2_score(holdout_rows[TARGET], predictions):.4f}")
    print(f"  Median baseline MAE: ${mean_absolute_error(holdout_rows[TARGET], baseline):,.2f}")


def add_december_coordinates(
    december: pd.DataFrame, train: pd.DataFrame
) -> pd.DataFrame:
    result = december.copy()
    for city_column, latitude_column, longitude_column in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        coordinates = train.groupby(city_column)[[latitude_column, longitude_column]].median()
        result[latitude_column] = result[city_column].map(coordinates[latitude_column])
        result[longitude_column] = result[city_column].map(coordinates[longitude_column])
        if result[[latitude_column, longitude_column]].isna().any().any():
            raise ValueError(f"No training coordinates found for a December {city_column} city")
    return result


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    train = add_date_features(train)
    report_holdout_metrics(train)

    model = make_model()
    model.fit(train[FEATURES], train[TARGET])

    validation = add_date_features(pd.read_csv(VALIDATION_PATH))
    validation_prediction = model.predict(validation[FEATURES])
    if not np.isfinite(validation_prediction).all() or (validation_prediction <= 0).any():
        raise ValueError("Validation predictions must all be finite and positive")

    template = pd.read_csv(TEMPLATE_PATH)
    predicted = pd.DataFrame(
        {"load_id": validation["load_id"], "predicted_rate": validation_prediction}
    )
    completed = template[["load_id"]].merge(
        predicted, on="load_id", how="left", validate="one_to_one", sort=False
    )
    if len(completed) != len(template) or completed["predicted_rate"].isna().any():
        raise ValueError("Could not produce one prediction for every template load_id")
    completed["predicted_rate"] = completed["predicted_rate"].round(2)
    completed.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(completed):,} validation predictions to {PREDICTIONS_PATH.name}")

    december = pd.read_csv(DECEMBER_PATH)
    december = add_date_features(add_december_coordinates(december, train))
    december["market_index"] = np.nan
    december["quote_signal"] = np.nan
    december["predicted_rate"] = model.predict(december[FEATURES]).round(2)
    if not np.isfinite(december["predicted_rate"]).all() or (december["predicted_rate"] <= 0).any():
        raise ValueError("December predictions must all be finite and positive")
    december[
        ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    ].to_csv(DECEMBER_PATH, index=False)
    print(f"Wrote {len(december):,} December predictions to {DECEMBER_PATH.name}")


if __name__ == "__main__":
    main()


    