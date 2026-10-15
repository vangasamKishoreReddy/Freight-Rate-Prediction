# Freight-Rate-Prediction
Machine learning model for freight rate prediction using data preprocessing, feature engineering, model validation, and predictive analytics.
# Freight Rate Prediction Using Machine Learning

## Project Overview

This project predicts freight rates using historical shipment data and machine learning. It includes data preprocessing, feature engineering, model training, evaluation, and prediction generation for validation data.

The project uses **ExtraTreesRegressor** to learn patterns from historical freight shipments and predict rates for new loads.

## Objectives

- Analyze and preprocess freight shipment data.
- Perform feature engineering to improve model performance.
- Train a machine learning model using historical data.
- Evaluate performance using MAE, RMSE, and R².
- Generate predictions for validation records.
- Provide API endpoints using FastAPI.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## Project Structure

```text
Freight-Rate-ML/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   └── validation_predictions_template.csv
├── train_model.py
├── validation_predictions.csv
├── requirements.txt
├── README.md
└── .venv/
```

*Note: Update the filenames above if your actual project uses different names or stores the CSV files in the root directory.*

## Model Performance

The model was evaluated on a chronological holdout from October 2025.

| Metric | Result |
|---|---:|
| Model | ExtraTreesRegressor |
| Mean Absolute Error (MAE) | $153.81 |
| Root Mean Squared Error (RMSE) | $684.19 |
| R² Score | 0.7996 |
| Median Baseline MAE | $1,146.79 |

These results represent the reported holdout evaluation and may differ on unseen validation data.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Freight-Rate-ML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Train the model and evaluate performance

```bash
python train_model.py
```

### Start the FastAPI server

```bash
uvicorn train_model:app --reload
```

Open the API documentation in your browser:

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Available endpoints depend on the implementation in `train_model.py`. The API includes model training, evaluation, and prediction functionality.

## Generate Validation Predictions

Use the project's prediction endpoint or assessment workflow to generate predictions for the validation dataset.

The expected output is:

```text
validation_predictions.csv
```

The output should preserve the required template structure and contain the `predicted_rate` column for all validation records. Verify the row count and column names before submitting the assessment.

## Evaluation Metrics

- **MAE:** Average absolute difference between actual and predicted rates.
- **RMSE:** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures how much variation in the target is explained by the model.

## Future Improvements

- Tune hyperparameters using time-aware validation.
- Analyze large prediction errors and outliers.
- Compare ExtraTrees with Random Forest and gradient-boosting models.
- Improve categorical and date-based feature engineering.
- Monitor performance on unseen data.

## Author

**Kishore Reddy V**

Machine Learning | Computer Vision | AI/ML

