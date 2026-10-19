"""Data loading and preprocessing helpers for the Telco churn dataset."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_dataset(path=DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the IBM Telco Customer Churn CSV and save it to "
            "data/raw/Telco-Customer-Churn.csv. See data/README.md."
        )
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Expected a target column named 'Churn'.")
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # In the common Telco CSV, TotalCharges is text and blank for some new customers.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df[df["Churn"].notna()].copy()
    y = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No'.")
    X = df.drop(columns=["Churn"])
    return X, y.astype(int), df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ], remainder="drop")
    return preprocessor, numeric_features, categorical_features
