
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place creditcard.csv inside data/raw/"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    if df["Class"].nunique() != 2:
        raise ValueError("Expected binary target: Class 0 and Class 1.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Legitimate transactions: {(y == 0).sum():,}")
    print(f"Fraud transactions: {(y == 1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
        "threshold": 0.5,
    }

    joblib.dump(model, ARTIFACTS_DIR / "fraud_model.joblib")

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel training complete!")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {ARTIFACTS_DIR / 'fraud_model.joblib'}")
    print(f"Metrics saved to: {ARTIFACTS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
