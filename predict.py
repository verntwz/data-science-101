"""Score loan applications with the trained model.

Usage:
    python predict.py applications.csv [output.csv]

The input needs the same columns as data/loan-approval.csv (Loan_Status is optional).
Adds Approval_Probability and Predicted_Status columns.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "loan_classifier.joblib"


def main(input_path: str, output_path: str | None) -> None:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_path)
    X = df.drop(columns=["Loan_ID", "Loan_Status"], errors="ignore")
    df["Approval_Probability"] = model.predict_proba(X)[:, 1].round(3)
    df["Predicted_Status"] = model.predict(X)
    df["Predicted_Status"] = df["Predicted_Status"].map({1: "Y", 0: "N"})
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(df)} predictions to {output_path}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
