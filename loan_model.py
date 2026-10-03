"""Feature engineering and preprocessing pipeline for the loan-approval classifier.

Kept in its own module so a saved pipeline can be unpickled from any script.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Credit_History",
]
NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "LogTotalIncome",
    "LogLoanAmount",
    "LoanToIncome",
    "EMIToIncome",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive affordability features. Missing inputs propagate as NaN and are imputed later."""
    df = df.copy()
    # Credit_History is a 0/1 flag with many blanks; missing applicants approve at
    # roughly the same rate as "1", so keep "missing" as its own category.
    df["Credit_History"] = df["Credit_History"].map({1.0: "1", 0.0: "0"}).fillna("missing")
    total_income = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LogTotalIncome"] = np.log1p(total_income)
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    df["LoanToIncome"] = df["LoanAmount"] / total_income.replace(0, np.nan)
    df["EMIToIncome"] = (df["LoanAmount"] / df["Loan_Amount_Term"]) / total_income.replace(0, np.nan)
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC)])


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
