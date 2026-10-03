"""Train and evaluate a loan-approval classifier.

Usage:
    python train.py [path/to/loan-approval.csv]

Compares a few models with stratified cross-validation on a training split,
refits the best one, reports hold-out test metrics, and saves the fitted
pipeline to models/loan_classifier.joblib.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from loan_model import make_pipeline

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "loan-approval.csv"
MODEL_PATH = Path(__file__).parent / "models" / "loan_classifier.joblib"

CANDIDATES = {
    "baseline (always approve)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE),
    "random forest": RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
    ),
    "gradient boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
    ),
}


def main(data_path: Path) -> None:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = (df["Loan_Status"] == "Y").astype(int)  # 1 = approved

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Data: {len(df)} rows, approval rate {y.mean():.1%}")
    print(f"Train {len(X_train)} / test {len(X_test)}\n")
    print("5-fold CV on training split:")
    print(f"  {'model':<28}{'accuracy':>16}{'ROC AUC':>16}{'F1 (reject)':>16}")

    results = {}
    for name, model in CANDIDATES.items():
        scores = cross_validate(
            make_pipeline(model),
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
                # F1 on the minority (rejected) class shows whether a model beats "approve everyone".
                "f1_reject": make_scorer(f1_score, pos_label=0),
            },
        )
        results[name] = scores
        print(
            f"  {name:<28}"
            f"{scores['test_accuracy'].mean():>9.3f} ± {scores['test_accuracy'].std():.3f}"
            f"{scores['test_roc_auc'].mean():>9.3f} ± {scores['test_roc_auc'].std():.3f}"
            f"{scores['test_f1_reject'].mean():>9.3f} ± {scores['test_f1_reject'].std():.3f}"
        )

    best_name = max(
        (n for n in results if not n.startswith("baseline")),
        key=lambda n: results[n]["test_roc_auc"].mean(),
    )
    print(f"\nBest by CV ROC AUC: {best_name}")

    best = make_pipeline(CANDIDATES[best_name]).fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = best.predict(X_test)

    print("\nHold-out test set:")
    print(f"  accuracy {accuracy_score(y_test, pred):.3f}   ROC AUC {roc_auc_score(y_test, proba):.3f}")
    print("  confusion matrix (rows = actual N/Y, cols = predicted N/Y):")
    for row in confusion_matrix(y_test, pred):
        print("   ", row)
    print(classification_report(y_test, pred, target_names=["N (rejected)", "Y (approved)"], digits=3))

    # Refit on all data for the saved model.
    final = make_pipeline(CANDIDATES[best_name]).fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved {best_name} pipeline (trained on all {len(X)} rows) to {MODEL_PATH}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH)
