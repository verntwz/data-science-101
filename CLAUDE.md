# CLAUDE.md

## Project

Binary classifier predicting `Loan_Status` (Y → 1 approved, N → 0 rejected) from
`data/loan-approval.csv` (614 rows, 69% approved). Several scikit-learn models are compared
with cross-validation; the best (currently random forest) is saved as a pipeline.

## Commands

```bash
pip install -r requirements.txt
python train.py [path/to/data.csv]        # CV comparison, hold-out eval, saves models/loan_classifier.joblib
python predict.py in.csv [out.csv]        # score applications (needs the saved model)
jupyter nbconvert --to notebook --execute --inplace loan_classifier_walkthrough.ipynb   # re-run notebook
```

There is no test suite. Verify changes by running `train.py` and `predict.py`, and by
re-executing the notebook with no errors.

## Layout

- `loan_model.py`: feature engineering (`add_features`), the `CATEGORICAL`/`NUMERIC` column lists, `make_pipeline`
- `train.py`: candidate models (`CANDIDATES`), `RANDOM_STATE`, and the training/evaluation `main`
- `predict.py`: loads the saved pipeline and adds `Approval_Probability` / `Predicted_Status`
- `loan_classifier_walkthrough.ipynb`: beginner walkthrough of the whole process
- `data/`: the raw CSV
- `models/`: gitignored; created by `train.py`

## Conventions and gotchas

- Anything the saved pipeline references (e.g. `add_features`) must live in an importable
  module such as `loan_model.py`, never in a script run as `__main__`. Otherwise `joblib.load` fails.
- Keep all preprocessing (imputation, encoding, scaling) inside the sklearn pipeline so nothing
  leaks from test data. Evaluate the hold-out test split once. Tune thresholds and
  hyperparameters with CV on the training split, not on the test set.
- Keep `RANDOM_STATE = 42` and the stratified 80/20 split so results are reproducible.
- A blank `Credit_History` is deliberately kept as its own `"missing"` category: those applicants
  are approved at about the same rate as `1`. Don't impute it as 0/1.
- The notebook imports from `loan_model.py` and `train.py`. After changing either, re-execute the
  notebook and commit it **with outputs**, so it reads on GitHub.
- `README.md` and the notebook's markdown quote specific results (CV ROC AUC 0.770, test accuracy
  0.854, 21 of 38 rejections caught, etc.). Update them if the results change.
- The notebook is for beginners. Explain each new term in plain language. Chart colors:
  approved `#2a78d6`, rejected `#eb6834`.
- `Gender` and `Married` are model inputs. Flag fairness/legal concerns for any real-world use.
