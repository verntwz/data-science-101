# data-science-101

## Loan approval classifier

Predicts `Loan_Status` (Y = approved, N = rejected) from the applicant data in
`data/loan-approval.csv` (614 rows, 69% approved).

**New to this?** Start with [`loan_classifier_walkthrough.ipynb`](loan_classifier_walkthrough.ipynb),
a step-by-step notebook that explains every stage in plain language, with charts.
Open it with `jupyter notebook` (or directly on GitHub to read it with outputs).

```bash
pip install -r requirements.txt
python train.py                      # compare models, evaluate, save models/loan_classifier.joblib
python predict.py new_apps.csv out.csv   # score new applications
```

### Approach

- **Missing values:** categorical columns get the most frequent value and numeric
  columns get the median, both learned inside the pipeline so nothing leaks from test data.
  A blank `Credit_History` is kept as its own `missing` category, because those
  applicants are approved at about the same rate as `1` (74% vs 80%).
- **Engineered features:** log total income, log loan amount, loan-to-income, and
  monthly installment (EMI) to income.
- **Models compared** with 5-fold stratified CV on an 80% training split: always-approve
  baseline, logistic regression, random forest, gradient boosting. The best by ROC AUC
  is evaluated once on the 20% hold-out, then refit on all data and saved.

### Results

| Model | CV accuracy | CV ROC AUC | CV F1 (reject) |
|---|---|---|---|
| Baseline (always approve) | 0.686 | 0.500 | 0.000 |
| Logistic regression | 0.790 | 0.725 | 0.542 |
| **Random forest** | **0.794** | **0.770** | 0.540 |
| Gradient boosting | 0.788 | 0.761 | 0.552 |

Random forest on the hold-out set: **accuracy 0.854, ROC AUC 0.868**. It catches 55% of
rejections (21 of 38) with 95% precision, and flags 99% of approvals correctly.

`Credit_History` dominates: 92% of applicants with a bad credit history are rejected.
After that, the income and affordability ratios matter most.

### Limitations

- With 614 rows the CV scores vary by ±0.05–0.09, so the three real models are
  statistically very close; the hold-out score is probably on the lucky side.
- The model leans heavily on credit history and misses many rejections of applicants
  with good credit. To catch more rejections, lower the decision threshold on
  `predict_proba` below 0.5 (at the cost of more false rejections).
- `Gender` and `Married` are used as features. Drop them if this is used for real lending
  decisions, where they may be protected attributes.
