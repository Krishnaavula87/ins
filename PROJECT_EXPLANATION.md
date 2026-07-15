# PRCP-1010 Insurance Claim Prediction — Project Explanation

This document explains the project in simple words (internship / beginner level).

---

## 1. What is this project?

Insurance companies want to know which customers are more likely to be linked with a claim / risky outcome.

In this project we build a **simple predictive model** that:
1. Learns patterns from past customer data
2. Predicts the `target` column (`0` = no claim-related event, `1` = claim-related event)
3. Helps the marketing / insurance team focus on the right customers

Domain: **Finance / Insurance**

---

## 2. Project goals (from the problem statement)

### Task 1 — Predictive model
Create a model that helps identify which customers are more likely related to the product/claim outcome.

### Task 2 — Business suggestions
Give practical suggestions to the insurance marketing team based on model findings.

### Extra required reports
- Model comparison report
- Challenges faced report

All main code is kept in **one Python file**: `ins_claim_pred.py`

---

## 3. About the dataset

- File: `train.csv`
- Roughly **595,212 rows** and **59 columns**
- Feature names are anonymized (for privacy), e.g. `ps_ind_*`, `ps_car_*`, `ps_reg_*`
- Target column: `target`
- Class imbalance:
  - About **96.4%** are class `0`
  - About **3.6%** are class `1`
- Missing values are stored as `-1` instead of blank cells

Because of privacy rules, feature names do not explain the real meaning of each column.

---

## 4. Files in this repository

| File | Purpose |
|------|---------|
| `PRCP_1010_InsClaimPred.ipynb` | **Main submission notebook** (recommended) |
| `ins_claim_pred.py` | Same project logic in one Python file |
| `requirements.txt` | Python packages needed to run the project |
| `PROJECT_EXPLANATION.md` | This file — explains the project |
| `PRCP-1010-InsClaimPred (5).docx` | Original problem statement document |
| `.gitignore` | Ignores large dataset / temporary files |

Generated when you run the script/notebook:
- `model_comparison_results.csv` — saved model scores

---

## 5. How the code works (step by step)

### Step A — Load data
- Downloads the dataset automatically if it is not already present
- Optionally uses a sample (default 80,000 rows) so training is faster on a laptop

### Step B — Quick data check
- Shows class imbalance
- Shows which columns have many `-1` missing values

### Step C — Preprocessing (simple)
- Drops `id`
- Replaces `-1` with missing values
- Fills numeric missing values with median
- Fills categorical / binary missing values with mode
- One-hot encodes small categorical columns

### Step D — Train simple models
Only beginner-friendly models are used:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

No complex models (like XGBoost / deep learning) were used.

### Step E — Compare models
Metrics shown:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**ROC-AUC** is the main score used to choose the best model because the data is imbalanced.

### Step F — Marketing suggestions
Uses feature importance / coefficients from the best model to suggest:
- Focus on high-score customer groups
- Do not market randomly
- Combine model score with business rules

### Step G — Challenges report
Prints common problems and how they were handled:
- Class imbalance
- `-1` missing values
- Anonymized features
- Large data size
- High-cardinality categories

---

## 6. How to run the project

### Option A — Jupyter notebook (recommended for submission)
```bash
pip install -r requirements.txt
jupyter notebook PRCP_1010_InsClaimPred.ipynb
```
Then run all cells from top to bottom.

### Option B — Python file
```bash
pip install -r requirements.txt
python ins_claim_pred.py
```

Optional:
- To use the full dataset, open the notebook/file and set:
  ```python
  SAMPLE_SIZE = None
  ```

---

## 7. Expected output

When you run the script, you will see:
1. Data overview
2. Preprocessing summary
3. Confusion matrix + classification report for each model
4. Model comparison table
5. Best model recommendation
6. Marketing suggestions
7. Challenges report

A CSV file `model_comparison_results.csv` is also saved.

---

## 8. Typical result (with sample run)

On a sample of about 80,000 rows, Logistic Regression often gets the best ROC-AUC among the three simple models.

Important note for learning:
- Accuracy can look okay even if the rare class is hard to predict
- That is why we also check Precision, Recall, F1, and ROC-AUC

---

## 9. Why this is intern-friendly

- Everything is in one readable Python file
- Only simple sklearn models
- Clear comments and printed explanations
- No advanced ML libraries
- Easy to understand, run, and explain in a viva / interview

---

## 10. What you can improve later (optional)

If you continue learning, you can try later:
- Better handling of imbalance (SMOTE / undersampling)
- Cross-validation
- Hyperparameter tuning
- Stronger models (XGBoost / LightGBM)
- Convert the script into a Jupyter notebook

For now, this project stays simple and clear on purpose.
