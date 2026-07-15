"""
PRCP-1010 Insurance Claim Prediction
====================================
Intern-friendly project in ONE Python file.

Tasks covered:
1) Predict which customer is more likely related to a claim / product outcome
2) Give simple suggestions to the insurance marketing team
3) Compare a few easy ML models and pick a best one
4) Write a short challenges report

How to run:
    pip install -r requirements.txt
    python ins_claim_pred.py
"""

import os
import zipfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Config (kept simple on purpose)
# ------------------------------------------------------------
DATA_URL = (
    "https://d3ilbtxij3aepc.cloudfront.net/projects/"
    "CDS-Capstone-Projects/PRCP-1010-InsClaimPred.zip"
)
ZIP_PATH = "PRCP-1010-InsClaimPred.zip"
CSV_PATH = os.path.join("data", "Data", "train.csv")

# Use a sample so the script finishes quickly on a laptop.
# Set to None to use the full dataset (~595k rows).
SAMPLE_SIZE = 80000
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
def download_and_extract_data():
    """Download zip if needed and extract train.csv."""
    if os.path.exists(CSV_PATH):
        print(f"[OK] Found dataset at: {CSV_PATH}")
        return

    print("[INFO] Dataset not found. Downloading...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("[OK] Download complete. Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("data")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "train.csv was not found after extraction. "
            "Please check the zip contents."
        )
    print(f"[OK] Dataset ready at: {CSV_PATH}")


def load_data():
    """Load CSV and optionally take a stratified sample."""
    download_and_extract_data()
    print("[INFO] Reading CSV (this may take a moment)...")
    df = pd.read_csv(CSV_PATH)
    print(f"[OK] Loaded shape: {df.shape}")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        # Keep both classes in roughly the same ratio (simple stratified sample)
        df_0 = df[df["target"] == 0]
        df_1 = df[df["target"] == 1]
        n1 = max(1, int(SAMPLE_SIZE * len(df_1) / len(df)))
        n0 = SAMPLE_SIZE - n1
        df = pd.concat(
            [
                df_0.sample(n=n0, random_state=RANDOM_STATE),
                df_1.sample(n=n1, random_state=RANDOM_STATE),
            ],
            ignore_index=True,
        )
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[OK] Using sample shape: {df.shape}")

    return df


# ------------------------------------------------------------
# 2) Simple EDA summary (project says EDA can be light)
# ------------------------------------------------------------
def quick_data_overview(df):
    print("\n" + "=" * 60)
    print("QUICK DATA OVERVIEW")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nTarget class counts:")
    print(df["target"].value_counts())
    print("\nTarget class percentage:")
    print((df["target"].value_counts(normalize=True) * 100).round(2))

    missing_like = (df == -1).mean().sort_values(ascending=False)
    print("\nTop columns where -1 is used as missing value:")
    print((missing_like[missing_like > 0].head(10) * 100).round(2))


# ------------------------------------------------------------
# 3) Preprocessing
# ------------------------------------------------------------
def preprocess(df):
    """
    Intern-simple cleaning:
    - drop id
    - treat -1 as missing and fill with median / mode
    - keep feature names for later importance report
    """
    data = df.copy()
    data = data.drop(columns=["id"])

    target = data["target"]
    features = data.drop(columns=["target"])

    # Separate column types by naming convention used in this dataset
    cat_cols = [c for c in features.columns if c.endswith("_cat")]
    bin_cols = [c for c in features.columns if c.endswith("_bin")]
    num_cols = [c for c in features.columns if c not in cat_cols + bin_cols]

    # Replace -1 with NaN, then fill simply
    features = features.replace(-1, np.nan)

    for col in num_cols:
        features[col] = features[col].fillna(features[col].median())

    for col in cat_cols + bin_cols:
        # mode can be empty for rare edge cases, so fall back to 0
        mode_vals = features[col].mode(dropna=True)
        fill_value = mode_vals.iloc[0] if len(mode_vals) else 0
        features[col] = features[col].fillna(fill_value)

    # One-hot encode low-cardinality categoricals only (keeps it simple)
    small_cat_cols = [c for c in cat_cols if features[c].nunique() <= 20]
    large_cat_cols = [c for c in cat_cols if c not in small_cat_cols]

    # For high-cardinality cats, keep them as numbers (simple approach)
    features = pd.get_dummies(features, columns=small_cat_cols, drop_first=True)

    print("\n[OK] Preprocessing done")
    print(f"    Numeric cols: {len(num_cols)}")
    print(f"    Binary cols: {len(bin_cols)}")
    print(f"    One-hot cat cols: {len(small_cat_cols)}")
    print(f"    Left as numeric cat cols: {len(large_cat_cols)}")
    print(f"    Final feature matrix shape: {features.shape}")

    return features, target


# ------------------------------------------------------------
# 4) Train + evaluate simple models
# ------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Some models support predict_proba; if not, use decision scores if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n--- {name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics


def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale for Logistic Regression only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simple models only (intern level)
    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            True,  # needs scaling
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }

    results = []
    trained = {}

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, (model, needs_scale) in models.items():
        print(f"\n[INFO] Training {name}...")
        if needs_scale:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(name, model, X_test, y_test)

        results.append(metrics)
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT (higher ROC-AUC is better here)")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    print(f"\n[BEST MODEL FOR PRODUCTION] {best_name}")
    print(
        "Reason: Best ROC-AUC among simple models. "
        "ROC-AUC is useful for imbalanced claim data."
    )

    return trained, results_df, X_train, best_name


# ------------------------------------------------------------
# 5) Marketing suggestions from feature importance
# ------------------------------------------------------------
def marketing_suggestions(trained_models, feature_names, best_name):
    print("\n" + "=" * 60)
    print("TASK 2: SUGGESTIONS TO INSURANCE MARKETING TEAM")
    print("=" * 60)

    model = trained_models[best_name]

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        importance = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    else:
        print("No feature importance available for the best model.")
        return

    top_features = importance.sort_values(ascending=False).head(10)
    print("\nTop 10 important features:")
    print(top_features.round(4))

    print(
        """
Practical suggestions:
1) Focus campaigns on customer groups linked to the top features above
   (individual/car/region style signals in this anonymized data).
2) Because very few customers have target=1, do not market randomly.
   Score customers first, then contact high-probability customers.
3) Use a soft offer / education message for medium-score customers
   instead of expensive high-touch sales for everyone.
4) Re-check model scores every few months because customer risk
   and buying behavior can change over time.
5) Combine model score with business rules (age band, vehicle type,
   region, previous claim history) before final outreach.
"""
    )


# ------------------------------------------------------------
# 6) Challenges report
# ------------------------------------------------------------
def challenges_report():
    print("\n" + "=" * 60)
    print("REPORT ON CHALLENGES FACED")
    print("=" * 60)
    print(
        """
1) Class imbalance
   - Only about 3-4% of rows are target=1.
   - Technique: used class_weight='balanced' and stratified split.
   - Reason: plain accuracy looks high even if the model misses claims.

2) Missing values coded as -1
   - Dataset uses -1 instead of blank/NaN.
   - Technique: replaced -1 with NaN, then filled median/mode.
   - Reason: models should not treat -1 as a real numeric meaning.

3) Feature names are anonymized
   - Columns like ps_car_*, ps_ind_* do not have business labels.
   - Technique: used feature importance from simple models.
   - Reason: still helps marketing prioritize strongly linked signals.

4) Large dataset size
   - Full data has ~595k rows.
   - Technique: used a stratified sample for faster intern demos.
   - Reason: full-data RF can be slow on a normal laptop.

5) High-cardinality categories
   - Some _cat columns have many unique values.
   - Technique: one-hot encode only small categories; keep large ones numeric.
   - Reason: avoids creating too many columns / memory issues.
"""
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("PRCP-1010 Insurance Claim Prediction - Single File Project")
    print("Internship / beginner level models only\n")

    df = load_data()
    quick_data_overview(df)
    X, y = preprocess(df)

    trained_models, results_df, X_train, best_name = train_and_compare(X, y)
    marketing_suggestions(trained_models, X.columns, best_name)
    challenges_report()

    # Save comparison table for submission / report use
    out_csv = "model_comparison_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved model comparison to: {out_csv}")
    print("[DONE] Project finished successfully.")


if __name__ == "__main__":
    main()
