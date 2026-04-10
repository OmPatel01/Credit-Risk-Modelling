"""
train.py
------------------
Training script for the WOE Logistic Regression Scorecard model and XGBoost model.

This script reproduces the full scorecard development pipeline
from the notebook in a clean, runnable Python file.

Steps:
    1.  Load and clean raw data
    2.  Feature engineering
    3.  Feature selection (drop redundant/weak features)
    4.  Train/test split
    5.  Initial WOE binning + IV analysis
    6.  Remove negative-coefficient features
    7.  Remove highly correlated features
    8.  Fix non-monotonic WOE bins
    9.  Final WOE transform
    10. Fit final Logistic Regression
    11. Coefficient validation
    12. Build scorecard points table
    13. Score test set + validate monotonicity
    14. Save all artifacts

Usage:
    python mlops/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import scorecardpy as sc
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import yaml
import mlflow
import mlflow.sklearn
from mlops.evaluate import evaluate_model

mlflow.set_experiment("credit_risk_model")

# ── Paths ────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).resolve().parent.parent
MLFLOW_DIR       = ROOT_DIR / "mlflow"
DATA_PATH        = str(ROOT_DIR / "data" / "raw" / "credit_risk.xls")
ARTIFACTS_MODELS = str(ROOT_DIR / "artifacts" / "models")
ARTIFACTS_PREP   = str(ROOT_DIR / "artifacts" / "preprocessing")

# ── MLflow setup ─────────────────────────────────────────────────
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Point to absolute folder path — no SQLite, no database
mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment("credit_risk_model")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


with open("params.yaml") as f:
    params = yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_PATH        = "data/raw/credit_risk.xls"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_PREP   = "artifacts/preprocessing"

# Scorecard calibration
POINTS0     = 600   # base score for average-risk client
PDO         = 50    # points to double odds — higher = wider range
# TEST_SIZE   = 0.2
# RANDOM_SEED = 42
# LR_C        = 1.0   # logistic regression regularization
# LR_MAX_ITER = 1000
TEST_SIZE   = params["data"]["test_size"]
RANDOM_SEED = params["data"]["random_seed"]
LR_C        = params["logistic"]["C"]
LR_MAX_ITER = params["logistic"]["max_iter"]
# Features removed because of negative WOE coefficients or low IV
NEGATIVE_COEF_FEATURES = [
    "PAY_BILL_RATIO_1",  # negative coef — overlaps NUM_ZERO_PAYMENTS
    "AVG_BILL_AMT",      # negative coef — overlaps UTILIZATION
    "EDUCATION",         # near-zero coef — low IV (0.036)
]

# Features removed due to high correlation with a stronger feature
CORRELATED_FEATURES = [
    "NUM_DELAYS",        # r=0.75 with MAX_DELAY — MAX_DELAY is stronger
]

# Manual WOE bin break points to enforce monotonic risk ordering
# UTILIZATION bins were non-monotonic in auto-binning
MANUAL_BREAKS = {
    "UTILIZATION": [0.10, 0.50, 0.80],
}


# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def calculate_ks(y_true, y_pred_prob: np.ndarray) -> float:
    """
    KS (Kolmogorov-Smirnov) statistic.
    Measures maximum separation between default and non-default
    cumulative distributions.
    Industry benchmark: KS > 0.40 = good model.
    """
    data = pd.DataFrame({"target": y_true, "prob": y_pred_prob})
    data = data.sort_values("prob", ascending=False)
    data["cum_event"]     = np.cumsum(data["target"]) / data["target"].sum()
    data["cum_non_event"] = (np.cumsum(1 - data["target"])
                             / (1 - data["target"]).sum())
    data["ks"] = data["cum_event"] - data["cum_non_event"]
    return float(np.max(data["ks"]))


def gini(auc: float) -> float:
    """Gini = 2 × AUC − 1. Used in Basel II IRB context."""
    return round(2 * auc - 1, 4)


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND CLEAN RAW DATA
# ══════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load raw XLS file and apply data cleaning:
    - Remove duplicate header row (row 0 is a string copy of headers)
    - Cast numeric columns
    - Remap undocumented EDUCATION codes (0,5,6 → 4 = Others)
    - Remap undocumented MARRIAGE code  (0 → 3 = Others)
    - Drop ID column
    """
    print("Loading data...")
    df = pd.read_excel(path)

    # Rename X1..X23 columns to human-readable names
    rename_dict = {
        "Unnamed: 0": "ID",
        "X1": "LIMIT_BAL", "X2": "GENDER",   "X3": "EDUCATION",
        "X4": "MARRIAGE",  "X5": "AGE",
        "X6": "PAY_0",  "X7": "PAY_2",  "X8": "PAY_3",
        "X9": "PAY_4",  "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",  "X19": "PAY_AMT2",  "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",  "X22": "PAY_AMT5",  "X23": "PAY_AMT6",
        "Y":   "DEFAULT_NEXT_MONTH",
    }
    df = df.rename(columns=rename_dict)

    # Remove duplicate header row
    df = df.iloc[1:].reset_index(drop=True)

    # Cast numeric columns
    num_cols = [
        "LIMIT_BAL", "AGE",
        "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
        "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
        "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["DEFAULT_NEXT_MONTH"] = df["DEFAULT_NEXT_MONTH"].astype(int)

    # Remap undocumented category codes
    # EDUCATION: codes 0, 5, 6 are not in the data dictionary → map to Others (4)
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)

    # MARRIAGE: code 0 is not in the data dictionary → map to Others (3)
    df["MARRIAGE"]  = df["MARRIAGE"].replace(0, 3)

    # Drop ID — not a predictor
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Default rate: {df['DEFAULT_NEXT_MONTH'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features used in the scorecard model.
    Same logic as the notebook — centralised here for consistency
    with the API preprocessing module.
    """
    df = df.copy()

    pay_cols     = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    bill_cols    = ["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                    "BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                    "PAY_AMT4","PAY_AMT5","PAY_AMT6"]

    # ── Repayment behaviour ──────────────────────────────────────
    # MAX_DELAY: worst-case delinquency event across 6 months
    df["MAX_DELAY"]      = df[pay_cols].max(axis=1)

    # NUM_DELAYS: frequency of delinquency (months with delay >= 1)
    df["NUM_DELAYS"]     = (df[pay_cols] >= 1).sum(axis=1)

    # ANY_DELAY_FLAG: binary clean vs delinquent segmentation
    df["ANY_DELAY_FLAG"] = (df[pay_cols] >= 1).any(axis=1).astype(int)

    # PAST_DELAY_AVG: average delay excluding most recent month
    df["PAST_DELAY_AVG"] = df[["PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)

    # ── Billing features ─────────────────────────────────────────
    # AVG_BILL_AMT: smoothed debt exposure (replaces 6 collinear BILL_AMTs)
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)

    # BILL_GROWTH: is the outstanding balance growing or shrinking?
    df["BILL_GROWTH"]  = df["BILL_AMT1"] - df["BILL_AMT6"]

    # ── Payment features ─────────────────────────────────────────
    # Zero-payment flags per month
    for col in pay_amt_cols:
        df[col + "_ZERO_FLAG"] = (df[col] == 0).astype(int)

    zero_flag_cols = [c + "_ZERO_FLAG" for c in pay_amt_cols]

    # NUM_ZERO_PAYMENTS: how many months had zero payment (strong default signal)
    df["NUM_ZERO_PAYMENTS"] = df[zero_flag_cols].sum(axis=1)

    # PAYMENT_STD: payment volatility
    df["PAYMENT_STD"] = df[pay_amt_cols].std(axis=1)

    # ── Ratio features ───────────────────────────────────────────
    # PAY_BILL_RATIO: what fraction of the bill was repaid each month
    # A ratio near 0 means the client is accumulating debt without repaying
    for i in range(1, 7):
        df[f"PAY_BILL_RATIO_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0,
        )
        # Clip extreme outliers (max p99 ~ 15-17 in training data)
        df[f"PAY_BILL_RATIO_{i}"] = df[f"PAY_BILL_RATIO_{i}"].clip(upper=15.0)

    ratio_cols = [f"PAY_BILL_RATIO_{i}" for i in range(1, 7)]

    # AVG_PAY_BILL_RATIO: smoothed repayment coverage across all months
    df["AVG_PAY_BILL_RATIO"] = df[ratio_cols].mean(axis=1)

    # ── Exposure features ────────────────────────────────────────
    # UTILIZATION: how maxed-out is the client relative to their limit?
    df["UTILIZATION"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, 1)
    df["UTILIZATION"] = df["UTILIZATION"].clip(upper=1.05)

    return df


# ══════════════════════════════════════════════════════════════════
# STEP 3: FEATURE SELECTION
# Drop EDA-only columns and redundant raw features
# ══════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that should not enter the model:
    - EDA bucket columns (string categories, not model inputs)
    - Raw bill amounts (replaced by AVG_BILL_AMT + BILL_GROWTH)
    - Old PAY status columns (replaced by MAX_DELAY, NUM_DELAYS)
    - Individual zero flags (replaced by NUM_ZERO_PAYMENTS)
    - Old pay amount columns (replaced by ratios)
    - RECENT_DELAY (exact duplicate of PAY_0, r=1.0)
    - Weak demographic features (GENDER IV=0.011, MARRIAGE IV=0.005)
    - PAYMENT_STD (removing it improved AUC in testing)
    """
    drop_cols = [
        # EDA-only bucket columns — string labels not useful for model
        "LIMIT_BUCKET", "BILL_GROWTH_BUCKET",
        "AVG_BILL_BUCKET", "PAYMENT_STD_BUCKET",

        # Individual zero flags — NUM_ZERO_PAYMENTS captures all of them
        "PAY_AMT1_ZERO_FLAG", "PAY_AMT2_ZERO_FLAG", "PAY_AMT3_ZERO_FLAG",
        "PAY_AMT4_ZERO_FLAG", "PAY_AMT5_ZERO_FLAG", "PAY_AMT6_ZERO_FLAG",

        # Raw bill amounts — r>0.90 between adjacent months (severe collinearity)
        # Replaced by AVG_BILL_AMT and BILL_GROWTH
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",

        # Older PAY status columns — MAX_DELAY and NUM_DELAYS summarise them
        "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",

        # Older pay amounts — ratios capture repayment capacity better
        "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",

        # Older PAY_BILL_RATIOs — keep only 1 and 2; average covers the rest
        "PAY_BILL_RATIO_2", "PAY_BILL_RATIO_3",
        "PAY_BILL_RATIO_4", "PAY_BILL_RATIO_5", "PAY_BILL_RATIO_6",

        # RECENT_DELAY = PAY_0 exactly (r=1.0) — pure duplicate
        "RECENT_DELAY",

        # ANY_DELAY_FLAG = (NUM_DELAYS > 0) — redundant with NUM_DELAYS
        "ANY_DELAY_FLAG",

        # Weak demographic features (IV < 0.02, regulatory risk)
        "GENDER", "MARRIAGE",

        # PAYMENT_STD: removing improved AUC by 0.0012 in testing
        "PAYMENT_STD",
    ]

    # Only drop columns that actually exist in the dataframe
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    print(f"Features after selection: {df.shape[1] - 1}")
    return df


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════

def train_scorecard():

    # ── Create output directories ────────────────────────────────
    os.makedirs(ARTIFACTS_MODELS, exist_ok=True)
    os.makedirs(ARTIFACTS_PREP,   exist_ok=True)

    # ── Step 1: Load and clean ───────────────────────────────────
    df = load_and_clean(DATA_PATH)

    # ── Step 2: Feature engineering ──────────────────────────────
    print("\nEngineering features...")
    df = engineer_features(df)

    # ── Step 3: Feature selection ─────────────────────────────────
    print("Selecting features...")
    df_model = select_features(df)

    print("\nFinal feature set:")
    feature_list = [c for c in df_model.columns if c != "DEFAULT_NEXT_MONTH"]
    for f in feature_list:
        print(f"  - {f}")

    # ── Step 4: Train / test split ────────────────────────────────
    print("\nSplitting data...")
    train, test = train_test_split(
        df_model,
        test_size=TEST_SIZE,
        stratify=df_model["DEFAULT_NEXT_MONTH"],
        random_state=RANDOM_SEED,
    )
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Default rate — Train: {train['DEFAULT_NEXT_MONTH'].mean():.3f} "
          f"| Test: {test['DEFAULT_NEXT_MONTH'].mean():.3f}")

    # ── Step 5: Initial WOE binning + IV analysis ─────────────────
    print("\nComputing IV and initial WOE bins...")
    iv_table = sc.iv(train, y="DEFAULT_NEXT_MONTH")
    print("\nInformation Value (IV):")
    print(iv_table.sort_values("info_value", ascending=False).to_string())

    initial_bins = sc.woebin(train, y="DEFAULT_NEXT_MONTH", no_cores=1)

    # ── Step 6: Remove negative-coefficient features ──────────────
    print("\nFirst-pass LR to identify negative coefficients...")
    train_woe_init = sc.woebin_ply(train, initial_bins)
    test_woe_init  = sc.woebin_ply(test,  initial_bins)

    X_tr_init = train_woe_init.drop(columns=["DEFAULT_NEXT_MONTH"])
    y_tr_init = train_woe_init["DEFAULT_NEXT_MONTH"].astype(int)

    lr_init = LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C, random_state=RANDOM_SEED)
    lr_init.fit(X_tr_init, y_tr_init)

    coef_init = pd.DataFrame({
        "feature":     X_tr_init.columns,
        "coefficient": lr_init.coef_[0],
    }).sort_values("coefficient", ascending=False)

    negatives = coef_init[coef_init["coefficient"] < 0]["feature"].tolist()
    print(f"Negative coefficients found: {negatives}")

    print(f"Removing: {NEGATIVE_COEF_FEATURES}")
    train_clean = train.drop(columns=NEGATIVE_COEF_FEATURES, errors="ignore")
    test_clean  = test.drop(columns=NEGATIVE_COEF_FEATURES,  errors="ignore")

    # ── Step 7: Remove highly correlated features ─────────────────
    print(f"\nCorrelation check before removing {CORRELATED_FEATURES}:")
    check_cols = ["NUM_DELAYS", "MAX_DELAY", "PAY_0"]
    existing_check = [c for c in check_cols if c in train_clean.columns]
    print(train_clean[existing_check].corr().round(3))

    print(f"Removing highly correlated: {CORRELATED_FEATURES}")
    train_final = train_clean.drop(columns=CORRELATED_FEATURES, errors="ignore")
    test_final  = test_clean.drop(columns=CORRELATED_FEATURES,  errors="ignore")

    print(f"\nFinal scorecard features: {train_final.shape[1] - 1}")

    # ── Step 8: WOE binning with monotonic fix ────────────────────
    print("\nApplying WOE binning with monotonic correction for UTILIZATION...")
    bins_final = sc.woebin(
        train_final,
        y="DEFAULT_NEXT_MONTH",
        breaks_list=MANUAL_BREAKS,
        no_cores=1,
    )

    print("\nUTILIZATION bins after monotonic fix:")
    print(bins_final["UTILIZATION"][["bin", "badprob", "woe"]].to_string())

    # ── Step 9: Final WOE transformation ─────────────────────────
    print("\nApplying WOE transformation...")
    train_woe = sc.woebin_ply(train_final, bins_final)
    test_woe  = sc.woebin_ply(test_final,  bins_final)

    X_train = train_woe.drop(columns=["DEFAULT_NEXT_MONTH"])
    y_train = train_woe["DEFAULT_NEXT_MONTH"].astype(int)
    X_test  = test_woe.drop(columns=["DEFAULT_NEXT_MONTH"])
    y_test  = test_woe["DEFAULT_NEXT_MONTH"].astype(int)

    print(f"WOE features: {X_train.shape[1]}")

    # ── Step 10: Fit final Logistic Regression ────────────────────
    print("\nFitting Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=RANDOM_SEED,
    )
    lr_model.fit(X_train, y_train)

    pred_prob = lr_model.predict_proba(X_test)[:, 1]

    # ── FIX: evaluate before any print that uses metrics ─────────
    # metrics   = evaluate_model(y_test, pred_prob, model_name="scorecard_lr")
    # gini_coef = gini(metrics["auc"])

    # print(f"\n── Model Performance ─────────────────────")
    # print(f"  AUC:   {metrics['auc']:.4f}")
    # print(f"  KS:    {metrics['ks']:.4f}")
    # print(f"  Gini:  {gini_coef:.4f}")

    # ── Step 11: Coefficient validation ──────────────────────────
    print("\n── Coefficient Check ─────────────────────")
    coef_df = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": lr_model.coef_[0],
    }).sort_values("coefficient", ascending=False)

    print(coef_df.to_string())

    neg_coefs = coef_df[coef_df["coefficient"] < 0]
    if len(neg_coefs) > 0:
        print(f"\n⚠️  WARNING — Negative coefficients: {neg_coefs['feature'].tolist()}")
        print("   These features need investigation before production deployment.")
    else:
        print("\n✅ All coefficients positive — scorecard is clean")

    # ── Step 12: Build scorecard points table ─────────────────────
    print("\nBuilding scorecard points table...")
    n_default     = y_train.sum()
    n_non_default = len(y_train) - n_default
    actual_odds   = n_non_default / n_default
    print(f"  Actual odds (non-default / default): {actual_odds:.2f}")
    print(f"  Points0: {POINTS0} | PDO: {PDO}")

    scorecard_table = sc.scorecard(
        bins_final,
        lr_model,
        X_train.columns.tolist(),
        points0=POINTS0,
        odds0=actual_odds,
        pdo=PDO,
    )

    print("\n── Scorecard Points ──────────────────────")
    for feature, table in scorecard_table.items():
        print(f"\n  {feature}:")
        print(table[["bin", "points"]].to_string(index=False))

    # ── Step 13: Score test set + validate monotonicity ───────────
    print("\n── Scorecard Validation ──────────────────")
    test_scores = sc.scorecard_ply(test_final, scorecard_table, print_step=0)
    test_scores["DEFAULT_NEXT_MONTH"] = y_test.values

    print("\nScore distribution:")
    print(test_scores["score"].describe().round(1))

    test_scores["score_band"] = pd.cut(
        test_scores["score"],
        bins=[0, 650, 700, 750, 800, 850, 1000],
        labels=["<650", "650-700", "700-750", "750-800", "800-850", ">850"],
    )

    band_summary = test_scores.groupby("score_band", observed=True).agg(
        count=("score", "count"),
        defaults=("DEFAULT_NEXT_MONTH", "sum"),
        default_rate=("DEFAULT_NEXT_MONTH", "mean"),
    ).round(3)

    print("\nDefault Rate by Score Band:")
    print(band_summary[band_summary["count"] > 0].to_string())

    rates = band_summary[band_summary["count"] > 0]["default_rate"].values
    is_monotonic = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    if is_monotonic:
        print("\n✅ Monotonic — higher score = lower default rate")
    else:
        print("\n⚠️  WARNING — Score bands are not fully monotonic")

    # ── Step 14: Save all artifacts ───────────────────────────────
    print("\n── Saving Artifacts ──────────────────────")

    lr_path = os.path.join(ARTIFACTS_MODELS, "scorecard_lr_model.joblib")
    joblib.dump(lr_model, lr_path)
    print(f"  Saved: {lr_path}")

    bins_path = os.path.join(ARTIFACTS_PREP, "woe_bins.joblib")
    joblib.dump(bins_final, bins_path)
    print(f"  Saved: {bins_path}")

    scorecard_path = os.path.join(ARTIFACTS_PREP, "scorecard.joblib")
    joblib.dump(scorecard_table, scorecard_path)
    print(f"  Saved: {scorecard_path}")

    feature_cols = X_train.columns.tolist()
    feat_cols_path = os.path.join(ARTIFACTS_PREP, "feature_columns_scorecard.json")
    with open(feat_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ── MLflow logging ────────────────────────────────────────────
    with mlflow.start_run(run_name="scorecard_model"):

        # evaluate_model must be INSIDE the run — it calls mlflow.log_metrics internally
        metrics   = evaluate_model(y_test, pred_prob, model_name="scorecard_lr")
        gini_coef = gini(metrics["auc"])
        mlflow.log_metric("gini", gini_coef)

        print(f"\n── Model Performance ─────────────────────")
        print(f"  AUC:   {metrics['auc']:.4f}")
        print(f"  KS:    {metrics['ks']:.4f}")
        print(f"  Gini:  {gini_coef:.4f}")
        
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("approach", "scorecard_woe")

        mlflow.log_params(params["logistic"])
        mlflow.log_params(params["data"])

        mlflow.set_tag("project", "credit_risk")
        mlflow.set_tag("model_family", "scorecard")
        mlflow.set_tag("stage", "training")

        mlflow.sklearn.log_model(
            lr_model,
            "scorecard_model",
            registered_model_name="credit_risk_scorecard",
        )

        mlflow.log_artifact(lr_path)
        mlflow.log_artifact(bins_path)
        mlflow.log_artifact(scorecard_path)
        mlflow.log_artifact(feat_cols_path)

    print(f"  Saved: {feat_cols_path}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── Final summary ─────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  TRAINING COMPLETE")
    print("══════════════════════════════════════════")
    print(f"  AUC:              {metrics['auc']:.4f}")
    print(f"  KS Statistic:     {metrics['ks']:.4f}")
    print(f"  Gini Coefficient: {gini_coef:.4f}")
    print(f"  Score range:      {test_scores['score'].min():.0f} – "
          f"{test_scores['score'].max():.0f}")
    print(f"  Features used:    {len(feature_cols)}")
    print(f"  Monotonic bands:  {'✅' if is_monotonic else '⚠️ '}")
    print("══════════════════════════════════════════")

    return {
        "auc":       metrics["auc"],
        "ks":        metrics["ks"],
        "gini":      gini_coef,
        "features":  feature_cols,
        "monotonic": is_monotonic,
    }


def train_xgb(df_model: pd.DataFrame):
    """
    Train XGBoost model using already processed dataframe.

    Assumes:
        df_model is already:
            load_and_clean → engineer_features → select_features
    """

    # ── Step 1: Split X / y ─────────────────────────
    X = df_model.drop(columns=["DEFAULT_NEXT_MONTH"])
    y = df_model["DEFAULT_NEXT_MONTH"].astype(int)

    categorical_cols = ["EDUCATION"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"Total features: {X.shape[1]}")
    print(f"Numerical: {len(numerical_cols)} | Categorical: {len(categorical_cols)}")

    # ── Step 2: Train/Test Split ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Step 3: Pipeline ───────────────────────────
    xgb_preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ])

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", xgb_preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=3.52,
            n_estimators=params["xgboost"]["n_estimators"],
            learning_rate=params["xgboost"]["learning_rate"],
            max_depth=params["xgboost"]["max_depth"],
            min_child_weight=params["xgboost"]["min_child_weight"],
            subsample=params["xgboost"]["subsample"],
            colsample_bytree=params["xgboost"]["colsample_bytree"],
            gamma=params["xgboost"]["gamma"],
            reg_alpha=params["xgboost"]["reg_alpha"],
            reg_lambda=params["xgboost"]["reg_lambda"],
            random_state=RANDOM_SEED,
            eval_metric="auc",
        ))
    ])

    # ── Step 4: Train ──────────────────────────────
    print("\nTraining XGBoost...")
    xgb_pipeline.fit(X_train, y_train)

    # ── Step 5: Evaluation ─────────────────────────
    xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    # ── FIX: evaluate before any print that uses metrics ─────────
    # metrics = evaluate_model(y_test, xgb_proba, model_name="xgboost")

    # print("\n── Model Performance ─────────────────────")
    # print(f"  AUC: {metrics['auc']:.4f}")
    # print(f"  KS:  {metrics['ks']:.4f}")

    # ── Cross Validation ───────────────────────────
    print("\n⏳ Running 5-fold CV...")
    cv_scores = cross_val_score(
        xgb_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(f"CV Mean: {cv_scores.mean():.4f}")
    print(f"CV Std:  {cv_scores.std():.4f}")

    # ── Step 6: Feature Importance ─────────────────
    print("\n── Feature Importance ────────────────────")

    feature_names = numerical_cols + categorical_cols

    feat_imp = pd.Series(
        xgb_pipeline.named_steps["model"].feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("\nTop 10 Features:")
    print(feat_imp.head(10).to_string())

    print("\nBottom 5 Features:")
    print(feat_imp.tail(5).to_string())

    # ── Step 7: Save Artifacts ─────────────────────
    print("\n── Saving Artifacts ─────────────────────")

    os.makedirs(ARTIFACTS_MODELS, exist_ok=True)
    os.makedirs(ARTIFACTS_PREP, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_MODELS, "xgb_pipeline.joblib")
    joblib.dump(xgb_pipeline, model_path)
    print(f"  Saved: {model_path}")

    feature_cols_path = os.path.join(ARTIFACTS_PREP, "feature_columns_xgb.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Saved: {feature_cols_path}")

    # ── MLflow logging ────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost_model"):

        # evaluate_model must be INSIDE the run — it calls mlflow.log_metrics internally
        metrics = evaluate_model(y_test, xgb_proba, model_name="xgboost")

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(params["xgboost"])
        mlflow.log_metric("cv_mean_auc", cv_scores.mean())
        mlflow.log_metric("cv_mean_auc", cv_scores.mean())  # FIX: removed duplicate pair
        mlflow.log_metric("cv_std_auc", cv_scores.std())

        mlflow.set_tag("project", "credit_risk")
        mlflow.set_tag("model_family", "tree_model")
        mlflow.set_tag("stage", "training")

        mlflow.sklearn.log_model(
            xgb_pipeline,
            "xgb_model",
            registered_model_name="credit_risk_xgboost",
        )

        mlflow.log_artifact(feature_cols_path)

    print("\n══════════════════════════════════════════")
    print("  XGBOOST TRAINING COMPLETE")
    print("══════════════════════════════════════════")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  KS:       {metrics['ks']:.4f}")
    print(f"  CV Mean:  {cv_scores.mean():.4f}")
    print(f"  CV Std:   {cv_scores.std():.4f}")
    print("══════════════════════════════════════════")

    return {
        "auc":      float(metrics["auc"]),
        "ks":       float(metrics["ks"]),
        "cv_mean":  float(cv_scores.mean()),
        "cv_std":   float(cv_scores.std()),
        "features": feature_names,
    }

def main():
    print("Starting full training pipeline...\n")

    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    df_model = select_features(df)

    # Scorecard (already uses full pipeline internally)
    train_scorecard()

    # XGBoost (use processed data)
    train_xgb(df_model)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("credit_risk_model")
    print("Experiment ID:", experiment.experiment_id)

    runs = client.search_runs(experiment.experiment_id)
    for r in runs:
        print(f"Run: {r.info.run_name} | AUC: {r.data.metrics.get('auc')}")


if __name__ == "__main__":
    main()