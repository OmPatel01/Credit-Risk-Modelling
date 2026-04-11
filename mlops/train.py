"""
mlops/train.py
--------------
Full end-to-end training pipeline for both the champion scorecard model and the
XGBoost challenger model. Running this script reproduces all training artifacts
from scratch using only the raw data file.

Pipeline stages (in order):
    1.  Load + clean raw XLS data (remap undocumented category codes, drop ID)
    2.  Feature engineering (derive 15+ behavioural features from raw columns)
    3.  Feature selection (remove collinear, weak, and leaky columns)
    4.  Train/test split (stratified on target to preserve default rate)
    5.  Initial WOE binning + IV analysis to identify informative features
    6.  First-pass LR to detect and remove negative-coefficient features
    7.  Remove highly correlated features (per correlation matrix analysis)
    8.  Re-bin with manual breaks to enforce WOE monotonicity (UTILIZATION)
    9.  Final WOE transformation of train and test sets
    10. Fit final Logistic Regression on WOE features
    11. Validate all coefficients are positive (required for a valid scorecard)
    12. Build scorecard points table from LR coefficients + WOE bins
    13. Score test set and verify default rate decreases as score increases (monotonicity)
    14. Save all artifacts to disk + log to MLflow
    Then repeat steps 1–3 and run XGBoost training on the same processed data.

Usage:
    python mlops/train.py

Artifacts produced:
    artifacts/models/scorecard_lr_model.joblib
    artifacts/models/xgb_pipeline.joblib
    artifacts/preprocessing/woe_bins.joblib
    artifacts/preprocessing/scorecard.joblib
    artifacts/preprocessing/feature_columns_scorecard.json
    artifacts/preprocessing/feature_columns_xgb.json
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

# ── Path setup — all paths resolved from this file's location so the script
# works regardless of the working directory it's launched from
ROOT_DIR         = Path(__file__).resolve().parent.parent
MLFLOW_DIR       = ROOT_DIR / "mlflow"
DATA_PATH        = str(ROOT_DIR / "data" / "raw" / "credit_risk.xls")
ARTIFACTS_MODELS = str(ROOT_DIR / "artifacts" / "models")
ARTIFACTS_PREP   = str(ROOT_DIR / "artifacts" / "preprocessing")

# ── MLflow — use a local filesystem URI (no database server required)
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment("credit_risk_model")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ── Hyperparameters — loaded from params.yaml so DVC can track and version them.
# Never hardcode hyperparameters in training code; params.yaml is the single source.
with open("params.yaml") as f:
    params = yaml.safe_load(f)

DATA_PATH        = "data/raw/credit_risk.xls"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_PREP   = "artifacts/preprocessing"

# ── Scorecard calibration — controls the score range and sensitivity
# POINTS0: score assigned to a borrower with average odds (non-default/default ratio)
# PDO: how many points correspond to a doubling of odds (higher = wider score spread)
POINTS0 = 600
PDO     = 50

TEST_SIZE   = params["data"]["test_size"]
RANDOM_SEED = params["data"]["random_seed"]
LR_C        = params["logistic"]["C"]
LR_MAX_ITER = params["logistic"]["max_iter"]

# ── Features excluded after first-pass LR coefficient inspection
# These three were identified during initial model development:
#   PAY_BILL_RATIO_1 → negative coefficient; its variance overlaps with NUM_ZERO_PAYMENTS
#   AVG_BILL_AMT     → negative coefficient; captured better by UTILIZATION (credit-limit-normalised)
#   EDUCATION        → near-zero coefficient and low IV (0.036); not worth the regulatory scrutiny
NEGATIVE_COEF_FEATURES = [
    "PAY_BILL_RATIO_1",
    "AVG_BILL_AMT",
    "EDUCATION",
]

# ── Features excluded due to high inter-feature correlation
# NUM_DELAYS (r=0.75 with MAX_DELAY) is redundant when MAX_DELAY is present;
# keeping both would inflate the coefficient of the collinear pair and reduce interpretability
CORRELATED_FEATURES = [
    "NUM_DELAYS",
]

# ── Manual WOE bin boundaries to enforce monotonic risk ordering
# Auto-binning produced non-monotonic WOE for UTILIZATION (some middle bins had lower
# default rates than adjacent bins, which violates scorecard intuition).
# Manual breaks force the bins into a sensible low→medium→high→very-high structure.
MANUAL_BREAKS = {
    "UTILIZATION": [0.10, 0.50, 0.80],
}


# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def calculate_ks(y_true, y_pred_prob: np.ndarray) -> float:
    """
    KS statistic — maximum gap between cumulative default and non-default rate curves.

    Duplicated here (also in core/utils.py and mlops/evaluate.py) so this script
    remains self-contained and can be run without importing from the API layer.
    Industry benchmark: KS > 0.40 = good; KS > 0.50 = strong.
    """
    data = pd.DataFrame({"target": y_true, "prob": y_pred_prob})
    data = data.sort_values("prob", ascending=False)
    data["cum_event"]     = np.cumsum(data["target"]) / data["target"].sum()
    data["cum_non_event"] = (np.cumsum(1 - data["target"])
                             / (1 - data["target"]).sum())
    data["ks"] = data["cum_event"] - data["cum_non_event"]
    return float(np.max(data["ks"]))


def gini(auc: float) -> float:
    """
    Gini coefficient derived from AUC: Gini = 2 × AUC − 1.

    Gini is the standard discrimination metric in Basel II IRB models.
    A Gini of 0 = random model; 1 = perfect model; typical retail models: 0.40–0.65.
    """
    return round(2 * auc - 1, 4)


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND CLEAN RAW DATA
# ══════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the raw XLS file, rename cryptic column codes, and fix data quality issues.

    Data quality issues addressed:
        - Row 0 is a duplicate of the header (string copy) — removed via iloc[1:]
        - EDUCATION codes 0, 5, 6 are undocumented in the dataset description
          → remapped to 4 (Others) to avoid spurious category splits during WOE binning
        - MARRIAGE code 0 is similarly undocumented → remapped to 3 (Others)
        - ID column provides no predictive signal and must be dropped before any modelling
    """
    print("Loading data...")
    df = pd.read_excel(path)

    # Original columns are labelled X1..X23 — rename to human-readable names
    # that match the preprocessing pipeline's expected column names
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

    # First data row is actually a second header (string copies of column names) — drop it
    df = df.iloc[1:].reset_index(drop=True)

    num_cols = [
        "LIMIT_BAL", "AGE",
        "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
        "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
        "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["DEFAULT_NEXT_MONTH"] = df["DEFAULT_NEXT_MONTH"].astype(int)

    # Remap undocumented category codes to the catch-all "Others" bucket
    # so WOE binning doesn't create meaningless singleton bins
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
    df["MARRIAGE"]  = df["MARRIAGE"].replace(0, 3)

    df = df.drop(columns=["ID"]) if "ID" in df.columns else df

    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Default rate: {df['DEFAULT_NEXT_MONTH'].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features used by both models.

    Must be kept in exact sync with core/preprocessing.py:engineer_features().
    Any divergence between training and inference engineering silently degrades predictions.
    See core/preprocessing.py for detailed comments on each feature's purpose.

    Note: training adds a few extra columns (PAYMENT_STD, per-month zero flags) that are
    later dropped in select_features(). These are omitted from the inference version to
    avoid computing unused columns at prediction time.
    """
    df = df.copy()

    pay_cols     = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    bill_cols    = ["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                    "BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                    "PAY_AMT4","PAY_AMT5","PAY_AMT6"]

    # ── Repayment behaviour
    df["MAX_DELAY"]      = df[pay_cols].max(axis=1)
    df["NUM_DELAYS"]     = (df[pay_cols] >= 1).sum(axis=1)
    df["ANY_DELAY_FLAG"] = (df[pay_cols] >= 1).any(axis=1).astype(int)
    df["PAST_DELAY_AVG"] = df[["PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)

    # ── Billing
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["BILL_GROWTH"]  = df["BILL_AMT1"] - df["BILL_AMT6"]

    # ── Per-month zero payment flags — only used in training to compute NUM_ZERO_PAYMENTS;
    # dropped in select_features() so they never reach the model directly
    for col in pay_amt_cols:
        df[col + "_ZERO_FLAG"] = (df[col] == 0).astype(int)

    zero_flag_cols = [c + "_ZERO_FLAG" for c in pay_amt_cols]
    df["NUM_ZERO_PAYMENTS"] = df[zero_flag_cols].sum(axis=1)

    # PAYMENT_STD measures monthly payment volatility; included here for EDA but
    # removed in select_features() because it marginally hurt AUC in cross-validation
    df["PAYMENT_STD"] = df[pay_amt_cols].std(axis=1)

    # ── Pay-to-bill ratios — clipped at p99 (~15) to match inference-time clipping
    for i in range(1, 7):
        df[f"PAY_BILL_RATIO_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0,
        )
        df[f"PAY_BILL_RATIO_{i}"] = df[f"PAY_BILL_RATIO_{i}"].clip(upper=15.0)

    ratio_cols = [f"PAY_BILL_RATIO_{i}" for i in range(1, 7)]
    df["AVG_PAY_BILL_RATIO"] = df[ratio_cols].mean(axis=1)

    # ── Credit utilisation — capped at 1.05 to handle balances slightly over limit
    df["UTILIZATION"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, 1)
    df["UTILIZATION"] = df["UTILIZATION"].clip(upper=1.05)

    return df


# ══════════════════════════════════════════════════════════════════
# STEP 3: FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all columns that should not enter the WOE binning or model fitting stages.

    Removal rationale for each group:
        EDA bucket columns    → string labels created during analysis notebooks, not model inputs
        Individual zero flags → their aggregate (NUM_ZERO_PAYMENTS) is already in the dataset
        Raw BILL_AMTs         → r > 0.90 between adjacent months; replaced by AVG_BILL_AMT + BILL_GROWTH
        Older PAY_X cols      → MAX_DELAY and NUM_DELAYS summarise them more robustly
        Older PAY_AMTs        → pay-to-bill ratios are more informative than raw amounts
        Older PAY_BILL_RATIOs → AVG_PAY_BILL_RATIO covers months 2–6; month 1 is kept separately
        RECENT_DELAY          → exact duplicate of PAY_0 (r=1.0); would cause perfect collinearity
        ANY_DELAY_FLAG        → binary version of NUM_DELAYS; redundant once NUM_DELAYS is present
        GENDER, MARRIAGE      → IV < 0.02 and legally sensitive features in credit models
        PAYMENT_STD           → removing improved AUC by 0.0012; adds noise without signal
    """
    drop_cols = [
        "LIMIT_BUCKET", "BILL_GROWTH_BUCKET",
        "AVG_BILL_BUCKET", "PAYMENT_STD_BUCKET",
        "PAY_AMT1_ZERO_FLAG", "PAY_AMT2_ZERO_FLAG", "PAY_AMT3_ZERO_FLAG",
        "PAY_AMT4_ZERO_FLAG", "PAY_AMT5_ZERO_FLAG", "PAY_AMT6_ZERO_FLAG",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "PAY_BILL_RATIO_2", "PAY_BILL_RATIO_3",
        "PAY_BILL_RATIO_4", "PAY_BILL_RATIO_5", "PAY_BILL_RATIO_6",
        "RECENT_DELAY",
        "ANY_DELAY_FLAG",
        "GENDER", "MARRIAGE",
        "PAYMENT_STD",
    ]

    # Drop only what actually exists — prevents crashes if upstream steps changed
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    print(f"Features after selection: {df.shape[1] - 1}")  # -1 excludes the target column
    return df


# ══════════════════════════════════════════════════════════════════
# SCORECARD TRAINING
# ══════════════════════════════════════════════════════════════════

def train_scorecard():
    """
    Train the full WOE Logistic Regression scorecard and save all artifacts.

    The scorecard approach requires two passes of LR:
        Pass 1 (initial_bins): identify negative coefficients to remove.
        Pass 2 (bins_final):   fit the production model on the cleaned feature set.
    This is standard scorecard development practice — a single pass is not sufficient
    because removing features changes WOE values for correlated features.

    MLflow logging happens inside this function; it must be called within a run context
    (see the `with mlflow.start_run()` block at the bottom of this function).
    """
    os.makedirs(ARTIFACTS_MODELS, exist_ok=True)
    os.makedirs(ARTIFACTS_PREP,   exist_ok=True)

    # ── Steps 1–3: load, engineer, select (full pipeline)
    df       = load_and_clean(DATA_PATH)
    df       = engineer_features(df)
    df_model = select_features(df)

    print("\nFinal feature set:")
    feature_list = [c for c in df_model.columns if c != "DEFAULT_NEXT_MONTH"]
    for f in feature_list:
        print(f"  - {f}")

    # ── Step 4: Stratified split — preserves default rate in both train and test
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

    # ── Step 5: IV analysis — Information Value measures each feature's predictive power.
    # IV < 0.02 = useless, 0.02–0.10 = weak, 0.10–0.30 = medium, > 0.30 = strong.
    # WOE binning is only run on training data — never on test data (leakage prevention).
    print("\nComputing IV and initial WOE bins...")
    iv_table     = sc.iv(train, y="DEFAULT_NEXT_MONTH")
    initial_bins = sc.woebin(train, y="DEFAULT_NEXT_MONTH", no_cores=1)

    print("\nInformation Value (IV):")
    print(iv_table.sort_values("info_value", ascending=False).to_string())

    # ── Step 6: First-pass LR to surface negative coefficients
    # In a valid scorecard, ALL coefficients must be positive (higher WOE = higher risk score).
    # A negative coefficient means the feature's WOE relationship is inverted, which breaks
    # the monotonicity requirement and makes the score uninterpretable.
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

    # Remove the predetermined set (not the dynamically detected list) to ensure
    # reproducibility — dynamic detection can vary with data partitioning
    train_clean = train.drop(columns=NEGATIVE_COEF_FEATURES, errors="ignore")
    test_clean  = test.drop(columns=NEGATIVE_COEF_FEATURES,  errors="ignore")

    # ── Step 7: Remove highly correlated features
    # NUM_DELAYS and MAX_DELAY both measure payment delinquency but from different angles
    # (count vs severity). They have r=0.75; keeping both inflates their joint coefficient
    # and makes the scorecard harder to explain to regulators.
    print(f"\nCorrelation check before removing {CORRELATED_FEATURES}:")
    check_cols    = ["NUM_DELAYS", "MAX_DELAY", "PAY_0"]
    existing_check = [c for c in check_cols if c in train_clean.columns]
    print(train_clean[existing_check].corr().round(3))
    print(f"Removing highly correlated: {CORRELATED_FEATURES}")

    train_final = train_clean.drop(columns=CORRELATED_FEATURES, errors="ignore")
    test_final  = test_clean.drop(columns=CORRELATED_FEATURES,  errors="ignore")
    print(f"\nFinal scorecard features: {train_final.shape[1] - 1}")

    # ── Step 8: Re-bin with MANUAL_BREAKS for UTILIZATION
    # scorecardpy's automatic binning produced non-monotonic WOE for UTILIZATION
    # (a middle bin had lower badprob than its neighbours). Manual breaks enforce
    # the intuitive ordering: low utilisation → low risk, high utilisation → high risk.
    print("\nApplying WOE binning with monotonic correction for UTILIZATION...")
    bins_final = sc.woebin(
        train_final,
        y="DEFAULT_NEXT_MONTH",
        breaks_list=MANUAL_BREAKS,
        no_cores=1,
    )

    print("\nUTILIZATION bins after monotonic fix:")
    print(bins_final["UTILIZATION"][["bin", "badprob", "woe"]].to_string())

    # ── Step 9: Final WOE transformation — applied to both train and test using TRAINING bins only
    print("\nApplying WOE transformation...")
    train_woe = sc.woebin_ply(train_final, bins_final)
    test_woe  = sc.woebin_ply(test_final,  bins_final)

    X_train = train_woe.drop(columns=["DEFAULT_NEXT_MONTH"])
    y_train = train_woe["DEFAULT_NEXT_MONTH"].astype(int)
    X_test  = test_woe.drop(columns=["DEFAULT_NEXT_MONTH"])
    y_test  = test_woe["DEFAULT_NEXT_MONTH"].astype(int)

    print(f"WOE features: {X_train.shape[1]}")

    # ── Step 10: Final LR fit — this model goes to production
    print("\nFitting Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=RANDOM_SEED,
    )
    lr_model.fit(X_train, y_train)
    pred_prob = lr_model.predict_proba(X_test)[:, 1]

    # ── Step 11: Coefficient validation — ALL must be positive for a valid scorecard
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

    # ── Step 12: Scorecard points table
    # sc.scorecard converts LR coefficients + WOE bins into additive integer points per bin.
    # The calibration (points0=600, pdo=50) anchors the scale:
    #   A borrower at average odds gets 600 points; each halving of odds costs 50 points.
    # actual_odds is computed from training data to correctly anchor the scale.
    print("\nBuilding scorecard points table...")
    n_default     = y_train.sum()
    n_non_default = len(y_train) - n_default
    actual_odds   = n_non_default / n_default
    print(f"  Actual odds (non-default / default): {actual_odds:.2f}")

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

    # ── Step 13: Monotonicity validation
    # Score bands must have strictly decreasing default rates as scores increase.
    # Any violation (higher score band with higher default rate) is a scorecard defect.
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

    rates       = band_summary[band_summary["count"] > 0]["default_rate"].values
    is_monotonic = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    print("\n✅ Monotonic" if is_monotonic else "\n⚠️  WARNING — Score bands are not fully monotonic")

    # ── Step 14: Save artifacts — the exact set the inference API loads at startup
    print("\n── Saving Artifacts ──────────────────────")

    lr_path = os.path.join(ARTIFACTS_MODELS, "scorecard_lr_model.joblib")
    joblib.dump(lr_model, lr_path)

    bins_path = os.path.join(ARTIFACTS_PREP, "woe_bins.joblib")
    joblib.dump(bins_final, bins_path)

    scorecard_path = os.path.join(ARTIFACTS_PREP, "scorecard.joblib")
    joblib.dump(scorecard_table, scorecard_path)

    # Feature column list is saved as JSON (not joblib) so it's human-readable and
    # easy to inspect without Python. The API loads this to know which WOE columns to select.
    feature_cols   = X_train.columns.tolist()
    feat_cols_path = os.path.join(ARTIFACTS_PREP, "feature_columns_scorecard.json")
    with open(feat_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ── MLflow run — evaluate_model must be inside the run because it calls mlflow.log_metrics()
    with mlflow.start_run(run_name="scorecard_model"):
        metrics   = evaluate_model(y_test, pred_prob, model_name="scorecard_lr")
        gini_coef = gini(metrics["auc"])

        print(f"\n── Model Performance ─────────────────────")
        print(f"  AUC:   {metrics['auc']:.4f}")
        print(f"  KS:    {metrics['ks']:.4f}")
        print(f"  Gini:  {gini_coef:.4f}")

        mlflow.log_metric("gini", gini_coef)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("approach", "scorecard_woe")
        mlflow.log_params(params["logistic"])
        mlflow.log_params(params["data"])

        mlflow.set_tag("project", "credit_risk")
        mlflow.set_tag("model_family", "scorecard")
        mlflow.set_tag("stage", "training")

        # Log the fitted model object to MLflow model registry for versioning
        mlflow.sklearn.log_model(
            lr_model,
            "scorecard_model",
            registered_model_name="credit_risk_scorecard",
        )

        mlflow.log_artifact(lr_path)
        mlflow.log_artifact(bins_path)
        mlflow.log_artifact(scorecard_path)
        mlflow.log_artifact(feat_cols_path)

    print(f"\n  Feature columns ({len(feature_cols)}): {feature_cols}")
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


# ══════════════════════════════════════════════════════════════════
# XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════════

def train_xgb(df_model: pd.DataFrame):
    """
    Train the XGBoost challenger model on the already-processed feature set.

    Expects df_model to have already passed through:
        load_and_clean → engineer_features → select_features

    Design decisions:
        - XGBoost is wrapped in a sklearn Pipeline with OrdinalEncoder so the single
          pipeline object handles both preprocessing and inference at serve time.
          This avoids a separate "fit encoder, save encoder, load encoder" workflow.
        - scale_pos_weight=3.52 corrects for the ~22% default rate; the value equals
          (n_non_default / n_default) computed from the training distribution.
        - 5-fold CV is run after training (not used for hyperparameter tuning here)
          to verify the held-out AUC is stable and the model isn't overfitting to
          the specific train/test partition.
        - Feature importance is printed at the end to sanity-check that the model
          is relying on the same signals the business understands as drivers.
    """
    X = df_model.drop(columns=["DEFAULT_NEXT_MONTH"])
    y = df_model["DEFAULT_NEXT_MONTH"].astype(int)

    # EDUCATION is the only categorical feature; all others are already numeric
    categorical_cols = ["EDUCATION"]
    numerical_cols   = [c for c in X.columns if c not in categorical_cols]

    print(f"Total features: {X.shape[1]}")
    print(f"Numerical: {len(numerical_cols)} | Categorical: {len(categorical_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # OrdinalEncoder inside ColumnTransformer handles unseen categories gracefully
    # (unknown_value=-1) — important for inference on edge-case EDUCATION values
    xgb_preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ])

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", xgb_preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=3.52,                        # corrects class imbalance (~78% non-default)
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

    print("\nTraining XGBoost...")
    xgb_pipeline.fit(X_train, y_train)

    xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    # 5-fold CV on training set — confirms the test AUC isn't a lucky split
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

    # Feature importance — XGBoost uses gain-based importance (not permutation importance)
    # The top features should mirror what domain experts expect: delay behaviour > billing > amounts
    print("\n── Feature Importance ────────────────────")
    feature_names = numerical_cols + categorical_cols  # must match ColumnTransformer column order

    feat_imp = pd.Series(
        xgb_pipeline.named_steps["model"].feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("\nTop 10 Features:")
    print(feat_imp.head(10).to_string())
    print("\nBottom 5 Features:")
    print(feat_imp.tail(5).to_string())

    # ── Save artifacts
    os.makedirs(ARTIFACTS_MODELS, exist_ok=True)
    os.makedirs(ARTIFACTS_PREP, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_MODELS, "xgb_pipeline.joblib")
    joblib.dump(xgb_pipeline, model_path)
    print(f"  Saved: {model_path}")

    # feature_names order must match the ColumnTransformer column order exactly;
    # the inference pipeline (prepare_xgb_input) selects columns in this order
    feature_cols_path = os.path.join(ARTIFACTS_PREP, "feature_columns_xgb.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Saved: {feature_cols_path}")

    with mlflow.start_run(run_name="xgboost_model"):
        metrics = evaluate_model(y_test, xgb_proba, model_name="xgboost")

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(params["xgboost"])
        mlflow.log_metric("cv_mean_auc", cv_scores.mean())
        mlflow.log_metric("cv_std_auc",  cv_scores.std())

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


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Run the full training pipeline: scorecard first, then XGBoost.

    Steps 1–3 (load, engineer, select) run twice — once inside train_scorecard() which
    manages its own data internally, and once here for the XGBoost branch.
    The duplication is intentional: both models need the same base data but train_scorecard()
    applies additional scorecard-specific filtering (NEGATIVE_COEF_FEATURES, CORRELATED_FEATURES)
    that should not affect the XGBoost feature set.
    """
    print("Starting full training pipeline...\n")

    # ── Shared preprocessing for XGBoost (scorecard does its own internally)
    df       = load_and_clean(DATA_PATH)
    df       = engineer_features(df)
    df_model = select_features(df)

    train_scorecard()
    train_xgb(df_model)

    # ── Print MLflow run summary so CI logs show which runs completed
    client     = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("credit_risk_model")
    print("Experiment ID:", experiment.experiment_id)

    runs = client.search_runs(experiment.experiment_id)
    for r in runs:
        print(f"Run: {r.info.run_name} | AUC: {r.data.metrics.get('auc')}")


if __name__ == "__main__":
    main()