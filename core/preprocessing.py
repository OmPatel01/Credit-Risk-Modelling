"""
core/preprocessing.py
---------------------
All data transformation logic that runs between raw API input and model inference.

Responsibilities:
    1. Load frozen training artifacts (WOE bins, feature column lists)
    2. Apply the same feature engineering that was used during model training
    3. Apply WOE transformation required by the scorecard/LR model
    4. Assemble the final feature matrices for each model

IMPORTANT — data leakage guard:
    WOE bins and the scorecard table are NEVER recomputed on inference data.
    They are loaded from disk (fitted on training data only) and applied as
    a fixed lookup. Any change to binning logic must go through mlops/train.py.
"""

import json
import joblib
import pandas as pd
import numpy as np
import scorecardpy as sc

from core.config import (
    WOE_BINS_PATH,
    SCORECARD_FEATURE_COLS_PATH,
    XGB_FEATURE_COLS_PATH,
)


# ── Artifact loaders — thin wrappers so callers don't need to know file formats

def load_woe_bins():
    """Deserialise the scorecardpy bin definitions saved during training."""
    return joblib.load(WOE_BINS_PATH)


def load_scorecard_feature_columns():
    """Return the ordered list of WOE column names expected by the LR model.
    Note: names already carry the '_woe' suffix (e.g. 'PAY_0_woe') — do not append it again."""
    with open(SCORECARD_FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def load_xgb_feature_columns():
    """Return the ordered list of raw engineered feature names expected by the XGBoost pipeline."""
    with open(XGB_FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


# ── Feature engineering — must mirror mlops/train.py:engineer_features() exactly.
# Any divergence between training and inference engineering causes silent model degradation.

def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Derive all model features from the raw 21-column client record.

    The raw input contains the original dataset columns (LIMIT_BAL, AGE,
    PAY_0..PAY_6, BILL_AMT1..6, PAY_AMT1..6, EDUCATION).
    This function adds 15+ engineered columns that capture repayment behaviour,
    billing trends, and utilisation — the same ones built during training.

    Returns a single-row DataFrame ready for either WOE transform (scorecard)
    or direct ingestion (XGBoost pipeline).
    """
    df = pd.DataFrame([raw_input])

    pay_cols      = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols     = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                     "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols  = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                     "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    # ── Repayment behaviour — captures severity, frequency, and trend of delinquency

    # Worst single month across the full 6-month window — strongest default predictor
    df["MAX_DELAY"]        = df[pay_cols].max(axis=1)

    # Count of months where client was actually overdue (PAY_X >= 1 means n months late)
    df["NUM_DELAYS"]       = (df[pay_cols] >= 1).sum(axis=1)

    # Binary flag: 1 if any delinquency occurred in the window, 0 if clean throughout
    df["ANY_DELAY_FLAG"]   = (df[pay_cols] >= 1).any(axis=1).astype(int)

    # Average delay over older months (PAY_2..PAY_6); PAY_0 excluded to avoid
    # double-counting the most recent month which is already in MAX_DELAY
    df["PAST_DELAY_AVG"]   = df[["PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)

    # ── Billing features — tracks outstanding balance level and direction

    # Mean bill across 6 months; replaces 6 highly collinear raw BILL_AMT columns (r > 0.90)
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)

    # Positive = balance growing (client accumulating debt); negative = shrinking
    df["BILL_GROWTH"]  = df["BILL_AMT1"] - df["BILL_AMT6"]

    # ── Payment features — how many months did the client make zero payment?
    # NUM_ZERO_PAYMENTS is a strong default signal: chronic non-payers default far more often
    df["NUM_ZERO_PAYMENTS"] = (df[pay_amt_cols] == 0).sum(axis=1)

    # ── Ratio features — what fraction of each bill was actually repaid?
    # Ratio = 0 when BILL_AMT = 0 (no outstanding balance); ratio is capped at p99
    # of training data (≈15) to prevent extreme outliers from distorting the model
    for i in range(1, 7):
        df[f"PAY_BILL_RATIO_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0
        )
        p99 = 15.0  # hard cap matching training-time clipping; do NOT change without retraining
        df[f"PAY_BILL_RATIO_{i}"] = df[f"PAY_BILL_RATIO_{i}"].clip(upper=p99)

    ratio_cols = [f"PAY_BILL_RATIO_{i}" for i in range(1, 7)]

    # Smoothed repayment coverage across all 6 months — reduces noise from single-month anomalies
    df["AVG_PAY_BILL_RATIO"] = df[ratio_cols].mean(axis=1)

    # ── Exposure features — how much of the credit limit is being used?
    # Capped at 1.05 to handle edge cases where balance slightly exceeds the limit
    df["UTILIZATION"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, 1)
    df["UTILIZATION"] = df["UTILIZATION"].clip(upper=1.05)

    return df


def apply_woe_transform(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    """
    Replace raw feature values with their Weight-of-Evidence scores using frozen training bins.

    scorecardpy's woebin_ply expects a target column to be present even at inference time —
    we inject a dummy zero column that is immediately dropped after transformation.
    The resulting _woe columns are what the LR model was trained on.
    """
    df["DEFAULT_NEXT_MONTH"] = 0   # placeholder — required by scorecardpy API, not used in transform
    woe_df = sc.woebin_ply(df, bins)
    woe_df = woe_df.drop(columns=["DEFAULT_NEXT_MONTH"], errors="ignore")
    return woe_df


def prepare_scorecard_input(raw_input: dict, bins: dict,
                             feature_cols: list) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the scorecard (LR) model.

    Pipeline: raw dict → engineer_features → WOE transform → select & order columns.

    feature_cols comes from feature_columns_scorecard.json and already contains the
    '_woe' suffix — do NOT add '_woe' here or columns will not be found.

    Raises ValueError if none of the expected WOE columns exist in the transformed
    output, which indicates a mismatch between saved bins and the current feature set.
    """
    df_engineered = engineer_features(raw_input)
    df_woe        = apply_woe_transform(df_engineered.copy(), bins)

    # feature_cols already contains _woe suffix (e.g. 'PAY_0_woe')
    available = [c for c in feature_cols if c in df_woe.columns]

    if not available:
        raise ValueError(
            f"No matching WOE columns found.\n"
            f"Expected: {feature_cols}\n"
            f"Got: {df_woe.columns.tolist()}"
        )

    return df_woe[available]


def prepare_xgb_input(raw_input: dict, feature_cols: list) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the XGBoost model.

    XGBoost's Pipeline object handles OrdinalEncoding internally, so only
    feature engineering is needed here — no WOE transformation.
    """
    df_engineered = engineer_features(raw_input)

    # Silently skip any column in feature_cols that didn't survive engineering
    # (shouldn't happen in production, but avoids a hard crash during debugging)
    available = [c for c in feature_cols if c in df_engineered.columns]
    return df_engineered[available]