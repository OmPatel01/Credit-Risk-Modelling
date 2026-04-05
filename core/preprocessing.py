"""
preprocessing.py
----------------
Handles all data transformation before prediction.

This module is responsible for:
1. Loading WOE bins from artifacts
2. Applying WOE transformation to new input data
3. Validating that required features are present

The WOE bins were fitted on the training set in the notebook.
They must be loaded here — never recomputed on new data.
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


def load_woe_bins():
    """Load WOE bins from artifact file."""
    return joblib.load(WOE_BINS_PATH)


def load_scorecard_feature_columns():
    """Load the exact feature columns used in the scorecard model."""
    with open(SCORECARD_FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def load_xgb_feature_columns():
    """Load the exact feature columns used in the XGBoost model."""
    with open(XGB_FEATURE_COLS_PATH, "r") as f:
        return json.load(f)


def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Apply feature engineering to raw input — same logic as the notebook.

    Parameters
    ----------
    raw_input : dict
        Raw input fields from the API request.
        Must contain the original dataset features.

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with all engineered features.
    """
    df = pd.DataFrame([raw_input])

    pay_cols      = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols     = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                     "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols  = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                     "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    # ── Repayment behaviour features ─────────────────────────────
    df["MAX_DELAY"]        = df[pay_cols].max(axis=1)
    df["NUM_DELAYS"]       = (df[pay_cols] >= 1).sum(axis=1)
    df["ANY_DELAY_FLAG"]   = (df[pay_cols] >= 1).any(axis=1).astype(int)
    df["PAST_DELAY_AVG"]   = df[["PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].mean(axis=1)

    # ── Billing features ──────────────────────────────────────────
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["BILL_GROWTH"]  = df["BILL_AMT1"] - df["BILL_AMT6"]

    # ── Payment features ──────────────────────────────────────────
    df["NUM_ZERO_PAYMENTS"] = (df[pay_amt_cols] == 0).sum(axis=1)

    # ── Ratio features ────────────────────────────────────────────
    for i in range(1, 7):
        df[f"PAY_BILL_RATIO_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0
        )
        # Clip extreme outliers — same as notebook
        p99 = 15.0
        df[f"PAY_BILL_RATIO_{i}"] = df[f"PAY_BILL_RATIO_{i}"].clip(upper=p99)

    ratio_cols = [f"PAY_BILL_RATIO_{i}" for i in range(1, 7)]
    df["AVG_PAY_BILL_RATIO"] = df[ratio_cols].mean(axis=1)

    # ── Exposure features ─────────────────────────────────────────
    df["UTILIZATION"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, 1)
    df["UTILIZATION"] = df["UTILIZATION"].clip(upper=1.05)

    return df


def apply_woe_transform(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    """
    Apply WOE transformation using pre-fitted bins.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe (output of engineer_features).
    bins : dict
        WOE bins loaded from artifacts.

    Returns
    -------
    pd.DataFrame
        WOE-transformed dataframe ready for scorecard LR model.
    """
    # scorecardpy expects a dummy target column for woebin_ply
    # we add a placeholder — it is not used during transform
    df["DEFAULT_NEXT_MONTH"] = 0
    woe_df = sc.woebin_ply(df, bins)
    woe_df = woe_df.drop(columns=["DEFAULT_NEXT_MONTH"], errors="ignore")
    return woe_df


# def prepare_scorecard_input(raw_input: dict, bins: dict,
#                              feature_cols: list) -> pd.DataFrame:
#     """
#     Full preprocessing pipeline for the scorecard model.

#     Steps:
#     1. Engineer features
#     2. Apply WOE transformation
#     3. Select and order scorecard feature columns

#     Returns
#     -------
#     pd.DataFrame
#         Single-row WOE-transformed dataframe with scorecard features only.
#     """
#     df_engineered = engineer_features(raw_input)
#     df_woe        = apply_woe_transform(df_engineered.copy(), bins)

#     # Keep only scorecard features in correct order
#     # woe_cols = [f"{c}_woe" for c in feature_cols]
#     available = [c for c in woe_cols if c in df_woe.columns]
#     return df_woe[available]

def prepare_scorecard_input(raw_input: dict, bins: dict,
                             feature_cols: list) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the scorecard model.

    feature_cols from JSON already have _woe suffix.
    Do NOT add _woe again — use them directly as column names.
    """
    df_engineered = engineer_features(raw_input)
    df_woe        = apply_woe_transform(df_engineered.copy(), bins)

    # feature_cols already contains _woe suffix (e.g. 'PAY_0_woe')
    # so use them directly — do NOT do f"{c}_woe"
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

    Steps:
    1. Engineer features
    2. Select XGBoost feature columns (no WOE needed)

    Returns
    -------
    pd.DataFrame
        Single-row dataframe ready for the XGBoost pipeline.
    """
    df_engineered = engineer_features(raw_input)
    available = [c for c in feature_cols if c in df_engineered.columns]
    return df_engineered[available]
