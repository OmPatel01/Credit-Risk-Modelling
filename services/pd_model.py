"""
services/pd_model.py
--------------------
Inference layer for both the champion (scorecard) and challenger (XGBoost) models.

Design decisions:
    - All artifacts are loaded ONCE at module import time (startup), not per request.
      This avoids ~500ms joblib.load overhead on every API call.
    - Module-level variables (prefixed with _) are treated as private singletons.
    - If any artifact fails to load, the entire app startup fails fast with a clear
      exception rather than silently serving wrong predictions.

Model summary:
    Scorecard (champion):
        Input  → feature engineering → WOE transform → Logistic Regression
        Output → credit score (int) + default probability (float) + risk band + per-feature WOE values + per-feature point contributions

    XGBoost (challenger):
        Input  → feature engineering → XGBoost Pipeline (handles encoding internally)
        Output → default probability (float) + risk band
        Note   → no credit score; XGBoost is not interpretable as a points table
"""

import joblib
import scorecardpy as sc
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

from core.config import (
    XGB_MODEL_PATH,
    SCORECARD_LR_MODEL_PATH,
    WOE_BINS_PATH,
    SCORECARD_PATH,
    SCORECARD_FEATURE_COLS_PATH,
    XGB_FEATURE_COLS_PATH,
    SCORE_BANDS,
)
from core.preprocessing import (
    load_woe_bins,
    load_scorecard_feature_columns,
    load_xgb_feature_columns,
    prepare_scorecard_input,
    prepare_xgb_input,
    engineer_features,
)

logger = logging.getLogger(__name__)

# ── Startup artifact loading — happens once when the module is first imported.
logger.info("Loading models and artifacts...")

try:
    _xgb_pipeline        = joblib.load(XGB_MODEL_PATH)
    _scorecard_lr_model  = joblib.load(SCORECARD_LR_MODEL_PATH)
    _woe_bins            = load_woe_bins()
    _scorecard           = joblib.load(SCORECARD_PATH)
    _scorecard_feat_cols = load_scorecard_feature_columns()
    _xgb_feat_cols       = load_xgb_feature_columns()

    logger.info("All models and artifacts loaded successfully.")

except Exception as e:
    logger.exception("Failed to load models/artifacts")
    raise


def get_risk_band(score: float) -> tuple:
    """
    Map a numeric credit score to a (risk_level, decision) tuple using SCORE_BANDS config.
    Returns ("Unknown", "Review") if the score falls outside all defined bands.
    """
    for low, high, risk, decision in SCORE_BANDS:
        if low <= score < high:
            return risk, decision

    logger.warning(f"Score {score} did not match any band")
    return "Unknown", "Review"


def _compute_feature_contributions(
    X_woe: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute each feature's contribution to the LR log-odds.

    Contribution_i = coefficient_i × WOE_i

    This is the standard decomposition for a WOE logistic regression:
    log-odds = intercept + sum(coef_i × woe_i).
    A positive contribution pushes toward default (higher risk);
    a negative contribution pushes toward non-default (lower risk).

    Returns a dict keyed by the raw feature name (without _woe suffix),
    value is the signed contribution float.
    """
    coefs = _scorecard_lr_model.coef_[0]  # shape: (n_features,)
    contributions = {}

    for col, coef in zip(X_woe.columns, coefs):
        woe_val = float(X_woe[col].iloc[0])
        raw_name = col.replace("_woe", "")
        contributions[raw_name] = round(coef * woe_val, 6)

    return contributions


def _compute_point_contributions(
    df_engineered: pd.DataFrame,
) -> Dict[str, int]:
    """
    Compute per-feature credit score points using the scorecard points table.

    scorecardpy stores the scorecard as a dict of DataFrames, one per feature.
    Each DataFrame has 'bin' and 'points' columns. We find which bin the borrower
    falls into for each feature and return the corresponding integer points.

    Returns a dict keyed by raw feature name, value is integer points.
    """
    point_contributions = {}

    for feature, table in _scorecard.items():
        if feature == "basepoints":
            # The intercept term — store separately so callers can use it
            try:
                point_contributions["__basepoints__"] = int(table["points"].iloc[0])
            except Exception:
                point_contributions["__basepoints__"] = 0
            continue

        if feature not in df_engineered.columns:
            logger.warning(f"Feature '{feature}' in scorecard not found in engineered df")
            continue

        val = df_engineered[feature].iloc[0]

        # Find matching bin — scorecardpy stores bin boundaries as strings like "[0,0.5)"
        # The scorecard_ply function handles this internally; we replicate the lookup here.
        matched_points = None
        for _, row in table.iterrows():
            bin_str = str(row["bin"])
            try:
                pts = int(row["points"])
            except (ValueError, TypeError):
                pts = 0

            # scorecardpy bin strings for numeric features look like "[low,high)"
            # For categorical/integer features they are exact values like "0" or "-2"
            if bin_str.startswith("[") or bin_str.startswith("("):
                # Numeric interval bin
                try:
                    # Strip brackets and split
                    clean = bin_str.strip("[]()").replace("Inf", "inf")
                    lo_str, hi_str = clean.split(",")
                    lo = float(lo_str.strip())
                    hi = float(hi_str.strip())
                    left_closed  = bin_str.startswith("[")
                    right_closed = bin_str.endswith("]")

                    in_left  = (val >= lo) if left_closed  else (val > lo)
                    in_right = (val <= hi) if right_closed else (val < hi)

                    if in_left and in_right:
                        matched_points = pts
                        break
                except Exception:
                    continue
            else:
                # Categorical / exact-match bin (e.g. "missing" or exact integer)
                try:
                    if float(bin_str) == float(val):
                        matched_points = pts
                        break
                except (ValueError, TypeError):
                    if bin_str == str(val):
                        matched_points = pts
                        break

        if matched_points is None:
            logger.warning(
                f"No bin matched for feature '{feature}', value={val}. "
                "Defaulting to 0 points."
            )
            matched_points = 0

        point_contributions[feature] = matched_points

    return point_contributions


def scorecard_predict(raw_input: dict) -> dict:
    """
    Run the full champion model pipeline and return credit score + default probability
    + per-feature WOE contributions + per-feature point contributions.

    Steps:
        1. WOE-transform engineered features using frozen training bins
        2. LR model outputs P(default) from the WOE feature matrix
        3. Compute per-feature log-odds contributions (coef × WOE)
        4. scorecardpy converts the same engineered data into additive score points
        5. Compute per-feature point contributions from the scorecard table
        6. Score is mapped to a risk band and lending decision via SCORE_BANDS

    raw_input must be the 21-column dict produced by input_mapper or ClientInput.dict().
    """
    logger.info("Scorecard prediction started")

    try:
        # Step 1: WOE transform
        X_woe = prepare_scorecard_input(
            raw_input, _woe_bins, _scorecard_feat_cols
        )
        logger.debug("WOE transformation completed")

        # Step 2: LR probability
        default_prob = float(
            _scorecard_lr_model.predict_proba(X_woe)[:, 1][0]
        )

        # Step 3: Per-feature WOE × coefficient contributions
        feature_contributions = _compute_feature_contributions(X_woe)

        # Step 4: Credit score via scorecardpy
        df_engineered = engineer_features(raw_input)

        scores_df = sc.scorecard_ply(
            df_engineered,
            _scorecard,
            print_step=0
        )
        credit_score = int(scores_df["score"].values[0])

        # Step 5: Per-feature point contributions
        point_contributions = _compute_point_contributions(df_engineered)

        # Step 6: Risk band
        risk_level, decision = get_risk_band(credit_score)

        # Build ranked top risk drivers (features with highest positive contribution
        # to default risk, i.e. largest positive coef×WOE values)
        # Exclude the basepoints entry
        ranked = sorted(
            [(k, v) for k, v in feature_contributions.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_risk_drivers = [
            {"feature": feat, "contribution": contrib}
            for feat, contrib in ranked[:3]
            if contrib > 0  # only features actively increasing risk
        ]
        # If fewer than 3 positive drivers, fill from the ranked list regardless of sign
        if len(top_risk_drivers) < 3:
            top_risk_drivers = [
                {"feature": feat, "contribution": contrib}
                for feat, contrib in ranked[:3]
            ]

        logger.info(
            f"Scorecard prediction completed | "
            f"Score={credit_score}, PD={default_prob:.4f}, "
            f"Risk={risk_level}, Decision={decision}"
        )

        return {
            "model":                  "Logistic Regression + WOE Scorecard",
            "credit_score":           credit_score,
            "default_probability":    round(default_prob, 4),
            "risk_level":             risk_level,
            "decision":               decision,
            "feature_contributions":  feature_contributions,   # coef × WOE per feature
            "point_contributions":    point_contributions,     # integer score points per feature
            "top_risk_drivers":       top_risk_drivers,        # ranked top 3 risk factors
        }

    except Exception as e:
        logger.exception("Error during scorecard prediction")
        raise


def xgb_predict(raw_input: dict) -> dict:
    """
    Run the challenger XGBoost pipeline and return default probability.

    raw_input must be the 21-column dict produced by input_mapper or ClientInput.dict().
    """
    logger.info("XGBoost prediction started")

    try:
        X = prepare_xgb_input(raw_input, _xgb_feat_cols)
        logger.debug("XGBoost input prepared")

        default_prob = float(
            _xgb_pipeline.predict_proba(X)[:, 1][0]
        )

        if default_prob >= 0.60:
            risk_level, decision = "High Risk", "Decline"
        elif default_prob >= 0.40:
            risk_level, decision = "Elevated",  "Decline"
        elif default_prob >= 0.25:
            risk_level, decision = "Moderate",  "Review"
        elif default_prob >= 0.12:
            risk_level, decision = "Low",       "Approve"
        else:
            risk_level, decision = "Minimal",   "Approve"

        logger.info(
            f"XGBoost prediction completed | "
            f"PD={default_prob:.4f}, "
            f"Risk={risk_level}, Decision={decision}"
        )

        return {
            "model":               "XGBoost (Challenger)",
            "default_probability": round(default_prob, 4),
            "risk_level":          risk_level,
            "decision":            decision,
        }

    except Exception as e:
        logger.exception("Error during XGBoost prediction")
        raise