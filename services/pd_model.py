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
        Output → credit score (int) + default probability (float) + risk band

    XGBoost (challenger):
        Input  → feature engineering → XGBoost Pipeline (handles encoding internally)
        Output → default probability (float) + risk band
        Note   → no credit score; XGBoost is not interpretable as a points table
"""

import joblib
import scorecardpy as sc
import pandas as pd
import logging

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
)

logger = logging.getLogger(__name__)


# ── Startup artifact loading — happens once when the module is first imported.
# Wrapped in try/except so a missing or corrupt file produces a readable error
# rather than an AttributeError later when a route tries to call predict.
logger.info("Loading models and artifacts...")

try:
    _xgb_pipeline        = joblib.load(XGB_MODEL_PATH)           # sklearn Pipeline (OrdinalEncoder + XGBClassifier)
    _scorecard_lr_model  = joblib.load(SCORECARD_LR_MODEL_PATH)  # fitted LogisticRegression on WOE features
    _woe_bins            = load_woe_bins()                        # scorecardpy bin definitions for WOE transform
    _scorecard           = joblib.load(SCORECARD_PATH)            # points table: maps WOE bins to score points
    _scorecard_feat_cols = load_scorecard_feature_columns()       # ordered list of _woe column names for LR
    _xgb_feat_cols       = load_xgb_feature_columns()            # ordered list of raw feature names for XGBoost

    logger.info("All models and artifacts loaded successfully.")

except Exception as e:
    logger.exception("Failed to load models/artifacts")
    raise  # re-raise so FastAPI startup fails visibly rather than running in a broken state


def get_risk_band(score: float) -> tuple:
    """
    Map a numeric credit score to a (risk_level, decision) tuple using SCORE_BANDS config.

    Iterates bands in order; the first matching [low, high) interval wins.
    Returns ("Unknown", "Review") if the score falls outside all defined bands —
    this should never happen in production but guards against misconfigured bands.
    """
    for low, high, risk, decision in SCORE_BANDS:
        if low <= score < high:
            return risk, decision

    logger.warning(f"Score {score} did not match any band")
    return "Unknown", "Review"


def scorecard_predict(raw_input: dict) -> dict:
    """
    Run the full champion model pipeline and return credit score + default probability.

    Steps:
        1. WOE-transform engineered features using frozen training bins
        2. LR model outputs P(default) from the WOE feature matrix
        3. scorecardpy converts the same engineered data into additive score points
        4. Score is mapped to a risk band and lending decision via SCORE_BANDS

    raw_input must be the 21-column dict produced by input_mapper or ClientInput.dict().
    """
    logger.info("Scorecard prediction started")

    try:
        # Step 1: WOE transform produces the feature matrix the LR model was trained on
        X_woe = prepare_scorecard_input(
            raw_input, _woe_bins, _scorecard_feat_cols
        )
        logger.debug("WOE transformation completed")

        # Step 2: LR model probability — [:, 1] selects the default class column
        default_prob = float(
            _scorecard_lr_model.predict_proba(X_woe)[:, 1][0]
        )

        # Step 3: Credit score — scorecard_ply needs the RAW engineered features
        # (not WOE), because it maps each feature's raw bin to its pre-computed points
        from core.preprocessing import engineer_features
        df_engineered = engineer_features(raw_input)

        scores_df = sc.scorecard_ply(
            df_engineered,
            _scorecard,
            print_step=0  # suppress scorecardpy's verbose stdout output
        )
        credit_score = int(scores_df["score"].values[0])

        # Step 4: Translate numeric score to business decision
        risk_level, decision = get_risk_band(credit_score)

        logger.info(
            f"Scorecard prediction completed | "
            f"Score={credit_score}, PD={default_prob:.4f}, "
            f"Risk={risk_level}, Decision={decision}"
        )

        return {
            "model":               "Logistic Regression + WOE Scorecard",
            "credit_score":        credit_score,
            "default_probability": round(default_prob, 4),
            "risk_level":          risk_level,
            "decision":            decision,
        }

    except Exception as e:
        logger.exception("Error during scorecard prediction")
        raise  # let the route handler catch and return HTTP 500


def xgb_predict(raw_input: dict) -> dict:
    """
    Run the challenger XGBoost pipeline and return default probability.

    Steps:
        1. Feature engineering (same as scorecard step 3 above)
        2. XGBoost Pipeline handles OrdinalEncoding and prediction internally
        3. Probability is mapped to a risk band using hardcoded probability thresholds
           (XGBoost has no scorecard table, so score-band mapping isn't applicable)

    raw_input must be the 21-column dict produced by input_mapper or ClientInput.dict().
    """
    logger.info("XGBoost prediction started")

    try:
        X = prepare_xgb_input(raw_input, _xgb_feat_cols)
        logger.debug("XGBoost input prepared")

        # Pipeline.predict_proba passes X through OrdinalEncoder then XGBClassifier
        default_prob = float(
            _xgb_pipeline.predict_proba(X)[:, 1][0]
        )

        # XGBoost risk bands are probability-based, not score-based.
        # Thresholds were set during model validation by reviewing the P(default)
        # distribution across known good and bad accounts.
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