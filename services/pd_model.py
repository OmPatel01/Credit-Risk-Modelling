"""
pd_model.py
----------
Prediction logic for both models.

Two prediction functions:
1. scorecard_predict  → LR + WOE → credit score + default probability
2. xgb_predict        → XGBoost pipeline → default probability only

Models and artifacts are loaded once at module level.
This avoids reloading on every API request (expensive operation).
"""

# import joblib
# import scorecardpy as sc
# import pandas as pd

# from core.config import (
#     XGB_MODEL_PATH,
#     SCORECARD_LR_MODEL_PATH,
#     WOE_BINS_PATH,
#     SCORECARD_PATH,
#     SCORECARD_FEATURE_COLS_PATH,
#     XGB_FEATURE_COLS_PATH,
#     SCORE_BANDS,
# )
# from core.preprocessing import (
#     load_woe_bins,
#     load_scorecard_feature_columns,
#     load_xgb_feature_columns,
#     prepare_scorecard_input,
#     prepare_xgb_input,
# )

# # ── Load all artifacts once at startup ───────────────────────────
# # These are module-level variables — loaded when the app starts
# # NOT reloaded on every request

# print("Loading models and artifacts...")

# _xgb_pipeline        = joblib.load(XGB_MODEL_PATH)
# _scorecard_lr_model  = joblib.load(SCORECARD_LR_MODEL_PATH)
# _woe_bins            = load_woe_bins()
# _scorecard           = joblib.load(SCORECARD_PATH)
# _scorecard_feat_cols = load_scorecard_feature_columns()
# _xgb_feat_cols       = load_xgb_feature_columns()

# print("All artifacts loaded successfully.")


# def get_risk_band(score: float) -> tuple:
#     """
#     Map a credit score to a risk level and decision.

#     Returns
#     -------
#     tuple : (risk_level, decision)
#     """
#     for low, high, risk, decision in SCORE_BANDS:
#         if low <= score < high:
#             return risk, decision
#     return "Unknown", "Review"


# def scorecard_predict(raw_input: dict) -> dict:
#     """
#     Run the full scorecard prediction pipeline.

#     Flow:
#       raw_input → feature engineering → WOE transform
#       → LR model → default probability
#       → scorecard_ply → credit score
#       → score band → risk level + decision

#     Parameters
#     ----------
#     raw_input : dict
#         Raw feature values from the API request.

#     Returns
#     -------
#     dict
#         credit_score, default_probability, risk_level, decision
#     """
#     # Step 1: Prepare WOE-transformed features for LR model
#     X_woe = prepare_scorecard_input(
#         raw_input, _woe_bins, _scorecard_feat_cols
#     )

#     # Step 2: Get default probability from LR model
#     default_prob = float(
#         _scorecard_lr_model.predict_proba(X_woe)[:, 1][0]
#     )

#     # Step 3: Get credit score from scorecard points table
#     # scorecard_ply needs RAW engineered data (not WOE)
#     from core.preprocessing import engineer_features
#     df_engineered = engineer_features(raw_input)

#     scores_df = sc.scorecard_ply(
#         df_engineered,
#         _scorecard,
#         print_step=0
#     )
#     credit_score = int(scores_df["score"].values[0])

#     # Step 4: Map score to risk band and decision
#     risk_level, decision = get_risk_band(credit_score)

#     return {
#         "model":               "Logistic Regression + WOE Scorecard",
#         "credit_score":        credit_score,
#         "default_probability": round(default_prob, 4),
#         "risk_level":          risk_level,
#         "decision":            decision,
#     }


# def xgb_predict(raw_input: dict) -> dict:
#     """
#     Run the XGBoost prediction pipeline.

#     Flow:
#       raw_input → feature engineering
#       → XGBoost pipeline (handles scaling internally)
#       → default probability

#     Parameters
#     ----------
#     raw_input : dict
#         Raw feature values from the API request.

#     Returns
#     -------
#     dict
#         default_probability, risk_level, decision
#     """
#     # Prepare features for XGBoost
#     X = prepare_xgb_input(raw_input, _xgb_feat_cols)

#     # Get default probability
#     default_prob = float(
#         _xgb_pipeline.predict_proba(X)[:, 1][0]
#     )

#     # Map probability to a simple risk band
#     if default_prob >= 0.60:
#         risk_level, decision = "High Risk", "Decline"
#     elif default_prob >= 0.40:
#         risk_level, decision = "Elevated",  "Decline"
#     elif default_prob >= 0.25:
#         risk_level, decision = "Moderate",  "Review"
#     elif default_prob >= 0.12:
#         risk_level, decision = "Low",       "Approve"
#     else:
#         risk_level, decision = "Minimal",   "Approve"

#     return {
#         "model":               "XGBoost (Challenger)",
#         "default_probability": round(default_prob, 4),
#         "risk_level":          risk_level,
#         "decision":            decision,
#     }

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

# ────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Load Artifacts (Startup Phase)
# ────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────
# Risk Band Mapping
# ────────────────────────────────────────────────────────────────
def get_risk_band(score: float) -> tuple:
    for low, high, risk, decision in SCORE_BANDS:
        if low <= score < high:
            return risk, decision

    logger.warning(f"Score {score} did not match any band")
    return "Unknown", "Review"


# ────────────────────────────────────────────────────────────────
# Scorecard Prediction
# ────────────────────────────────────────────────────────────────
def scorecard_predict(raw_input: dict) -> dict:
    logger.info("Scorecard prediction started")

    try:
        # Step 1: Prepare WOE features
        X_woe = prepare_scorecard_input(
            raw_input, _woe_bins, _scorecard_feat_cols
        )
        logger.debug("WOE transformation completed")

        # Step 2: Predict probability
        default_prob = float(
            _scorecard_lr_model.predict_proba(X_woe)[:, 1][0]
        )

        # Step 3: Compute credit score
        from core.preprocessing import engineer_features
        df_engineered = engineer_features(raw_input)

        scores_df = sc.scorecard_ply(
            df_engineered,
            _scorecard,
            print_step=0
        )
        credit_score = int(scores_df["score"].values[0])

        # Step 4: Risk mapping
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
        raise


# ────────────────────────────────────────────────────────────────
# XGBoost Prediction
# ────────────────────────────────────────────────────────────────
def xgb_predict(raw_input: dict) -> dict:
    logger.info("XGBoost prediction started")

    try:
        # Prepare features
        X = prepare_xgb_input(raw_input, _xgb_feat_cols)
        logger.debug("XGBoost input prepared")

        # Predict probability
        default_prob = float(
            _xgb_pipeline.predict_proba(X)[:, 1][0]
        )

        # Risk band logic
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