"""
services/explain_service.py
----------------------------
Computes a full feature-level explanation of a scorecard prediction.

How scorecard explainability works:
    A WOE logistic regression decomposes into:
        log-odds(default) = intercept + Σ (coefficient_i × WOE_i)

    This means each feature's contribution to the final risk verdict is:
        contribution_i = coefficient_i × WOE_i

    A positive contribution pushes toward default (increases risk).
    A negative contribution pushes toward non-default (decreases risk).

    In parallel, the scorecard points table assigns integer points to each
    feature bin. Higher score = lower risk, so a feature in a risky bin
    contributes fewer points (or negative points relative to the base).

This service orchestrates:
    1. Running scorecard_predict() to get all raw prediction data
    2. Applying policy rules via policy_engine
    3. Structuring the per-feature data into ExplainResponse objects
    4. Sorting features by absolute contribution (most impactful first)
"""

import logging
from typing import Dict, Any, List

from core.input_mapper import map_business_to_raw
from core.preprocessing import engineer_features
from services.pd_model import scorecard_predict
from services.policy_engine import apply_policy_rules

logger = logging.getLogger(__name__)

# Human-readable labels for raw feature names.
# This mapping is used to make explanation output comprehensible to non-technical users.
FEATURE_LABELS: Dict[str, str] = {
    "PAY_0":             "Most Recent Payment Status",
    "MAX_DELAY":         "Worst Payment Delay (months)",
    "PAST_DELAY_AVG":    "Average Past Payment Delay",
    "NUM_DELAYS":        "Number of Delayed Months",
    "NUM_ZERO_PAYMENTS": "Months with Zero Payment",
    "AVG_PAY_BILL_RATIO":"Avg Payment-to-Bill Ratio",
    "PAY_AMT1":          "Most Recent Payment Amount (NT$)",
    "UTILIZATION":       "Credit Utilisation",
    "LIMIT_BAL":         "Credit Limit (NT$)",
    "BILL_GROWTH":       "Bill Growth (Recent − Oldest, NT$)",
    "AGE":               "Borrower Age",
    "EDUCATION":         "Education Level",
    "AVG_BILL_AMT":      "Average Monthly Bill (NT$)",
}


def _direction(contribution: float) -> str:
    """Classify a contribution as risk-increasing, risk-decreasing, or neutral."""
    if contribution > 0.001:
        return "increases_risk"
    elif contribution < -0.001:
        return "decreases_risk"
    return "neutral"


def explain_prediction(business_input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a full feature-level explanation for a scorecard prediction.

    Parameters
    ----------
    business_input_dict : dict
        A BusinessInput payload as a plain dict (call .dict() on the Pydantic model
        before passing here).

    Returns
    -------
    dict matching the ExplainResponse schema:
        credit_score, default_probability, risk_level, decision,
        base_points, feature_explanations (list), top_risk_drivers (list)

    The policy engine is also applied so the returned decision reflects
    any hard business rule overrides.
    """
    logger.info("[EXPLAIN] Explanation request started")

    # Step 1: Map business input → raw 21-column dict
    raw_input = map_business_to_raw(business_input_dict)

    # Step 2: Run full scorecard prediction (includes feature_contributions,
    # point_contributions, top_risk_drivers from pd_model.py)
    pred = scorecard_predict(raw_input)

    credit_score        = pred["credit_score"]
    default_probability = pred["default_probability"]
    risk_level          = pred["risk_level"]
    model_decision      = pred["decision"]
    feature_contribs    = pred.get("feature_contributions", {})   # coef × WOE per feature
    point_contribs      = pred.get("point_contributions", {})     # integer points per feature
    top_risk_drivers    = pred.get("top_risk_drivers", [])

    # Step 3: Apply policy rules using engineered features
    df_engineered  = engineer_features(raw_input)
    features_dict  = df_engineered.iloc[0].to_dict()
    policy_result  = apply_policy_rules(model_decision, features_dict)
    final_decision = policy_result["final_decision"]

    # Step 4: Retrieve base points (scorecard intercept)
    base_points = int(point_contribs.get("__basepoints__", 0))

    # Step 5: Build per-feature explanation list
    # We iterate over feature_contribs which covers all scorecard features
    feature_explanations = []

    # We need WOE values and coefficients per feature.
    # scorecard_predict stores contribution = coef × woe.
    # We can back out the approximate WOE from the raw df_engineered via bins,
    # but the cleanest approach is to use the contributions directly and note
    # that we don't store woe/coef separately outside pd_model.
    # To keep this service self-contained we call prepare_scorecard_input
    # to get X_woe, then pull coefficients from the loaded LR model.
    from services.pd_model import (
        _woe_bins,
        _scorecard_feat_cols,
        _scorecard_lr_model,
    )
    from core.preprocessing import prepare_scorecard_input

    X_woe = prepare_scorecard_input(raw_input, _woe_bins, _scorecard_feat_cols)
    coefs = _scorecard_lr_model.coef_[0]

    # Build a lookup: raw_feature_name → (woe_value, coefficient)
    woe_coef_lookup: Dict[str, tuple] = {}
    for col, coef in zip(X_woe.columns, coefs):
        raw_name = col.replace("_woe", "")
        woe_val  = float(X_woe[col].iloc[0])
        woe_coef_lookup[raw_name] = (woe_val, float(coef))

    for feat_name, contribution in feature_contribs.items():
        woe_val, coef_val = woe_coef_lookup.get(feat_name, (0.0, 0.0))
        score_pts = int(point_contribs.get(feat_name, 0))

        feature_explanations.append({
            "feature":        feat_name,
            "label":          FEATURE_LABELS.get(feat_name, feat_name),
            "woe_value":      round(woe_val, 6),
            "coefficient":    round(coef_val, 6),
            "contribution":   round(contribution, 6),
            "score_points":   score_pts,
            "risk_direction": _direction(contribution),
        })

    # Sort descending by absolute contribution so most impactful features appear first
    feature_explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    logger.info(
        f"[EXPLAIN] Completed | Score={credit_score}, "
        f"Decision={final_decision}, Features={len(feature_explanations)}"
    )

    return {
        "credit_score":        credit_score,
        "default_probability": default_probability,
        "risk_level":          risk_level,
        "decision":            final_decision,
        "policy_overridden":   policy_result["policy_overridden"],
        "hard_rule_triggered": policy_result["hard_rule_triggered"],
        "soft_flags":          policy_result["soft_flags"],
        "base_points":         base_points,
        "feature_explanations": feature_explanations,
        "top_risk_drivers":    top_risk_drivers,
    }