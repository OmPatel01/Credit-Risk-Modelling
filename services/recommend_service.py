"""
services/recommend_service.py
------------------------------
Maps the top risk drivers identified by the scorecard explanation into
actionable, borrower-facing improvement recommendations.

Design:
    Each scorecard feature that is "increasing risk" (positive coef×WOE contribution)
    is looked up in RECOMMENDATION_MAP, which contains:
        - A human-readable description of what the current value means
        - A specific action the borrower can take
        - An expected qualitative impact tier (High / Medium / Low)

    Recommendations are only generated for features where the model identified
    the borrower's current bin as risky (positive contribution). Features with
    neutral or negative contributions are not actioned — there's nothing to improve.

    Priority is assigned by contribution magnitude: the feature contributing most
    to default risk gets priority 1, and so on.

    The service also provides the raw engineered feature values alongside the
    recommendations so the frontend can display "current value → target value" pairs.
"""

import logging
from typing import Dict, Any, List, Optional

from core.input_mapper import map_business_to_raw
from core.preprocessing import engineer_features
from services.pd_model import scorecard_predict
from services.policy_engine import apply_policy_rules

logger = logging.getLogger(__name__)


# ── Recommendation map ────────────────────────────────────────────────────────
#
# Key: raw feature name (as stored in feature_contributions dict)
# Value: dict with keys:
#   label          — plain-English feature name shown to the borrower
#   action         — specific thing the borrower should do
#   impact         — High / Medium / Low (qualitative estimate of score improvement)
#   detail_fn      — optional callable(value) → str that provides a value-specific message
#                    If None, a generic action string is used.
#
# All action strings should be written in second-person imperative form
# so they read naturally in a borrower-facing UI.
# ─────────────────────────────────────────────────────────────────────────────

def _pay0_detail(val: float) -> str:
    months = int(val)
    if months >= 3:
        return (
            f"Your most recent payment is {months} months overdue. "
            "Bring this account current immediately — this single factor has the "
            "largest impact on your credit score."
        )
    elif months == 2:
        return (
            "Your most recent payment is 2 months overdue. "
            "Make the overdue payment now to prevent the delay from worsening."
        )
    elif months == 1:
        return (
            "Your most recent payment is 1 month overdue. "
            "Clear the overdue balance as soon as possible."
        )
    return "Ensure your next payment is made on time."


def _max_delay_detail(val: float) -> str:
    months = int(val)
    return (
        f"Your worst recorded delay was {months} months. "
        "Consistent on-time payments over the next 6–12 months will gradually "
        "improve this indicator."
    )


def _utilisation_detail(val: float) -> str:
    pct = val * 100
    return (
        f"Your credit utilisation is {pct:.0f}% of your limit. "
        "Aim to keep utilisation below 50% — ideally below 30%. "
        "You can do this by paying down the outstanding balance or requesting "
        "a credit limit increase (only if you do not increase spending)."
    )


def _zero_payments_detail(val: float) -> str:
    months = int(val)
    return (
        f"You made zero payment in {months} of the last 6 months. "
        "Even the minimum payment each month keeps your account active and "
        "prevents the zero-payment flag from accumulating further."
    )


def _past_delay_avg_detail(val: float) -> str:
    return (
        f"Your average delay over recent months is {val:.1f} months. "
        "Set up automatic minimum payments to eliminate avoidable late payments."
    )


def _pay_bill_ratio_detail(val: float) -> str:
    pct = val * 100
    return (
        f"You are paying back approximately {pct:.0f}% of your statement balance each month. "
        "Try to increase your monthly payment — paying more than the minimum "
        "reduces your outstanding balance faster and signals lower risk."
    )


def _bill_growth_detail(val: float) -> str:
    if val > 0:
        return (
            f"Your outstanding balance has grown by NT${val:,.0f} over the past 6 months. "
            "Avoid new charges and focus on reducing the balance."
        )
    return "Your balance trend is stable or improving. Maintain this pattern."


RECOMMENDATION_MAP: Dict[str, Dict[str, Any]] = {
    "PAY_0": {
        "label":     "Most Recent Payment Status",
        "action":    "Bring your most recent payment up to date immediately.",
        "impact":    "High",
        "detail_fn": _pay0_detail,
    },
    "MAX_DELAY": {
        "label":     "Worst Payment Delay",
        "action":    "Build a consistent on-time payment record over the next 6–12 months.",
        "impact":    "High",
        "detail_fn": _max_delay_detail,
    },
    "UTILIZATION": {
        "label":     "Credit Utilisation",
        "action":    "Reduce your outstanding balance to below 50% of your credit limit.",
        "impact":    "High",
        "detail_fn": _utilisation_detail,
    },
    "NUM_ZERO_PAYMENTS": {
        "label":     "Months with Zero Payment",
        "action":    "Make at least the minimum payment every month without exception.",
        "impact":    "High",
        "detail_fn": _zero_payments_detail,
    },
    "PAST_DELAY_AVG": {
        "label":     "Average Past Payment Delay",
        "action":    "Set up automatic payments to eliminate late payments going forward.",
        "impact":    "Medium",
        "detail_fn": _past_delay_avg_detail,
    },
    "AVG_PAY_BILL_RATIO": {
        "label":     "Average Payment-to-Bill Ratio",
        "action":    "Increase your monthly payment amount above the minimum required.",
        "impact":    "Medium",
        "detail_fn": _pay_bill_ratio_detail,
    },
    "BILL_GROWTH": {
        "label":     "Bill Growth Trend",
        "action":    "Avoid new charges and focus on reducing your outstanding balance.",
        "impact":    "Medium",
        "detail_fn": _bill_growth_detail,
    },
    "NUM_DELAYS": {
        "label":     "Number of Delayed Months",
        "action":    "Aim for zero late payments over the next 6 months.",
        "impact":    "Medium",
        "detail_fn": lambda v: (
            f"You had delays in {int(v)} of the last 6 months. "
            "Eliminating further delays will progressively improve this figure."
        ),
    },
    "PAY_AMT1": {
        "label":     "Most Recent Payment Amount",
        "action":    "Increase the amount you pay each month — even small increases help.",
        "impact":    "Low",
        "detail_fn": lambda v: (
            f"Your most recent payment was NT${v:,.0f}. "
            "Paying a larger amount reduces your balance faster and improves your ratio."
        ),
    },
    "LIMIT_BAL": {
        "label":     "Credit Limit",
        "action":    "If eligible, request a credit limit increase without increasing spending.",
        "impact":    "Low",
        "detail_fn": lambda v: (
            f"Your current credit limit is NT${v:,.0f}. "
            "A higher limit (without additional spending) lowers your utilisation ratio."
        ),
    },
    "AGE": {
        "label":     "Borrower Age",
        "action":    "Age is not directly actionable; focus on improving payment behaviour.",
        "impact":    "Low",
        "detail_fn": lambda v: (
            "Age contributes modestly to the risk score but cannot be changed. "
            "Focus on actionable factors like payment history and utilisation."
        ),
    },
}


def _format_current_value(feature: str, value: float) -> str:
    """Return a human-readable string for the borrower's current value of a feature."""
    if feature == "PAY_0":
        if value <= 0:
            return "Paid on time / revolving"
        return f"{int(value)} month(s) overdue"
    elif feature == "UTILIZATION":
        return f"{value * 100:.0f}% of credit limit"
    elif feature in ("LIMIT_BAL", "PAY_AMT1", "BILL_GROWTH", "AVG_BILL_AMT"):
        return f"NT${value:,.0f}"
    elif feature == "AVG_PAY_BILL_RATIO":
        return f"{value * 100:.0f}% of bill paid each month"
    elif feature == "MAX_DELAY":
        return f"{int(value)} month(s) worst delay"
    elif feature == "NUM_DELAYS":
        return f"{int(value)} months with delay"
    elif feature == "NUM_ZERO_PAYMENTS":
        return f"{int(value)} months with no payment"
    elif feature == "PAST_DELAY_AVG":
        return f"{value:.1f} months average delay"
    return str(round(value, 3))


def generate_recommendations(business_input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate prioritised, actionable recommendations for a borrower.

    Parameters
    ----------
    business_input_dict : dict
        A BusinessInput payload as a plain dict.

    Returns
    -------
    dict matching the RecommendResponse schema:
        credit_score, default_probability, risk_level, decision,
        recommendations (list of Recommendation dicts, sorted by priority)
    """
    logger.info("[RECOMMEND] Recommendation request started")

    # Step 1: Map and predict
    raw_input = map_business_to_raw(business_input_dict)
    pred      = scorecard_predict(raw_input)

    credit_score        = pred["credit_score"]
    default_probability = pred["default_probability"]
    risk_level          = pred["risk_level"]
    model_decision      = pred["decision"]
    feature_contribs    = pred.get("feature_contributions", {})

    # Step 2: Apply policy
    df_engineered = engineer_features(raw_input)
    features_dict = df_engineered.iloc[0].to_dict()
    policy_result = apply_policy_rules(model_decision, features_dict)
    final_decision = policy_result["final_decision"]

    # Step 3: Filter to risk-increasing features only (positive contribution)
    risky_features = sorted(
        [(feat, contrib) for feat, contrib in feature_contribs.items() if contrib > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    # Step 4: Build recommendations
    recommendations: List[Dict[str, Any]] = []
    priority = 1

    for feat_name, contribution in risky_features:
        if feat_name not in RECOMMENDATION_MAP:
            # Feature exists in model but has no recommendation entry — skip
            logger.debug(f"[RECOMMEND] No entry in RECOMMENDATION_MAP for '{feat_name}'")
            continue

        rec_template = RECOMMENDATION_MAP[feat_name]
        current_raw  = features_dict.get(feat_name, 0.0)
        current_str  = _format_current_value(feat_name, current_raw)

        # Use the detail_fn for a value-specific message if available
        detail_fn = rec_template.get("detail_fn")
        action_str = detail_fn(current_raw) if detail_fn else rec_template["action"]

        recommendations.append({
            "feature":         feat_name,
            "label":           rec_template["label"],
            "current_value":   current_str,
            "action":          action_str,
            "expected_impact": rec_template["impact"],
            "priority":        priority,
            "contribution":    round(contribution, 6),
        })
        priority += 1

    # Cap at top 5 recommendations to avoid overwhelming the borrower
    recommendations = recommendations[:5]

    logger.info(
        f"[RECOMMEND] Completed | Score={credit_score}, "
        f"Decision={final_decision}, Recommendations={len(recommendations)}"
    )

    return {
        "credit_score":        credit_score,
        "default_probability": default_probability,
        "risk_level":          risk_level,
        "decision":            final_decision,
        "policy_overridden":   policy_result["policy_overridden"],
        "hard_rule_triggered": policy_result["hard_rule_triggered"],
        "recommendations":     recommendations,
    }