"""
services/policy_engine.py
--------------------------
Hard business rule overrides applied on top of the model's score-based decision.

Why this layer exists:
    A statistical model optimises for average-case accuracy. It cannot encode
    absolute business rules such as "always reject a borrower who is currently
    3+ months overdue regardless of their historical score". These non-negotiable
    constraints live here, evaluated AFTER the model produces a decision, and can
    override it in either direction (force Decline or force Approve).

Design principles:
    - Rules are pure functions of the raw engineered input dict — no model calls.
    - Each rule returns a (triggered: bool, reason: str) tuple.
    - apply_policy_rules() evaluates all rules and returns the final overridden
      decision along with which rule fired (if any).
    - Rules are evaluated in priority order; the first triggered hard rule wins.
    - "Soft" rules (advisory only) are flagged but do not change the decision.

Rule catalogue (extend by adding to HARD_RULES or SOFT_RULES lists):
    Hard rules (override decision):
        R001 — Reject if PAY_0 >= 3 (currently 3+ months overdue)
        R002 — Reject if MAX_DELAY >= 6 (6+ months overdue in any recent month)
        R003 — Reject if NUM_ZERO_PAYMENTS >= 5 (chronic non-payer)
        R004 — Reject if UTILIZATION > 0.95 (severely over-extended)

    Soft rules (flag only, no override):
        S001 — Flag if PAY_0 == 2 (2 months overdue — borderline)
        S002 — Flag if UTILIZATION > 0.80 (high utilisation — watch)
        S003 — Flag if NUM_DELAYS >= 4 (frequent delays — watch)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Type alias for a rule result
RuleResult = Tuple[bool, str]   # (triggered, human-readable reason)


# ─────────────────────────────────────────────────────────────────────────────
# Individual rule functions
# Each accepts the engineered feature dict and returns (triggered, reason).
# Use raw input field names exactly as they appear after engineer_features().
# ─────────────────────────────────────────────────────────────────────────────

def _rule_r001_recent_severe_overdue(features: Dict[str, Any]) -> RuleResult:
    """R001: Reject if the most recent payment is 3+ months overdue."""
    pay_0 = features.get("PAY_0", 0)
    if pay_0 >= 3:
        return True, (
            f"R001: Automatic decline — borrower is currently {pay_0} months overdue "
            "(PAY_0 >= 3). Hard policy threshold."
        )
    return False, ""


def _rule_r002_historical_severe_overdue(features: Dict[str, Any]) -> RuleResult:
    """R002: Reject if any historical month shows 6+ months overdue."""
    max_delay = features.get("MAX_DELAY", 0)
    if max_delay >= 6:
        return True, (
            f"R002: Automatic decline — borrower had a {max_delay}-month delay "
            "in recent history (MAX_DELAY >= 6). Hard policy threshold."
        )
    return False, ""


def _rule_r003_chronic_non_payer(features: Dict[str, Any]) -> RuleResult:
    """R003: Reject if the borrower made zero payment in 5 or more of the last 6 months."""
    num_zero = features.get("NUM_ZERO_PAYMENTS", 0)
    if num_zero >= 5:
        return True, (
            f"R003: Automatic decline — borrower made zero payment in {num_zero} "
            "of the last 6 months (NUM_ZERO_PAYMENTS >= 5). Hard policy threshold."
        )
    return False, ""


def _rule_r004_extreme_utilisation(features: Dict[str, Any]) -> RuleResult:
    """R004: Reject if credit utilisation exceeds 95% (severely over-extended)."""
    utilisation = features.get("UTILIZATION", 0.0)
    if utilisation > 0.95:
        return True, (
            f"R004: Automatic decline — credit utilisation is {utilisation:.1%} "
            "(UTILIZATION > 0.95). Borrower is severely over-extended."
        )
    return False, ""


def _soft_s001_borderline_overdue(features: Dict[str, Any]) -> RuleResult:
    """S001: Flag if most recent payment is exactly 2 months overdue."""
    pay_0 = features.get("PAY_0", 0)
    if pay_0 == 2:
        return True, (
            "S001: Advisory flag — borrower is 2 months overdue on most recent payment. "
            "Consider manual review before approving."
        )
    return False, ""


def _soft_s002_high_utilisation(features: Dict[str, Any]) -> RuleResult:
    """S002: Flag if credit utilisation exceeds 80%."""
    utilisation = features.get("UTILIZATION", 0.0)
    if utilisation > 0.80:
        return True, (
            f"S002: Advisory flag — credit utilisation is {utilisation:.1%}. "
            "Borrower is carrying a heavy balance relative to their limit."
        )
    return False, ""


def _soft_s003_frequent_delays(features: Dict[str, Any]) -> RuleResult:
    """S003: Flag if the borrower had 4 or more delayed months in the window."""
    num_delays = features.get("NUM_DELAYS", 0)
    if num_delays >= 4:
        return True, (
            f"S003: Advisory flag — borrower had delays in {num_delays} of the last "
            "6 months. Pattern suggests chronic repayment difficulty."
        )
    return False, ""


# ── Rule registries — order matters for hard rules (first match wins)
HARD_RULES = [
    _rule_r001_recent_severe_overdue,
    _rule_r002_historical_severe_overdue,
    _rule_r003_chronic_non_payer,
    _rule_r004_extreme_utilisation,
]

SOFT_RULES = [
    _soft_s001_borderline_overdue,
    _soft_s002_high_utilisation,
    _soft_s003_frequent_delays,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_policy_rules(
    model_decision: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate all hard and soft policy rules against the engineered feature dict.

    Parameters
    ----------
    model_decision : str
        The decision produced by the scorecard model ("Approve", "Review", "Decline").
        Hard rules may override this.
    features : Dict[str, Any]
        The engineered feature dict as produced by core.preprocessing.engineer_features().
        Must contain at minimum: PAY_0, MAX_DELAY, NUM_ZERO_PAYMENTS, UTILIZATION, NUM_DELAYS.

    Returns
    -------
    dict with keys:
        final_decision     : str   — post-policy decision (may equal model_decision)
        model_decision     : str   — original model decision (before policy)
        policy_overridden  : bool  — True if a hard rule changed the decision
        hard_rule_triggered: Optional[str] — rule ID + reason if a hard rule fired
        soft_flags         : List[str]     — reasons for any soft rules that fired
    """
    logger.debug(f"[POLICY] Evaluating rules. Model decision: {model_decision}")

    hard_rule_triggered: Optional[str] = None
    final_decision = model_decision
    policy_overridden = False

    # ── Evaluate hard rules — first match overrides and stops evaluation
    for rule_fn in HARD_RULES:
        triggered, reason = rule_fn(features)
        if triggered:
            hard_rule_triggered = reason
            final_decision = "Decline"
            policy_overridden = (model_decision != "Decline")
            logger.info(f"[POLICY] Hard rule fired: {reason}")
            break  # first hard rule wins; no point evaluating further

    # ── Evaluate soft rules — all are checked, none override
    soft_flags: List[str] = []
    for rule_fn in SOFT_RULES:
        triggered, reason = rule_fn(features)
        if triggered:
            soft_flags.append(reason)
            logger.debug(f"[POLICY] Soft flag: {reason}")

    logger.info(
        f"[POLICY] Result → model={model_decision}, final={final_decision}, "
        f"overridden={policy_overridden}, soft_flags={len(soft_flags)}"
    )

    return {
        "final_decision":      final_decision,
        "model_decision":      model_decision,
        "policy_overridden":   policy_overridden,
        "hard_rule_triggered": hard_rule_triggered,
        "soft_flags":          soft_flags,
    }