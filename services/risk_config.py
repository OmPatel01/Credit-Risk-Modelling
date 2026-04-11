"""
services/risk_config.py
------------------------
Central configuration for all credit risk analytics services.

All tunable risk parameters live here — never hardcode thresholds or scenarios
inside service logic. Changing a scenario or default requires editing only this file.

Guidelines for modifying values:
    STRESS_SCENARIOS    — add/remove scenarios here; keys must be unique strings;
                          the "base" key is required (scenario_service uses it as the anchor).
    SENSITIVITY_*_SHIFTS — list of floats; negative = improvement, positive = deterioration;
                          keep lists short (≤6) to avoid bloated API responses.
    DEFAULT_LGD         — 0.45 reflects typical unsecured consumer credit recovery rates;
                          update if the product mix or collateral assumptions change.
"""

# ── Risk Segmentation ─────────────────────────────────────────────────────────

# Labels used when the caller doesn't supply custom bucket names (A = best, E = worst)
DEFAULT_SEGMENT_LABELS = ["A", "B", "C", "D", "E"]

# Boundaries for fixed-threshold segmentation; creates 5 buckets:
#   A=[0, 0.05)  B=[0.05, 0.15)  C=[0.15, 0.30)  D=[0.30, 0.50)  E=[0.50, 1.0]
DEFAULT_PD_THRESHOLDS = [0.05, 0.15, 0.30, 0.50]

# Default number of quantile buckets when method="quantile"
DEFAULT_NUM_QUANTILES = 5  # quintiles; use 10 for deciles

# ── ECL Defaults ──────────────────────────────────────────────────────────────

# 45% is the industry standard LGD for unsecured retail credit under Basel II;
# override at the request level for specific portfolio segments
DEFAULT_LGD = 0.45

# ── Monte Carlo Defaults ──────────────────────────────────────────────────────

DEFAULT_NUM_SIMULATIONS = 10_000  # balances statistical accuracy (~0.5% VaR error) with response time
DEFAULT_RANDOM_SEED     = 42      # fixed seed for reproducibility in tests; override for fresh runs
DEFAULT_VAR_CONFIDENCE  = 0.95   # 95th percentile is the standard for retail credit VaR

# ── Stress Testing Scenarios ──────────────────────────────────────────────────

# pd_multiplier: scales all borrower PDs (1.0 = no change, 1.8 = 80% increase)
# lgd_override:  replaces the caller-supplied LGD; None = use input LGD unchanged
# The "base" key must exist; scenario_service anchors all % change calculations against it.
STRESS_SCENARIOS = {
    "base": {
        "label":        "Base",
        "pd_multiplier": 1.0,
        "lgd_override":  None,   # keeps whatever LGD the caller supplied
    },
    "mild": {
        "label":        "Mild Stress",
        "pd_multiplier": 1.3,    # PDs increase by 30% — early recession conditions
        "lgd_override":  0.45,   # same as default LGD; minor collateral deterioration assumed
    },
    "severe": {
        "label":        "Severe Stress",
        "pd_multiplier": 1.8,    # PDs increase by 80% — deep recession / systemic shock
        "lgd_override":  0.55,   # higher LGD reflects lower recovery in distressed markets
    },
}

# ── Sensitivity Analysis Defaults ─────────────────────────────────────────────

# PD shifts: relative (e.g. -0.20 = PDs are 20% lower than estimated)
SENSITIVITY_PD_SHIFTS  = [-0.20, -0.10, 0.10, 0.20]

# LGD shifts: absolute percentage points (e.g. 0.10 = LGD is 10pp higher than assumed)
SENSITIVITY_LGD_SHIFTS = [-0.10, 0.10]