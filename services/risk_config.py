"""
risk_config.py
--------------
Central configuration for all credit risk analytics modules.

All tunable parameters are defined here — no hardcoding in service logic.
"""

# ── Risk Segmentation Defaults ───────────────────────────────────
DEFAULT_SEGMENT_LABELS = ["A", "B", "C", "D", "E"]

DEFAULT_PD_THRESHOLDS = [0.05, 0.15, 0.30, 0.50]
# Buckets:  A=[0, 0.05)  B=[0.05, 0.15)  C=[0.15, 0.30)  D=[0.30, 0.50)  E=[0.50, 1.0]

DEFAULT_NUM_QUANTILES = 5  # deciles=10, quintiles=5

# ── ECL Defaults ─────────────────────────────────────────────────
DEFAULT_LGD = 0.45  # 45% — typical for unsecured consumer credit

# ── Monte Carlo Defaults ─────────────────────────────────────────
DEFAULT_NUM_SIMULATIONS = 10_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_VAR_CONFIDENCE = 0.95  # 95th percentile VaR

# ── Stress Testing Scenarios ─────────────────────────────────────
STRESS_SCENARIOS = {
    "base": {
        "label": "Base",
        "pd_multiplier": 1.0,
        "lgd_override": None,   # use input LGD
    },
    "mild": {
        "label": "Mild Stress",
        "pd_multiplier": 1.3,
        "lgd_override": 0.45,
    },
    "severe": {
        "label": "Severe Stress",
        "pd_multiplier": 1.8,
        "lgd_override": 0.55,
    },
}

# ── Sensitivity Analysis Defaults ────────────────────────────────
SENSITIVITY_PD_SHIFTS = [-0.20, -0.10, 0.10, 0.20]   # ±10–20%
SENSITIVITY_LGD_SHIFTS = [-0.10, 0.10]                 # ±10%
