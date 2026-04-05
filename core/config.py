"""
config.py
---------
Central configuration — all paths and constants in one place.
Import this in any module instead of hardcoding paths.
"""

from pathlib import Path

# ── Root directory ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Artifact paths ────────────────────────────────────────────────
ARTIFACTS_DIR      = ROOT_DIR / "artifacts"
MODELS_DIR         = ARTIFACTS_DIR / "models"
PREPROCESSING_DIR  = ARTIFACTS_DIR / "preprocessing"

# ── Model files ───────────────────────────────────────────────────
XGB_MODEL_PATH              = MODELS_DIR / "xgb_pipeline.joblib"
SCORECARD_LR_MODEL_PATH     = MODELS_DIR / "scorecard_lr_model.joblib"

# ── Preprocessing files ───────────────────────────────────────────
WOE_BINS_PATH               = PREPROCESSING_DIR / "woe_bins.joblib"
SCORECARD_PATH              = PREPROCESSING_DIR / "scorecard.joblib"
SCORECARD_FEATURE_COLS_PATH = PREPROCESSING_DIR / "feature_columns_scorecard.json"
XGB_FEATURE_COLS_PATH       = PREPROCESSING_DIR / "feature_columns_xgb.json"

# ── Score band thresholds ─────────────────────────────────────────
SCORE_BANDS = [
    (0,   650, "High Risk",  "Decline"),
    (650, 700, "Elevated",   "Decline"),
    (700, 750, "Moderate",   "Review"),
    (750, 800, "Low",        "Approve"),
    (800, 850, "Lower",      "Approve"),
    (850, 9999,"Minimal",    "Approve"),
]
