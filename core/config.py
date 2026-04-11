"""
core/config.py
--------------
Single source of truth for all file paths and model constants.
Every module imports from here — never hardcode paths elsewhere.

Architecture note:
    Artifacts are split into two folders intentionally:
        models/        → trained model objects (.joblib)
        preprocessing/ → transformation artifacts fitted on training data
                         (WOE bins, scorecard table, feature column lists)
    This separation makes it easy to retrain only one model without
    accidentally overwriting preprocessing artifacts used by the other.
"""

from pathlib import Path

# ── Root directory — resolved relative to this file so the app runs correctly
# regardless of where it's launched from (uvicorn, pytest, mlops scripts)
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Artifact directories — created during training (mlops/train.py)
ARTIFACTS_DIR      = ROOT_DIR / "artifacts"
MODELS_DIR         = ARTIFACTS_DIR / "models"
PREPROCESSING_DIR  = ARTIFACTS_DIR / "preprocessing"

# ── Model binaries — joblib-serialised sklearn-compatible objects
# xgb_pipeline wraps OrdinalEncoder + XGBClassifier in a single Pipeline object,
# so no separate preprocessing step is needed at inference time for XGBoost
XGB_MODEL_PATH              = MODELS_DIR / "xgb_pipeline.joblib"
SCORECARD_LR_MODEL_PATH     = MODELS_DIR / "scorecard_lr_model.joblib"

# ── Preprocessing artifacts — must be fitted on training data ONLY,
# then frozen here; applying them at inference prevents data leakage
WOE_BINS_PATH               = PREPROCESSING_DIR / "woe_bins.joblib"       # scorecardpy bin definitions
SCORECARD_PATH              = PREPROCESSING_DIR / "scorecard.joblib"       # points table built from LR coefficients
SCORECARD_FEATURE_COLS_PATH = PREPROCESSING_DIR / "feature_columns_scorecard.json"  # WOE column names (already have _woe suffix)
XGB_FEATURE_COLS_PATH       = PREPROCESSING_DIR / "feature_columns_xgb.json"        # raw engineered feature names

# ── Credit score bands — maps a scalar score to a business decision
# Range is [0, 9999]; upper bound is intentionally large to catch any score above 850.
# Order matters: bands are evaluated top-to-bottom so ranges must not overlap.
# (low_inclusive, high_exclusive, risk_label, lending_decision)
SCORE_BANDS = [
    (0,   650, "High Risk",  "Decline"),
    (650, 700, "Elevated",   "Decline"),
    (700, 750, "Moderate",   "Review"),
    (750, 800, "Low",        "Approve"),
    (800, 850, "Lower",      "Approve"),
    (850, 9999,"Minimal",    "Approve"),
]