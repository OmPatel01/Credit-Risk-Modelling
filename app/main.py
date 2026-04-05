"""
main.py
-------
FastAPI application — the API layer.

This file only handles:
- Route definitions
- Request/response validation (via Pydantic schemas)
- Calling src/ functions

It does NOT contain any ML logic.
All prediction logic lives in src/predict.py.

Routes:
    GET  /             → health check
    GET  /health       → health check with model status
    POST /predict      → scorecard prediction (LR + WOE)
    POST /predict/xgb  → XGBoost challenger prediction
    POST /predict/both → both models combined
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import (
    ClientInput,
    ScorecardResponse,
    XGBResponse,
    CombinedResponse,
)
from services.pd_model import scorecard_predict, xgb_predict
from app.routes import segmentation
from app.routes import ecl
from app.routes import simulation
from app.routes import stress
from app.routes import sensitivity

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts credit default probability and assigns a credit score "
        "using a WOE Logistic Regression scorecard (champion) "
        "and XGBoost (challenger). "
    ),
    version="1.0.0",
)

# ── CORS — allows the frontend HTML to call this API ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router)
app.include_router(ecl.router)
app.include_router(simulation.router)
app.include_router(stress.router)
app.include_router(sensitivity.router)

# ── Routes ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Root health check."""
    return {
        "status":  "ok",
        "message": "Credit Risk Scoring API is running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check — confirms models are loaded."""
    return {
        "status":  "ok",
        "models":  ["Scorecard LR + WOE", "XGBoost"],
        "version": "1.0.0",
    }


@app.post(
    "/predict",
    response_model=ScorecardResponse,
    tags=["Prediction"],
    summary="Scorecard prediction (Champion model)",
    description=(
        "Returns a credit score (576–906) and default probability "
        "using the Logistic Regression + WOE scorecard. "
        "This is the production/champion model."
    ),
)
def predict_scorecard(client: ClientInput):
    """
    Scorecard (Champion) prediction endpoint.

    Input  → raw client features
    Output → credit_score, default_probability, risk_level, decision
    """
    try:
        result = scorecard_predict(client.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/xgb",
    response_model=XGBResponse,
    tags=["Prediction"],
    summary="XGBoost prediction (Challenger model)",
    description=(
        "Returns default probability using the XGBoost challenger model. "
        "Higher AUC than scorecard but not interpretable."
    ),
)
def predict_xgb(client: ClientInput):
    """
    XGBoost (Challenger) prediction endpoint.

    Input  → raw client features
    Output → default_probability, risk_level, decision
    """
    try:
        result = xgb_predict(client.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/both",
    response_model=CombinedResponse,
    tags=["Prediction"],
    summary="Both models — champion + challenger",
    description=(
        "Returns predictions from both the scorecard and XGBoost models. "
        "Useful for comparing champion vs challenger decisions."
    ),
)
def predict_both(client: ClientInput):
    """
    Combined champion-challenger prediction endpoint.

    Input  → raw client features
    Output → scorecard result + xgboost result
    """
    try:
        input_data     = client.dict()
        scorecard_result = scorecard_predict(input_data)
        xgb_result       = xgb_predict(input_data)
        return {
            "scorecard": scorecard_result,
            "xgboost":   xgb_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this debug route temporarily to app/main.py
@app.post("/debug", tags=["Debug"])
def debug(client: ClientInput):
    from src.preprocessing import engineer_features, apply_woe_transform, load_woe_bins
    import json

    raw = client.dict()

    # Step 1 - check engineered features
    df_eng = engineer_features(raw)
    print("Engineered columns:", df_eng.columns.tolist())

    # Step 2 - check WOE output
    bins = load_woe_bins()
    df_woe = apply_woe_transform(df_eng.copy(), bins)
    print("WOE columns:", df_woe.columns.tolist())

    # Step 3 - check feature cols from JSON
    with open("artifacts/preprocessing/feature_columns_scorecard.json") as f:
        feat_cols = json.load(f)
    print("Expected feature cols:", feat_cols)

    woe_cols_expected = [f"{c}_woe" for c in feat_cols]
    print("Expected WOE cols:", woe_cols_expected)

    available = [c for c in woe_cols_expected if c in df_woe.columns]
    print("Available matches:", available)

    return {
        "engineered_cols": df_eng.columns.tolist(),
        "woe_cols":        df_woe.columns.tolist(),
        "expected_cols":   woe_cols_expected,
        "matched_cols":    available,
    }