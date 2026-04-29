"""
app/main.py
-----------
FastAPI application entry point — defines all HTTP routes and wires them to service functions.

Responsibilities:
    - Route registration (raw ClientInput routes, BusinessInput routes, new analytics routes)
    - Request/response validation via Pydantic schemas (happens automatically before handlers run)
    - Calling service/prediction functions and wrapping results in typed response models
    - Error handling: ValueError → HTTP 422, all others → HTTP 500

NOT responsible for:
    - ML logic (lives in services/pd_model.py)
    - Feature engineering (lives in core/preprocessing.py)
    - Input translation (lives in core/input_mapper.py)
    - Explainability (lives in services/explain_service.py)
    - Recommendations (lives in services/recommend_service.py)
    - Policy overrides (lives in services/policy_engine.py)

Route groups:
    /predict/*          — raw 21-column ClientInput (technical consumers)
    /predict/business/* — 10-field BusinessInput (business-friendly API)
    /whatif/*           — compare two BusinessInput scenarios; returns score/PD delta + decision flip
    /explain            — per-feature WOE contribution breakdown (BusinessInput)
    /recommend          — actionable improvement recommendations (BusinessInput)
    /portfolio/summary  — aggregate portfolio KPIs from stored metadata
    Risk analytics routes registered via sub-routers from app/routes/
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import logging

from app.schema import (
    ClientInput,
    BusinessInput,
    ScorecardResponse,
    XGBResponse,
    CombinedResponse,
    WhatIfResponse,
    WhatIfRequest,
)
from services.pd_model import scorecard_predict, xgb_predict
from services.policy_engine import apply_policy_rules
from core.input_mapper import map_business_to_raw
from core.preprocessing import engineer_features
from app.routes import segmentation, ecl, simulation, stress, sensitivity
from app.routes import explain, recommend, portfolio

# ── Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts credit default probability and assigns a credit score "
        "using a WOE Logistic Regression scorecard (champion) "
        "and XGBoost (challenger). Includes explainability, recommendations, "
        "and portfolio analytics."
    ),
    version="1.1.0",
)

# ── CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sub-router registration
app.include_router(segmentation.router)
app.include_router(ecl.router)
app.include_router(simulation.router)
app.include_router(stress.router)
app.include_router(sensitivity.router)
app.include_router(explain.router)
app.include_router(recommend.router)
app.include_router(portfolio.router)


# ══════════════════════════════════════════════════════════════════
# Health Routes
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """Lightweight liveness check — confirms the API process is running."""
    logger.info("[ROOT] Health check called")
    return {
        "status":  "ok",
        "message": "Credit Risk Scoring API is running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Extended health check — confirms models are loaded and lists available model names."""
    logger.info("[HEALTH] Detailed health check")
    return {
        "status":  "ok",
        "models":  ["Scorecard LR + WOE", "XGBoost"],
        "version": "1.1.0",
    }


@app.get("/model-info")
def model_info():
    """Return training metadata (AUC, features used) from the artifact saved during training."""
    try:
        with open("artifacts/model_metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not found. Run mlops/train.py first.",
        )

    return {
        "model_type":    "Credit Risk Scorecard + XGBoost",
        "target":        metadata.get("target", "DEFAULT_NEXT_MONTH"),
        "features_used": len(metadata.get("features_used", [])),
        "metrics": {
            "logistic_auc":  metadata.get("logistic_auc"),
            "xgboost_auc":   metadata.get("xgboost_auc"),
            "scorecard_auc": metadata.get("scorecard_auc"),
        },
    }


# ══════════════════════════════════════════════════════════════════
# Raw Input Prediction Routes (ClientInput — technical consumers)
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    response_model=ScorecardResponse,
    tags=["Prediction"],
    summary="Scorecard prediction (Champion model)",
)
def predict_scorecard(client: ClientInput):
    """Run the champion scorecard model on raw ClientInput and return score + PD + decision."""
    try:
        logger.info("[PREDICT] Scorecard request received")
        result_dict = scorecard_predict(client.dict())

        # Apply policy engine
        df_eng       = engineer_features(client.dict())
        feat_dict    = df_eng.iloc[0].to_dict()
        policy       = apply_policy_rules(result_dict["decision"], feat_dict)
        result_dict["decision"] = policy["final_decision"]

        return ScorecardResponse(**result_dict)

    except Exception as e:
        logger.error(f"[PREDICT] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/xgb",
    response_model=XGBResponse,
    tags=["Prediction"],
    summary="XGBoost prediction (Challenger model)",
)
def predict_xgb(client: ClientInput):
    """Run the XGBoost challenger model on raw ClientInput and return PD + decision."""
    try:
        logger.info("[PREDICT-XGB] Request received")
        result_dict = xgb_predict(client.dict())
        return XGBResponse(**result_dict)

    except Exception as e:
        logger.error(f"[PREDICT-XGB] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/both",
    response_model=CombinedResponse,
    tags=["Prediction"],
    summary="Both models — champion + challenger",
)
def predict_both(client: ClientInput):
    """Run both models on the same raw input for champion/challenger comparison."""
    try:
        logger.info("[PREDICT-BOTH] Request received")
        input_data = client.dict()

        sc_dict  = scorecard_predict(input_data)
        xgb_dict = xgb_predict(input_data)

        # Apply policy to scorecard decision
        df_eng    = engineer_features(input_data)
        feat_dict = df_eng.iloc[0].to_dict()
        policy    = apply_policy_rules(sc_dict["decision"], feat_dict)
        sc_dict["decision"] = policy["final_decision"]

        return CombinedResponse(
            scorecard=ScorecardResponse(**sc_dict),
            xgboost=XGBResponse(**xgb_dict),
        )

    except Exception as e:
        logger.error(f"[PREDICT-BOTH] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev-only debug route
@app.post("/debug", tags=["Debug"])
def debug(client: ClientInput):
    """Return engineered columns, WOE columns, and feature matching diagnostics — dev use only."""
    from core.preprocessing import engineer_features as ef, apply_woe_transform, load_woe_bins

    raw    = client.dict()
    df_eng = ef(raw)
    bins   = load_woe_bins()
    df_woe = apply_woe_transform(df_eng.copy(), bins)

    with open("artifacts/preprocessing/feature_columns_scorecard.json") as f:
        feat_cols = json.load(f)

    woe_cols_expected = [f"{c}_woe" for c in feat_cols]
    available         = [c for c in woe_cols_expected if c in df_woe.columns]

    return {
        "engineered_cols": df_eng.columns.tolist(),
        "woe_cols":        df_woe.columns.tolist(),
        "expected_cols":   woe_cols_expected,
        "matched_cols":    available,
    }


# ══════════════════════════════════════════════════════════════════
# Business Input Routes (BusinessInput — user-friendly API)
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/predict/business",
    response_model=ScorecardResponse,
    tags=["Prediction (Business Input)"],
    summary="Scorecard prediction from business inputs",
)
def predict_scorecard_business(business: BusinessInput):
    """Translate BusinessInput → raw features → scorecard prediction + policy check."""
    try:
        logger.info("[BUSINESS-PREDICT] Scorecard request received")

        raw_input   = map_business_to_raw(business)
        result_dict = scorecard_predict(raw_input)

        # Apply policy engine using engineered features
        df_eng    = engineer_features(raw_input)
        feat_dict = df_eng.iloc[0].to_dict()
        policy    = apply_policy_rules(result_dict["decision"], feat_dict)
        result_dict["decision"] = policy["final_decision"]

        result = ScorecardResponse(**result_dict)
        logger.info(
            f"[BUSINESS-PREDICT] Success → PD={result.default_probability}, "
            f"Decision={result.decision}, PolicyOverride={policy['policy_overridden']}"
        )
        return result

    except ValueError as e:
        logger.warning(f"[BUSINESS-PREDICT] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"[BUSINESS-PREDICT] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/business/xgb",
    response_model=XGBResponse,
    tags=["Prediction (Business Input)"],
    summary="XGBoost prediction from business inputs",
)
def predict_xgb_business(business: BusinessInput):
    """Translate BusinessInput → raw features → XGBoost prediction."""
    try:
        logger.info("[BUSINESS-XGB] Request received")
        raw_input   = map_business_to_raw(business)
        result_dict = xgb_predict(raw_input)
        result      = XGBResponse(**result_dict)
        logger.info(
            f"[BUSINESS-XGB] Success → PD={result.default_probability}, "
            f"Decision={result.decision}"
        )
        return result

    except ValueError as e:
        logger.warning(f"[BUSINESS-XGB] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"[BUSINESS-XGB] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/business/both",
    response_model=CombinedResponse,
    tags=["Prediction (Business Input)"],
    summary="Both models from business inputs",
)
def predict_both_business(business: BusinessInput):
    """Translate BusinessInput → raw features → both model predictions."""
    try:
        logger.info("[BUSINESS-BOTH] Request received")
        raw_input = map_business_to_raw(business)

        sc_dict  = scorecard_predict(raw_input)
        xgb_dict = xgb_predict(raw_input)

        # Apply policy to scorecard only
        df_eng    = engineer_features(raw_input)
        feat_dict = df_eng.iloc[0].to_dict()
        policy    = apply_policy_rules(sc_dict["decision"], feat_dict)
        sc_dict["decision"] = policy["final_decision"]

        logger.info(
            f"[BUSINESS-BOTH] Success → "
            f"SC PD={sc_dict['default_probability']}, XGB PD={xgb_dict['default_probability']}"
        )
        return CombinedResponse(
            scorecard=ScorecardResponse(**sc_dict),
            xgboost=XGBResponse(**xgb_dict),
        )

    except ValueError as e:
        logger.warning(f"[BUSINESS-BOTH] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"[BUSINESS-BOTH] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
# What-If Analysis Route
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/whatif/scorecard",
    response_model=WhatIfResponse,
    tags=["What-If Analysis"],
    summary="Compare two scenarios — score/PD delta + decision flip detection",
    description=(
        "Runs the scorecard on both the base and modified BusinessInput scenarios "
        "and returns the score delta, PD delta, and whether the lending decision "
        "changed (e.g. Decline → Approve). The policy engine is applied to both "
        "scenarios so the decisions reflect hard business rules."
    ),
)
def whatif_scorecard(data: WhatIfRequest):
    """
    Compare scorecard output for two BusinessInput scenarios.

    base_input    → current applicant profile
    modified_input → proposed / improved profile

    Returns:
        base_score, new_score, delta_score
        base_pd, new_pd, delta_pd
        base_decision, new_decision, decision_flipped
    """
    try:
        logger.info("[WHATIF] Request received")

        # ── Base scenario
        base_raw    = map_business_to_raw(data.base_input.dict())
        base_result = scorecard_predict(base_raw)

        base_df_eng    = engineer_features(base_raw)
        base_feat_dict = base_df_eng.iloc[0].to_dict()
        base_policy    = apply_policy_rules(base_result["decision"], base_feat_dict)
        base_decision  = base_policy["final_decision"]

        # ── Modified scenario
        mod_raw    = map_business_to_raw(data.modified_input.dict())
        mod_result = scorecard_predict(mod_raw)

        mod_df_eng    = engineer_features(mod_raw)
        mod_feat_dict = mod_df_eng.iloc[0].to_dict()
        mod_policy    = apply_policy_rules(mod_result["decision"], mod_feat_dict)
        new_decision  = mod_policy["final_decision"]

        # ── Deltas
        delta_score = mod_result["credit_score"] - base_result["credit_score"]
        delta_pd    = round(
            mod_result["default_probability"] - base_result["default_probability"], 4
        )

        # decision_flipped = True when the final decision category changed
        decision_flipped = base_decision != new_decision

        logger.info(
            f"[WHATIF] Score Δ={delta_score}, PD Δ={delta_pd}, "
            f"Base decision={base_decision}, New decision={new_decision}, "
            f"Flipped={decision_flipped}"
        )

        return {
            "base_score":      base_result["credit_score"],
            "new_score":       mod_result["credit_score"],
            "delta_score":     delta_score,
            "base_pd":         base_result["default_probability"],
            "new_pd":          mod_result["default_probability"],
            "delta_pd":        delta_pd,
            "base_decision":   base_decision,
            "new_decision":    new_decision,
            "decision_flipped": decision_flipped,
        }

    except Exception as e:
        logger.error(f"[WHATIF] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))