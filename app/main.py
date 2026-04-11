"""
app/main.py
-----------
FastAPI application entry point — defines all HTTP routes and wires them to service functions.

Responsibilities:
    - Route registration (both raw ClientInput routes and business-friendly BusinessInput routes)
    - Request/response validation via Pydantic schemas (happens automatically before handlers run)
    - Calling service/prediction functions and wrapping results in typed response models
    - Error handling: ValueError → HTTP 422, all others → HTTP 500

NOT responsible for:
    - ML logic (lives in services/pd_model.py)
    - Feature engineering (lives in core/preprocessing.py)
    - Input translation (lives in core/input_mapper.py)

Route groups:
    /predict/*          — accept raw 21-column ClientInput (technical consumers)
    /predict/business/* — accept 10-field BusinessInput (business-friendly API)
    /whatif/*           — compare two BusinessInput scenarios and return score/PD delta
    Risk analytics routes are registered via sub-routers from app/routes/
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json

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
from core.input_mapper import map_business_to_raw
from app.routes import segmentation, ecl, simulation, stress, sensitivity

import logging

# ── Logging — configure once at app startup; all modules inherit this config via getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG for detailed per-request traces
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ── App setup — metadata appears in the auto-generated /docs Swagger UI
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts credit default probability and assigns a credit score "
        "using a WOE Logistic Regression scorecard (champion) "
        "and XGBoost (challenger). "
    ),
    version="1.0.0",
)

# ── CORS — allow any origin so the standalone frontend HTML file can call this API
# In production, replace "*" with the specific frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sub-router registration — each analytics feature has its own router module
app.include_router(segmentation.router)
app.include_router(ecl.router)
app.include_router(simulation.router)
app.include_router(stress.router)
app.include_router(sensitivity.router)


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
        "version": "1.0.0",
    }


@app.get("/model-info")
def model_info():
    """Return training metadata (AUC, features used) from the artifact saved during training."""
    with open("artifacts/model_metadata.json") as f:
        metadata = json.load(f)

    return {
        "model_type":     "Credit Risk Scorecard + XGBoost",
        "target":         metadata["target"],
        "features_used":  len(metadata["features_used"]),
        "metrics": {
            "logistic_auc": metadata.get("logistic_auc"),
            "xgboost_auc":  metadata.get("xgboost_auc"),
            "scorecard_auc": metadata.get("scorecard_auc"),
        }
    }


# ══════════════════════════════════════════════════════════════════
# Raw Input Prediction Routes (ClientInput — technical consumers)
# These routes accept the full 21-column raw dataset format.
# Intended for system integrations that already have the structured data.
# ══════════════════════════════════════════════════════════════════

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
    """Run the champion scorecard model on raw ClientInput and return score + PD + decision."""
    try:
        logger.info("[PREDICT] Scorecard request received")
        logger.debug(f"[PREDICT] Payload: {client.dict()}")

        result_dict = scorecard_predict(client.dict())
        result      = ScorecardResponse(**result_dict)

        logger.info(
            f"[BUSINESS-PREDICT] Success → PD: {result.default_probability}, "
            f"Decision: {result.decision}"
        )
        return result

    except Exception as e:
        logger.error(f"[PREDICT] Error: {str(e)}", exc_info=True)
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
    """Run the XGBoost challenger model on raw ClientInput and return PD + decision."""
    try:
        logger.info("[PREDICT-XGB] Request received")

        result_dict = xgb_predict(client.dict())
        result      = XGBResponse(**result_dict)

        logger.info(
            f"[PREDICT-XGB] Success → PD: {result.default_probability}, "
            f"Decision: {result.decision}"
        )
        return result

    except Exception as e:
        logger.error(f"[PREDICT-XGB] Error: {str(e)}", exc_info=True)
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
    """Run both models on the same raw input to enable champion/challenger comparison."""
    try:
        logger.info("[PREDICT-BOTH] Request received")

        input_data = client.dict()

        scorecard_dict = scorecard_predict(input_data)
        xgb_dict       = xgb_predict(input_data)

        scorecard_result = ScorecardResponse(**scorecard_dict)
        xgb_result       = XGBResponse(**xgb_dict)

        logger.info(
            f"[PREDICT-BOTH] Scorecard PD: {scorecard_result.default_probability}, "
            f"XGB PD: {xgb_result.default_probability}"
        )
        return CombinedResponse(
            scorecard=scorecard_result,
            xgboost=xgb_result
        )

    except Exception as e:
        logger.error(f"[PREDICT-BOTH] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Debug route — exposes intermediate preprocessing steps to help diagnose
# WOE column mismatches or feature engineering issues during development.
# Remove or protect this route before production deployment.
@app.post("/debug", tags=["Debug"])
def debug(client: ClientInput):
    """Return engineered columns, WOE columns, and feature matching diagnostics — dev use only."""
    from core.preprocessing import engineer_features, apply_woe_transform, load_woe_bins

    raw      = client.dict()
    df_eng   = engineer_features(raw)
    bins     = load_woe_bins()
    df_woe   = apply_woe_transform(df_eng.copy(), bins)

    with open("artifacts/preprocessing/feature_columns_scorecard.json") as f:
        feat_cols = json.load(f)

    # feat_cols already contains _woe suffix; test old (wrong) pattern to show the mismatch
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
# Accept intuitive business fields; map_business_to_raw converts them
# to raw dataset format before passing to the prediction functions.
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/predict/business",
    response_model=ScorecardResponse,
    tags=["Prediction (Business Input)"],
    summary="Scorecard prediction from business inputs",
    description=(
        "Scorecard prediction using business-friendly inputs. "
        "Inputs are automatically mapped to raw dataset features. "
        "Returns a credit score (576–906) and default probability."
    ),
)
def predict_scorecard_business(business: BusinessInput):
    """Translate BusinessInput → raw features → scorecard prediction."""
    try:
        logger.info("[BUSINESS-PREDICT] Scorecard request received")
        logger.debug(f"[BUSINESS-PREDICT] Input: {business.dict()}")

        raw_input   = map_business_to_raw(business)
        result_dict = scorecard_predict(raw_input)
        result      = ScorecardResponse(**result_dict)

        logger.info(
            f"[BUSINESS-PREDICT] Success → PD: {result.default_probability}, "
            f"Decision: {result.decision}"
        )
        return result

    except ValueError as e:
        # Validation errors from the mapper (out-of-range fields) → 422 Unprocessable Entity
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
    description=(
        "XGBoost challenger model prediction using business-friendly inputs. "
        "Inputs are automatically mapped to raw dataset features. "
        "Returns default probability."
    ),
)
def predict_xgb_business(business: BusinessInput):
    """Translate BusinessInput → raw features → XGBoost prediction."""
    try:
        logger.info("[BUSINESS-XGB] Request received")
        logger.debug(f"[BUSINESS-XGB] Input: {business.dict()}")

        raw_input   = map_business_to_raw(business)
        result_dict = xgb_predict(raw_input)
        result      = XGBResponse(**result_dict)

        logger.info(
            f"[BUSINESS-XGB] Success → PD: {result.default_probability}, "
            f"Decision: {result.decision}"
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
    description=(
        "Both scorecard and XGBoost predictions using business-friendly inputs. "
        "Useful for comparing champion vs challenger decisions."
    ),
)
def predict_both_business(business: BusinessInput):
    """Translate BusinessInput → raw features → both model predictions in one call."""
    try:
        logger.info("[BUSINESS-BOTH] Request received")
        logger.debug(f"[BUSINESS-BOTH] Input: {business.dict()}")

        raw_input = map_business_to_raw(business)

        scorecard_dict = scorecard_predict(raw_input)
        xgb_dict       = xgb_predict(raw_input)

        scorecard_result = ScorecardResponse(**scorecard_dict)
        xgb_result       = XGBResponse(**xgb_dict)

        logger.info(
            f"[BUSINESS-BOTH] Success → "
            f"Scorecard PD: {scorecard_result.default_probability}, "
            f"XGB PD: {xgb_result.default_probability}"
        )
        return CombinedResponse(
            scorecard=scorecard_result,
            xgboost=xgb_result
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
)
def whatif_scorecard(data: WhatIfRequest):
    """
    Compare scorecard output for two BusinessInput scenarios and return score/PD deltas.

    Use case: a loan officer wants to know "if this applicant reduces their bill amount
    by NT$5,000, how much does their credit score improve?"

    base_input    → current applicant profile
    modified_input → proposed / hypothetical profile
    Response includes raw scores, probabilities, and the signed differences (delta).
    Positive delta_score = score improved. Negative delta_pd = default risk decreased.
    """
    try:
        logger.info("[WHATIF] Request received")

        base_raw    = map_business_to_raw(data.base_input.dict())
        base_result = scorecard_predict(base_raw)

        mod_raw    = map_business_to_raw(data.modified_input.dict())
        mod_result = scorecard_predict(mod_raw)

        delta_score = mod_result["credit_score"]        - base_result["credit_score"]
        delta_pd    = round(
            mod_result["default_probability"] - base_result["default_probability"], 4
        )

        logger.info(f"[WHATIF] Score Δ={delta_score}, PD Δ={delta_pd}")

        return {
            "base_score":  base_result["credit_score"],
            "new_score":   mod_result["credit_score"],
            "delta_score": delta_score,
            "base_pd":     base_result["default_probability"],
            "new_pd":      mod_result["default_probability"],
            "delta_pd":    delta_pd,
        }

    except Exception as e:
        logger.error(f"[WHATIF] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))