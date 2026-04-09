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
from app.routes import segmentation
from app.routes import ecl
from app.routes import simulation
from app.routes import stress
from app.routes import sensitivity

import logging

# ── Logging Configuration ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for deep logs
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

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
    allow_origins=["*"],
    allow_credentials=True,
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
    logger.info("[ROOT] Health check called")
    return {
        "status":  "ok",
        "message": "Credit Risk Scoring API is running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    logger.info("[HEALTH] Detailed health check")
    return {
        "status":  "ok",
        "models":  ["Scorecard LR + WOE", "XGBoost"],
        "version": "1.0.0",
    }

@app.get("/model-info")
def model_info():
    with open("artifacts/model_metadata.json") as f:
        metadata = json.load(f)

    return {
        "model_type": "Credit Risk Scorecard + XGBoost",
        "target": metadata["target"],
        "features_used": len(metadata["features_used"]),
        "metrics": {
            "logistic_auc": metadata.get("logistic_auc"),
            "xgboost_auc": metadata.get("xgboost_auc"),
            "scorecard_auc": metadata.get("scorecard_auc"),
        }
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
    try:
        logger.info("[PREDICT] Scorecard request received")
        logger.debug(f"[PREDICT] Payload: {client.dict()}")

        result_dict = scorecard_predict(client.dict())
        result = ScorecardResponse(**result_dict)

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
    try:
        logger.info("[PREDICT-XGB] Request received")

        result_dict = xgb_predict(client.dict())
        result = XGBResponse(**result_dict)

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
    

# Add this debug route temporarily to app/main.py
@app.post("/debug", tags=["Debug"])
def debug(client: ClientInput):
    from core.preprocessing import engineer_features, apply_woe_transform, load_woe_bins
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


# ═══════════════════════════════════════════════════════════════════
# Business Input Routes (NEW)
# ═══════════════════════════════════════════════════════════════════
# These routes accept user-friendly business inputs and map them to
# the internal raw dataset format before prediction.

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
    try:
        logger.info("[BUSINESS-PREDICT] Scorecard request received")
        logger.debug(f"[BUSINESS-PREDICT] Input: {business.dict()}")

        raw_input = map_business_to_raw(business)

        logger.debug("[BUSINESS-PREDICT] Mapping completed")

        result_dict = scorecard_predict(raw_input)
        result = ScorecardResponse(**result_dict)

        logger.info(
            f"[BUSINESS-PREDICT] Success → PD: {result.default_probability}, "
            f"Decision: {result.decision}"
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
    description=(
        "XGBoost challenger model prediction using business-friendly inputs. "
        "Inputs are automatically mapped to raw dataset features. "
        "Returns default probability."
    ),
)
def predict_xgb_business(business: BusinessInput):
    try:
        logger.info("[BUSINESS-XGB] Request received")
        logger.debug(f"[BUSINESS-XGB] Input: {business.dict()}")

        # 🔹 Mapping
        raw_input = map_business_to_raw(business)
        logger.debug("[BUSINESS-XGB] Mapping completed")

        # 🔹 Prediction
        result_dict = xgb_predict(raw_input)
        result = XGBResponse(**result_dict)

        # 🔹 Success log
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
    try:
        logger.info("[BUSINESS-BOTH] Request received")
        logger.debug(f"[BUSINESS-BOTH] Input: {business.dict()}")

        # 🔹 Mapping
        raw_input = map_business_to_raw(business)
        logger.debug("[BUSINESS-BOTH] Mapping completed")

        # 🔹 Predictions
        scorecard_dict = scorecard_predict(raw_input)
        xgb_dict       = xgb_predict(raw_input)

        scorecard_result = ScorecardResponse(**scorecard_dict)
        xgb_result       = XGBResponse(**xgb_dict)

        # 🔹 Success log (important comparison insight)
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
    
@app.post(
    "/whatif/scorecard",
    response_model=WhatIfResponse,
    tags=["What-If Analysis"],
)
def whatif_scorecard(data: WhatIfRequest):
    try:
        logger.info("[WHATIF] Request received")

        # ── Base Input ─────────────────────
        base_raw = map_business_to_raw(data.base_input.dict())
        base_result = scorecard_predict(base_raw)

        # ── Modified Input ─────────────────
        mod_raw  = map_business_to_raw(data.modified_input.dict())
        mod_result = scorecard_predict(mod_raw)

        # ── Compute delta ──────────────────
        delta_score = mod_result["credit_score"] - base_result["credit_score"]
        delta_pd    = round(
            mod_result["default_probability"] - base_result["default_probability"], 4
        )

        logger.info(
            f"[WHATIF] Score Δ={delta_score}, PD Δ={delta_pd}"
        )

        return {
            "base_score": base_result["credit_score"],
            "new_score": mod_result["credit_score"],
            "delta_score": delta_score,

            "base_pd": base_result["default_probability"],
            "new_pd": mod_result["default_probability"],
            "delta_pd": delta_pd,
        }

    except Exception as e:
        logger.error(f"[WHATIF] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))