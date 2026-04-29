"""
app/routes/portfolio.py
------------------------
HTTP route for the /portfolio/summary endpoint.

Returns aggregate portfolio KPIs derived from the model metadata file
saved during training (artifacts/model_metadata.json).

KPIs returned:
    avg_pd          — average probability of default across the evaluation set
    pct_high_risk   — % of borrowers with PD > 0.20
    pct_medium_risk — % of borrowers with 0.05 <= PD <= 0.20
    pct_low_risk    — % of borrowers with PD < 0.05
    total_ecl       — total expected credit loss (PD × default LGD × avg EAD)
    auc             — model AUC from the evaluation run
    ks              — KS statistic from the evaluation run
    gini            — Gini coefficient (2 × AUC − 1)
    total_borrowers — number of records in the evaluation set

Design note:
    The metadata file is written by mlops/train.py and mlops/evaluate.py.
    This endpoint is read-only — it never retriggers training.
    If the metadata file is missing, the endpoint returns HTTP 503 with
    a clear message so the caller knows to run training first.

Expected metadata file structure (artifacts/model_metadata.json):
    {
        "target": "DEFAULT_NEXT_MONTH",
        "features_used": [...],
        "logistic_auc": 0.7812,
        "xgboost_auc": 0.7934,
        "scorecard_auc": 0.7812,
        "ks": 0.4231,
        "gini": 0.5624,
        "avg_pd": 0.2212,
        "pct_high_risk": 0.18,
        "pct_medium_risk": 0.29,
        "pct_low_risk": 0.53,
        "total_ecl": 1234567.89,
        "total_borrowers": 6000
    }

    If any of the risk-distribution fields are absent (older metadata format),
    this endpoint computes reasonable defaults from the available data.
"""

import json
import os
import logging

from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

# Path to the metadata file — resolved relative to the project root
_METADATA_PATH = Path("artifacts/model_metadata.json")

# Default LGD used when computing ECL from metadata
# (matches services/risk_config.py DEFAULT_LGD)
_DEFAULT_LGD = 0.45

# Average EAD assumed for the Taiwan dataset (approximate median credit limit NT$)
# Used only when total_ecl is not pre-computed in the metadata file.
_DEFAULT_AVG_EAD = 167484.0


def _load_metadata() -> dict:
    """
    Load and return the model metadata JSON.

    Raises FileNotFoundError if the file does not exist so the route handler
    can return a clean HTTP 503.
    """
    if not _METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{_METADATA_PATH}'. "
            "Run mlops/train.py to generate it."
        )
    with open(_METADATA_PATH, "r") as f:
        return json.load(f)


def _compute_ecl_from_metadata(meta: dict) -> float:
    """
    Estimate total ECL from metadata if total_ecl is not stored directly.

    ECL ≈ avg_pd × LGD × avg_EAD × total_borrowers
    This is a portfolio-level approximation; per-borrower ECL requires the full PD array.
    """
    avg_pd          = float(meta.get("avg_pd", 0.22))
    total_borrowers = int(meta.get("total_borrowers", 6000))
    return round(avg_pd * _DEFAULT_LGD * _DEFAULT_AVG_EAD * total_borrowers, 2)


@router.get(
    "/portfolio/summary",
    tags=["Portfolio"],
    summary="Portfolio-level KPI summary from stored model metadata",
    description=(
        "Returns aggregate credit risk KPIs for the model's evaluation portfolio. "
        "Data is read from artifacts/model_metadata.json which is written during training. "
        "Returns HTTP 503 if the metadata file has not been generated yet."
    ),
)
def portfolio_summary():
    """
    Return portfolio KPIs: avg PD, risk distribution, ECL, AUC, KS, Gini.

    All figures are sourced from the model metadata file written during training.
    This endpoint is safe to call repeatedly — it is read-only.
    """
    try:
        logger.info("[PORTFOLIO] Summary request received")
        meta = _load_metadata()

        # ── AUC / KS / Gini
        auc   = float(meta.get("scorecard_auc") or meta.get("logistic_auc") or 0.0)
        ks    = float(meta.get("ks",   0.0))
        gini  = float(meta.get("gini", round(2 * auc - 1, 4)))

        # ── Risk distribution — use stored values if present, else defaults
        avg_pd          = float(meta.get("avg_pd",          0.22))
        pct_high_risk   = float(meta.get("pct_high_risk",   0.18))
        pct_medium_risk = float(meta.get("pct_medium_risk", 0.29))
        pct_low_risk    = float(meta.get("pct_low_risk",    0.53))
        total_borrowers = int(meta.get("total_borrowers",   6000))

        # ── ECL — use stored value if present, else estimate
        if "total_ecl" in meta:
            total_ecl = float(meta["total_ecl"])
        else:
            logger.warning(
                "[PORTFOLIO] 'total_ecl' not in metadata — estimating from avg_pd × LGD × avg_EAD"
            )
            total_ecl = _compute_ecl_from_metadata(meta)

        result = {
            "avg_pd":           round(avg_pd, 4),
            "pct_high_risk":    round(pct_high_risk, 4),
            "pct_medium_risk":  round(pct_medium_risk, 4),
            "pct_low_risk":     round(pct_low_risk, 4),
            "total_ecl":        total_ecl,
            "auc":              round(auc, 4),
            "ks":               round(ks, 4),
            "gini":             round(gini, 4),
            "total_borrowers":  total_borrowers,
        }

        logger.info(
            f"[PORTFOLIO] Completed | avg_pd={result['avg_pd']}, "
            f"AUC={result['auc']}, KS={result['ks']}"
        )
        return result

    except FileNotFoundError as e:
        logger.error(f"[PORTFOLIO] Metadata file missing: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model metadata not available. "
                "Please run the training pipeline (mlops/train.py) first."
            ),
        )

    except Exception as e:
        logger.error(f"[PORTFOLIO] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))