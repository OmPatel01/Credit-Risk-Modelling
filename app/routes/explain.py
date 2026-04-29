"""
app/routes/explain.py
---------------------
HTTP route for the /explain endpoint.

Accepts a BusinessInput payload and returns a full feature-level breakdown
of why the scorecard produced the score it did, including:
    - Per-feature WOE values, coefficients, and log-odds contributions
    - Per-feature integer score points
    - Top 3 ranked risk drivers
    - Policy rule results (hard overrides + soft flags)

Delegates all computation to services/explain_service.py.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schema import BusinessInput
from services.explain_service import explain_prediction

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/explain",
    tags=["Explainability"],
    summary="Feature-level explanation of scorecard prediction",
    description=(
        "Returns a per-feature breakdown of the WOE scorecard decision. "
        "Shows each feature's WOE value, LR coefficient, log-odds contribution, "
        "and integer score points. Also applies policy engine rules and returns "
        "the top 3 risk drivers ranked by contribution magnitude."
    ),
)
def explain(business: BusinessInput):
    """
    Explain the scorecard prediction for a given BusinessInput.

    Returns feature-level contributions sorted by absolute impact,
    top 3 risk drivers, base points, and any policy rule overrides.
    """
    try:
        logger.info("[EXPLAIN ROUTE] Request received")
        result = explain_prediction(business.dict())
        logger.info(
            f"[EXPLAIN ROUTE] Completed | Score={result['credit_score']}, "
            f"Decision={result['decision']}"
        )
        return result

    except ValueError as e:
        logger.warning(f"[EXPLAIN ROUTE] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"[EXPLAIN ROUTE] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))