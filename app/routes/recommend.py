"""
app/routes/recommend.py
------------------------
HTTP route for the /recommend endpoint.

Accepts a BusinessInput payload and returns prioritised, actionable
improvement recommendations based on the top risk drivers identified
by the scorecard explanation layer.

Delegates all computation to services/recommend_service.py.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schema import BusinessInput
from services.recommend_service import generate_recommendations

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/recommend",
    tags=["Explainability"],
    summary="Actionable improvement recommendations for a borrower",
    description=(
        "Identifies which features are most responsible for the borrower's risk level "
        "and returns specific, prioritised actions the borrower can take to improve "
        "their credit score. Only risk-increasing features are actioned. "
        "Recommendations are capped at 5 to avoid overwhelming the borrower."
    ),
)
def recommend(business: BusinessInput):
    """
    Generate actionable recommendations for a given BusinessInput.

    Returns up to 5 prioritised recommendations sorted by risk contribution,
    along with the current prediction (score, PD, decision).
    """
    try:
        logger.info("[RECOMMEND ROUTE] Request received")
        result = generate_recommendations(business.dict())
        logger.info(
            f"[RECOMMEND ROUTE] Completed | Score={result['credit_score']}, "
            f"Recommendations={len(result['recommendations'])}"
        )
        return result

    except ValueError as e:
        logger.warning(f"[RECOMMEND ROUTE] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"[RECOMMEND ROUTE] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))