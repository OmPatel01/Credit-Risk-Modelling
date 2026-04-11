"""
app/routes/ecl.py
-----------------
HTTP route for the Expected Credit Loss (ECL) endpoint.

Thin handler: validates the request via Pydantic, delegates all computation to
ecl_service.compute_ecl, and returns the typed ECLResponse.
Error handling uses HTTP 400 (Bad Request) because ECL failures are typically
due to invalid input arrays (mismatched lengths, out-of-range values) rather
than server-side errors.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import ECLRequest, ECLResponse
from services.ecl_service import compute_ecl

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/ecl",
    response_model=ECLResponse,
    tags=["Risk Analytics"],
    summary="Expected Credit Loss (ECL)",
)
def calculate_ecl(request: ECLRequest):
    """Compute ECL = PD × LGD × EAD for each borrower and return portfolio totals and optional segment breakdown."""
    try:
        logger.info("[ECL] Request received")
        logger.debug(
            f"[ECL] Inputs → PD count: {len(request.pd_values)}, "
            f"LGD: {request.lgd}, "
            f"EAD count: {len(request.ead_values)}"
        )

        logger.info("[ECL] Computing ECL...")
        result = compute_ecl(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            segment_labels=request.segment_labels,
        )

        logger.info(
            f"[ECL] Computation successful → Total ECL: {result['total_ecl']}, "
            f"Mean ECL: {result['mean_ecl']}"
        )
        return result

    except Exception as e:
        logger.error(f"[ECL] Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))