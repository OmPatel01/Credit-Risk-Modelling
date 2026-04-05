from fastapi import APIRouter, HTTPException

from app.schemas.risk_schemas import (
    ECLRequest,
    ECLResponse,
)
from services.ecl_service import compute_ecl

router = APIRouter()


@router.post(
    "/ecl",
    response_model=ECLResponse,
    tags=["Risk Analytics"],
    summary="Expected Credit Loss (ECL)",
)
def calculate_ecl(request: ECLRequest):
    try:
        result = compute_ecl(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            segment_labels=request.segment_labels,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))