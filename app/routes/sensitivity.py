from fastapi import APIRouter, HTTPException

from app.schemas.risk_schemas import (
    SensitivityRequest,
    SensitivityResponse,
)
from services.sensitivity_service import run_sensitivity_analysis

router = APIRouter()


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    tags=["Risk Analytics"],
    summary="Sensitivity Analysis",
)
def sensitivity_analysis(request: SensitivityRequest):
    try:
        result = run_sensitivity_analysis(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            pd_shifts=request.pd_shifts,
            lgd_shifts=request.lgd_shifts,
            run_simulation=request.run_simulation,
            num_simulations=request.num_simulations,
            seed=request.seed,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))