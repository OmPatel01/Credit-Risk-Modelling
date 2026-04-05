from fastapi import APIRouter, HTTPException

from app.schemas.risk_schemas import (
    StressTestRequest,
    StressTestResponse,
)
from services.scenario_service import run_stress_test

router = APIRouter()


@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    tags=["Risk Analytics"],
    summary="Scenario Analysis (Stress Testing)",
)
def stress_test(request: StressTestRequest):
    try:
        result = run_stress_test(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            run_simulation=request.run_simulation,
            num_simulations=request.num_simulations,
            seed=request.seed,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))