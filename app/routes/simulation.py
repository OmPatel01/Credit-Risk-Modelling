from fastapi import APIRouter, HTTPException

from app.schemas.risk_schemas import (
    SimulationRequest,
    SimulationResponse,
)
from services.monte_carlo_service import run_monte_carlo_simulation

router = APIRouter()


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["Risk Analytics"],
    summary="Monte Carlo Simulation",
)
def simulate_loss(request: SimulationRequest):
    try:
        result = run_monte_carlo_simulation(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            num_simulations=request.num_simulations,
            confidence_level=request.confidence_level,
            seed=request.seed,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))