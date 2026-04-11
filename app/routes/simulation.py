"""
app/routes/simulation.py
------------------------
HTTP route for the Monte Carlo loss simulation endpoint.

Delegates to monte_carlo_service.run_monte_carlo_simulation.
The endpoint is intentionally stateless — each call is independent and reproducible
via the seed parameter.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import SimulationRequest, SimulationResponse
from services.monte_carlo_service import run_monte_carlo_simulation

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["Risk Analytics"],
    summary="Monte Carlo Simulation",
)
def simulate_loss(request: SimulationRequest):
    """Simulate portfolio losses across num_simulations scenarios and return VaR, CVaR, and the loss distribution."""
    try:
        logger.info("[SIMULATION] Request received")
        logger.debug(
            f"[SIMULATION] Inputs → "
            f"PD count: {len(request.pd_values)}, "
            f"EAD count: {len(request.ead_values)}, "
            f"LGD: {request.lgd}, "
            f"Simulations: {request.num_simulations}, "
            f"Confidence: {request.confidence_level}, "
            f"Seed: {request.seed}"
        )

        logger.info("[SIMULATION] Running Monte Carlo simulation...")
        result = run_monte_carlo_simulation(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            num_simulations=request.num_simulations,
            confidence_level=request.confidence_level,
            seed=request.seed,
        )

        logger.info(
            f"[SIMULATION] Completed → "
            f"Expected Loss: {result['expected_loss']}, "
            f"VaR: {result['var']}, "
            f"CVaR: {result['cvar']}"
        )
        return result

    except Exception as e:
        logger.error(f"[SIMULATION] Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))