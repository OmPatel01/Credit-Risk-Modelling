# simulation.py
from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import (
    SimulationRequest,
    SimulationResponse,
)
from services.monte_carlo_service import run_monte_carlo_simulation

router = APIRouter()

# 🔥 Create logger
logger = logging.getLogger(__name__)


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["Risk Analytics"],
    summary="Monte Carlo Simulation",
)
def simulate_loss(request: SimulationRequest):
    try:
        # 🔹 Log request received
        logger.info("[SIMULATION] Request received")

        # 🔹 Log input summary (NOT full data)
        logger.debug(
            f"[SIMULATION] Inputs → "
            f"PD count: {len(request.pd_values)}, "
            f"EAD count: {len(request.ead_values)}, "
            f"LGD: {request.lgd}, "
            f"Simulations: {request.num_simulations}, "
            f"Confidence: {request.confidence_level}, "
            f"Seed: {request.seed}"
        )

        # 🔹 Run simulation
        logger.info("[SIMULATION] Running Monte Carlo simulation...")
        result = run_monte_carlo_simulation(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            num_simulations=request.num_simulations,
            confidence_level=request.confidence_level,
            seed=request.seed,
        )

        # 🔹 Log result summary
        logger.info(
            f"[SIMULATION] Completed → "
            f"Expected Loss: {result['expected_loss']}, "
            f"VaR: {result['var']}, "
            f"CVaR: {result['cvar']}"
        )
                

        return result

    except Exception as e:
        # 🔴 Proper error logging
        logger.error(
            f"[SIMULATION] Error occurred: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))