#   app/routes/stress.py
from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import (
    StressTestRequest,
    StressTestResponse,
)
from services.scenario_service import run_stress_test

router = APIRouter()

# 🔥 Create logger
logger = logging.getLogger(__name__)


@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    tags=["Risk Analytics"],
    summary="Scenario Analysis (Stress Testing)",
)
def stress_test(request: StressTestRequest):
    try:
        # 🔹 Log request received
        logger.info("[STRESS] Request received")

        # 🔹 Log input summary (avoid full data dump)
        logger.debug(
            f"[STRESS] Inputs → "
            f"PD count: {len(request.pd_values)}, "
            f"EAD count: {len(request.ead_values)}, "
            f"LGD: {request.lgd}, "
            f"Run simulation: {request.run_simulation}, "
            f"Simulations: {request.num_simulations}, "
            f"Seed: {request.seed}"
        )

        # 🔹 Run stress test
        logger.info("[STRESS] Running stress scenario analysis...")
        result = run_stress_test(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            run_simulation=request.run_simulation,
            num_simulations=request.num_simulations,
            seed=request.seed,
        )

        # 🔹 Log result summary (safe fields only)
        logger.info(f"[STRESS] Completed → scenarios: {len(result['scenarios'])}")

        return result

    except Exception as e:
        # 🔴 Proper error logging
        logger.error(
            f"[STRESS] Error occurred: {str(e)}",
            exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(e))