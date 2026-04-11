"""
app/routes/stress.py
--------------------
HTTP route for the stress testing (scenario analysis) endpoint.

Delegates to scenario_service.run_stress_test.
Stress scenarios are predefined in services/risk_config.py — callers cannot
define custom scenarios via this API; they can only configure PD/LGD inputs
and whether to also run Monte Carlo under each scenario.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import StressTestRequest, StressTestResponse
from services.scenario_service import run_stress_test

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    tags=["Risk Analytics"],
    summary="Scenario Analysis (Stress Testing)",
)
def stress_test(request: StressTestRequest):
    """Apply predefined stress scenarios (Base / Mild / Severe) to the portfolio and return ECL changes."""
    try:
        logger.info("[STRESS] Request received")
        logger.debug(
            f"[STRESS] Inputs → "
            f"PD count: {len(request.pd_values)}, "
            f"EAD count: {len(request.ead_values)}, "
            f"LGD: {request.lgd}, "
            f"Run simulation: {request.run_simulation}, "
            f"Simulations: {request.num_simulations}, "
            f"Seed: {request.seed}"
        )

        logger.info("[STRESS] Running stress scenario analysis...")
        result = run_stress_test(
            pd_values=request.pd_values,
            lgd=request.lgd,
            ead_values=request.ead_values,
            run_simulation=request.run_simulation,
            num_simulations=request.num_simulations,
            seed=request.seed,
        )

        logger.info(f"[STRESS] Completed → scenarios: {len(result['scenarios'])}")
        return result

    except Exception as e:
        logger.error(f"[STRESS] Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))