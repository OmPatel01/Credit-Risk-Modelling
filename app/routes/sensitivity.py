"""
app/routes/sensitivity.py
--------------------------
HTTP route for the sensitivity analysis endpoint.

Delegates to sensitivity_service.run_sensitivity_analysis.
Unlike stress testing (which uses fixed named scenarios), sensitivity analysis
accepts custom shift arrays from the caller — useful for ad-hoc "what if PD
were 5% higher?" exploration without modifying config files.
"""

from fastapi import APIRouter, HTTPException
import logging

from app.schemas.risk_schemas import SensitivityRequest, SensitivityResponse
from services.sensitivity_service import run_sensitivity_analysis

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    tags=["Risk Analytics"],
    summary="Sensitivity Analysis",
)
def sensitivity_analysis(request: SensitivityRequest):
    """Sweep PD and LGD shifts and return ECL change at each shift level relative to the unmodified baseline."""
    try:
        logger.info("[SENSITIVITY] Request received")
        logger.debug(
            f"[SENSITIVITY] Inputs → "
            f"PD count: {len(request.pd_values)}, "
            f"EAD count: {len(request.ead_values)}, "
            f"LGD: {request.lgd}, "
            f"PD shifts: {request.pd_shifts}, "
            f"LGD shifts: {request.lgd_shifts}, "
            f"Run simulation: {request.run_simulation}, "
            f"Simulations: {request.num_simulations}, "
            f"Seed: {request.seed}"
        )

        logger.info("[SENSITIVITY] Running sensitivity analysis...")
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

        logger.info(f"[SENSITIVITY] Completed → Results: {len(result['results'])}")
        return result

    except Exception as e:
        logger.error(f"[SENSITIVITY] Error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))