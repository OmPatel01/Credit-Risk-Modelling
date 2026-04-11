"""
services/sensitivity_service.py
--------------------------------
Sensitivity analysis: measures how much ECL changes when PD or LGD shift by small amounts.

Difference from stress testing:
    Stress testing   → applies predefined named scenarios (Base / Mild / Severe)
                        with large, realistic shocks to simulate economic downturns.
    Sensitivity analysis → sweeps a range of small shifts (e.g. ±10%, ±20%)
                        to understand model sensitivity and which driver (PD vs LGD)
                        has the bigger impact on ECL for a given portfolio.

The result is a table of (driver, shift_amount, resulting_ECL, % change vs baseline)
that risk managers use to understand which assumptions the portfolio is most sensitive to.

Default shifts come from risk_config.py and can be overridden by the caller.
"""

from typing import List, Dict, Optional
import logging

from services.ecl_service import compute_ecl
from services.monte_carlo_service import run_monte_carlo_simulation
from services.risk_config import (
    SENSITIVITY_PD_SHIFTS,
    SENSITIVITY_LGD_SHIFTS,
)

logger = logging.getLogger(__name__)


def run_sensitivity_analysis(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    pd_shifts: Optional[List[float]],
    lgd_shifts: Optional[List[float]],
    run_simulation: bool,
    num_simulations: int,
    seed: int,
) -> Dict:
    """
    Sweep PD and LGD shifts, compute ECL for each, and report change vs baseline.

    PD shifts are applied as relative multipliers: adjusted_PD = PD × (1 + shift).
        e.g. shift=+0.20 means "PDs are 20% higher than current estimates"
    LGD shifts are applied as absolute additions: adjusted_LGD = LGD + shift.
        e.g. shift=+0.10 means "LGD is 10 percentage points higher than assumed"
    This asymmetry is intentional — PD is a rate (scales naturally) while LGD is
    an absolute fraction (additive shifts are easier to interpret for underwriters).

    Both PD and LGD are clamped to [0, 1] after adjustment to stay in valid range.
    """
    logger.info("[SENSITIVITY] Sensitivity analysis started")

    # Fall back to config defaults if caller did not supply custom shifts
    pd_shifts  = pd_shifts  or SENSITIVITY_PD_SHIFTS
    lgd_shifts = lgd_shifts or SENSITIVITY_LGD_SHIFTS

    logger.debug(
        f"[SENSITIVITY] Input summary → "
        f"Borrowers: {len(pd_values)}, "
        f"LGD: {lgd}, "
        f"PD shifts: {pd_shifts}, "
        f"LGD shifts: {lgd_shifts}, "
        f"Run simulation: {run_simulation}"
    )

    # Baseline ECL with the unmodified PD and LGD — all results are expressed relative to this
    base         = compute_ecl(pd_values, lgd, ead_values)
    baseline_ecl = base["total_ecl"]

    logger.info(f"[SENSITIVITY] Baseline ECL: {baseline_ecl}")

    results = []

    # ── PD sensitivity — how much does ECL change if our PD estimates are off by X%?
    for shift in pd_shifts:
        logger.info(f"[SENSITIVITY] PD shift: {shift * 100:.0f}%")

        # Clamp to [0, 1] — PD cannot exceed 100% or drop below 0%
        adjusted_pd = [min(max(p * (1 + shift), 0), 1) for p in pd_values]

        ecl_result = compute_ecl(adjusted_pd, lgd, ead_values)
        total_ecl  = ecl_result["total_ecl"]

        change_pct = round(
            ((total_ecl - baseline_ecl) / baseline_ecl) * 100,
            2
        )

        logger.info(f"[SENSITIVITY] PD Impact → ECL: {total_ecl}, Change: {change_pct}%")

        simulation_result = None
        if run_simulation:
            logger.debug("[SENSITIVITY] Running simulation for PD shift")
            simulation_result = run_monte_carlo_simulation(
                pd_values=adjusted_pd,
                lgd=lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,
                seed=seed,
            )

        results.append({
            "driver":       "PD",
            "shift":        shift,
            "shift_label":  f"{int(shift * 100)}%",  # human-readable label for frontend tables
            "total_ecl":    total_ecl,
            "ecl_change_pct": change_pct,
            "simulation":   simulation_result,
        })

    # ── LGD sensitivity — how much does ECL change if our recovery assumptions are wrong?
    for shift in lgd_shifts:
        logger.info(f"[SENSITIVITY] LGD shift: {shift * 100:.0f}%")

        adjusted_lgd = min(max(lgd + shift, 0), 1)  # absolute shift, clamped to [0, 1]

        ecl_result = compute_ecl(pd_values, adjusted_lgd, ead_values)
        total_ecl  = ecl_result["total_ecl"]

        change_pct = round(
            ((total_ecl - baseline_ecl) / baseline_ecl) * 100,
            2
        )

        logger.info(f"[SENSITIVITY] LGD Impact → ECL: {total_ecl}, Change: {change_pct}%")

        simulation_result = None
        if run_simulation:
            logger.debug("[SENSITIVITY] Running simulation for LGD shift")
            simulation_result = run_monte_carlo_simulation(
                pd_values=pd_values,
                lgd=adjusted_lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,
                seed=seed,
            )

        results.append({
            "driver":         "LGD",
            "shift":          shift,
            "shift_label":    f"{int(shift * 100)}%",
            "total_ecl":      total_ecl,
            "ecl_change_pct": change_pct,
            "simulation":     simulation_result,
        })

    logger.info("[SENSITIVITY] Sensitivity analysis completed")

    return {
        "baseline_ecl": baseline_ecl,
        "results":      results,
    }