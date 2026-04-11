"""
services/scenario_service.py
-----------------------------
Stress testing: evaluates how the portfolio's ECL changes under predefined adverse scenarios.

What stress testing answers:
    "If economic conditions deteriorate, how much worse does our expected loss get?"

Each scenario applies a PD multiplier (e.g. ×1.8 for Severe) and optionally overrides
the LGD. The stressed ECL is then compared against the Base scenario to produce a
percentage change — the key metric regulators and risk committees care about.

Scenarios are defined in services/risk_config.py and can be changed there without
touching this service logic. Adding a new scenario requires only a new entry in
STRESS_SCENARIOS — no code changes here.

Optional Monte Carlo:
    When run_simulation=True, each scenario also runs a full Monte Carlo simulation
    using the stressed PD/LGD values. This reveals not just the expected loss under
    stress, but the tail risk (VaR, CVaR) that could materialise in the worst cases.
"""

from typing import List, Dict
import numpy as np
import logging

from services.risk_config import STRESS_SCENARIOS
from services.ecl_service import compute_ecl
from services.monte_carlo_service import run_monte_carlo_simulation

logger = logging.getLogger(__name__)


def run_stress_test(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    run_simulation: bool,
    num_simulations: int,
    seed: int,
) -> Dict:
    """
    Apply every scenario in STRESS_SCENARIOS and return ECL results with % change vs Base.

    The Base scenario (pd_multiplier=1.0, no LGD override) must appear in STRESS_SCENARIOS
    and must be keyed as "base". All other scenarios are compared against it.
    The function processes scenarios in dict iteration order, so Base should be defined first.
    """
    logger.info("[STRESS] Stress test started")

    results        = []
    base_total_ecl = None  # set on first "base" scenario; used for % change calculation in all others

    logger.debug(
        f"[STRESS] Input summary → "
        f"Borrowers: {len(pd_values)}, "
        f"LGD: {lgd}, "
        f"Run simulation: {run_simulation}"
    )

    for key, scenario in STRESS_SCENARIOS.items():

        pd_multiplier = scenario["pd_multiplier"]
        lgd_override  = scenario["lgd_override"]

        logger.info(f"[STRESS] Running scenario: {scenario['label']}")

        # Apply PD stress: multiply each borrower's PD by the scenario multiplier,
        # clipping at 1.0 since probability cannot exceed 100%
        stressed_pd  = [min(p * pd_multiplier, 1.0) for p in pd_values]

        # Use overridden LGD if the scenario specifies one; otherwise keep the input LGD
        # (Base scenario has lgd_override=None to use the caller's value unchanged)
        stressed_lgd = lgd_override if lgd_override is not None else lgd

        logger.debug(
            f"[STRESS] Scenario params → "
            f"PD multiplier: {pd_multiplier}, "
            f"LGD used: {stressed_lgd}"
        )

        ecl_result = compute_ecl(
            pd_values=stressed_pd,
            lgd=stressed_lgd,
            ead_values=ead_values,
        )

        total_ecl = ecl_result["total_ecl"]
        mean_ecl  = ecl_result["mean_ecl"]
        avg_pd    = sum(stressed_pd) / len(stressed_pd)

        logger.info(f"[STRESS] ECL → Total: {total_ecl}, Mean: {mean_ecl}")

        # Anchor the baseline so all subsequent scenarios can report their relative change
        if key == "base":
            base_total_ecl = total_ecl
            logger.debug(f"[STRESS] Base scenario set → {base_total_ecl}")

        # % change is only meaningful for non-base scenarios; None signals "N/A" to the frontend
        ecl_change_pct = None
        if base_total_ecl is not None and key != "base":
            ecl_change_pct = round(
                ((total_ecl - base_total_ecl) / base_total_ecl) * 100,
                2,
            )
            logger.info(f"[STRESS] Change vs Base → {ecl_change_pct}%")

        simulation_result = None
        if run_simulation:
            logger.info("[STRESS] Running simulation for scenario")
            simulation_result = run_monte_carlo_simulation(
                pd_values=stressed_pd,
                lgd=stressed_lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,  # hardcoded for stress tests; sensitivity uses caller-supplied level
                seed=seed,
            )

        results.append({
            "scenario":       scenario["label"],
            "pd_multiplier":  pd_multiplier,
            "lgd_used":       stressed_lgd,
            "total_ecl":      total_ecl,
            "mean_ecl":       mean_ecl,
            "avg_pd":         avg_pd,
            "ecl_change_pct": ecl_change_pct,
            "simulation":     simulation_result,
        })

    logger.info("[STRESS] Stress test completed")

    return {"scenarios": results}