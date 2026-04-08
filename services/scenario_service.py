#   services/scenario_service.py
from typing import List, Dict
import numpy as np
import logging

from services.risk_config import STRESS_SCENARIOS
from services.ecl_service import compute_ecl
from services.monte_carlo_service import run_monte_carlo_simulation

# 🔥 Logger
logger = logging.getLogger(__name__)


def run_stress_test(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    run_simulation: bool,
    num_simulations: int,
    seed: int,
) -> Dict:

    logger.info("[STRESS] Stress test started")

    results = []
    base_total_ecl = None

    logger.debug(
        f"[STRESS] Input summary → "
        f"Borrowers: {len(pd_values)}, "
        f"LGD: {lgd}, "
        f"Run simulation: {run_simulation}"
    )

    for key, scenario in STRESS_SCENARIOS.items():

        pd_multiplier = scenario["pd_multiplier"]
        lgd_override = scenario["lgd_override"]

        logger.info(f"[STRESS] Running scenario: {scenario['label']}")

        # ── Adjust PD and LGD ─────────────────────────
        stressed_pd = [min(p * pd_multiplier, 1.0) for p in pd_values]
        stressed_lgd = lgd_override if lgd_override is not None else lgd

        logger.debug(
            f"[STRESS] Scenario params → "
            f"PD multiplier: {pd_multiplier}, "
            f"LGD used: {stressed_lgd}"
        )

        # ── Compute ECL ───────────────────────────────
        ecl_result = compute_ecl(
            pd_values=stressed_pd,
            lgd=stressed_lgd,
            ead_values=ead_values,
        )

        total_ecl = ecl_result["total_ecl"]
        mean_ecl = ecl_result["mean_ecl"]
        avg_pd = sum(stressed_pd) / len(stressed_pd)

        logger.info(
            f"[STRESS] ECL → Total: {total_ecl}, Mean: {mean_ecl}"
        )

        # Store base for comparison
        if key == "base":
            base_total_ecl = total_ecl
            logger.debug(f"[STRESS] Base scenario set → {base_total_ecl}")

        # ── % Change vs Base ──────────────────────────
        ecl_change_pct = None
        if base_total_ecl is not None and key != "base":
            ecl_change_pct = round(
                ((total_ecl - base_total_ecl) / base_total_ecl) * 100,
                2,
            )

            logger.info(
                f"[STRESS] Change vs Base → {ecl_change_pct}%"
            )

        # ── Optional Monte Carlo ──────────────────────
        simulation_result = None
        if run_simulation:
            logger.info("[STRESS] Running simulation for scenario")

            simulation_result = run_monte_carlo_simulation(
                pd_values=stressed_pd,
                lgd=stressed_lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,
                seed=seed,
            )

        results.append({
            "scenario": scenario["label"],
            "pd_multiplier": pd_multiplier,
            "lgd_used": stressed_lgd,
            "total_ecl": total_ecl,
            "mean_ecl": mean_ecl,
            "avg_pd": avg_pd,
            "ecl_change_pct": ecl_change_pct,
            "simulation": simulation_result,
        })

    logger.info("[STRESS] Stress test completed")

    return {
        "scenarios": results
    }