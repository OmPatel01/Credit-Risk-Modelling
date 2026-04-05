from typing import List, Dict
import numpy as np

from services.risk_config import STRESS_SCENARIOS
from services.ecl_service import compute_ecl
from services.monte_carlo_service import run_monte_carlo_simulation


def run_stress_test(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    run_simulation: bool,
    num_simulations: int,
    seed: int,
) -> Dict:

    results = []

    base_total_ecl = None

    for key, scenario in STRESS_SCENARIOS.items():

        pd_multiplier = scenario["pd_multiplier"]
        lgd_override = scenario["lgd_override"]

        # ── Adjust PD and LGD ─────────────────────────
        stressed_pd = [min(p * pd_multiplier, 1.0) for p in pd_values]
        stressed_lgd = lgd_override if lgd_override is not None else lgd

        # ── Compute ECL ───────────────────────────────
        ecl_result = compute_ecl(
            pd_values=stressed_pd,
            lgd=stressed_lgd,
            ead_values=ead_values,
        )

        total_ecl = ecl_result["total_ecl"]
        mean_ecl = ecl_result["mean_ecl"]

        # Store base for comparison
        if key == "base":
            base_total_ecl = total_ecl

        # ── % Change vs Base ──────────────────────────
        ecl_change_pct = None
        if base_total_ecl is not None and key != "base":
            ecl_change_pct = round(
                ((total_ecl - base_total_ecl) / base_total_ecl) * 100,
                2,
            )

        # ── Optional Monte Carlo ──────────────────────
        simulation_result = None
        if run_simulation:
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
            "ecl_change_pct": ecl_change_pct,
            "simulation": simulation_result,
        })

    return {
        "scenarios": results
    }