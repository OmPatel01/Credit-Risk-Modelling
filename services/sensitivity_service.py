from typing import List, Dict, Optional

from services.ecl_service import compute_ecl
from services.monte_carlo_service import run_monte_carlo_simulation
from services.risk_config import (
    SENSITIVITY_PD_SHIFTS,
    SENSITIVITY_LGD_SHIFTS,
)


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

    # ── Defaults ─────────────────────────
    pd_shifts = pd_shifts or SENSITIVITY_PD_SHIFTS
    lgd_shifts = lgd_shifts or SENSITIVITY_LGD_SHIFTS

    # ── Baseline ECL ─────────────────────
    base = compute_ecl(pd_values, lgd, ead_values)
    baseline_ecl = base["total_ecl"]

    results = []

    # ── PD Sensitivity ───────────────────
    for shift in pd_shifts:

        adjusted_pd = [min(max(p * (1 + shift), 0), 1) for p in pd_values]

        ecl_result = compute_ecl(adjusted_pd, lgd, ead_values)
        total_ecl = ecl_result["total_ecl"]

        change_pct = round(
            ((total_ecl - baseline_ecl) / baseline_ecl) * 100,
            2
        )

        simulation_result = None
        if run_simulation:
            simulation_result = run_monte_carlo_simulation(
                pd_values=adjusted_pd,
                lgd=lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,
                seed=seed,
            )

        results.append({
            "driver": "PD",
            "shift": shift,
            "shift_label": f"{int(shift * 100)}%",
            "total_ecl": total_ecl,
            "ecl_change_pct": change_pct,
            "simulation": simulation_result,
        })

    # ── LGD Sensitivity ──────────────────
    for shift in lgd_shifts:

        adjusted_lgd = min(max(lgd + shift, 0), 1)

        ecl_result = compute_ecl(pd_values, adjusted_lgd, ead_values)
        total_ecl = ecl_result["total_ecl"]

        change_pct = round(
            ((total_ecl - baseline_ecl) / baseline_ecl) * 100,
            2
        )

        simulation_result = None
        if run_simulation:
            simulation_result = run_monte_carlo_simulation(
                pd_values=pd_values,
                lgd=adjusted_lgd,
                ead_values=ead_values,
                num_simulations=num_simulations,
                confidence_level=0.95,
                seed=seed,
            )

        results.append({
            "driver": "LGD",
            "shift": shift,
            "shift_label": f"{int(shift * 100)}%",
            "total_ecl": total_ecl,
            "ecl_change_pct": change_pct,
            "simulation": simulation_result,
        })

    return {
        "baseline_ecl": baseline_ecl,
        "results": results,
    }