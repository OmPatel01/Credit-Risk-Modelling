import numpy as np
from typing import List, Dict


def _validate_inputs(pd_values: List[float], ead_values: List[float]):
    if len(pd_values) != len(ead_values):
        raise ValueError("pd_values and ead_values must have the same length")

    if len(pd_values) == 0:
        raise ValueError("Input lists cannot be empty")


def run_monte_carlo_simulation(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    num_simulations: int,
    confidence_level: float,
    seed: int,
) -> Dict:

    # ── Validation ─────────────────────────
    _validate_inputs(pd_values, ead_values)

    pd_array = np.array(pd_values)
    ead_array = np.array(ead_values)

    # ── Set random seed ────────────────────
    if seed is not None:
        np.random.seed(seed)

    # ── Simulation ─────────────────────────
    # Shape: (num_simulations, num_borrowers)
    random_matrix = np.random.rand(num_simulations, len(pd_array))

    # Default occurs if random < PD
    defaults = (random_matrix < pd_array).astype(int)

    # Loss = default × LGD × EAD
    losses = defaults * lgd * ead_array

    # Portfolio loss per simulation
    portfolio_losses = losses.sum(axis=1)

    # ── Metrics ────────────────────────────
    expected_loss = np.mean(portfolio_losses)
    unexpected_loss = np.std(portfolio_losses)

    # VaR
    var = np.percentile(portfolio_losses, confidence_level * 100)

    # CVaR (Expected Shortfall)
    tail_losses = portfolio_losses[portfolio_losses >= var]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

    # Additional stats
    min_loss = np.min(portfolio_losses)
    max_loss = np.max(portfolio_losses)
    percentile_5 = np.percentile(portfolio_losses, 5)
    percentile_95 = np.percentile(portfolio_losses, 95)

    # ── Round outputs ──────────────────────
    return {
        "num_simulations": num_simulations,
        "expected_loss": round(float(expected_loss), 2),
        "unexpected_loss": round(float(unexpected_loss), 2),
        "var": round(float(var), 2),
        "cvar": round(float(cvar), 2),
        "confidence_level": confidence_level,
        "min_loss": round(float(min_loss), 2),
        "max_loss": round(float(max_loss), 2),
        "percentile_5": round(float(percentile_5), 2),
        "percentile_95": round(float(percentile_95), 2),
    }