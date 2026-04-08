# monte_carlo_service.py
import numpy as np
import logging
from typing import List, Dict

# 🔥 Create logger
logger = logging.getLogger(__name__)


def _validate_inputs(pd_values: List[float], ead_values: List[float], lgd: float):
    if len(pd_values) == 0:
        logger.error("[SIMULATION] Validation failed: empty inputs")
        raise ValueError("Input lists cannot be empty")

    if len(pd_values) != len(ead_values):
        logger.error("[SIMULATION] Validation failed: length mismatch")
        raise ValueError("pd_values and ead_values must have the same length")
    
    if any(p < 0 or p > 1 for p in pd_values):
        logger.error("[SIMULATION] Validation failed: PD out of range")
        raise ValueError("PD values must be between 0 and 1")

    # Validate EAD
    if any(e < 0 for e in ead_values):
        logger.error("[SIMULATION] Validation failed: negative EAD")
        raise ValueError("EAD values must be non-negative")
    
    if lgd < 0 or lgd > 1:
        logger.error("[SIMULATION] Validation failed: LGD out of range")
        raise ValueError("LGD must be between 0 and 1")

def run_monte_carlo_simulation(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    num_simulations: int,
    confidence_level: float,
    seed: int,
) -> Dict:

    logger.info("[SIMULATION] Monte Carlo simulation started")

    # ── Validation ─────────────────────────
    _validate_inputs(pd_values, ead_values, lgd)

    pd_array = np.array(pd_values)
    ead_array = np.array(ead_values)

    logger.debug(
        f"[SIMULATION] Input summary → "
        f"Borrowers: {len(pd_array)}, "
        f"Simulations: {num_simulations}, "
        f"LGD: {lgd}, "
        f"Confidence: {confidence_level}"
    )

    # ── Set random seed ────────────────────
    if seed is not None:
        np.random.seed(seed)
        logger.debug(f"[SIMULATION] Seed set to {seed}")

    # ── Simulation ─────────────────────────
    logger.info("[SIMULATION] Generating random scenarios")

    random_matrix = np.random.rand(num_simulations, len(pd_array))
    defaults = (random_matrix < pd_array).astype(int)

    losses = defaults * lgd * ead_array
    portfolio_losses = losses.sum(axis=1)

    # Ensure no negative losses
    portfolio_losses = np.maximum(portfolio_losses, 0)

    # NOW compute default rate
    portfolio_defaults = (portfolio_losses > 0).astype(int)
    default_rate = np.mean(portfolio_defaults)

    logger.debug(
        f"[SIMULATION] Sample losses → first 5: {portfolio_losses[:5]}"
    )

    # ── Metrics ────────────────────────────
    logger.info("[SIMULATION] Calculating risk metrics")

    expected_loss = np.mean(portfolio_losses)
    unexpected_loss = np.std(portfolio_losses)

    var = np.percentile(portfolio_losses, confidence_level * 100)

    tail_losses = portfolio_losses[portfolio_losses >= var]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var

    min_loss = np.min(portfolio_losses)
    max_loss = np.max(portfolio_losses)
    percentile_5 = np.percentile(portfolio_losses, 5)
    percentile_95 = np.percentile(portfolio_losses, 95)

    # ── Final log summary ──────────────────
    logger.info(
        f"[SIMULATION] Completed → "
        f"EL: {expected_loss:.2f}, "
        f"UL: {unexpected_loss:.2f}, "
        f"VaR: {var:.2f}, "
        f"CVaR: {cvar:.2f}"
    )

    # Limit distribution size for frontend
    sample_size = min(200, len(portfolio_losses))
    sample_indices = np.random.choice(len(portfolio_losses), size=sample_size, replace=False)
    loss_distribution = portfolio_losses[sample_indices].tolist()
    # ── Return ────────────────────────────
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
        "default_rate": round(float(default_rate), 4),
        "loss_distribution": loss_distribution
    }