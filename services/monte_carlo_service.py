"""
services/monte_carlo_service.py
--------------------------------
Simulates portfolio credit losses using Monte Carlo methods to produce
risk metrics that deterministic ECL cannot capture (tail risk, volatility).

Why Monte Carlo for credit risk?
    ECL = PD × LGD × EAD gives the *expected* loss, but lenders also need to know
    the *distribution* of losses — especially the tail (worst-case scenarios).
    Monte Carlo simulates thousands of independent default/no-default outcomes
    per borrower and measures where losses concentrate in bad scenarios.

Key metrics produced:
    Expected Loss (EL)     — mean simulated loss; should approximately equal deterministic ECL
    Unexpected Loss (UL)   — standard deviation of losses; measures volatility around EL
    VaR (Value at Risk)    — loss threshold not exceeded in X% of simulations (e.g. 95th percentile)
    CVaR (Conditional VaR) — average loss in the simulations that DID exceed VaR (Expected Shortfall)

Simulation approach:
    For each simulation:
        - Draw a random U[0,1] for each borrower
        - If U < PD → borrower defaults in this simulation
        - Loss = default_indicator × LGD × EAD
        - Portfolio loss = sum across all borrowers
    After num_simulations runs, compute statistics over the portfolio loss distribution.

Independence assumption:
    Borrower defaults are treated as independent events. This is a simplification —
    in practice, defaults correlate during economic downturns (systematic risk).
    A correlated model (e.g. Vasicek) would require a shared factor, which is not
    implemented here.
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _validate_inputs(pd_values: List[float], ead_values: List[float], lgd: float):
    """Fail fast with a specific error message before allocating the large random matrix."""
    if len(pd_values) == 0:
        logger.error("[SIMULATION] Validation failed: empty inputs")
        raise ValueError("Input lists cannot be empty")

    if len(pd_values) != len(ead_values):
        logger.error("[SIMULATION] Validation failed: length mismatch")
        raise ValueError("pd_values and ead_values must have the same length")

    if any(p < 0 or p > 1 for p in pd_values):
        logger.error("[SIMULATION] Validation failed: PD out of range")
        raise ValueError("PD values must be between 0 and 1")

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
    """
    Run Monte Carlo simulation and return the full set of portfolio loss statistics.

    Matrix layout:
        random_matrix  shape: (num_simulations, num_borrowers)  — uniform random draws
        defaults       shape: (num_simulations, num_borrowers)  — binary default indicators
        losses         shape: (num_simulations, num_borrowers)  — per-borrower losses
        portfolio_losses shape: (num_simulations,)              — total portfolio loss per simulation

    The loss_distribution returned in the response is a random sample of 200 portfolio
    loss values (not all num_simulations) to keep the API response payload manageable
    while still giving the frontend enough data points to render a histogram.
    """
    logger.info("[SIMULATION] Monte Carlo simulation started")

    _validate_inputs(pd_values, ead_values, lgd)

    pd_array  = np.array(pd_values)
    ead_array = np.array(ead_values)

    logger.debug(
        f"[SIMULATION] Input summary → "
        f"Borrowers: {len(pd_array)}, "
        f"Simulations: {num_simulations}, "
        f"LGD: {lgd}, "
        f"Confidence: {confidence_level}"
    )

    # Seed before any random operation to guarantee reproducibility across calls
    # with the same inputs — important for regression testing and audit trails
    if seed is not None:
        np.random.seed(seed)
        logger.debug(f"[SIMULATION] Seed set to {seed}")

    logger.info("[SIMULATION] Generating random scenarios")

    # Vectorised simulation: each row is one scenario, each column is one borrower.
    # Broadcasting pd_array (shape: num_borrowers) across all rows at once is
    # significantly faster than a Python loop over num_simulations.
    random_matrix = np.random.rand(num_simulations, len(pd_array))
    defaults      = (random_matrix < pd_array).astype(int)  # 1 = default, 0 = no default

    losses            = defaults * lgd * ead_array  # element-wise: only defaulted borrowers incur loss
    portfolio_losses  = losses.sum(axis=1)          # total portfolio loss for each simulation
    portfolio_losses  = np.maximum(portfolio_losses, 0)  # numerical safety — losses cannot be negative

    # default_rate: fraction of simulations where at least one borrower defaulted
    # (useful for understanding portfolio-level default frequency)
    portfolio_defaults = (portfolio_losses > 0).astype(int)
    default_rate       = np.mean(portfolio_defaults)

    logger.debug(f"[SIMULATION] Sample losses → first 5: {portfolio_losses[:5]}")
    logger.info("[SIMULATION] Calculating risk metrics")

    expected_loss   = np.mean(portfolio_losses)
    unexpected_loss = np.std(portfolio_losses)

    # VaR: the loss value that is NOT exceeded in (confidence_level × 100)% of simulations.
    # e.g. at 95% confidence, VaR is the 95th percentile of portfolio_losses.
    var = np.percentile(portfolio_losses, confidence_level * 100)

    # CVaR (Expected Shortfall): average loss across the worst-case simulations that exceeded VaR.
    # Gives a better picture of tail severity than VaR alone.
    tail_losses = portfolio_losses[portfolio_losses >= var]
    cvar        = np.mean(tail_losses) if len(tail_losses) > 0 else var

    min_loss       = np.min(portfolio_losses)
    max_loss       = np.max(portfolio_losses)
    percentile_5   = np.percentile(portfolio_losses, 5)
    percentile_95  = np.percentile(portfolio_losses, 95)

    logger.info(
        f"[SIMULATION] Completed → "
        f"EL: {expected_loss:.2f}, "
        f"UL: {unexpected_loss:.2f}, "
        f"VaR: {var:.2f}, "
        f"CVaR: {cvar:.2f}"
    )

    # Downsample loss distribution to 200 points for frontend histogram rendering;
    # sending all num_simulations values (up to 500k) would bloat the HTTP response
    sample_size     = min(200, len(portfolio_losses))
    sample_indices  = np.random.choice(len(portfolio_losses), size=sample_size, replace=False)
    loss_distribution = portfolio_losses[sample_indices].tolist()

    return {
        "num_simulations":  num_simulations,
        "expected_loss":    round(float(expected_loss),   2),
        "unexpected_loss":  round(float(unexpected_loss), 2),
        "var":              round(float(var),             2),
        "cvar":             round(float(cvar),            2),
        "confidence_level": confidence_level,
        "min_loss":         round(float(min_loss),        2),
        "max_loss":         round(float(max_loss),        2),
        "percentile_5":     round(float(percentile_5),    2),
        "percentile_95":    round(float(percentile_95),   2),
        "default_rate":     round(float(default_rate),    4),
        "loss_distribution": loss_distribution,  # sampled subset for frontend charting
    }