"""
services/ecl_service.py
-----------------------
Computes Expected Credit Loss (ECL) at both individual borrower and portfolio level.

ECL formula (IFRS 9 / Basel): ECL = PD × LGD × EAD
    PD  — Probability of Default (output of scorecard or XGBoost model)
    LGD — Loss Given Default (what fraction of the exposure is lost if the borrower defaults)
    EAD — Exposure at Default (outstanding balance at the time of default)

The service accepts parallel arrays (one entry per borrower) and a scalar LGD,
which is the typical structure for a homogeneous portfolio segment where all borrowers
share the same product-level loss assumption.

Optional segment_labels allow the caller to group borrowers (e.g. by risk bucket or region)
and receive a per-segment ECL breakdown alongside the portfolio totals.
"""

from typing import List, Dict, Optional
import pandas as pd


def _validate_inputs(pd_values: List[float], ead_values: List[float]):
    """Guard against mismatched or empty arrays before any computation."""
    if len(pd_values) != len(ead_values):
        raise ValueError("pd_values and ead_values must have the same length")

    if len(pd_values) == 0:
        raise ValueError("Input lists cannot be empty")


def _compute_individual_ecl(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
) -> List[float]:
    """Apply ECL = PD × LGD × EAD to each borrower and round to 2 decimal places (financial precision)."""
    ecl_values = []
    for pd, ead in zip(pd_values, ead_values):
        ecl = pd * lgd * ead
        ecl_values.append(round(ecl, 2))
    return ecl_values


def _compute_segment_ecl(
    ecl_values: List[float],
    segment_labels: List[str],
) -> List[Dict]:
    """
    Aggregate individual ECL values by segment label.

    Groups by segment, computes count, total ECL, and average ECL per group.
    Returns a list of dicts sorted alphabetically by segment name —
    consistent ordering matters for downstream dashboards.
    """
    df = pd.DataFrame({
        "segment": segment_labels,
        "ecl": ecl_values
    })

    grouped = (
        df.groupby("segment")
        .agg(
            count=("ecl", "count"),
            total_ecl=("ecl", "sum"),
            avg_ecl=("ecl", "mean"),
        )
        .reset_index()
    )

    grouped["total_ecl"] = grouped["total_ecl"].round(2)
    grouped["avg_ecl"]   = grouped["avg_ecl"].round(2)

    return grouped.to_dict(orient="records")


def compute_ecl(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    segment_labels: Optional[List[str]] = None,
) -> Dict:
    """
    Orchestrate full ECL computation: validate → individual ECL → portfolio totals → optional segment breakdown.

    Returns a dict with individual_ecl list, total_ecl, mean_ecl, and optionally segment_ecl.
    segment_ecl is None when no labels are provided, so callers can check its presence.
    """
    _validate_inputs(pd_values, ead_values)

    # segment_labels, if provided, must align 1:1 with borrowers
    if segment_labels is not None and len(segment_labels) != len(pd_values):
        raise ValueError("segment_labels must match length of pd_values")

    individual_ecl = _compute_individual_ecl(pd_values, lgd, ead_values)

    total_ecl = round(sum(individual_ecl), 2)
    mean_ecl  = round(total_ecl / len(individual_ecl), 2)

    segment_ecl = None
    if segment_labels is not None:
        segment_ecl = _compute_segment_ecl(individual_ecl, segment_labels)

    return {
        "individual_ecl": individual_ecl,
        "total_ecl":      total_ecl,
        "mean_ecl":       mean_ecl,
        "segment_ecl":    segment_ecl,
    }