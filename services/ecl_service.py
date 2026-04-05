from typing import List, Dict, Optional
import pandas as pd


def _validate_inputs(pd_values: List[float], ead_values: List[float]):
    if len(pd_values) != len(ead_values):
        raise ValueError("pd_values and ead_values must have the same length")

    if len(pd_values) == 0:
        raise ValueError("Input lists cannot be empty")


def _compute_individual_ecl(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
) -> List[float]:

    ecl_values = []

    for pd, ead in zip(pd_values, ead_values):
        ecl = pd * lgd * ead
        ecl_values.append(round(ecl, 2))  # financial rounding

    return ecl_values


def _compute_segment_ecl(
    ecl_values: List[float],
    segment_labels: List[str],
) -> List[Dict]:

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

    # Round results
    grouped["total_ecl"] = grouped["total_ecl"].round(2)
    grouped["avg_ecl"] = grouped["avg_ecl"].round(2)

    return grouped.to_dict(orient="records")


def compute_ecl(
    pd_values: List[float],
    lgd: float,
    ead_values: List[float],
    segment_labels: Optional[List[str]] = None,
) -> Dict:

    # ── Validation ─────────────────────────
    _validate_inputs(pd_values, ead_values)

    if segment_labels is not None and len(segment_labels) != len(pd_values):
        raise ValueError("segment_labels must match length of pd_values")

    # ── Individual ECL ─────────────────────
    individual_ecl = _compute_individual_ecl(
        pd_values,
        lgd,
        ead_values
    )

    # ── Portfolio metrics ──────────────────
    total_ecl = round(sum(individual_ecl), 2)
    mean_ecl = round(total_ecl / len(individual_ecl), 2)

    # ── Segment-level ECL (optional) ───────
    segment_ecl = None
    if segment_labels is not None:
        segment_ecl = _compute_segment_ecl(
            individual_ecl,
            segment_labels
        )

    return {
        "individual_ecl": individual_ecl,
        "total_ecl": total_ecl,
        "mean_ecl": mean_ecl,
        "segment_ecl": segment_ecl,
    }