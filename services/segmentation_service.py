"""
services/segmentation_service.py
---------------------------------
Groups borrowers into risk buckets based on their predicted PD values.

Two segmentation strategies are supported:

    quantile  — divides borrowers into equal-sized groups by PD rank.
                Each bucket has roughly the same number of borrowers.
                Useful when you want evenly distributed monitoring groups
                or when the PD distribution is highly skewed.

    fixed     — assigns borrowers to buckets using predefined PD thresholds.
                Useful when thresholds carry regulatory or policy meaning
                (e.g. "any borrower above PD=0.30 goes to collections").

The segment labels (A–E or custom) are ordered from best to worst risk.
Both strategies return the same output structure so downstream consumers
don't need to know which strategy was used.
"""

import pandas as pd
from typing import List, Dict, Optional

from services.risk_config import (
    DEFAULT_PD_THRESHOLDS,
    DEFAULT_SEGMENT_LABELS,
    DEFAULT_NUM_QUANTILES,
)


def _validate_inputs(pd_values: List[float]):
    """Ensure the input list is non-empty before any computation."""
    if len(pd_values) == 0:
        raise ValueError("pd_values cannot be empty")


def _fixed_segmentation(
    pd_values: List[float],
    thresholds: List[float],
    labels: List[str]
) -> List[str]:
    """
    Assign each borrower to a bucket using hardcoded PD cut-points.

    The last label catches all borrowers whose PD exceeds every threshold.
    Example with DEFAULT_PD_THRESHOLDS=[0.05, 0.15, 0.30, 0.50]:
        PD < 0.05  → labels[0] (A, lowest risk)
        PD < 0.15  → labels[1] (B)
        PD < 0.30  → labels[2] (C)
        PD < 0.50  → labels[3] (D)
        PD >= 0.50 → labels[4] (E, highest risk)
    """
    if len(labels) != len(thresholds) + 1:
        raise ValueError("labels must be len(thresholds) + 1")

    buckets = []
    for pd in pd_values:
        assigned = False
        for i, threshold in enumerate(thresholds):
            if pd < threshold:
                buckets.append(labels[i])
                assigned = True
                break
        if not assigned:
            buckets.append(labels[-1])  # PD exceeded all thresholds → worst bucket

    return buckets


def _quantile_segmentation(
    pd_values: List[float],
    num_quantiles: int,
    labels: List[str]
) -> List[str]:
    """
    Assign borrowers to equal-frequency buckets using pandas qcut.

    duplicates="drop" handles edge cases where many borrowers share the exact
    same PD (e.g. a model that outputs many zeroes), which would otherwise cause
    qcut to fail with "bins are not unique" — it silently merges identical boundaries.
    """
    if len(labels) != num_quantiles:
        raise ValueError("labels must match num_quantiles")

    series = pd.Series(pd_values)

    buckets = pd.qcut(
        series,
        q=num_quantiles,
        labels=labels,
        duplicates="drop",  # merge duplicate bin edges instead of raising an error
    )

    return buckets.astype(str).tolist()


def _compute_summary(
    pd_values: List[float],
    buckets: List[str]
) -> List[Dict]:
    """
    Compute count and average PD for each bucket — gives a profile of each risk group.

    Sorted alphabetically by bucket label (A → E) so the response is consistently
    ordered from low to high risk regardless of the input PD distribution.
    """
    df = pd.DataFrame({"pd": pd_values, "bucket": buckets})

    summary = (
        df.groupby("bucket")
        .agg(
            count=("pd", "count"),
            avg_pd=("pd", "mean"),
        )
        .reset_index()
        .sort_values("bucket")
    )

    return summary.to_dict(orient="records")


def perform_segmentation(
    pd_values: List[float],
    method: str,
    num_quantiles: Optional[int],
    thresholds: Optional[List[float]],
    labels: Optional[List[str]],
) -> Dict:
    """
    Route to the correct segmentation strategy and return bucket assignments + summary stats.

    Falls back to config defaults when the caller omits optional parameters.
    Raises ValueError for unsupported method strings (caught and re-raised as HTTP 400).
    """
    _validate_inputs(pd_values)

    if method == "quantile":
        num_quantiles = num_quantiles or DEFAULT_NUM_QUANTILES
        labels        = labels or DEFAULT_SEGMENT_LABELS[:num_quantiles]

        buckets = _quantile_segmentation(pd_values, num_quantiles, labels)

    elif method == "fixed":
        thresholds = thresholds or DEFAULT_PD_THRESHOLDS
        labels     = labels or DEFAULT_SEGMENT_LABELS

        buckets = _fixed_segmentation(pd_values, thresholds, labels)

    else:
        raise ValueError(f"Invalid segmentation method: '{method}'. Must be 'quantile' or 'fixed'.")

    summary = _compute_summary(pd_values, buckets)

    return {
        "bucket_assignments": buckets,
        "summary":            summary,
    }