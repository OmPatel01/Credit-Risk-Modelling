import pandas as pd
from typing import List, Dict, Optional

from services.risk_config import (
    DEFAULT_PD_THRESHOLDS,
    DEFAULT_SEGMENT_LABELS,
    DEFAULT_NUM_QUANTILES,
)


def _validate_inputs(pd_values: List[float]):
    if len(pd_values) == 0:
        raise ValueError("pd_values cannot be empty")


def _fixed_segmentation(
    pd_values: List[float],
    thresholds: List[float],
    labels: List[str]
) -> List[str]:

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
            buckets.append(labels[-1])

    return buckets


def _quantile_segmentation(
    pd_values: List[float],
    num_quantiles: int,
    labels: List[str]
) -> List[str]:

    if len(labels) != num_quantiles:
        raise ValueError("labels must match num_quantiles")

    series = pd.Series(pd_values)

    buckets = pd.qcut(
        series,
        q=num_quantiles,
        labels=labels,
        duplicates="drop"
    )

    return buckets.astype(str).tolist()


def _compute_summary(
    pd_values: List[float],
    buckets: List[str]
) -> List[Dict]:

    df = pd.DataFrame({
        "pd": pd_values,
        "bucket": buckets
    })

    summary = (
        df.groupby("bucket")
        .agg(
            count=("pd", "count"),
            avg_pd=("pd", "mean")
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

    _validate_inputs(pd_values)

    # ── Defaults handling ─────────────────────────
    if method == "quantile":
        num_quantiles = num_quantiles or DEFAULT_NUM_QUANTILES
        labels = labels or DEFAULT_SEGMENT_LABELS[:num_quantiles]

        buckets = _quantile_segmentation(
            pd_values,
            num_quantiles,
            labels
        )

    elif method == "fixed":
        thresholds = thresholds or DEFAULT_PD_THRESHOLDS
        labels = labels or DEFAULT_SEGMENT_LABELS

        buckets = _fixed_segmentation(
            pd_values,
            thresholds,
            labels
        )

    else:
        raise ValueError("Invalid method")

    summary = _compute_summary(pd_values, buckets)

    return {
        "bucket_assignments": buckets,
        "summary": summary
    }