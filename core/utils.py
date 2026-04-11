"""
core/utils.py
-------------
Shared utility functions used across training (mlops/) and evaluation workflows.

These are pure functions with no side effects — safe to import anywhere.
They are NOT used during API inference (no imports from services/ or app/).
"""

import numpy as np
import pandas as pd


def calculate_ks(y_true, y_pred_prob) -> float:
    """
    Compute the KS (Kolmogorov-Smirnov) statistic for a binary classifier.

    KS measures the maximum vertical gap between the cumulative default rate
    and the cumulative non-default rate when borrowers are ordered by predicted
    probability (highest risk first). A larger gap means the model is better at
    separating good from bad borrowers.

    Industry benchmark:
        KS > 0.40 → good separation
        KS > 0.50 → strong model
        KS < 0.20 → weak model, investigate

    Returns a float in [0, 1]; 1.0 = perfect separation (never achieved in practice).
    """
    data = pd.DataFrame({"target": y_true, "prob": y_pred_prob})

    # Sort descending by predicted probability so high-risk borrowers appear first
    data = data.sort_values("prob", ascending=False)

    # Cumulative share of actual defaults and non-defaults captured at each threshold
    data["cum_event"]     = (np.cumsum(data["target"])
                             / data["target"].sum())
    data["cum_non_event"] = (np.cumsum(1 - data["target"])
                             / (1 - data["target"]).sum())

    # KS = maximum gap between the two cumulative curves
    data["ks"] = data["cum_event"] - data["cum_non_event"]

    return float(np.max(data["ks"]))


def validate_input_fields(data: dict, required_fields: list) -> list:
    """
    Return a list of field names that are present in required_fields but missing from data.

    Returns an empty list when all required fields are present — callers check `if missing: raise`.
    Used in ad-hoc scripts and tests; API validation is handled by Pydantic schemas instead.
    """
    return [f for f in required_fields if f not in data]