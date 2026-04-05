"""
utils.py
--------
Shared utility functions used across the project.
"""

import numpy as np
import pandas as pd


def calculate_ks(y_true, y_pred_prob) -> float:
    """
    Calculate the KS (Kolmogorov-Smirnov) statistic.

    KS measures the maximum separation between the cumulative
    distribution of defaults and non-defaults across all score thresholds.

    Industry benchmark: KS > 0.40 = good model

    Parameters
    ----------
    y_true : array-like
        Binary target (1 = default, 0 = no default)
    y_pred_prob : array-like
        Predicted probability of default

    Returns
    -------
    float
        KS statistic (0 to 1)
    """
    data = pd.DataFrame({"target": y_true, "prob": y_pred_prob})
    data = data.sort_values("prob", ascending=False)

    data["cum_event"]     = (np.cumsum(data["target"])
                             / data["target"].sum())
    data["cum_non_event"] = (np.cumsum(1 - data["target"])
                             / (1 - data["target"]).sum())
    data["ks"]            = data["cum_event"] - data["cum_non_event"]

    return float(np.max(data["ks"]))


def validate_input_fields(data: dict, required_fields: list) -> list:
    """
    Check that all required input fields are present.

    Parameters
    ----------
    data : dict
        Input data dictionary.
    required_fields : list
        List of required field names.

    Returns
    -------
    list
        List of missing field names (empty if all present).
    """
    return [f for f in required_fields if f not in data]
