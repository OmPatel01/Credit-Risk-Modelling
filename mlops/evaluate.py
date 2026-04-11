"""
mlops/evaluate.py
-----------------
Model evaluation utilities called during training to compute, visualise, and log metrics.

This module runs INSIDE an active MLflow run — it calls mlflow.log_metrics() and
mlflow.log_artifact() directly. Always call evaluate_model() inside a `with mlflow.start_run():`
block (see mlops/train.py). Calling it outside a run will raise an MlflowException.

Outputs:
    - Standard classification metrics (accuracy, precision, recall, F1, AUC, KS)
    - Confusion matrix PNG saved to artifacts/plots/
    - ROC curve PNG saved to artifacts/plots/
    - All metrics and plots logged to the active MLflow run
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import mlflow


def calculate_ks(y_true, y_pred_prob):
    """
    Compute KS statistic — the maximum separation between cumulative default and non-default rates.

    Identical to core/utils.py:calculate_ks but implemented here with numpy directly
    to avoid importing from the API layer (mlops/ should be independent of app/).
    """
    data = np.column_stack((y_true, y_pred_prob))

    # Sort descending by predicted probability: riskiest borrowers first
    data         = data[data[:, 1].argsort()[::-1]]
    y_true_sorted = data[:, 0]

    cum_event     = np.cumsum(y_true_sorted)       / np.sum(y_true)
    cum_non_event = np.cumsum(1 - y_true_sorted)   / np.sum(1 - y_true)

    ks = np.max(cum_event - cum_non_event)
    return float(ks)


def evaluate_model(y_true, y_pred_prob, threshold=0.5, model_name="model"):
    """
    Compute all evaluation metrics, save visualisation artifacts, and log everything to MLflow.

    threshold controls the cut-off for converting probabilities to binary predictions
    (affects accuracy, precision, recall, F1 but NOT AUC or KS which are threshold-free).

    Plots are written to artifacts/plots/ before being logged to MLflow so they are
    also available locally without opening the MLflow UI.

    Returns the metrics dict so the caller can print a summary or log additional derived metrics.
    """
    os.makedirs("artifacts/plots", exist_ok=True)

    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred),
        "auc":       roc_auc_score(y_true, y_pred_prob),  # threshold-free ranking metric
        "ks":        calculate_ks(y_true, y_pred_prob),   # threshold-free separation metric
    }

    # ── Confusion matrix — shows actual distribution of TP/FP/TN/FN
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    cm_path = f"artifacts/plots/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ── ROC curve — visualises tradeoff between true positive and false positive rates
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])  # diagonal = random classifier baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    roc_path = f"artifacts/plots/{model_name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # ── Log to active MLflow run — must be called inside `with mlflow.start_run()`
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)

    return metrics