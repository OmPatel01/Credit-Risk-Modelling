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
    data = np.column_stack((y_true, y_pred_prob))
    data = data[data[:, 1].argsort()[::-1]]

    y_true_sorted = data[:, 0]

    cum_event = np.cumsum(y_true_sorted) / np.sum(y_true)
    cum_non_event = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true)

    ks = np.max(cum_event - cum_non_event)
    return float(ks)


def evaluate_model(y_true, y_pred_prob, threshold=0.5, model_name="model"):
    """
    Evaluate classification model and log everything to MLflow
    """

    os.makedirs("artifacts/plots", exist_ok=True)

    # ── Predictions ─────────────────────────────
    y_pred = (y_pred_prob >= threshold).astype(int)

    # ── Metrics ────────────────────────────────
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_prob),
        "ks": calculate_ks(y_true, y_pred_prob),
    }

    # ── Confusion Matrix ───────────────────────
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    cm_path = f"artifacts/plots/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ── ROC Curve ──────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")

    roc_path = f"artifacts/plots/{model_name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # ── MLflow Logging ─────────────────────────
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)

    return metrics