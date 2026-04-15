"""
metrics.py
──────────
Evaluation metrics for the test set.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


CLASS_NAMES = ["CN", "MCI", "AD"]


def compute_all_metrics(y_true, y_pred):
    """
    Parameters
    ----------
    y_true, y_pred : list or np.array of integer class indices

    Returns dict with accuracy, macro precision/recall/F1, confusion matrix.
    """
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    f1    = f1_score(y_true, y_pred,        average="macro", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "confusion_matrix": cm,
        "report":    report,
    }


def print_metrics(metrics_dict):
    print(f"\n  Accuracy : {metrics_dict['accuracy']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall   : {metrics_dict['recall']:.4f}")
    print(f"  F1-score : {metrics_dict['f1']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics_dict['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics_dict['report']}")