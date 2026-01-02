"""
Comprehensive metrics calculation for model evaluation.
Includes classification metrics, per-class metrics, and statistical validation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, labels=None):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC calculation)
        labels: Label names (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["weighted_precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["weighted_recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    if labels is None:
        labels = [f"class_{i}" for i in range(len(precision_per_class))]

    for i, label in enumerate(labels):
        metrics[f"precision_{label}"] = precision_per_class[i]
        metrics[f"recall_{label}"] = recall_per_class[i]
        metrics[f"f1_{label}"] = f1_per_class[i]

    if y_proba is not None:
        try:
            if len(np.unique(y_true)) > 2:
                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true, y_proba, average="macro", multi_class="ovr"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true, y_proba, average="weighted", multi_class="ovr"
                )
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                )
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    metrics["total_samples"] = len(y_true)
    metrics["num_classes"] = len(np.unique(y_true))

    return metrics


def calculate_class_distribution(y):
    """Calculate class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts, strict=False))

    stats = {
        "class_counts": distribution,
        "class_proportions": {k: v / len(y) for k, v in distribution.items()},
        "total_samples": len(y),
        "num_classes": len(unique),
        "class_imbalance_ratio": max(counts) / min(counts) if len(counts) > 1 else 1.0,
    }

    return stats


def calculate_prediction_statistics(y_pred, y_proba=None):
    """Calculate statistics about predictions."""
    stats = {
        "prediction_distribution": dict(
            zip(*np.unique(y_pred, return_counts=True), strict=False)
        ),
        "total_predictions": len(y_pred),
    }

    if y_proba is not None:
        stats["mean_confidence"] = float(np.mean(np.max(y_proba, axis=1)))
        stats["std_confidence"] = float(np.std(np.max(y_proba, axis=1)))
        stats["min_confidence"] = float(np.min(np.max(y_proba, axis=1)))
        stats["max_confidence"] = float(np.max(np.max(y_proba, axis=1)))

    return stats


def compare_metrics(reference_metrics, production_metrics, threshold=0.05):
    """
    Compare reference and production metrics to detect degradation.

    Args:
        reference_metrics: Dictionary of reference metrics
        production_metrics: Dictionary of production metrics
        threshold: Relative change threshold for alerting (default 5%)

    Returns:
        Dictionary with comparison results and alerts
    """
    comparison = {
        "metrics_comparison": {},
        "alerts": [],
        "degradation_detected": False,
    }

    metrics_to_compare = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    for metric_name in metrics_to_compare:
        if metric_name in reference_metrics and metric_name in production_metrics:
            ref_value = reference_metrics[metric_name]
            prod_value = production_metrics[metric_name]

            if ref_value > 0:
                relative_change = (prod_value - ref_value) / ref_value
                absolute_change = prod_value - ref_value

                comparison["metrics_comparison"][metric_name] = {
                    "reference": ref_value,
                    "production": prod_value,
                    "absolute_change": absolute_change,
                    "relative_change": relative_change,
                    "degradation": relative_change < -threshold,
                }

                if relative_change < -threshold:
                    comparison["alerts"].append(
                        {
                            "metric": metric_name,
                            "severity": (
                                "high" if abs(relative_change) > 0.1 else "medium"
                            ),
                            "message": f"{metric_name} degraded by {abs(relative_change)*100:.2f}%",
                            "reference": ref_value,
                            "production": prod_value,
                        }
                    )
                    comparison["degradation_detected"] = True

    return comparison



