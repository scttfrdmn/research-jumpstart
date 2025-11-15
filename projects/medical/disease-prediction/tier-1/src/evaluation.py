"""
Clinical evaluation metrics for medical imaging models.

Includes AUC-ROC, sensitivity, specificity, Dice coefficient, etc.
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


def calculate_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate AUC-ROC score."""
    return roc_auc_score(y_true, y_pred)


def calculate_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    """
    Calculate sensitivity (recall) and specificity.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Tuple of (sensitivity, specificity)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


def calculate_dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Dice coefficient for segmentation.

    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask

    Returns:
        Dice coefficient
    """
    intersection = np.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Comprehensive classification evaluation.

    Returns:
        Dictionary with all metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
    }

    sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_pred, threshold)
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity

    return metrics


if __name__ == "__main__":
    print("Clinical Evaluation Metrics")
