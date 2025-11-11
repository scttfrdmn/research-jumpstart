#!/usr/bin/env python3
"""
Evaluation Metrics for Medical Image Classification

Clinical metrics (sensitivity, specificity, PPV, NPV), ROC curves,
confusion matrices, and visualization tools.

Usage:
    from evaluation_metrics import compute_clinical_metrics, plot_roc_curve

    metrics = compute_clinical_metrics(y_true, y_pred)
    plot_roc_curve(y_true, y_probs, class_names=['Benign', 'Malignant'])
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)


def compute_clinical_metrics(y_true, y_pred, pos_label=1):
    """
    Compute clinical performance metrics.

    For medical applications, we care about:
    - Sensitivity (recall): True positive rate
    - Specificity: True negative rate
    - PPV (precision): Positive predictive value
    - NPV: Negative predictive value
    - F1 score: Harmonic mean of precision and recall

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    pos_label : int
        Label for positive class (disease/malignant)

    Returns:
    --------
    metrics : dict
        Dictionary of clinical metrics
    """
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

    # Balanced accuracy (useful for imbalanced datasets)
    balanced_acc = (sensitivity + specificity) / 2

    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # Also called recall or TPR
        'specificity': specificity,  # TNR
        'ppv': ppv,  # Also called precision
        'npv': npv,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_acc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
                         figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    class_names : list of str, optional
        Names of classes
    normalize : bool
        Whether to normalize (show percentages)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_probs, class_names=None, figsize=(8, 6),
                  save_path=None):
    """
    Plot ROC curve for binary or multiclass classification.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_probs : array-like
        Predicted probabilities (shape: [n_samples, n_classes])
    class_names : list of str, optional
        Names of classes
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=figsize)

    # Convert to numpy if needed
    y_probs = np.array(y_probs)

    # Binary classification
    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        if y_probs.ndim == 2:
            y_probs = y_probs[:, 1]  # Probability of positive class

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

    # Multiclass classification
    else:
        n_classes = y_probs.shape[1]

        # One-vs-rest ROC for each class
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_probs, class_names=None,
                                figsize=(8, 6), save_path=None):
    """
    Plot precision-recall curve.

    Useful for imbalanced datasets where ROC may be misleading.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_probs : array-like
        Predicted probabilities
    class_names : list of str, optional
        Names of classes
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=figsize)

    y_probs = np.array(y_probs)

    # Binary classification
    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        if y_probs.ndim == 2:
            y_probs = y_probs[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)

        plt.plot(recall, precision, lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')

    # Multiclass
    else:
        n_classes = y_probs.shape[1]

        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, i])
            avg_precision = average_precision_score(y_true_binary, y_probs[:, i])

            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(recall, precision, lw=2,
                    label=f'{class_name} (AP = {avg_precision:.3f})')

    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_summary(y_true, y_pred, y_probs=None,
                                class_names=None):
    """
    Print comprehensive classification summary.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities (for AUC)
    class_names : list of str, optional
        Names of classes
    """
    print("Classification Performance Summary")
    print("=" * 60)

    # Clinical metrics
    metrics = compute_clinical_metrics(y_true, y_pred)

    print("\nClinical Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.3f}")
    print(f"  Sensitivity (TPR):  {metrics['sensitivity']:.3f}")
    print(f"  Specificity (TNR):  {metrics['specificity']:.3f}")
    print(f"  PPV (Precision):    {metrics['ppv']:.3f}")
    print(f"  NPV:                {metrics['npv']:.3f}")
    print(f"  F1 Score:           {metrics['f1_score']:.3f}")

    print("\nConfusion Matrix Components:")
    print(f"  True Positives:     {metrics['true_positives']}")
    print(f"  True Negatives:     {metrics['true_negatives']}")
    print(f"  False Positives:    {metrics['false_positives']}")
    print(f"  False Negatives:    {metrics['false_negatives']}")

    # AUC if probabilities provided
    if y_probs is not None:
        y_probs = np.array(y_probs)
        if y_probs.ndim == 2 and y_probs.shape[1] == 2:
            y_probs = y_probs[:, 1]

        try:
            auc_score = roc_auc_score(y_true, y_probs)
            print(f"\n  ROC AUC:            {auc_score:.3f}")
        except Exception as e:
            print(f"\n  ROC AUC: Unable to compute ({e})")

    # Detailed classification report
    print("\n" + "=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def compute_bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000,
                        confidence=0.95, random_state=42):
    """
    Compute confidence interval for a metric using bootstrap.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    metric_func : callable
        Function that computes metric (takes y_true, y_pred)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed

    Returns:
    --------
    ci : tuple
        (lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]

        # Compute metric
        score = metric_func(y_true_boot, y_pred_boot)
        scores.append(score)

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return (lower, upper)


def analyze_errors(y_true, y_pred, y_probs, image_paths=None, top_k=10):
    """
    Analyze misclassified examples.

    Identifies the most confidently wrong predictions.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like
        Prediction probabilities
    image_paths : list of str, optional
        Paths to images
    top_k : int
        Number of top errors to return

    Returns:
    --------
    errors : list of dict
        List of error information dictionaries
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Find misclassifications
    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]

    if len(error_indices) == 0:
        return []

    # Get confidence for predicted class
    if y_probs.ndim == 2:
        confidence = y_probs[error_indices, y_pred[error_indices]]
    else:
        confidence = y_probs[error_indices]

    # Sort by confidence (most confident errors first)
    sorted_indices = error_indices[np.argsort(-confidence)][:top_k]

    errors = []
    for idx in sorted_indices:
        error_info = {
            'index': int(idx),
            'true_label': int(y_true[idx]),
            'pred_label': int(y_pred[idx]),
            'confidence': float(y_probs[idx, y_pred[idx]] if y_probs.ndim == 2 else y_probs[idx])
        }

        if image_paths is not None:
            error_info['image_path'] = image_paths[idx]

        errors.append(error_info)

    return errors


if __name__ == '__main__':
    # Example usage with synthetic data
    print("Medical Image Classification Evaluation Metrics")
    print("=" * 60)

    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    y_probs = np.random.rand(n_samples, 2)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    y_pred = y_probs.argmax(axis=1)

    class_names = ['Benign', 'Malignant']

    # Print classification summary
    print_classification_summary(y_true, y_pred, y_probs, class_names)

    # Compute confidence interval for accuracy
    print("\n" + "=" * 60)
    print("\nBootstrap Confidence Interval for Accuracy:")
    from sklearn.metrics import accuracy_score
    ci = compute_bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstrap=1000)
    print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Analyze errors
    print("\n" + "=" * 60)
    print("\nTop 5 Most Confident Errors:")
    errors = analyze_errors(y_true, y_pred, y_probs, top_k=5)
    for i, error in enumerate(errors, 1):
        print(f"  {i}. Index {error['index']}: "
              f"True={error['true_label']}, "
              f"Pred={error['pred_label']}, "
              f"Confidence={error['confidence']:.3f}")

    print("\n✓ Evaluation metrics ready")
