"""
Multi-modal fusion strategies for affect recognition.

This module implements different fusion approaches to combine
EEG, facial, and physiological modalities.
"""

from typing import Optional

import numpy as np


def early_fusion(
    eeg_features: np.ndarray,
    facial_features: np.ndarray,
    physio_features: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Early fusion: Concatenate features from all modalities.

    Args:
        eeg_features: EEG feature matrix, shape (n_samples, n_eeg_features)
        facial_features: Facial feature matrix, shape (n_samples, n_facial_features)
        physio_features: Physiological feature matrix, shape (n_samples, n_physio_features)
        normalize: Whether to normalize each modality before fusion

    Returns:
        Fused feature matrix, shape (n_samples, total_features)
    """
    features = [eeg_features, facial_features, physio_features]

    if normalize:
        # Normalize each modality to zero mean and unit variance
        from sklearn.preprocessing import StandardScaler

        normalized_features = []
        for feat in features:
            if feat is not None and len(feat) > 0:
                scaler = StandardScaler()
                normalized = scaler.fit_transform(feat)
                normalized_features.append(normalized)
        features = normalized_features

    # Concatenate along feature dimension
    fused = np.concatenate([f for f in features if f is not None], axis=1)

    return fused


def late_fusion(
    eeg_predictions: np.ndarray,
    facial_predictions: np.ndarray,
    physio_predictions: np.ndarray,
    method: str = "average",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Late fusion: Combine predictions from individual modality classifiers.

    Args:
        eeg_predictions: EEG classifier predictions, shape (n_samples, n_classes)
        facial_predictions: Facial classifier predictions, shape (n_samples, n_classes)
        physio_predictions: Physiological classifier predictions, shape (n_samples, n_classes)
        method: Fusion method ('average', 'weighted', 'max', 'product')
        weights: Weights for each modality (only used if method='weighted')

    Returns:
        Fused predictions, shape (n_samples, n_classes)
    """
    predictions = [eeg_predictions, facial_predictions, physio_predictions]
    predictions = [p for p in predictions if p is not None]

    if len(predictions) == 0:
        raise ValueError("At least one modality prediction is required")

    if method == "average":
        # Simple average
        fused = np.mean(predictions, axis=0)

    elif method == "weighted":
        # Weighted average
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

        fused = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            fused += weight * pred

    elif method == "max":
        # Maximum rule
        fused = np.maximum.reduce(predictions)

    elif method == "product":
        # Product rule
        fused = np.prod(predictions, axis=0)
        # Renormalize
        fused = fused / fused.sum(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return fused


def hybrid_fusion(
    eeg_features: np.ndarray,
    facial_features: np.ndarray,
    physio_features: np.ndarray,
    eeg_predictions: np.ndarray,
    facial_predictions: np.ndarray,
    physio_predictions: np.ndarray,
    feature_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hybrid fusion: Combine both feature-level and decision-level fusion.

    Args:
        eeg_features: EEG features
        facial_features: Facial features
        physio_features: Physiological features
        eeg_predictions: EEG predictions
        facial_predictions: Facial predictions
        physio_predictions: Physiological predictions
        feature_weight: Weight for feature-level fusion (0-1)

    Returns:
        Tuple of (fused_features, fused_predictions)
    """
    # Feature-level fusion
    fused_features = early_fusion(eeg_features, facial_features, physio_features)

    # Decision-level fusion
    fused_predictions = late_fusion(eeg_predictions, facial_predictions, physio_predictions)

    return fused_features, fused_predictions


def attention_fusion(
    eeg_features: np.ndarray,
    facial_features: np.ndarray,
    physio_features: np.ndarray,
    attention_model=None,
) -> np.ndarray:
    """
    Attention-based fusion: Learn importance weights for each modality.

    Args:
        eeg_features: EEG feature matrix
        facial_features: Facial feature matrix
        physio_features: Physiological feature matrix
        attention_model: Pre-trained attention model (optional)

    Returns:
        Attention-weighted fused features
    """
    # Stack modalities
    modalities = [eeg_features, facial_features, physio_features]
    modalities = [m for m in modalities if m is not None]

    if attention_model is not None:
        # Use learned attention weights
        attention_weights = attention_model.predict(modalities)
    else:
        # Default: uniform attention
        attention_weights = np.ones((len(modalities), 1)) / len(modalities)

    # Apply attention weights
    weighted_modalities = []
    for modality, weight in zip(modalities, attention_weights):
        weighted = modality * weight
        weighted_modalities.append(weighted)

    # Concatenate
    fused = np.concatenate(weighted_modalities, axis=1)

    return fused


def ensemble_fusion(models: list, features: list[np.ndarray], method: str = "voting") -> np.ndarray:
    """
    Ensemble fusion: Combine multiple models trained on different modalities.

    Args:
        models: List of trained models
        features: List of feature matrices for each model
        method: Ensemble method ('voting', 'stacking')

    Returns:
        Ensemble predictions
    """
    if len(models) != len(features):
        raise ValueError("Number of models must match number of feature sets")

    predictions = []
    for model, feat in zip(models, features):
        pred = model.predict(feat)
        predictions.append(pred)

    if method == "voting":
        # Majority voting (for classification)
        if len(predictions[0].shape) == 1:
            # Hard predictions
            stacked = np.stack(predictions, axis=1)
            fused = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=stacked)
        else:
            # Soft predictions (probabilities)
            fused = np.mean(predictions, axis=0)

    elif method == "stacking":
        # Stack predictions as features for meta-learner
        # This requires a separate meta-model (not implemented here)
        fused = np.concatenate(predictions, axis=1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return fused


def compute_modality_importance(
    X_train_modalities: list[np.ndarray], y_train: np.ndarray, base_model_class, n_folds: int = 5
) -> dict[str, float]:
    """
    Compute importance of each modality using cross-validation.

    Args:
        X_train_modalities: List of feature matrices (one per modality)
        y_train: Training labels
        base_model_class: Model class to use for evaluation
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary of modality importance scores
    """
    from sklearn.model_selection import cross_val_score

    modality_names = ["eeg", "facial", "physiological"]
    importance_scores = {}

    for modality_name, X_modality in zip(modality_names, X_train_modalities):
        if X_modality is not None and len(X_modality) > 0:
            model = base_model_class()
            scores = cross_val_score(model, X_modality, y_train, cv=n_folds)
            importance_scores[modality_name] = np.mean(scores)
        else:
            importance_scores[modality_name] = 0.0

    return importance_scores


def adaptive_fusion(
    eeg_features: np.ndarray,
    facial_features: np.ndarray,
    physio_features: np.ndarray,
    modality_confidences: dict[str, float],
) -> np.ndarray:
    """
    Adaptive fusion: Weight modalities based on their confidence scores.

    Args:
        eeg_features: EEG features
        facial_features: Facial features
        physio_features: Physiological features
        modality_confidences: Confidence score for each modality

    Returns:
        Adaptively fused features
    """
    # Normalize confidence scores
    total_confidence = sum(modality_confidences.values())
    normalized_confidences = {k: v / total_confidence for k, v in modality_confidences.items()}

    # Apply confidence weights
    fused_features = []

    if eeg_features is not None:
        weight = normalized_confidences.get("eeg", 0)
        fused_features.append(eeg_features * weight)

    if facial_features is not None:
        weight = normalized_confidences.get("facial", 0)
        fused_features.append(facial_features * weight)

    if physio_features is not None:
        weight = normalized_confidences.get("physiological", 0)
        fused_features.append(physio_features * weight)

    # Concatenate
    fused = np.concatenate(fused_features, axis=1)

    return fused
