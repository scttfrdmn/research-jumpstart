"""
Visualization utilities for multi-modal affect recognition.

This module provides plotting functions for EEG topographies, facial landmarks,
multi-modal timelines, and fusion performance analysis.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_eeg_topography(
    values: np.ndarray,
    title: str = "EEG Topography",
    cmap: str = "RdBu_r",
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot EEG topographic map.

    Args:
        values: Channel values to plot, shape (n_channels,)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
    """
    # Simplified 2D projection of electrode positions (for 32 channels)
    # In practice, would use actual electrode coordinates
    n_channels = len(values)

    # Create circular electrode layout
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    _fig, ax = plt.subplots(figsize=figsize)

    # Create interpolated topography
    from scipy.interpolate import griddata

    xi = np.linspace(-1.2, 1.2, 100)
    yi = np.linspace(-1.2, 1.2, 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), values, (xi, yi), method="cubic")

    # Mask outside head
    mask = (xi**2 + yi**2) > 1
    zi[mask] = np.nan

    # Plot
    im = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)
    ax.scatter(x, y, c="k", s=20, zorder=3)

    # Draw head outline
    circle = plt.Circle((0, 0), 1, fill=False, color="k", linewidth=2)
    ax.add_patch(circle)

    # Nose
    nose_x = [0, 0.18, -0.18]
    nose_y = [1, 1.15, 1.15]
    ax.plot(nose_x, nose_y, "k-", linewidth=2)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12)

    plt.colorbar(im, ax=ax, label="Amplitude (μV)")
    plt.tight_layout()


def plot_facial_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    title: str = "Facial Landmarks",
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot facial image with detected landmarks.

    Args:
        image: Facial image, shape (height, width, 3) or (height, width)
        landmarks: Facial landmarks, shape (n_landmarks, 2)
        title: Plot title
        figsize: Figure size
    """
    _fig, ax = plt.subplots(figsize=figsize)

    # Display image
    if len(image.shape) == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap="gray")

    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="red", s=30, alpha=0.7)

    # Connect landmarks for different facial features
    # Jaw (0-16)
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], "b-", linewidth=1, alpha=0.5)

    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], "g-", linewidth=1, alpha=0.5)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], "g-", linewidth=1, alpha=0.5)

    # Nose
    ax.plot(landmarks[27:36, 0], landmarks[27:36, 1], "m-", linewidth=1, alpha=0.5)

    # Eyes
    ax.plot(
        np.append(landmarks[36:42, 0], landmarks[36, 0]),
        np.append(landmarks[36:42, 1], landmarks[36, 1]),
        "c-",
        linewidth=1,
        alpha=0.5,
    )
    ax.plot(
        np.append(landmarks[42:48, 0], landmarks[42, 0]),
        np.append(landmarks[42:48, 1], landmarks[42, 1]),
        "c-",
        linewidth=1,
        alpha=0.5,
    )

    # Mouth
    ax.plot(
        np.append(landmarks[48:60, 0], landmarks[48, 0]),
        np.append(landmarks[48:60, 1], landmarks[48, 1]),
        "y-",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.axis("off")
    plt.tight_layout()


def plot_multimodal_timeline(
    eeg_signal: np.ndarray,
    gsr_signal: np.ndarray,
    resp_signal: np.ndarray,
    emotion_labels: Optional[np.ndarray] = None,
    sampling_rates: Optional[dict[str, int]] = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Plot multi-modal physiological signals over time.

    Args:
        eeg_signal: EEG signal (single channel), shape (n_samples,)
        gsr_signal: GSR signal, shape (n_samples,)
        resp_signal: Respiration signal, shape (n_samples,)
        emotion_labels: Optional emotion labels over time
        sampling_rates: Dictionary of sampling rates for each modality
        figsize: Figure size
    """
    if sampling_rates is None:
        sampling_rates = {"eeg": 256, "gsr": 32, "resp": 32}
    fig, axes = plt.subplots(
        4 if emotion_labels is not None else 3, 1, figsize=figsize, sharex=True
    )

    # Time axes
    time_eeg = np.arange(len(eeg_signal)) / sampling_rates["eeg"]
    time_gsr = np.arange(len(gsr_signal)) / sampling_rates["gsr"]
    time_resp = np.arange(len(resp_signal)) / sampling_rates["resp"]

    # Plot EEG
    axes[0].plot(time_eeg, eeg_signal, "b-", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("EEG\nAmplitude (μV)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot GSR
    axes[1].plot(time_gsr, gsr_signal, "g-", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("GSR\n(μS)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Plot Respiration
    axes[2].plot(time_resp, resp_signal, "r-", linewidth=1, alpha=0.7)
    axes[2].set_ylabel("Respiration\nAmplitude", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # Plot emotion labels if provided
    if emotion_labels is not None:
        axes[3].plot(time_resp, emotion_labels, "k-", linewidth=2)
        axes[3].set_ylabel("Emotion\nLabel", fontweight="bold")
        axes[3].set_xlabel("Time (seconds)", fontweight="bold")
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel("Time (seconds)", fontweight="bold")

    fig.suptitle("Multi-Modal Physiological Signals", fontweight="bold", fontsize=14)
    plt.tight_layout()


def plot_fusion_performance(
    results: dict[str, dict[str, float]],
    metric: str = "accuracy",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot performance comparison across fusion methods.

    Args:
        results: Dictionary of results, format:
                 {'method_name': {'accuracy': X, 'f1': Y, ...}}
        metric: Metric to plot
        figsize: Figure size
    """
    methods = list(results.keys())
    scores = [results[method][metric] for method in methods]

    _fig, ax = plt.subplots(figsize=figsize)

    colors = ["skyblue", "lightcoral", "lightgreen", "plum", "gold"]
    bars = ax.bar(methods, scores, color=colors[: len(methods)], alpha=0.8, edgecolor="black")

    ax.set_ylabel(metric.capitalize(), fontweight="bold", fontsize=12)
    ax.set_title(
        f"Fusion Method Comparison: {metric.capitalize()}", fontweight="bold", fontsize=14, pad=15
    )
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()


def plot_confusion_matrix_comparison(
    confusion_matrices: dict[str, np.ndarray],
    class_names: list[str],
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Plot comparison of confusion matrices from different methods.

    Args:
        confusion_matrices: Dictionary of confusion matrices
        class_names: List of class names
        figsize: Figure size
    """
    n_methods = len(confusion_matrices)
    _fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, cm) in zip(axes, confusion_matrices.items()):
        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Proportion"},
        )

        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("True", fontweight="bold")
        ax.set_title(f"{method_name}", fontweight="bold", fontsize=11)

    plt.tight_layout()


def plot_modality_importance(
    importance_scores: dict[str, float], figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot importance scores for each modality.

    Args:
        importance_scores: Dictionary of modality importance scores
        figsize: Figure size
    """
    modalities = list(importance_scores.keys())
    scores = list(importance_scores.values())

    _fig, ax = plt.subplots(figsize=figsize)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bars = ax.bar(
        modalities,
        scores,
        color=colors[: len(modalities)],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )

    ax.set_ylabel("Importance Score", fontweight="bold", fontsize=12)
    ax.set_title("Modality Importance Analysis", fontweight="bold", fontsize=14, pad=15)
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()


def plot_feature_correlation(
    features: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Correlation Matrix",
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """
    Plot correlation matrix of features.

    Args:
        features: Feature matrix, shape (n_samples, n_features)
        feature_names: List of feature names
        title: Plot title
        figsize: Figure size
    """
    # Compute correlation
    correlation = np.corrcoef(features.T)

    _fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        correlation,
        annot=False,
        cmap="coolwarm",
        center=0,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
    plt.tight_layout()


def plot_training_history(
    history: dict[str, list[float]], figsize: tuple[int, int] = (14, 5)
) -> None:
    """
    Plot training history (loss and accuracy).

    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if "loss" in history:
        ax1.plot(history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Model Loss", fontweight="bold", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    if "accuracy" in history:
        ax2.plot(history["accuracy"], label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history:
        ax2.plot(history["val_accuracy"], label="Validation Accuracy", linewidth=2)

    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy", fontweight="bold")
    ax2.set_title("Model Accuracy", fontweight="bold", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
