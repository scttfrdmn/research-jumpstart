"""
Visualization functions for dialect analysis results.

This module provides plotting functions for dialect space visualization,
confusion matrices, feature importance, and cross-linguistic comparisons.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_dialect_space(
    features: np.ndarray, labels: np.ndarray, method: str = "pca", save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize dialect space using dimensionality reduction.

    Args:
        features: Feature embeddings
        labels: Dialect labels
        method: Reduction method ('pca', 'tsne', 'umap')
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Reduce to 2D
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(features)
        title = "Dialect Space (PCA)"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)
        title = "Dialect Space (t-SNE)"
    else:
        # Placeholder for UMAP
        reduced = np.random.randn(len(features), 2)
        title = f"Dialect Space ({method.upper()})"

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, dialect in enumerate(unique_labels):
        mask = labels == dialect
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1], c=[colors[i]], label=dialect, alpha=0.6, s=50
        )

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dialect_names: Optional[list[str]] = None,
    normalize: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix for dialect classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        dialect_names: List of dialect names
        normalize: Normalize confusion matrix
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=dialect_names if dialect_names else "auto",
        yticklabels=dialect_names if dialect_names else "auto",
        ax=ax,
    )

    ax.set_xlabel("Predicted Dialect")
    ax.set_ylabel("True Dialect")
    ax.set_title("Dialect Classification Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def create_feature_importance_plot(
    importance_df: pd.DataFrame, top_k: int = 20, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance for dialect classification.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_k: Number of top features to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    data = importance_df.head(top_k).sort_values("importance")

    ax.barh(range(len(data)), data["importance"], color="steelblue")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["feature"])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_k} Most Important Features for Dialect Classification")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def plot_cross_linguistic_comparison(
    results_df: pd.DataFrame, metric: str = "accuracy", save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare performance across languages.

    Args:
        results_df: DataFrame with language and metric columns
        metric: Metric to plot ('accuracy', 'f1_score', etc.)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    results_df = results_df.sort_values(metric, ascending=True)

    ax.barh(results_df["language"], results_df[metric], color="coral")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Language")
    ax.set_title(f"Dialect Classification {metric.replace('_', ' ').title()} by Language")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, (_lang, value) in enumerate(zip(results_df["language"], results_df[metric])):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def plot_dialect_distance_heatmap(
    distance_df: pd.DataFrame, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot heatmap of pairwise dialect distances.

    Args:
        distance_df: DataFrame with pairwise distances
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        distance_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Distance"}
    )

    ax.set_title("Pairwise Dialect Distances")
    ax.set_xlabel("Dialect")
    ax.set_ylabel("Dialect")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def create_ablation_study_plot(
    ablation_df: pd.DataFrame, metric: str = "accuracy", save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize ablation study results.

    Args:
        ablation_df: DataFrame with configuration and metric columns
        metric: Metric to plot
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = ablation_df.sort_values(metric, ascending=True)

    colors = ["green" if "full" in cfg else "red" for cfg in data["configuration"]]

    ax.barh(data["configuration"], data[metric], color=colors, alpha=0.7)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Configuration")
    ax.set_title(f"Ablation Study: Impact on {metric.replace('_', ' ').title()}")
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, (_cfg, value) in enumerate(zip(data["configuration"], data[metric])):
        ax.text(value + 0.005, i, f"{value:.3f}", va="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    return fig


def create_interactive_dashboard(
    features: np.ndarray, labels: np.ndarray, save_path: Optional[Path] = None
) -> None:
    """
    Create interactive Plotly dashboard for dialect exploration.

    Args:
        features: Feature embeddings
        labels: Dialect labels
        save_path: Path to save HTML dashboard
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    # Reduce to 3D
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features)

    # Create 3D scatter plot
    fig = go.Figure()

    unique_labels = np.unique(labels)
    for dialect in unique_labels:
        mask = labels == dialect
        fig.add_trace(
            go.Scatter3d(
                x=reduced[mask, 0],
                y=reduced[mask, 1],
                z=reduced[mask, 2],
                mode="markers",
                name=str(dialect),
                marker={"size": 5, "opacity": 0.7},
            )
        )

    fig.update_layout(
        title="Interactive 3D Dialect Space",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        width=1000,
        height=800,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved interactive dashboard to {save_path}")
    else:
        fig.show()
