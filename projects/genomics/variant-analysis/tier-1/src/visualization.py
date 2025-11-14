"""
Visualization functions for genomic variant analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict
from sklearn.metrics import roc_curve, auc


def plot_read_depth(
    depths: np.ndarray,
    positions: np.ndarray,
    title: str = "Read Depth Along Chromosome",
    figsize: tuple = (14, 4)
) -> plt.Figure:
    """
    Plot read depth across genomic positions.

    Args:
        depths: Array of read depths
        positions: Genomic positions
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(positions, depths, alpha=0.5, color='steelblue')
    ax.plot(positions, depths, color='steelblue', linewidth=1)

    ax.set_xlabel('Genomic Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Read Depth', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_variant_density(
    variants_df: pd.DataFrame,
    window_size: int = 100000,
    title: str = "Variant Density",
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Plot variant density along chromosome.

    Args:
        variants_df: DataFrame with 'pos' column
        window_size: Window size for density calculation
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate density
    positions = variants_df['pos'].values
    min_pos = positions.min()
    max_pos = positions.max()

    bins = np.arange(min_pos, max_pos + window_size, window_size)
    counts, bin_edges = np.histogram(positions, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    ax.bar(bin_centers, counts, width=window_size * 0.8, alpha=0.7, color='coral', edgecolor='black')

    ax.set_xlabel('Genomic Position', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Variant Count (per {window_size/1000:.0f}kb)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve - Variant Calling",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve for variant calling performance.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_prob_flat)
    roc_auc = auc(fpr, tpr)

    # Plot
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_population_comparison(
    metrics_by_population: Dict[str, Dict[str, float]],
    metric_name: str = "f1",
    title: str = "Performance by Population",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare variant calling performance across populations.

    Args:
        metrics_by_population: Dict mapping population to metrics dict
        metric_name: Metric to plot (e.g., 'precision', 'recall', 'f1')
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    populations = list(metrics_by_population.keys())
    values = [metrics_by_population[pop][metric_name] for pop in populations]

    # Color by value
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(populations)))

    bars = ax.bar(populations, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_ylabel(metric_name.upper(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def create_manhattan_plot(
    variants_df: pd.DataFrame,
    quality_col: str = 'qual',
    threshold: float = 30.0,
    title: str = "Manhattan Plot - Variant Quality",
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Create Manhattan plot showing variant quality scores.

    Args:
        variants_df: DataFrame with 'pos' and quality columns
        quality_col: Column name for quality scores
        threshold: Quality threshold line
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = variants_df['pos'].values
    qualities = variants_df[quality_col].values

    # Color by quality
    colors = ['red' if q < threshold else 'blue' for q in qualities]

    ax.scatter(positions, qualities, c=colors, alpha=0.6, s=20)

    # Threshold line
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')

    ax.set_xlabel('Genomic Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Plot training history for variant caller model.

    Args:
        history: Training history dictionary from model.fit()
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]

    for idx, (metric, label) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        if metric in history:
            ax.plot(history[metric], label='Train', linewidth=2)
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label='Validation', linewidth=2)

            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.set_title(f'Training {label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ensemble_agreement(
    predictions: List[np.ndarray],
    model_names: List[str],
    positions: np.ndarray,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot agreement between ensemble models.

    Args:
        predictions: List of prediction arrays from each model
        model_names: Names of models
        positions: Genomic positions
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate agreement (fraction of models predicting variant)
    stacked = np.stack([p.flatten() for p in predictions], axis=0)
    agreement = np.mean(stacked, axis=0)

    # Plot
    scatter = ax.scatter(
        positions,
        agreement,
        c=agreement,
        cmap='RdYlGn',
        alpha=0.6,
        s=30,
        vmin=0,
        vmax=1
    )

    # Consensus line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Majority threshold')

    ax.set_xlabel('Genomic Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ensemble Agreement', fontsize=12, fontweight='bold')
    ax.set_title(f'Ensemble Agreement ({len(model_names)} models)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Agreement Score', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig


def create_interactive_dashboard(
    variants_df: pd.DataFrame,
    metrics: Dict[str, float],
    title: str = "Variant Calling Dashboard"
) -> go.Figure:
    """
    Create interactive Plotly dashboard for variant analysis.

    Args:
        variants_df: DataFrame with variant information
        metrics: Dictionary of performance metrics
        title: Dashboard title

    Returns:
        Plotly Figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Variant Positions',
            'Quality Score Distribution',
            'Performance Metrics',
            'Cumulative Variants'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ]
    )

    # Plot 1: Variant positions
    fig.add_trace(
        go.Scatter(
            x=variants_df['pos'],
            y=variants_df.get('qual', [1] * len(variants_df)),
            mode='markers',
            name='Variants',
            marker=dict(size=6, color='steelblue')
        ),
        row=1, col=1
    )

    # Plot 2: Quality distribution
    if 'qual' in variants_df.columns:
        fig.add_trace(
            go.Histogram(
                x=variants_df['qual'],
                name='Quality',
                marker=dict(color='coral')
            ),
            row=1, col=2
        )

    # Plot 3: Metrics
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=metric_values,
            name='Metrics',
            marker=dict(color='lightgreen')
        ),
        row=2, col=1
    )

    # Plot 4: Cumulative variants
    sorted_positions = np.sort(variants_df['pos'].values)
    cumulative = np.arange(1, len(sorted_positions) + 1)

    fig.add_trace(
        go.Scatter(
            x=sorted_positions,
            y=cumulative,
            mode='lines',
            name='Cumulative',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
        hovermode='closest'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Quality", row=1, col=1)
    fig.update_xaxes(title_text="Quality Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Metric", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Position", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Count", row=2, col=2)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix for variant calling.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix

    fig, ax = plt.subplots(figsize=figsize)

    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    # Plot
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticklabels(['Reference', 'Variant'])
    ax.set_yticklabels(['Reference', 'Variant'])

    plt.tight_layout()
    return fig
