"""
Visualization functions for climate data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


def plot_time_series(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Time Series",
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot one or more time series.

    Args:
        data: DataFrame containing data
        x_col: Column name for x-axis (typically Date or Year)
        y_cols: List of column names to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in y_cols:
        ax.plot(data[x_col], data[col], label=col, linewidth=2)

    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Create correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def create_interactive_dashboard(
    data: pd.DataFrame,
    date_col: str,
    variables: List[str],
    title: str = "Climate Dashboard"
) -> go.Figure:
    """
    Create interactive Plotly dashboard with multiple variables.

    Args:
        data: DataFrame containing data
        date_col: Column name for date/x-axis
        variables: List of variable column names
        title: Dashboard title

    Returns:
        Plotly Figure object
    """
    # Create subplots
    n_vars = len(variables)
    fig = make_subplots(
        rows=n_vars,
        cols=1,
        subplot_titles=variables,
        vertical_spacing=0.08
    )

    # Add traces for each variable
    for idx, var in enumerate(variables, 1):
        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data[var],
                name=var,
                mode='lines',
                line=dict(width=2)
            ),
            row=idx,
            col=1
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=300 * n_vars,
        showlegend=False,
        hovermode='x unified'
    )

    # Update axes
    for idx in range(1, n_vars + 1):
        fig.update_xaxes(title_text="Date" if idx == n_vars else "", row=idx, col=1)
        fig.update_yaxes(title_text=variables[idx - 1], row=idx, col=1)

    return fig


def plot_trend_comparison(
    data: pd.DataFrame,
    periods_col: str,
    values_col: str,
    title: str = "Trend Comparison",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create bar chart comparing trends across periods.

    Args:
        data: DataFrame with period-wise trends
        periods_col: Column name for period labels
        values_col: Column name for values to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(data)))
    bars = ax.bar(data[periods_col], data[values_col], color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_ylabel(values_col, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    return fig
