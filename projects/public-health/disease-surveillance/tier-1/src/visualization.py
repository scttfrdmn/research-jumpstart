"""
Visualization utilities for disease surveillance forecasting.

Provides functions for plotting time series, forecasts, heatmaps,
and interactive dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_time_series(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    title: str = "Disease Surveillance Time Series",
    ylabel: str = "Cases/Rate",
    threshold: Optional[float] = None,
    figsize: tuple = (14, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot disease surveillance time series.

    Args:
        dates: Date index
        values: Time series values
        title: Plot title
        ylabel: Y-axis label
        threshold: Optional epidemic threshold to plot
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(dates, values, color='steelblue', linewidth=2, label='Observed')

    # Add threshold if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Epidemic Threshold ({threshold:.1f})', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_forecast(
    dates: pd.DatetimeIndex,
    historical: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    forecast_mean: np.ndarray,
    forecast_lower: Optional[np.ndarray] = None,
    forecast_upper: Optional[np.ndarray] = None,
    true_future: Optional[np.ndarray] = None,
    title: str = "Disease Surveillance Forecast",
    ylabel: str = "Cases/Rate",
    threshold: Optional[float] = None,
    figsize: tuple = (14, 7)
) -> plt.Figure:
    """
    Plot historical data with forecast and uncertainty intervals.

    Args:
        dates: Historical dates
        historical: Historical values
        forecast_dates: Forecast dates
        forecast_mean: Mean forecast
        forecast_lower: Lower bound of prediction interval
        forecast_upper: Upper bound of prediction interval
        true_future: True future values (for evaluation)
        title: Plot title
        ylabel: Y-axis label
        threshold: Epidemic threshold
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(dates, historical, color='steelblue', linewidth=2, label='Historical', zorder=3)

    # Plot forecast
    ax.plot(forecast_dates, forecast_mean, color='orangered', linewidth=2.5,
            label='Forecast', marker='o', markersize=6, zorder=4)

    # Plot uncertainty interval
    if forecast_lower is not None and forecast_upper is not None:
        ax.fill_between(forecast_dates, forecast_lower, forecast_upper,
                        color='orangered', alpha=0.2, label='90% Prediction Interval')

    # Plot true future (if available)
    if true_future is not None:
        ax.plot(forecast_dates, true_future, color='green', linewidth=2,
                label='Actual', marker='s', markersize=5, linestyle='--', zorder=3)

    # Add threshold
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Epidemic Threshold')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_multi_horizon_evaluation(
    evaluation_df: pd.DataFrame,
    title: str = "Multi-Horizon Forecast Evaluation",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot evaluation metrics across forecast horizons.

    Args:
        evaluation_df: DataFrame with horizon and metric columns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metrics_to_plot = ['mae', 'rmse', 'mape', 'r2']
    metric_titles = ['Mean Absolute Error', 'Root Mean Squared Error',
                    'Mean Absolute Percentage Error', 'R² Score']

    for idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
        if metric in evaluation_df.columns:
            axes[idx].plot(evaluation_df['horizon'], evaluation_df[metric],
                          marker='o', linewidth=2, markersize=8, color='steelblue')
            axes[idx].set_xlabel('Forecast Horizon', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            axes[idx].set_title(metric_title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_outbreak_heatmap(
    data: pd.DataFrame,
    value_col: str,
    region_col: str = 'Region',
    date_col: str = 'Date',
    threshold: Optional[float] = None,
    title: str = "Disease Outbreak Heatmap",
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Create heatmap showing disease activity across regions and time.

    Args:
        data: DataFrame with region, date, and value columns
        value_col: Column name for values to plot
        region_col: Column name for regions
        date_col: Column name for dates
        threshold: Optional threshold for highlighting
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Pivot data for heatmap
    pivot = data.pivot(index=region_col, columns=date_col, values=value_col)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': value_col})

    # Add threshold line if provided
    if threshold is not None:
        # This is approximate - would need more sophisticated handling
        ax.set_title(f"{title}\n(Epidemic Threshold: {threshold:.1f})",
                    fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Region', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def create_forecast_dashboard(
    data: pd.DataFrame,
    forecast_results: Dict,
    disease: str,
    region: str
) -> Optional[go.Figure]:
    """
    Create interactive Plotly dashboard for forecast visualization.

    Args:
        data: Historical data
        forecast_results: Dictionary with forecast information
        disease: Disease name
        region: Region name

    Returns:
        Plotly figure (if Plotly available)
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Historical & Forecast',
            'Forecast Uncertainty',
            'Growth Rate',
            'Outbreak Probability'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Plot 1: Historical & Forecast
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Cases'],
            mode='lines',
            name='Historical',
            line=dict(color='steelblue', width=2)
        ),
        row=1, col=1
    )

    if 'forecast_dates' in forecast_results:
        fig.add_trace(
            go.Scatter(
                x=forecast_results['forecast_dates'],
                y=forecast_results['mean'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orangered', width=2)
            ),
            row=1, col=1
        )

    # Plot 2: Uncertainty bands
    if 'q05' in forecast_results and 'q95' in forecast_results:
        fig.add_trace(
            go.Scatter(
                x=forecast_results['forecast_dates'],
                y=forecast_results['q95'],
                mode='lines',
                name='95th percentile',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_results['forecast_dates'],
                y=forecast_results['q05'],
                mode='lines',
                name='90% Interval',
                fill='tonexty',
                line=dict(width=0),
                fillcolor='rgba(255, 69, 0, 0.2)'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_results['forecast_dates'],
                y=forecast_results['mean'],
                mode='lines',
                name='Mean',
                line=dict(color='orangered', width=2)
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text=f"{disease} Surveillance Dashboard - {region}",
        title_font_size=16,
        height=800,
        showlegend=True
    )

    return fig


def plot_ensemble_forecasts(
    dates: pd.DatetimeIndex,
    historical: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    ensemble_forecasts: np.ndarray,
    forecast_mean: np.ndarray,
    true_future: Optional[np.ndarray] = None,
    title: str = "Ensemble Forecast",
    figsize: tuple = (14, 7)
) -> plt.Figure:
    """
    Plot individual ensemble member forecasts.

    Args:
        dates: Historical dates
        historical: Historical values
        forecast_dates: Forecast dates
        ensemble_forecasts: All ensemble forecasts (n_models, n_time_steps)
        forecast_mean: Mean forecast
        true_future: True future values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical
    ax.plot(dates, historical, color='steelblue', linewidth=2, label='Historical', zorder=3)

    # Plot individual ensemble members
    for i, forecast in enumerate(ensemble_forecasts):
        ax.plot(forecast_dates, forecast, color='gray', linewidth=1,
                alpha=0.3, label='Ensemble member' if i == 0 else '')

    # Plot mean forecast
    ax.plot(forecast_dates, forecast_mean, color='orangered', linewidth=2.5,
            label='Ensemble Mean', marker='o', markersize=6, zorder=4)

    # Plot true future
    if true_future is not None:
        ax.plot(forecast_dates, true_future, color='green', linewidth=2,
                label='Actual', marker='s', markersize=5, linestyle='--', zorder=3)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cases/Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance for forecasting model.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(top_n), importances[indices], color='steelblue', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig
