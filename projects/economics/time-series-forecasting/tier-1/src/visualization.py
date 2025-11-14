"""
Visualization functions for economic time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict


def plot_time_series(
    data: pd.DataFrame,
    title: str = "Economic Time Series",
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot multiple time series on the same chart.

    Args:
        data: DataFrame with time series
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in data.columns:
        ax.plot(data.index, data[col], label=col, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (12, 10)
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
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax,
        vmin=-1,
        vmax=1
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_forecast_comparison(
    actual: pd.Series,
    forecasts: Dict[str, pd.Series],
    title: str = "Forecast Comparison",
    figsize: tuple = (14, 7)
) -> plt.Figure:
    """
    Compare multiple forecasts against actual values.

    Args:
        actual: Actual values
        forecasts: Dictionary of {model_name: forecast_series}
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual
    ax.plot(actual.index, actual.values, label='Actual',
            linewidth=2.5, color='black', alpha=0.8)

    # Plot forecasts
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (name, forecast), color in zip(forecasts.items(), colors):
        ax.plot(forecast.index, forecast.values, label=name,
                linewidth=1.5, alpha=0.7, color=color, linestyle='--')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_impulse_response(
    irf_data: pd.DataFrame,
    shock_var: str,
    response_var: str,
    title: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot impulse response function.

    Args:
        irf_data: IRF data from VAR model
        shock_var: Variable that receives shock
        response_var: Variable whose response is plotted
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    col_name = f"{shock_var}_to_{response_var}"
    if col_name not in irf_data.columns:
        print(f"Warning: {col_name} not found in IRF data")
        return fig

    irf = irf_data[col_name]
    periods = len(irf)

    ax.plot(range(periods), irf.values, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(range(periods), 0, irf.values, alpha=0.2, color='steelblue')

    if title is None:
        title = f"Impulse Response: {shock_var} → {response_var}"

    ax.set_xlabel('Periods', fontsize=12, fontweight='bold')
    ax.set_ylabel('Response', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_forecast_error_variance_decomposition(
    fevd_data: pd.DataFrame,
    response_var: str,
    periods: int = 20,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot forecast error variance decomposition as stacked area chart.

    Args:
        fevd_data: FEVD data from VAR model
        response_var: Response variable to plot
        periods: Number of periods to plot
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter for specific response variable
    var_data = fevd_data[fevd_data['response_variable'] == response_var].iloc[:periods]

    # Get shock columns
    shock_cols = [col for col in var_data.columns if col.startswith('shock_from_')]

    # Plot stacked area
    var_data[shock_cols].plot.area(ax=ax, alpha=0.7, stacked=True)

    ax.set_xlabel('Periods', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Share', fontsize=12, fontweight='bold')
    ax.set_title(f'Forecast Error Variance Decomposition: {response_var}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Shock from', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_interactive_dashboard(
    data: pd.DataFrame,
    title: str = "Economic Dashboard"
) -> go.Figure:
    """
    Create interactive Plotly dashboard with multiple panels.

    Args:
        data: DataFrame with multiple economic series
        title: Dashboard title

    Returns:
        Plotly Figure object
    """
    n_series = len(data.columns)
    fig = make_subplots(
        rows=n_series,
        cols=1,
        subplot_titles=data.columns.tolist(),
        vertical_spacing=0.05
    )

    colors = plt.cm.tab10(np.linspace(0, 1, n_series))

    for idx, col in enumerate(data.columns, 1):
        color_rgb = f"rgb({int(colors[idx-1][0]*255)}, {int(colors[idx-1][1]*255)}, {int(colors[idx-1][2]*255)})"

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                mode='lines',
                line=dict(width=2, color=color_rgb),
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=idx,
            col=1
        )

    fig.update_layout(
        title=title,
        height=300 * n_series,
        showlegend=False,
        hovermode='x unified',
        template='plotly_white'
    )

    for idx in range(1, n_series + 1):
        fig.update_xaxes(title_text="Date" if idx == n_series else "", row=idx, col=1)
        fig.update_yaxes(title_text="Value", row=idx, col=1)

    return fig


def plot_model_performance(
    results: pd.DataFrame,
    metric: str = 'RMSE',
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Compare model performance across different models.

    Args:
        results: DataFrame with model performance metrics
        metric: Metric to plot (column name in results)
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = results['Model'] if 'Model' in results.columns else results.index
    values = results[metric]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

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

    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Comparison ({metric})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_residual_diagnostics(
    residuals: pd.Series,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Create comprehensive residual diagnostic plots.

    Args:
        residuals: Model residuals
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Time series plot
    axes[0, 0].plot(residuals.index, residuals.values, linewidth=1, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(residuals.dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_title('Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation of Residuals', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_cross_country_comparison(
    data: pd.DataFrame,
    indicator: str,
    countries: List[str],
    title: Optional[str] = None,
    figsize: tuple = (14, 7)
) -> plt.Figure:
    """
    Compare a single indicator across multiple countries.

    Args:
        data: Panel data with multi-level columns
        indicator: Indicator name to plot
        countries: List of countries to compare
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for country in countries:
        if (country, indicator) in data.columns:
            series = data[(country, indicator)]
            ax.plot(series.index, series.values, label=country, linewidth=2, alpha=0.8)

    if title is None:
        title = f'{indicator} - Cross-Country Comparison'

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(indicator, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
