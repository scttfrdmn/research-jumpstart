"""
Visualization utilities for agricultural analysis.

Functions for plotting time series, maps, and model results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


def plot_timeseries(
    data: pd.DataFrame,
    columns: List[str],
    title: str = "Time Series",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot time series data.

    Args:
        data: DataFrame with DatetimeIndex
        columns: List of columns to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in columns:
        ax.plot(data.index, data[col], label=col, linewidth=2)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_field_map(
    image: np.ndarray,
    title: str = "Field Map",
    cmap: str = 'RdYlGn',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot spatial field map.

    Args:
        image: 2D array (height, width)
        title: Plot title
        cmap: Colormap
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=11)

    plt.tight_layout()
    return fig


def plot_yield_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Yield Prediction",
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Plot predicted vs actual yield.

    Args:
        y_true: True yield values
        y_pred: Predicted yield values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=50)

    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Add metrics to plot
    metrics_text = f'R² = {r2:.3f}\\nMAE = {mae:.1f}\\nRMSE = {rmse:.1f}'
    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('True Yield (kg/ha)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Yield (kg/ha)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importance_values: Importance values
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importance_values)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(top_n), importance_values[indices], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()

    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_confusion_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spatial_coords: np.ndarray,
    title: str = "Prediction Error Map",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot spatial map of prediction errors.

    Args:
        y_true: True values
        y_pred: Predicted values
        spatial_coords: Spatial coordinates (n_samples, 2)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Error map
    scatter1 = axes[0].scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        c=errors,
        cmap='RdBu_r',
        s=100,
        alpha=0.7,
        vmin=-abs_errors.max(),
        vmax=abs_errors.max()
    )
    axes[0].set_title('Prediction Error (Pred - True)', fontweight='bold')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    plt.colorbar(scatter1, ax=axes[0], label='Error')

    # Absolute error map
    scatter2 = axes[1].scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        c=abs_errors,
        cmap='Reds',
        s=100,
        alpha=0.7
    )
    axes[1].set_title('Absolute Error', fontweight='bold')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    plt.colorbar(scatter2, ax=axes[1], label='Absolute Error')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_interactive_map(
    data: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
    title: str = "Interactive Map"
):
    """
    Create interactive map using Folium.

    Args:
        data: DataFrame with location and value data
        lat_col: Latitude column name
        lon_col: Longitude column name
        value_col: Value column name
        title: Map title

    Returns:
        Folium map object
    """
    import folium
    from folium.plugins import HeatMap

    # Center map
    center_lat = data[lat_col].mean()
    center_lon = data[lon_col].mean()

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )

    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add markers
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            popup=f"{value_col}: {row[value_col]:.1f}",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)

    # Add heatmap layer
    heat_data = [[row[lat_col], row[lon_col], row[value_col]]
                 for idx, row in data.iterrows()]
    HeatMap(heat_data, radius=15, blur=25, max_zoom=1).add_to(m)

    return m


def plot_training_history(
    history,
    metrics: List[str] = ['loss', 'mae'],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training history.

    Args:
        history: Keras training history object
        metrics: List of metrics to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        axes[i].plot(history.history[metric], label='Training', linewidth=2)
        if f'val_{metric}' in history.history:
            axes[i].plot(history.history[f'val_{metric}'], label='Validation', linewidth=2)

        axes[i].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[i].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        axes[i].set_title(f'{metric.upper()} over Epochs', fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ndvi_evolution(
    ndvi_timeseries: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str = "NDVI Evolution",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot NDVI time series with phenology annotations.

    Args:
        ndvi_timeseries: NDVI values over time
        dates: Corresponding dates
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from src.features import extract_phenology_metrics

    fig, ax = plt.subplots(figsize=figsize)

    # Plot NDVI
    ax.plot(dates, ndvi_timeseries, 'o-', linewidth=2, markersize=6, label='NDVI')

    # Extract and plot phenology
    phenology = extract_phenology_metrics(ndvi_timeseries, dates)

    # Mark key dates
    ax.axvline(phenology['greenup_date'], color='green', linestyle='--',
               label='Green-up', alpha=0.7)
    ax.axvline(phenology['peak_date'], color='red', linestyle='--',
               label='Peak', alpha=0.7)
    ax.axvline(phenology['senescence_date'], color='orange', linestyle='--',
               label='Senescence', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text annotation
    metrics_text = (f"Peak NDVI: {phenology['peak_ndvi']:.2f}\\n"
                   f"Growing Season: {phenology['growing_season_length']} days")
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig
