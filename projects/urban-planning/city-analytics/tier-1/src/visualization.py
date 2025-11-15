"""
Visualization functions for urban planning analysis.
"""

from typing import TYPE_CHECKING

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

if TYPE_CHECKING:
    import geopandas


def plot_urban_growth(
    imagery_series: list[np.ndarray], years: list[int], city_name: str, figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot urban growth time series from satellite imagery.

    Parameters:
    -----------
    imagery_series : list of np.ndarray
        List of imagery arrays over time
    years : list of int
        Corresponding years
    city_name : str
        Name of city for title
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    n_images = len(imagery_series)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    if n_images == 1:
        axes = [axes]

    for idx, (img, year) in enumerate(zip(imagery_series, years)):
        axes[idx].imshow(img)
        axes[idx].set_title(f"{city_name} - {year}")
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_mobility_heatmap(
    traffic_data: pd.DataFrame,
    metric: str = "aadt",
    title: str = "Traffic Flow Heatmap",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create heatmap of mobility metrics.

    Parameters:
    -----------
    traffic_data : pandas.DataFrame
        Traffic data with spatial coordinates
    metric : str
        Metric to visualize ('aadt', 'v_c_ratio', 'speed')
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if "x" in traffic_data.columns and "y" in traffic_data.columns:
        scatter = ax.scatter(
            traffic_data["x"],
            traffic_data["y"],
            c=traffic_data[metric],
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label=metric.upper())

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.tight_layout()

    return fig


def create_city_comparison_dashboard(
    city_data: dict[str, pd.DataFrame], metrics: list[str]
) -> go.Figure:
    """
    Create interactive dashboard comparing multiple cities.

    Parameters:
    -----------
    city_data : dict
        Dictionary mapping city names to DataFrames with metrics
    metrics : list of str
        List of metrics to compare

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    n_metrics = len(metrics)
    n_cities = len(city_data)

    # Create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=n_metrics, subplot_titles=metrics, horizontal_spacing=0.1)

    colors = px.colors.qualitative.Set2[:n_cities]

    for col_idx, metric in enumerate(metrics, start=1):
        for city_idx, (city_name, df) in enumerate(city_data.items()):
            if metric in df.columns:
                fig.add_trace(
                    go.Bar(
                        name=city_name,
                        x=[city_name],
                        y=[df[metric].mean()],
                        marker_color=colors[city_idx],
                        showlegend=(col_idx == 1),  # Only show legend once
                    ),
                    row=1,
                    col=col_idx,
                )

    fig.update_layout(title_text="Multi-City Comparison Dashboard", height=500, showlegend=True)

    return fig


def create_interactive_map(
    gdf: "geopandas.GeoDataFrame", metric: str, center: tuple, zoom: int = 10, cmap: str = "YlOrRd"
) -> folium.Map:
    """
    Create interactive folium map with metric visualization.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometries and metrics
    metric : str
        Column name of metric to visualize
    center : tuple
        Map center coordinates (lat, lon)
    zoom : int
        Initial zoom level
    cmap : str
        Colormap name

    Returns:
    --------
    folium.Map
        Interactive map object
    """
    m = folium.Map(location=center, zoom_start=zoom)

    # Add choropleth layer
    if metric in gdf.columns:
        folium.Choropleth(
            geo_data=gdf,
            data=gdf,
            columns=["id", metric],
            key_on="feature.properties.id",
            fill_color=cmap,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=metric,
        ).add_to(m)

    return m


def plot_time_series_comparison(
    data_dict: dict[str, pd.Series],
    title: str = "Time Series Comparison",
    ylabel: str = "Value",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot multiple time series on same axes.

    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping series names to pandas Series
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, series in data_dict.items():
        ax.plot(series.index, series.values, label=name, marker="o", linewidth=2)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, title: str = "Feature Correlations", figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with numeric columns
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate correlation matrix
    corr = df.corr()

    # Create heatmap
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=1, ax=ax
    )

    ax.set_title(title)
    plt.tight_layout()

    return fig
