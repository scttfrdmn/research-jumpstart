"""
Visualization utilities for climate ensemble analysis.

This module provides functions to create publication-quality figures for
climate model ensemble results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_ensemble_timeseries(
    ensemble: xr.DataArray,
    ens_mean: Optional[xr.DataArray] = None,
    ens_std: Optional[xr.DataArray] = None,
    title: str = 'Multi-Model Ensemble Projection',
    ylabel: str = 'Temperature Anomaly (°C)',
    figsize: Tuple[float, float] = (12, 6),
    show_individual: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ensemble time series with individual models and statistics.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    ens_mean : xr.DataArray, optional
        Ensemble mean (calculated if not provided)
    ens_std : xr.DataArray, optional
        Ensemble standard deviation (calculated if not provided)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size (width, height)
    show_individual : bool, default True
        Whether to plot individual model traces
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_ensemble_timeseries(ensemble, title='US Southwest Projection')
    >>> plt.show()
    """
    logger.info("Creating ensemble time series plot")

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate statistics if not provided
    if ens_mean is None:
        ens_mean = ensemble.mean('model')
    if ens_std is None:
        ens_std = ensemble.std('model')

    time = ens_mean.time.values

    # Plot individual models
    if show_individual:
        for model in ensemble.model.values:
            model_data = ensemble.sel(model=model)
            ax.plot(
                time, model_data.values,
                alpha=0.3, linewidth=1.0, color='gray',
                label='_nolegend_'
            )

    # Plot ensemble mean
    ax.plot(
        time, ens_mean.values,
        color='black', linewidth=2.5,
        label='Ensemble Mean', zorder=10
    )

    # Plot uncertainty range (±1σ)
    ax.fill_between(
        time,
        (ens_mean - ens_std).values,
        (ens_mean + ens_std).values,
        alpha=0.3, color='blue',
        label='±1σ Spread', zorder=5
    )

    # Formatting
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add reference line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Adjust x-axis format for dates
    if len(time) > 20:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_model_agreement(
    ensemble: xr.DataArray,
    agreement: Optional[xr.DataArray] = None,
    title: str = 'Model Agreement Analysis',
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model agreement with decadal box plots.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    agreement : xr.DataArray, optional
        Model agreement metric (calculated if not provided)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> fig = plot_model_agreement(ensemble)
    >>> plt.show()
    """
    logger.info("Creating model agreement plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Decadal box plots
    # Group data by decade
    years = ensemble.time.dt.year.values
    decades = (years // 10) * 10
    unique_decades = np.unique(decades)

    box_data = []
    box_labels = []
    for decade in unique_decades:
        decade_mask = (decades >= decade) & (decades < decade + 10)
        decade_data = ensemble.isel(time=decade_mask).values.flatten()
        box_data.append(decade_data[~np.isnan(decade_data)])
        box_labels.append(f'{decade}s')

    bp = ax1.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=5)
    )

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax1.set_xlabel('Decade')
    ax1.set_ylabel('Value')
    ax1.set_title('Decadal Model Spread')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Panel 2: Model spread over time
    ens_mean = ensemble.mean('model')
    ens_min = ensemble.min('model')
    ens_max = ensemble.max('model')
    p25 = ensemble.quantile(0.25, dim='model')
    p75 = ensemble.quantile(0.75, dim='model')

    time = ens_mean.time.values

    # Plot range
    ax2.fill_between(
        time, ens_min.values, ens_max.values,
        alpha=0.2, color='gray', label='Full Range'
    )

    # Plot IQR
    ax2.fill_between(
        time, p25.values, p75.values,
        alpha=0.4, color='blue', label='Interquartile Range'
    )

    # Plot mean
    ax2.plot(time, ens_mean.values, 'k-', linewidth=2, label='Ensemble Mean')

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Value')
    ax2.set_title('Model Spread Evolution')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Adjust x-axis
    if len(time) > 20:
        ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_regional_map(
    data: xr.DataArray,
    region: Dict[str, float],
    title: str = 'Regional Analysis Domain',
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot map of regional analysis domain.

    Parameters
    ----------
    data : xr.DataArray
        Spatial data to plot (2D: lat, lon)
    region : dict
        Regional bounds with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> region = {'lat_min': 31, 'lat_max': 37, 'lon_min': -114, 'lon_max': -109}
    >>> fig = plot_regional_map(spatial_mean, region, title='US Southwest')
    """
    logger.info("Creating regional map")

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot data
    im = data.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        cbar_kwargs={'label': data.attrs.get('units', ''), 'shrink': 0.8}
    )

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')

    # Draw analysis region box
    ax.plot(
        [region['lon_min'], region['lon_max'], region['lon_max'], region['lon_min'], region['lon_min']],
        [region['lat_min'], region['lat_min'], region['lat_max'], region['lat_max'], region['lat_min']],
        color='red', linewidth=2, transform=ccrs.PlateCarree(), label='Analysis Region'
    )

    # Set extent (zoom to region with buffer)
    buffer = 5
    ax.set_extent([
        region['lon_min'] - buffer,
        region['lon_max'] + buffer,
        region['lat_min'] - buffer,
        region['lat_max'] + buffer
    ], crs=ccrs.PlateCarree())

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def plot_scenario_comparison(
    scenarios: Dict[str, xr.DataArray],
    title: str = 'Scenario Comparison',
    ylabel: str = 'Temperature Anomaly (°C)',
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple emissions scenarios.

    Parameters
    ----------
    scenarios : dict
        Dictionary mapping scenario names to ensemble means
    title : str
        Plot title
    ylabel : str
        Y-axis label
    colors : dict, optional
        Dictionary mapping scenario names to colors
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object

    Examples
    --------
    >>> scenarios = {
    ...     'SSP1-2.6': ssp126_mean,
    ...     'SSP2-4.5': ssp245_mean,
    ...     'SSP5-8.5': ssp585_mean
    ... }
    >>> fig = plot_scenario_comparison(scenarios)
    """
    logger.info(f"Creating scenario comparison plot for {len(scenarios)} scenarios")

    if colors is None:
        colors = {
            'SSP1-2.6': 'green',
            'SSP2-4.5': 'orange',
            'SSP5-8.5': 'red',
            'historical': 'black'
        }

    fig, ax = plt.subplots(figsize=figsize)

    for scenario_name, data in scenarios.items():
        color = colors.get(scenario_name, 'blue')
        ax.plot(
            data.time.values, data.values,
            label=scenario_name, linewidth=2.5, color=color
        )

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig


def create_summary_figure(
    ensemble: xr.DataArray,
    region: Dict[str, float],
    spatial_data: Optional[xr.DataArray] = None,
    title: str = 'Climate Projection Summary',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive multi-panel summary figure.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    region : dict
        Regional bounds
    spatial_data : xr.DataArray, optional
        Spatial data for map panel
    title : str
        Overall figure title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    logger.info("Creating comprehensive summary figure")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Time series
    ax1 = fig.add_subplot(gs[0, :])
    ens_mean = ensemble.mean('model')
    ens_std = ensemble.std('model')
    time = ens_mean.time.values

    for model in ensemble.model.values:
        ax1.plot(time, ensemble.sel(model=model).values, alpha=0.3, linewidth=1, color='gray')

    ax1.plot(time, ens_mean.values, 'k-', linewidth=2.5, label='Ensemble Mean')
    ax1.fill_between(time, (ens_mean - ens_std).values, (ens_mean + ens_std).values,
                      alpha=0.3, color='blue', label='±1σ')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature Anomaly (°C)')
    ax1.set_title('Ensemble Projection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Panel 2: Box plots by decade
    ax2 = fig.add_subplot(gs[1, 0])
    years = ensemble.time.dt.year.values
    decades = (years // 10) * 10
    unique_decades = np.unique(decades)
    box_data = []
    box_labels = []
    for decade in unique_decades:
        decade_mask = (decades >= decade) & (decades < decade + 10)
        decade_data = ensemble.isel(time=decade_mask).values.flatten()
        box_data.append(decade_data[~np.isnan(decade_data)])
        box_labels.append(f'{decade}s')

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Decade')
    ax2.set_ylabel('Temperature Anomaly (°C)')
    ax2.set_title('Decadal Spread')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Map (if spatial data provided)
    if spatial_data is not None:
        ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        spatial_data.plot(ax=ax3, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                          cbar_kwargs={'shrink': 0.8})
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax3.plot(
            [region['lon_min'], region['lon_max'], region['lon_max'], region['lon_min'], region['lon_min']],
            [region['lat_min'], region['lat_min'], region['lat_max'], region['lat_max'], region['lat_min']],
            'r-', linewidth=2, transform=ccrs.PlateCarree()
        )
        ax3.set_title('Analysis Region')

    fig.suptitle(title, fontsize=14, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return fig
