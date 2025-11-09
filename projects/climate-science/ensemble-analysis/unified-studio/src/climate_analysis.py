"""
Core climate data analysis functions.

This module provides functions for common climate analysis tasks including
regional averaging, anomaly calculation, and temporal processing.
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_regional_mean(
    ds: xr.Dataset,
    variable: str,
    region: Dict[str, float],
    weights: str = 'cosine'
) -> xr.DataArray:
    """
    Calculate area-weighted regional mean.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable
    variable : str
        Variable name (e.g., 'tas', 'pr')
    region : dict
        Regional bounds with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    weights : str, default 'cosine'
        Weighting method: 'cosine' (latitude cosine), 'area' (grid cell area),
        or 'none' (no weighting)

    Returns
    -------
    xr.DataArray
        Regional mean time series

    Examples
    --------
    >>> region = {'lat_min': 31, 'lat_max': 37, 'lon_min': -114, 'lon_max': -109}
    >>> regional_mean = calculate_regional_mean(ds, 'tas', region)
    """
    logger.info(f"Calculating regional mean for {variable}")

    # Extract variable
    data = ds[variable]

    # Spatial subset
    data_region = data.sel(
        lat=slice(region['lat_min'], region['lat_max']),
        lon=slice(region['lon_min'], region['lon_max'])
    )

    # Calculate weights
    if weights == 'cosine':
        # Cosine of latitude (accounts for grid convergence at poles)
        lat_weights = np.cos(np.deg2rad(data_region.lat))
        weighted_data = data_region.weighted(lat_weights)
    elif weights == 'area':
        # True grid cell area (requires bounds or calculation)
        if 'cell_area' in ds:
            area_weights = ds['cell_area'].sel(
                lat=slice(region['lat_min'], region['lat_max']),
                lon=slice(region['lon_min'], region['lon_max'])
            )
            weighted_data = data_region.weighted(area_weights)
        else:
            logger.warning("Cell area not found, falling back to cosine weighting")
            lat_weights = np.cos(np.deg2rad(data_region.lat))
            weighted_data = data_region.weighted(lat_weights)
    elif weights == 'none':
        weighted_data = data_region
    else:
        raise ValueError(f"Unknown weighting method: {weights}")

    # Calculate mean over spatial dimensions
    regional_mean = weighted_data.mean(['lat', 'lon'])

    logger.info(
        f"Regional mean calculated for "
        f"{region['lat_min']}°-{region['lat_max']}°N, "
        f"{region['lon_min']}°-{region['lon_max']}°E"
    )

    return regional_mean


def calculate_anomaly(
    data: xr.DataArray,
    baseline_period: Tuple[str, str],
    method: str = 'difference'
) -> xr.DataArray:
    """
    Calculate anomaly relative to baseline period.

    Parameters
    ----------
    data : xr.DataArray
        Input time series
    baseline_period : tuple of str
        (start, end) dates for baseline, e.g., ('1995-01-01', '2014-12-31')
    method : str, default 'difference'
        Anomaly method: 'difference' (data - baseline) or
        'percent' ((data - baseline) / baseline * 100)

    Returns
    -------
    xr.DataArray
        Anomaly time series

    Examples
    --------
    >>> anomaly = calculate_anomaly(tas_data, ('1995', '2014'))
    """
    logger.info(f"Calculating {method} anomaly relative to {baseline_period}")

    # Extract baseline period
    baseline = data.sel(time=slice(baseline_period[0], baseline_period[1]))

    # Calculate baseline climatology
    baseline_mean = baseline.mean('time')

    # Calculate anomaly
    if method == 'difference':
        anomaly = data - baseline_mean
    elif method == 'percent':
        anomaly = (data - baseline_mean) / baseline_mean * 100
    else:
        raise ValueError(f"Unknown anomaly method: {method}")

    logger.info(f"Baseline mean: {float(baseline_mean):.3f}")

    return anomaly


def annual_mean(data: xr.DataArray) -> xr.DataArray:
    """
    Calculate annual mean from monthly or daily data.

    Parameters
    ----------
    data : xr.DataArray
        Input data with time dimension

    Returns
    -------
    xr.DataArray
        Annual mean time series

    Examples
    --------
    >>> annual_tas = annual_mean(monthly_tas)
    """
    logger.info("Calculating annual means")

    # Group by year and calculate mean
    annual_data = data.groupby('time.year').mean('time')

    # Rename 'year' coordinate to 'time' for consistency
    annual_data = annual_data.rename({'year': 'time'})

    logger.info(f"Resampled to {len(annual_data.time)} years")

    return annual_data


def seasonal_mean(
    data: xr.DataArray,
    season: str = 'DJF'
) -> xr.DataArray:
    """
    Calculate seasonal mean.

    Parameters
    ----------
    data : xr.DataArray
        Input data with time dimension
    season : str, default 'DJF'
        Season code: 'DJF' (winter), 'MAM' (spring), 'JJA' (summer), 'SON' (fall)

    Returns
    -------
    xr.DataArray
        Seasonal mean time series

    Examples
    --------
    >>> summer_tas = seasonal_mean(monthly_tas, season='JJA')
    """
    logger.info(f"Calculating {season} seasonal means")

    # Group by season and calculate mean
    seasonal_data = data.groupby('time.season').mean('time')

    # Select requested season
    seasonal_data = seasonal_data.sel(season=season)

    logger.info(f"Extracted {season} season")

    return seasonal_data


def calculate_trend(
    data: xr.DataArray,
    time_period: Optional[Tuple[str, str]] = None
) -> Tuple[float, float]:
    """
    Calculate linear trend using least squares regression.

    Parameters
    ----------
    data : xr.DataArray
        Input time series
    time_period : tuple of str, optional
        (start, end) dates for trend calculation

    Returns
    -------
    tuple of float
        (slope, intercept) where slope is in units per year

    Examples
    --------
    >>> slope, intercept = calculate_trend(annual_anomaly, ('2015', '2050'))
    >>> print(f"Warming rate: {slope:.3f} °C/year")
    """
    if time_period is not None:
        data = data.sel(time=slice(time_period[0], time_period[1]))

    # Convert time to years since start for regression
    time_years = (data.time - data.time[0]) / np.timedelta64(365, 'D')

    # Remove NaN values
    valid_mask = ~np.isnan(data.values)
    x = time_years.values[valid_mask]
    y = data.values[valid_mask]

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)

    logger.info(f"Trend: {slope:.4f} units/year")

    return float(slope), float(intercept)


def running_mean(
    data: xr.DataArray,
    window: int = 10,
    center: bool = True
) -> xr.DataArray:
    """
    Calculate running mean (moving average).

    Parameters
    ----------
    data : xr.DataArray
        Input time series
    window : int, default 10
        Window size in time steps
    center : bool, default True
        If True, center the window on each point

    Returns
    -------
    xr.DataArray
        Smoothed time series

    Examples
    --------
    >>> smoothed = running_mean(annual_data, window=10)
    """
    logger.info(f"Calculating {window}-point running mean")

    smoothed = data.rolling(time=window, center=center).mean()

    return smoothed


def detrend(
    data: xr.DataArray,
    method: str = 'linear'
) -> xr.DataArray:
    """
    Remove linear or quadratic trend from time series.

    Parameters
    ----------
    data : xr.DataArray
        Input time series
    method : str, default 'linear'
        Detrending method: 'linear' or 'quadratic'

    Returns
    -------
    xr.DataArray
        Detrended time series

    Examples
    --------
    >>> detrended = detrend(temperature_data, method='linear')
    """
    logger.info(f"Detrending using {method} method")

    # Convert time to numeric for fitting
    time_numeric = (data.time - data.time[0]) / np.timedelta64(1, 'D')

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(data.values)
    x = time_numeric.values[valid_mask]
    y = data.values[valid_mask]

    # Fit polynomial
    if method == 'linear':
        coeffs = np.polyfit(x, y, 1)
    elif method == 'quadratic':
        coeffs = np.polyfit(x, y, 2)
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    # Calculate trend
    trend = np.polyval(coeffs, time_numeric.values)

    # Remove trend
    detrended = data - trend

    return detrended


def calculate_climatology(
    data: xr.DataArray,
    period: Optional[Tuple[str, str]] = None,
    groupby: str = 'month'
) -> xr.DataArray:
    """
    Calculate climatological mean (seasonal cycle).

    Parameters
    ----------
    data : xr.DataArray
        Input data with time dimension
    period : tuple of str, optional
        (start, end) dates for climatology calculation
    groupby : str, default 'month'
        Grouping: 'month', 'dayofyear', or 'season'

    Returns
    -------
    xr.DataArray
        Climatology (12 months, 365 days, or 4 seasons)

    Examples
    --------
    >>> monthly_clim = calculate_climatology(tas_data, ('1995', '2014'))
    """
    logger.info(f"Calculating climatology by {groupby}")

    if period is not None:
        data = data.sel(time=slice(period[0], period[1]))

    # Group by time component and calculate mean
    if groupby == 'month':
        climatology = data.groupby('time.month').mean('time')
    elif groupby == 'dayofyear':
        climatology = data.groupby('time.dayofyear').mean('time')
    elif groupby == 'season':
        climatology = data.groupby('time.season').mean('time')
    else:
        raise ValueError(f"Unknown groupby: {groupby}")

    logger.info(f"Climatology calculated over {len(data.time)} time steps")

    return climatology


def convert_units(
    data: xr.DataArray,
    from_unit: str,
    to_unit: str
) -> xr.DataArray:
    """
    Convert between common climate variable units.

    Parameters
    ----------
    data : xr.DataArray
        Input data
    from_unit : str
        Current unit (e.g., 'K', 'degC', 'mm/day', 'kg/m2/s')
    to_unit : str
        Target unit

    Returns
    -------
    xr.DataArray
        Data in new units

    Examples
    --------
    >>> tas_celsius = convert_units(tas_kelvin, 'K', 'degC')
    """
    logger.info(f"Converting from {from_unit} to {to_unit}")

    # Temperature conversions
    if from_unit == 'K' and to_unit in ['degC', 'C']:
        converted = data - 273.15
    elif from_unit in ['degC', 'C'] and to_unit == 'K':
        converted = data + 273.15

    # Precipitation conversions
    elif from_unit == 'kg/m2/s' and to_unit == 'mm/day':
        converted = data * 86400  # seconds per day
    elif from_unit == 'mm/day' and to_unit == 'kg/m2/s':
        converted = data / 86400

    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not implemented")

    # Update attributes
    converted.attrs = data.attrs.copy()
    converted.attrs['units'] = to_unit

    return converted
