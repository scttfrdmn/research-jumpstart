"""
Feature engineering for agricultural analysis.

Functions for calculating vegetation indices, phenology metrics,
and temporal features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy.signal import savgol_filter


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        red: Red band reflectance
        nir: Near-infrared band reflectance

    Returns:
        NDVI values (-1 to 1)
    """
    return (nir - red) / (nir + red + 1e-8)


def calculate_evi(
    red: np.ndarray,
    nir: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0
) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index.

    EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))

    Args:
        red: Red band reflectance
        nir: Near-infrared band reflectance
        blue: Blue band reflectance
        G: Gain factor
        C1, C2: Aerosol resistance coefficients
        L: Canopy background adjustment

    Returns:
        EVI values
    """
    return G * ((nir - red) / (nir + C1 * red - C2 * blue + L + 1e-8))


def calculate_savi(
    red: np.ndarray,
    nir: np.ndarray,
    L: float = 0.5
) -> np.ndarray:
    """
    Calculate Soil Adjusted Vegetation Index.

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Args:
        red: Red band reflectance
        nir: Near-infrared band reflectance
        L: Soil brightness correction factor

    Returns:
        SAVI values
    """
    return ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)


def calculate_lai(ndvi: np.ndarray) -> np.ndarray:
    """
    Estimate Leaf Area Index from NDVI.

    Uses empirical relationship: LAI = a * ln(1 - NDVI) + b

    Args:
        ndvi: NDVI values

    Returns:
        LAI estimates (m²/m²)
    """
    # Empirical coefficients (crop-specific)
    a = -1.0
    b = 3.0

    # Clip NDVI to valid range
    ndvi_clipped = np.clip(ndvi, 0, 0.95)

    # Calculate LAI
    lai = a * np.log(1 - ndvi_clipped) + b
    lai = np.clip(lai, 0, 8)  # Reasonable LAI range

    return lai


def extract_phenology_metrics(
    ndvi_timeseries: np.ndarray,
    dates: pd.DatetimeIndex,
    smooth: bool = True
) -> Dict[str, float]:
    """
    Extract crop phenology metrics from NDVI time series.

    Identifies key growth stages:
    - Green-up: Start of growing season
    - Peak: Maximum vegetation
    - Senescence: Start of decline
    - Harvest: End of season

    Args:
        ndvi_timeseries: NDVI values over time
        dates: Corresponding dates
        smooth: Apply smoothing filter

    Returns:
        Dictionary with phenology metrics
    """
    # Smooth time series
    if smooth and len(ndvi_timeseries) >= 5:
        ndvi_smooth = savgol_filter(ndvi_timeseries, window_length=5, polyorder=2)
    else:
        ndvi_smooth = ndvi_timeseries

    # Find key points
    peak_idx = np.argmax(ndvi_smooth)
    peak_ndvi = ndvi_smooth[peak_idx]
    peak_date = dates[peak_idx]

    # Green-up: first time NDVI crosses 50% of peak
    threshold_greenup = 0.5 * peak_ndvi
    greenup_idx = np.where(ndvi_smooth >= threshold_greenup)[0]
    if len(greenup_idx) > 0:
        greenup_date = dates[greenup_idx[0]]
    else:
        greenup_date = dates[0]

    # Senescence: after peak, first time NDVI drops below 80% of peak
    threshold_senescence = 0.8 * peak_ndvi
    senescence_candidates = np.where(
        (ndvi_smooth < threshold_senescence) & (np.arange(len(ndvi_smooth)) > peak_idx)
    )[0]
    if len(senescence_candidates) > 0:
        senescence_date = dates[senescence_candidates[0]]
    else:
        senescence_date = dates[-1]

    # Calculate derived metrics
    growing_season_length = (senescence_date - greenup_date).days
    time_to_peak = (peak_date - greenup_date).days

    # Integrated NDVI (area under curve)
    integrated_ndvi = np.trapz(ndvi_smooth, dx=1)

    return {
        'greenup_date': greenup_date,
        'peak_date': peak_date,
        'senescence_date': senescence_date,
        'peak_ndvi': peak_ndvi,
        'growing_season_length': growing_season_length,
        'time_to_peak': time_to_peak,
        'integrated_ndvi': integrated_ndvi,
    }


def create_temporal_features(
    timeseries_data: pd.DataFrame,
    window_sizes: list = [7, 14, 30]
) -> pd.DataFrame:
    """
    Create temporal features from time series data.

    Features include:
    - Moving averages
    - Rate of change
    - Cumulative sums
    - Lagged values

    Args:
        timeseries_data: DataFrame with time series columns
        window_sizes: Window sizes for moving averages (days)

    Returns:
        DataFrame with additional temporal features
    """
    result = timeseries_data.copy()

    # Numeric columns only
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Moving averages
        for window in window_sizes:
            result[f'{col}_ma_{window}'] = result[col].rolling(window=window).mean()

        # Rate of change
        result[f'{col}_rate'] = result[col].diff()
        result[f'{col}_rate_pct'] = result[col].pct_change()

        # Cumulative sum
        result[f'{col}_cumsum'] = result[col].cumsum()

        # Lagged values
        for lag in [1, 7, 14]:
            result[f'{col}_lag_{lag}'] = result[col].shift(lag)

    return result


def calculate_growing_degree_days(
    temp_min: np.ndarray,
    temp_max: np.ndarray,
    base_temp: float = 10.0,
    upper_limit: float = 30.0
) -> np.ndarray:
    """
    Calculate Growing Degree Days (GDD).

    GDD = max(0, (Tmax + Tmin)/2 - Tbase)
    Capped at upper_limit

    Args:
        temp_min: Daily minimum temperature (°C)
        temp_max: Daily maximum temperature (°C)
        base_temp: Base temperature for crop growth
        upper_limit: Upper temperature limit

    Returns:
        Daily GDD values
    """
    # Calculate mean temperature
    temp_mean = (temp_min + temp_max) / 2

    # Cap at upper limit
    temp_mean_capped = np.minimum(temp_mean, upper_limit)

    # Calculate GDD
    gdd = np.maximum(0, temp_mean_capped - base_temp)

    return gdd


def create_spatial_features(
    image: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    Create spatial context features from image.

    Features include:
    - Local mean
    - Local standard deviation
    - Texture metrics

    Args:
        image: Input image (height, width, bands)
        window_size: Window size for spatial features

    Returns:
        Feature array (height, width, n_features)
    """
    from scipy.ndimage import uniform_filter

    features = []

    for band in range(image.shape[2]):
        band_data = image[:, :, band]

        # Local mean
        local_mean = uniform_filter(band_data, size=window_size)
        features.append(local_mean)

        # Local standard deviation
        local_var = uniform_filter(band_data**2, size=window_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        features.append(local_std)

    # Stack features
    feature_stack = np.stack(features, axis=-1)

    return feature_stack


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize features to zero mean and unit variance.

    Args:
        features: Feature array (n_samples, n_features)

    Returns:
        Tuple of (normalized_features, normalization_params)
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    normalized = (features - mean) / std

    params = {
        'mean': mean,
        'std': std
    }

    return normalized, params
