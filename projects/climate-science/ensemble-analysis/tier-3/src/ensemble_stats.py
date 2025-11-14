"""
Statistical methods for multi-model ensemble analysis.

This module provides functions to compute ensemble statistics, assess model
agreement, and quantify uncertainty across climate model projections.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ensemble(
    model_data: Dict[str, xr.DataArray],
    align_time: bool = True
) -> xr.DataArray:
    """
    Combine multiple model outputs into an ensemble array.

    Parameters
    ----------
    model_data : dict
        Dictionary mapping model names to DataArrays
    align_time : bool, default True
        If True, align all models to common time axis

    Returns
    -------
    xr.DataArray
        Ensemble with 'model' dimension

    Examples
    --------
    >>> ensemble = create_ensemble({
    ...     'CESM2': cesm_data,
    ...     'GFDL-CM4': gfdl_data,
    ...     'UKESM1-0-LL': ukesm_data
    ... })
    """
    logger.info(f"Creating ensemble from {len(model_data)} models")

    # Convert dict to list of DataArrays with model coordinate
    model_list = []
    for model_name, data in model_data.items():
        data_copy = data.copy()
        data_copy.coords['model'] = model_name
        model_list.append(data_copy)

    # Concatenate along model dimension
    if align_time:
        # Find common time period
        time_starts = [data.time.values[0] for data in model_list]
        time_ends = [data.time.values[-1] for data in model_list]
        common_start = max(time_starts)
        common_end = min(time_ends)

        logger.info(f"Aligning to common period: {common_start} to {common_end}")

        # Subset all models to common period
        model_list = [
            data.sel(time=slice(common_start, common_end))
            for data in model_list
        ]

    ensemble = xr.concat(model_list, dim='model')

    logger.info(
        f"Ensemble shape: {ensemble.shape} "
        f"({len(ensemble.model)} models × {len(ensemble.time)} time steps)"
    )

    return ensemble


def ensemble_mean(ensemble: xr.DataArray) -> xr.DataArray:
    """
    Calculate ensemble mean.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    xr.DataArray
        Ensemble mean time series

    Examples
    --------
    >>> ens_mean = ensemble_mean(ensemble)
    """
    logger.info("Calculating ensemble mean")
    return ensemble.mean('model')


def ensemble_std(ensemble: xr.DataArray) -> xr.DataArray:
    """
    Calculate ensemble standard deviation.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    xr.DataArray
        Ensemble standard deviation time series

    Examples
    --------
    >>> ens_std = ensemble_std(ensemble)
    """
    logger.info("Calculating ensemble standard deviation")
    return ensemble.std('model')


def ensemble_percentiles(
    ensemble: xr.DataArray,
    percentiles: List[float] = [10, 25, 50, 75, 90]
) -> Dict[str, xr.DataArray]:
    """
    Calculate ensemble percentiles.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    percentiles : list of float, default [10, 25, 50, 75, 90]
        Percentiles to calculate (0-100)

    Returns
    -------
    dict
        Dictionary mapping percentile labels to DataArrays

    Examples
    --------
    >>> percentiles = ensemble_percentiles(ensemble, [10, 50, 90])
    >>> p10 = percentiles['p10']
    >>> p50 = percentiles['p50']
    >>> p90 = percentiles['p90']
    """
    logger.info(f"Calculating ensemble percentiles: {percentiles}")

    result = {}
    for p in percentiles:
        result[f'p{int(p)}'] = ensemble.quantile(p / 100, dim='model')

    return result


def ensemble_range(ensemble: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate ensemble minimum and maximum.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    tuple of xr.DataArray
        (minimum, maximum) time series

    Examples
    --------
    >>> ens_min, ens_max = ensemble_range(ensemble)
    """
    logger.info("Calculating ensemble range")
    return ensemble.min('model'), ensemble.max('model')


def model_agreement(
    ensemble: xr.DataArray,
    threshold: float = 0.0,
    method: str = 'sign'
) -> xr.DataArray:
    """
    Calculate model agreement on change signal.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    threshold : float, default 0.0
        Threshold for determining agreement
    method : str, default 'sign'
        Agreement method:
        - 'sign': Fraction agreeing on sign of change
        - 'threshold': Fraction exceeding threshold
        - 'robust': Fraction where mean exceeds 1 std dev

    Returns
    -------
    xr.DataArray
        Model agreement (0-1 or 0-100%)

    Examples
    --------
    >>> agreement = model_agreement(ensemble, method='sign')
    >>> print(f"Model agreement: {agreement.values[-1]:.1%}")
    """
    logger.info(f"Calculating model agreement using '{method}' method")

    if method == 'sign':
        # Fraction of models agreeing on sign
        positive = (ensemble > threshold).sum('model')
        negative = (ensemble < threshold).sum('model')
        agreement = xr.where(
            positive >= negative,
            positive / len(ensemble.model),
            negative / len(ensemble.model)
        )

    elif method == 'threshold':
        # Fraction of models exceeding threshold
        agreement = (ensemble > threshold).sum('model') / len(ensemble.model)

    elif method == 'robust':
        # Robust change: mean exceeds 1 standard deviation
        ens_mean = ensemble.mean('model')
        ens_std = ensemble.std('model')
        robust_mask = np.abs(ens_mean) > ens_std
        # Fraction agreeing with robust signal
        agreement = (
            xr.where(ens_mean > 0, ensemble > 0, ensemble < 0)
            .sum('model') / len(ensemble.model)
        )
        agreement = xr.where(robust_mask, agreement, np.nan)

    else:
        raise ValueError(f"Unknown agreement method: {method}")

    return agreement


def signal_to_noise(ensemble: xr.DataArray) -> xr.DataArray:
    """
    Calculate signal-to-noise ratio (SNR).

    SNR = ensemble_mean / ensemble_std

    High SNR indicates models agree on magnitude and direction of change.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    xr.DataArray
        Signal-to-noise ratio time series

    Examples
    --------
    >>> snr = signal_to_noise(ensemble)
    >>> high_confidence = snr > 2  # SNR > 2 indicates high confidence
    """
    logger.info("Calculating signal-to-noise ratio")

    ens_mean = ensemble.mean('model')
    ens_std = ensemble.std('model')

    # Avoid division by zero
    snr = xr.where(ens_std > 0, ens_mean / ens_std, np.nan)

    return snr


def coefficient_of_variation(ensemble: xr.DataArray) -> xr.DataArray:
    """
    Calculate coefficient of variation (CV).

    CV = ensemble_std / |ensemble_mean| * 100%

    Low CV indicates good model agreement relative to signal magnitude.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    xr.DataArray
        Coefficient of variation (percentage)

    Examples
    --------
    >>> cv = coefficient_of_variation(ensemble)
    """
    logger.info("Calculating coefficient of variation")

    ens_mean = ensemble.mean('model')
    ens_std = ensemble.std('model')

    # Avoid division by zero
    cv = xr.where(
        np.abs(ens_mean) > 0,
        ens_std / np.abs(ens_mean) * 100,
        np.nan
    )

    return cv


def calculate_spread(
    ensemble: xr.DataArray,
    method: str = 'iqr'
) -> xr.DataArray:
    """
    Calculate ensemble spread (uncertainty).

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    method : str, default 'iqr'
        Spread method:
        - 'std': Standard deviation
        - 'iqr': Interquartile range (75th - 25th percentile)
        - 'range': Full range (max - min)

    Returns
    -------
    xr.DataArray
        Ensemble spread time series

    Examples
    --------
    >>> spread = calculate_spread(ensemble, method='iqr')
    """
    logger.info(f"Calculating ensemble spread using '{method}' method")

    if method == 'std':
        spread = ensemble.std('model')

    elif method == 'iqr':
        p75 = ensemble.quantile(0.75, dim='model')
        p25 = ensemble.quantile(0.25, dim='model')
        spread = p75 - p25

    elif method == 'range':
        spread = ensemble.max('model') - ensemble.min('model')

    else:
        raise ValueError(f"Unknown spread method: {method}")

    return spread


def identify_outliers(
    ensemble: xr.DataArray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Dict[str, bool]:
    """
    Identify outlier models in the ensemble.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension
    method : str, default 'iqr'
        Outlier detection method:
        - 'iqr': IQR method (values beyond threshold * IQR from quartiles)
        - 'zscore': Z-score method (values beyond threshold std devs from mean)
    threshold : float, default 1.5
        Threshold multiplier (1.5 for IQR is standard, 3.0 for z-score)

    Returns
    -------
    dict
        Dictionary mapping model names to boolean (True if outlier)

    Examples
    --------
    >>> outliers = identify_outliers(ensemble, method='iqr')
    >>> outlier_models = [m for m, is_out in outliers.items() if is_out]
    """
    logger.info(f"Identifying outliers using '{method}' method")

    outliers = {}

    if method == 'iqr':
        # Calculate quartiles across time
        q1 = ensemble.quantile(0.25, dim='time')
        q3 = ensemble.quantile(0.75, dim='time')
        iqr = q3 - q1

        for model in ensemble.model.values:
            model_data = ensemble.sel(model=model)
            lower_bound = q1.sel(model=model) - threshold * iqr.sel(model=model)
            upper_bound = q3.sel(model=model) + threshold * iqr.sel(model=model)

            # Check if any time steps are outliers
            is_outlier = (
                (model_data < lower_bound).any() or
                (model_data > upper_bound).any()
            ).values

            outliers[str(model)] = bool(is_outlier)

    elif method == 'zscore':
        ens_mean = ensemble.mean('model').mean('time')
        ens_std = ensemble.mean('time').std('model')

        for model in ensemble.model.values:
            model_mean = ensemble.sel(model=model).mean('time')
            z_score = np.abs((model_mean - ens_mean) / ens_std)

            outliers[str(model)] = bool(z_score > threshold)

    else:
        raise ValueError(f"Unknown outlier method: {method}")

    n_outliers = sum(outliers.values())
    logger.info(f"Found {n_outliers} outlier model(s)")

    return outliers


def ensemble_summary_stats(ensemble: xr.DataArray) -> Dict:
    """
    Calculate comprehensive ensemble summary statistics.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble array with 'model' dimension

    Returns
    -------
    dict
        Dictionary of summary statistics

    Examples
    --------
    >>> stats = ensemble_summary_stats(ensemble)
    >>> print(f"Ensemble mean: {stats['mean']}")
    >>> print(f"Ensemble spread: {stats['std']}")
    """
    logger.info("Calculating comprehensive ensemble statistics")

    stats = {
        'n_models': len(ensemble.model),
        'mean': float(ensemble.mean(['model', 'time']).values),
        'std': float(ensemble.std(['model', 'time']).values),
        'min': float(ensemble.min(['model', 'time']).values),
        'max': float(ensemble.max(['model', 'time']).values),
        'p10': float(ensemble.quantile(0.10, dim=['model', 'time']).values),
        'p25': float(ensemble.quantile(0.25, dim=['model', 'time']).values),
        'p50': float(ensemble.quantile(0.50, dim=['model', 'time']).values),
        'p75': float(ensemble.quantile(0.75, dim=['model', 'time']).values),
        'p90': float(ensemble.quantile(0.90, dim=['model', 'time']).values),
    }

    # Calculate per-model statistics
    stats['model_means'] = {
        str(model): float(ensemble.sel(model=model).mean('time').values)
        for model in ensemble.model.values
    }

    return stats
