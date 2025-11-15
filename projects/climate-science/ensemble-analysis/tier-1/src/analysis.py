"""
Statistical analysis functions for climate data.
"""


import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def calculate_trend(data: pd.Series, x: pd.Series = None) -> dict[str, float]:
    """
    Calculate linear trend using least squares regression.

    Args:
        data: Time series data (y values)
        x: Optional x values (defaults to sequential indices)

    Returns:
        Dictionary with slope, intercept, r_value, p_value, std_err
    """
    if x is None:
        x = np.arange(len(data))

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
    }


def decompose_time_series(
    data: pd.Series, period: int = 12, model: str = "additive"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Args:
        data: Time series data
        period: Period of seasonal component (12 for monthly data)
        model: 'additive' or 'multiplicative'

    Returns:
        Tuple of (trend, seasonal, residual) Series
    """
    decomposition = seasonal_decompose(data, model=model, period=period)

    return (decomposition.trend, decomposition.seasonal, decomposition.resid)


def calculate_correlation_matrix(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified variables.

    Args:
        df: DataFrame containing variables
        variables: List of column names to correlate

    Returns:
        Correlation matrix DataFrame
    """
    return df[variables].corr()


def detect_change_points(data: pd.Series, window: int = 10) -> pd.Series:
    """
    Detect significant change points using rolling statistics.

    Args:
        data: Time series data
        window: Window size for rolling mean/std

    Returns:
        Series indicating change point scores
    """
    # Calculate rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    # Calculate z-scores
    z_scores = (data - rolling_mean) / rolling_std

    return z_scores.abs()


def calculate_acceleration(data: pd.Series, dates: pd.Series, periods: list) -> pd.DataFrame:
    """
    Calculate rate of change across different time periods.

    Args:
        data: Time series data
        dates: Corresponding dates
        periods: List of (start_year, end_year, label) tuples

    Returns:
        DataFrame with period-wise trends
    """
    results = []

    for start, end, label in periods:
        mask = (dates.dt.year >= start) & (dates.dt.year <= end)
        period_data = data[mask]
        period_years = dates[mask].dt.year.values

        if len(period_data) > 1:
            trend_info = calculate_trend(period_data, period_years)
            results.append(
                {
                    "Period": label,
                    "Start_Year": start,
                    "End_Year": end,
                    "Years": end - start,
                    "Slope": trend_info["slope"],
                    "Rate_per_decade": trend_info["slope"] * 10,
                    "R_squared": trend_info["r_squared"],
                    "P_value": trend_info["p_value"],
                }
            )

    return pd.DataFrame(results)
