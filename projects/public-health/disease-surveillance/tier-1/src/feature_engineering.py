"""
Feature engineering utilities for disease surveillance forecasting.

Creates lag features, rolling statistics, seasonal decomposition,
and spatiotemporal features for LSTM training.
"""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def create_lag_features(
    data: pd.DataFrame, target_col: str, lag_periods: list[int], group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create lagged features for time series forecasting.

    Args:
        data: Input DataFrame
        target_col: Column to create lags for
        lag_periods: List of lag periods (e.g., [1, 2, 3, 4] for 1-4 weeks)
        group_col: Optional grouping column (e.g., 'Region')

    Returns:
        DataFrame with lag features added
    """
    df = data.copy()

    if group_col:
        for lag in lag_periods:
            df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lag_periods:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    data: pd.DataFrame, target_col: str, windows: list[int], group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics (mean, std, min, max).

    Args:
        data: Input DataFrame
        target_col: Column to calculate rolling stats for
        windows: List of window sizes (e.g., [2, 4, 8] for 2, 4, 8 weeks)
        group_col: Optional grouping column

    Returns:
        DataFrame with rolling features added
    """
    df = data.copy()

    for window in windows:
        if group_col:
            df[f"{target_col}_rolling_mean_{window}"] = df.groupby(group_col)[target_col].transform(
                lambda x, window=window: x.rolling(window=window, min_periods=1).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = df.groupby(group_col)[target_col].transform(
                lambda x, window=window: x.rolling(window=window, min_periods=1).std()
            )
        else:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )

    return df


def seasonal_decomposition(
    data: pd.DataFrame, target_col: str, period: int = 52, group_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform seasonal decomposition (trend, seasonal, residual).

    Args:
        data: Input DataFrame with DatetimeIndex
        target_col: Column to decompose
        period: Seasonal period (52 weeks for yearly)
        group_col: Optional grouping column

    Returns:
        DataFrame with trend, seasonal, and residual components
    """
    df = data.copy()

    if group_col:
        # Decompose for each group
        trends = []
        seasonals = []
        residuals = []

        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][target_col]

            if len(group_data) >= 2 * period:
                decomp = seasonal_decompose(
                    group_data, model="additive", period=period, extrapolate_trend="freq"
                )
                trends.append(decomp.trend)
                seasonals.append(decomp.seasonal)
                residuals.append(decomp.resid)
            else:
                # Not enough data for decomposition
                trends.append(pd.Series([np.nan] * len(group_data), index=group_data.index))
                seasonals.append(pd.Series([np.nan] * len(group_data), index=group_data.index))
                residuals.append(pd.Series([np.nan] * len(group_data), index=group_data.index))

        df[f"{target_col}_trend"] = pd.concat(trends)
        df[f"{target_col}_seasonal"] = pd.concat(seasonals)
        df[f"{target_col}_residual"] = pd.concat(residuals)
    else:
        if len(df) >= 2 * period:
            decomp = seasonal_decompose(
                df.set_index("Date")[target_col],
                model="additive",
                period=period,
                extrapolate_trend="freq",
            )
            df[f"{target_col}_trend"] = decomp.trend.values
            df[f"{target_col}_seasonal"] = decomp.seasonal.values
            df[f"{target_col}_residual"] = decomp.resid.values

    return df


def create_date_features(data: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extract date-based features (week, month, year, holidays).

    Args:
        data: Input DataFrame
        date_col: Name of date column

    Returns:
        DataFrame with date features added
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic date features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    df["day_of_year"] = df[date_col].dt.dayofyear

    # Cyclic encoding for week of year (to capture cyclical nature)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Month cyclic encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Holiday indicators (major US holidays)
    df["is_holiday_week"] = df[date_col].apply(_is_holiday_week)

    # Season
    df["season"] = df["month"].apply(_get_season)

    return df


def create_growth_rate_features(
    data: pd.DataFrame,
    target_col: str,
    periods: Optional[list[int]] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate growth rates (percent change) over different periods.

    Args:
        data: Input DataFrame
        target_col: Column to calculate growth rates for
        periods: Periods for calculating growth rates
        group_col: Optional grouping column

    Returns:
        DataFrame with growth rate features
    """
    if periods is None:
        periods = [1, 2, 4]
    df = data.copy()

    for period in periods:
        if group_col:
            df[f"{target_col}_growth_{period}"] = (
                df.groupby(group_col)[target_col].pct_change(periods=period) * 100
            )
        else:
            df[f"{target_col}_growth_{period}"] = df[target_col].pct_change(periods=period) * 100

    return df


def create_spatiotemporal_features(
    data: pd.DataFrame,
    target_col: str,
    lag_weeks: Optional[list[int]] = None,
    rolling_windows: Optional[list[int]] = None,
    include_seasonal_decomp: bool = True,
    include_growth_rates: bool = True,
    group_col: str = "Region",
) -> pd.DataFrame:
    """
    Create comprehensive spatiotemporal feature set for forecasting.

    Args:
        data: Input DataFrame
        target_col: Target variable column
        lag_weeks: Lag periods to create
        rolling_windows: Rolling window sizes
        include_seasonal_decomp: Include seasonal decomposition
        include_growth_rates: Include growth rate features
        group_col: Grouping column for spatial dimension

    Returns:
        DataFrame with full feature set
    """
    if rolling_windows is None:
        rolling_windows = [2, 4, 8]
    if lag_weeks is None:
        lag_weeks = [1, 2, 3, 4]
    df = data.copy()

    print("Creating lag features...")
    df = create_lag_features(df, target_col, lag_weeks, group_col)

    print("Creating rolling features...")
    df = create_rolling_features(df, target_col, rolling_windows, group_col)

    print("Creating date features...")
    df = create_date_features(df)

    if include_growth_rates:
        print("Creating growth rate features...")
        df = create_growth_rate_features(df, target_col, [1, 2, 4], group_col)

    if include_seasonal_decomp:
        print("Performing seasonal decomposition...")
        df = seasonal_decomposition(df, target_col, period=52, group_col=group_col)

    # Fill NaN values (from lag/rolling operations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method="bfill").fillna(0)

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


def _is_holiday_week(date: pd.Timestamp) -> int:
    """Check if date falls within a major US holiday week."""

    # Major holidays that affect healthcare-seeking behavior
    holidays = [
        (1, 1),  # New Year's Day
        (7, 4),  # Independence Day
        (11, 22),  # Thanksgiving (approximate)
        (12, 25),  # Christmas
    ]

    # Check if within 7 days of any holiday
    for h_month, h_day in holidays:
        holiday_date = pd.Timestamp(year=date.year, month=h_month, day=h_day)
        if abs((date - holiday_date).days) <= 7:
            return 1

    return 0


def _get_season(month: int) -> str:
    """Get season from month."""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"
