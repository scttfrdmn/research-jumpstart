"""
Statistical analysis functions for economic time series.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR


def test_stationarity(series: pd.Series, significance_level: float = 0.05) -> dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test
        significance_level: Significance level for test

    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_1pct": result[4]["1%"],
        "critical_5pct": result[4]["5%"],
        "critical_10pct": result[4]["10%"],
        "is_stationary": result[1] < significance_level,
    }


def granger_causality_test(
    data: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 4
) -> pd.DataFrame:
    """
    Test if x Granger-causes y.

    Args:
        data: DataFrame with both series
        x_col: Name of potential cause variable
        y_col: Name of effect variable
        max_lag: Maximum number of lags to test

    Returns:
        DataFrame with test results for each lag
    """
    test_data = data[[y_col, x_col]].dropna()

    try:
        result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

        summary = []
        for lag in range(1, max_lag + 1):
            test_result = result[lag][0]
            summary.append(
                {
                    "lag": lag,
                    "ssr_ftest_pvalue": test_result["ssr_ftest"][1],
                    "ssr_chi2test_pvalue": test_result["ssr_chi2test"][1],
                    "lrtest_pvalue": test_result["lrtest"][1],
                    "params_ftest_pvalue": test_result["params_ftest"][1],
                }
            )

        return pd.DataFrame(summary)
    except Exception as e:
        print(f"Error in Granger causality test: {e}")
        return pd.DataFrame()


def fit_var_model(data: pd.DataFrame, max_lag: int = 4) -> tuple[VAR, int]:
    """
    Fit Vector Autoregression (VAR) model.

    Args:
        data: Multivariate time series DataFrame
        max_lag: Maximum lag order to consider

    Returns:
        Fitted VAR model and selected lag order
    """
    model = VAR(data.dropna())

    # Select lag order using AIC
    lag_order = model.select_order(maxlags=max_lag)
    optimal_lag = lag_order.aic

    # Fit model
    fitted_model = model.fit(optimal_lag)

    return fitted_model, optimal_lag


def impulse_response_analysis(
    var_model, periods: int = 20, shock_var: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute impulse response functions from VAR model.

    Args:
        var_model: Fitted VAR model
        periods: Number of periods for IRF
        shock_var: Variable to shock (if None, shock all)

    Returns:
        DataFrame with impulse responses
    """
    irf = var_model.irf(periods)

    # Extract IRF matrix
    irf_matrix = irf.irfs

    # Convert to DataFrame
    variables = var_model.names
    columns = [f"{shock}_to_{response}" for shock in variables for response in variables]

    irf_df = pd.DataFrame(irf_matrix.reshape(periods, -1), columns=columns, index=range(periods))

    return irf_df


def forecast_error_variance_decomposition(var_model, periods: int = 20) -> pd.DataFrame:
    """
    Compute forecast error variance decomposition (FEVD).

    Args:
        var_model: Fitted VAR model
        periods: Number of periods for decomposition

    Returns:
        DataFrame with FEVD results
    """
    fevd = var_model.fevd(periods)

    # Extract FEVD matrices for each variable
    variables = var_model.names
    results = []

    for i, var in enumerate(variables):
        fevd_var = fevd.decomp[:, i, :]
        fevd_df = pd.DataFrame(
            fevd_var, columns=[f"shock_from_{v}" for v in variables], index=range(periods)
        )
        fevd_df["response_variable"] = var
        results.append(fevd_df)

    return pd.concat(results)


def cross_correlation_analysis(data: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """
    Compute cross-correlations at different lags.

    Args:
        data: DataFrame with multiple time series
        max_lag: Maximum lag to compute

    Returns:
        DataFrame with cross-correlations
    """
    columns = data.columns
    results = []

    for i, col1 in enumerate(columns):
        for col2 in columns[i:]:
            series1 = data[col1].dropna()
            series2 = data[col2].dropna()

            # Align series
            common_idx = series1.index.intersection(series2.index)
            s1 = series1.loc[common_idx]
            s2 = series2.loc[common_idx]

            # Compute cross-correlations
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = s1.iloc[-lag:].corr(s2.iloc[:lag])
                elif lag > 0:
                    corr = s1.iloc[:-lag].corr(s2.iloc[lag:])
                else:
                    corr = s1.corr(s2)

                results.append({"series1": col1, "series2": col2, "lag": lag, "correlation": corr})

    return pd.DataFrame(results)


def business_cycle_synchronization(
    data: pd.DataFrame,
    country_col: str = "Country",
    indicator_col: str = "Indicator",
    use_hp_filter: bool = True,
) -> pd.DataFrame:
    """
    Measure business cycle synchronization across countries.

    Args:
        data: Panel data with country and indicator information
        country_col: Column name for country identifier
        indicator_col: Column name for indicator
        use_hp_filter: If True, apply HP filter to extract cycles

    Returns:
        Correlation matrix of business cycles
    """
    # This is a simplified implementation
    # Full implementation would use HP filter or band-pass filter

    if use_hp_filter:
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter

            cycles = data.apply(
                lambda x: hpfilter(x.dropna(), lamb=1600)[0] if x.notna().sum() > 20 else x
            )
        except:
            cycles = data  # Fall back to original data
    else:
        cycles = data

    # Compute correlation matrix
    corr_matrix = cycles.corr()

    return corr_matrix


def calculate_trend(data: pd.Series, x: Optional[pd.Series] = None) -> dict[str, float]:
    """
    Calculate linear trend using least squares regression.

    Args:
        data: Time series data (y values)
        x: Optional x values (defaults to sequential indices)

    Returns:
        Dictionary with slope, intercept, r_value, p_value, std_err
    """
    clean_data = data.dropna()

    if len(clean_data) < 2:
        return {}

    x_vals = np.arange(len(clean_data)) if x is None else x[clean_data.index].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, clean_data.values)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
    }


def decompose_time_series(
    data: pd.Series, period: int = 4, model: str = "additive"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Args:
        data: Time series data
        period: Period of seasonal component (4 for quarterly, 12 for monthly)
        model: 'additive' or 'multiplicative'

    Returns:
        Tuple of (trend, seasonal, residual) Series
    """
    decomposition = seasonal_decompose(
        data.dropna(), model=model, period=period, extrapolate_trend="freq"
    )

    return (decomposition.trend, decomposition.seasonal, decomposition.resid)


def detect_structural_breaks(data: pd.Series, min_size: int = 20) -> list[int]:
    """
    Detect structural breaks using Chow test.

    Args:
        data: Time series data
        min_size: Minimum segment size

    Returns:
        List of break point indices
    """
    # Simplified implementation using rolling statistics
    # Full implementation would use proper Chow test or Bai-Perron method

    clean_data = data.dropna()
    n = len(clean_data)

    if n < 2 * min_size:
        return []

    # Use rolling mean and std to detect breaks
    window = min_size
    rolling_mean = clean_data.rolling(window=window, center=True).mean()
    rolling_std = clean_data.rolling(window=window, center=True).std()

    # Calculate z-scores
    z_scores = np.abs((clean_data - rolling_mean) / rolling_std)

    # Find potential break points (high z-scores)
    breaks = []
    threshold = 2.5

    for i in range(min_size, n - min_size):
        if z_scores.iloc[i] > threshold:
            # Check if not too close to previous break
            if not breaks or i - breaks[-1] > min_size:
                breaks.append(i)

    return breaks
