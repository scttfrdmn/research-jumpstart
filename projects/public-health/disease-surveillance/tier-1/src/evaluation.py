"""
Evaluation metrics for disease surveillance forecasting models.

Includes forecast accuracy metrics, epidemiological metrics,
and probabilistic evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate standard forecast accuracy metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Mean Absolute Error
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    metrics[f'{prefix}mape'] = mape

    # R² Score
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)

    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    metrics[f'{prefix}smape'] = smape

    return metrics


def evaluate_peak_timing(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, float]:
    """
    Evaluate accuracy of peak timing and intensity prediction.

    Args:
        y_true: True time series
        y_pred: Predicted time series
        dates: Optional date index

    Returns:
        Dictionary with peak timing and intensity metrics
    """
    # Find peaks
    true_peak_idx = np.argmax(y_true)
    pred_peak_idx = np.argmax(y_pred)

    true_peak_value = y_true[true_peak_idx]
    pred_peak_value = y_pred[pred_peak_idx]

    metrics = {
        'peak_timing_error_weeks': abs(true_peak_idx - pred_peak_idx),
        'peak_intensity_error': abs(true_peak_value - pred_peak_value),
        'peak_intensity_error_pct': abs(true_peak_value - pred_peak_value) / true_peak_value * 100,
        'true_peak_idx': true_peak_idx,
        'pred_peak_idx': pred_peak_idx,
        'true_peak_value': true_peak_value,
        'pred_peak_value': pred_peak_value
    }

    if dates is not None:
        metrics['true_peak_date'] = dates[true_peak_idx]
        metrics['pred_peak_date'] = dates[pred_peak_idx]

    return metrics


def calculate_outbreak_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Calculate outbreak detection metrics (binary classification).

    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Outbreak threshold

    Returns:
        Dictionary with outbreak detection metrics
    """
    # Convert to binary (above/below threshold)
    true_outbreak = (y_true > threshold).astype(int)
    pred_outbreak = (y_pred > threshold).astype(int)

    # True positives, false positives, etc.
    tp = np.sum((true_outbreak == 1) & (pred_outbreak == 1))
    fp = np.sum((true_outbreak == 0) & (pred_outbreak == 1))
    tn = np.sum((true_outbreak == 0) & (pred_outbreak == 0))
    fn = np.sum((true_outbreak == 1) & (pred_outbreak == 0))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'outbreak_accuracy': accuracy,
        'outbreak_precision': precision,
        'outbreak_recall': recall,
        'outbreak_f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

    return metrics


def calculate_prediction_interval_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    nominal_coverage: float = 0.95
) -> Dict[str, float]:
    """
    Calculate prediction interval coverage probability.

    Args:
        y_true: True values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        nominal_coverage: Nominal coverage (e.g., 0.95 for 95% interval)

    Returns:
        Dictionary with coverage metrics
    """
    # Count how many true values fall within interval
    within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = np.mean(within_interval)

    # Interval width
    mean_width = np.mean(upper_bound - lower_bound)
    median_width = np.median(upper_bound - lower_bound)

    metrics = {
        'coverage': coverage,
        'nominal_coverage': nominal_coverage,
        'coverage_error': abs(coverage - nominal_coverage),
        'mean_interval_width': mean_width,
        'median_interval_width': median_width,
        'n_within_interval': np.sum(within_interval),
        'n_total': len(y_true)
    }

    return metrics


def calculate_crps(
    y_true: np.ndarray,
    ensemble_forecasts: np.ndarray
) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS).

    Lower is better. Measures how well the ensemble forecast
    predicts the true value.

    Args:
        y_true: True values (n_samples,)
        ensemble_forecasts: Ensemble forecasts (n_models, n_samples)

    Returns:
        Mean CRPS
    """
    n_models = ensemble_forecasts.shape[0]
    crps_values = []

    for i in range(len(y_true)):
        true_val = y_true[i]
        forecasts = ensemble_forecasts[:, i]

        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # where X, X' are independent samples from forecast distribution
        term1 = np.mean(np.abs(forecasts - true_val))
        term2 = 0
        for j in range(n_models):
            for k in range(n_models):
                term2 += np.abs(forecasts[j] - forecasts[k])
        term2 = term2 / (2 * n_models ** 2)

        crps = term1 - term2
        crps_values.append(crps)

    return np.mean(crps_values)


def evaluate_multi_horizon_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Evaluate forecast at multiple horizons (e.g., 1-week, 2-week, etc.).

    Args:
        y_true: True values (n_samples, n_horizons)
        y_pred: Predictions (n_samples, n_horizons)
        horizon_names: Names for each horizon (e.g., ['1-week', '2-week', ...])

    Returns:
        DataFrame with metrics for each horizon
    """
    n_horizons = y_true.shape[1]

    if horizon_names is None:
        horizon_names = [f'Week +{i+1}' for i in range(n_horizons)]

    results = []
    for i in range(n_horizons):
        metrics = calculate_forecast_metrics(
            y_true[:, i],
            y_pred[:, i],
            prefix=''
        )
        metrics['horizon'] = horizon_names[i]
        results.append(metrics)

    return pd.DataFrame(results)


def calculate_ensemble_diversity(ensemble_forecasts: np.ndarray) -> Dict[str, float]:
    """
    Calculate diversity metrics for ensemble forecasts.

    Higher diversity generally leads to better ensemble performance.

    Args:
        ensemble_forecasts: Ensemble forecasts (n_models, n_samples)

    Returns:
        Dictionary with diversity metrics
    """
    n_models = ensemble_forecasts.shape[0]

    # Pairwise correlation between models
    correlations = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = np.corrcoef(ensemble_forecasts[i], ensemble_forecasts[j])[0, 1]
            correlations.append(corr)

    # Variance within ensemble at each time point
    variance_across_models = np.var(ensemble_forecasts, axis=0)

    metrics = {
        'mean_pairwise_correlation': np.mean(correlations),
        'std_pairwise_correlation': np.std(correlations),
        'mean_forecast_variance': np.mean(variance_across_models),
        'median_forecast_variance': np.median(variance_across_models)
    }

    return metrics


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ensemble_forecasts: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    dates: Optional[pd.DatetimeIndex] = None
) -> Dict[str, any]:
    """
    Generate comprehensive evaluation report.

    Args:
        y_true: True values
        y_pred: Mean predictions
        ensemble_forecasts: Optional ensemble forecasts for probabilistic metrics
        threshold: Optional outbreak threshold
        dates: Optional date index

    Returns:
        Dictionary with all evaluation metrics
    """
    report = {}

    # Basic forecast metrics
    report['forecast_metrics'] = calculate_forecast_metrics(y_true, y_pred)

    # Peak timing and intensity
    report['peak_metrics'] = evaluate_peak_timing(y_true, y_pred, dates)

    # Outbreak detection (if threshold provided)
    if threshold is not None:
        report['outbreak_metrics'] = calculate_outbreak_metrics(y_true, y_pred, threshold)

    # Ensemble metrics (if ensemble forecasts provided)
    if ensemble_forecasts is not None:
        report['diversity_metrics'] = calculate_ensemble_diversity(ensemble_forecasts)
        report['crps'] = calculate_crps(y_true, ensemble_forecasts)

        # Prediction intervals (using 5th and 95th percentiles)
        lower = np.percentile(ensemble_forecasts, 5, axis=0)
        upper = np.percentile(ensemble_forecasts, 95, axis=0)
        report['interval_metrics'] = calculate_prediction_interval_coverage(
            y_true, lower, upper, nominal_coverage=0.90
        )

    return report


def print_evaluation_report(report: Dict[str, any]):
    """Pretty print evaluation report."""
    print("=" * 60)
    print("FORECAST EVALUATION REPORT")
    print("=" * 60)

    if 'forecast_metrics' in report:
        print("\nFORECAST ACCURACY:")
        for key, value in report['forecast_metrics'].items():
            print(f"  {key:20s}: {value:8.4f}")

    if 'peak_metrics' in report:
        print("\nPEAK PREDICTION:")
        for key, value in report['peak_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key:30s}: {value:8.4f}")

    if 'outbreak_metrics' in report:
        print("\nOUTBREAK DETECTION:")
        for key, value in report['outbreak_metrics'].items():
            print(f"  {key:20s}: {value:8.4f}")

    if 'diversity_metrics' in report:
        print("\nENSEMBLE DIVERSITY:")
        for key, value in report['diversity_metrics'].items():
            print(f"  {key:30s}: {value:8.4f}")

    if 'crps' in report:
        print(f"\nCRPS (lower is better): {report['crps']:8.4f}")

    if 'interval_metrics' in report:
        print("\nPREDICTION INTERVALS:")
        for key, value in report['interval_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key:25s}: {value:8.4f}")

    print("=" * 60)
