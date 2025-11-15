"""
Climate Ensemble Analysis Package

Production-ready modules for CMIP6 climate model ensemble analysis on AWS.
"""

from .bedrock_client import BedrockClimateAssistant
from .climate_analysis import (
    annual_mean,
    calculate_anomaly,
    calculate_climatology,
    calculate_regional_mean,
    calculate_trend,
    convert_units,
    detrend,
    running_mean,
    seasonal_mean,
)
from .data_access import CMIP6DataAccess, check_s3_access, validate_region
from .ensemble_stats import (
    calculate_spread,
    coefficient_of_variation,
    create_ensemble,
    ensemble_mean,
    ensemble_percentiles,
    ensemble_range,
    ensemble_std,
    ensemble_summary_stats,
    identify_outliers,
    model_agreement,
    signal_to_noise,
)
from .visualization import (
    create_summary_figure,
    plot_ensemble_timeseries,
    plot_model_agreement,
    plot_regional_map,
    plot_scenario_comparison,
)

__version__ = "1.0.0"
__all__ = [
    # AI assistance
    "BedrockClimateAssistant",
    # Data access
    "CMIP6DataAccess",
    "annual_mean",
    "calculate_anomaly",
    "calculate_climatology",
    # Climate analysis
    "calculate_regional_mean",
    "calculate_spread",
    "calculate_trend",
    "check_s3_access",
    "coefficient_of_variation",
    "convert_units",
    # Ensemble statistics
    "create_ensemble",
    "create_summary_figure",
    "detrend",
    "ensemble_mean",
    "ensemble_percentiles",
    "ensemble_range",
    "ensemble_std",
    "ensemble_summary_stats",
    "identify_outliers",
    "model_agreement",
    # Visualization
    "plot_ensemble_timeseries",
    "plot_model_agreement",
    "plot_regional_map",
    "plot_scenario_comparison",
    "running_mean",
    "seasonal_mean",
    "signal_to_noise",
    "validate_region",
]
