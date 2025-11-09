"""
Climate Ensemble Analysis Package

Production-ready modules for CMIP6 climate model ensemble analysis on AWS.
"""

from .data_access import CMIP6DataAccess, validate_region, check_s3_access
from .climate_analysis import (
    calculate_regional_mean,
    calculate_anomaly,
    annual_mean,
    seasonal_mean,
    calculate_trend,
    running_mean,
    detrend,
    calculate_climatology,
    convert_units
)
from .ensemble_stats import (
    create_ensemble,
    ensemble_mean,
    ensemble_std,
    ensemble_percentiles,
    ensemble_range,
    model_agreement,
    signal_to_noise,
    coefficient_of_variation,
    calculate_spread,
    identify_outliers,
    ensemble_summary_stats
)
from .visualization import (
    plot_ensemble_timeseries,
    plot_model_agreement,
    plot_regional_map,
    plot_scenario_comparison,
    create_summary_figure
)
from .bedrock_client import BedrockClimateAssistant

__version__ = '1.0.0'
__all__ = [
    # Data access
    'CMIP6DataAccess',
    'validate_region',
    'check_s3_access',
    # Climate analysis
    'calculate_regional_mean',
    'calculate_anomaly',
    'annual_mean',
    'seasonal_mean',
    'calculate_trend',
    'running_mean',
    'detrend',
    'calculate_climatology',
    'convert_units',
    # Ensemble statistics
    'create_ensemble',
    'ensemble_mean',
    'ensemble_std',
    'ensemble_percentiles',
    'ensemble_range',
    'model_agreement',
    'signal_to_noise',
    'coefficient_of_variation',
    'calculate_spread',
    'identify_outliers',
    'ensemble_summary_stats',
    # Visualization
    'plot_ensemble_timeseries',
    'plot_model_agreement',
    'plot_regional_map',
    'plot_scenario_comparison',
    'create_summary_figure',
    # AI assistance
    'BedrockClimateAssistant',
]
