"""
Public Health Disease Surveillance Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and forecasting disease surveillance data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_ili_data,
    load_covid_data,
    load_rsv_data,
    load_multi_disease_panel,
    calculate_epidemic_threshold,
)
from .feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_spatiotemporal_features,
    seasonal_decomposition,
)
from .forecasting import (
    train_lstm_model,
    train_lstm_ensemble,
    generate_forecast,
    generate_probabilistic_forecast,
)
from .evaluation import (
    calculate_forecast_metrics,
    evaluate_peak_timing,
    calculate_outbreak_metrics,
)
from .visualization import (
    plot_time_series,
    plot_forecast,
    create_forecast_dashboard,
    plot_outbreak_heatmap,
)

__all__ = [
    "load_ili_data",
    "load_covid_data",
    "load_rsv_data",
    "load_multi_disease_panel",
    "calculate_epidemic_threshold",
    "create_lag_features",
    "create_rolling_features",
    "create_spatiotemporal_features",
    "seasonal_decomposition",
    "train_lstm_model",
    "train_lstm_ensemble",
    "generate_forecast",
    "generate_probabilistic_forecast",
    "calculate_forecast_metrics",
    "evaluate_peak_timing",
    "calculate_outbreak_metrics",
    "plot_time_series",
    "plot_forecast",
    "create_forecast_dashboard",
    "plot_outbreak_heatmap",
]
