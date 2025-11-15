"""
Public Health Disease Surveillance Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and forecasting disease surveillance data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    calculate_epidemic_threshold,
    load_covid_data,
    load_ili_data,
    load_multi_disease_panel,
    load_rsv_data,
)
from .evaluation import (
    calculate_forecast_metrics,
    calculate_outbreak_metrics,
    evaluate_peak_timing,
)
from .feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_spatiotemporal_features,
    seasonal_decomposition,
)
from .forecasting import (
    generate_forecast,
    generate_probabilistic_forecast,
    train_lstm_ensemble,
    train_lstm_model,
)
from .visualization import (
    create_forecast_dashboard,
    plot_forecast,
    plot_outbreak_heatmap,
    plot_time_series,
)

__all__ = [
    "calculate_epidemic_threshold",
    "calculate_forecast_metrics",
    "calculate_outbreak_metrics",
    "create_forecast_dashboard",
    "create_lag_features",
    "create_rolling_features",
    "create_spatiotemporal_features",
    "evaluate_peak_timing",
    "generate_forecast",
    "generate_probabilistic_forecast",
    "load_covid_data",
    "load_ili_data",
    "load_multi_disease_panel",
    "load_rsv_data",
    "plot_forecast",
    "plot_outbreak_heatmap",
    "plot_time_series",
    "seasonal_decomposition",
    "train_lstm_ensemble",
    "train_lstm_model",
]
