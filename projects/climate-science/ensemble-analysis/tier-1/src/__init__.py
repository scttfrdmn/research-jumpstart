"""
Climate Science Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing climate datasets.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .analysis import (
    calculate_correlation_matrix,
    calculate_trend,
    decompose_time_series,
)
from .data_utils import (
    calculate_anomalies,
    load_co2_data,
    load_sea_level_data,
    load_temperature_data,
)
from .visualization import (
    create_interactive_dashboard,
    plot_correlation_heatmap,
    plot_time_series,
)

__all__ = [
    "calculate_anomalies",
    "calculate_correlation_matrix",
    "calculate_trend",
    "create_interactive_dashboard",
    "decompose_time_series",
    "load_co2_data",
    "load_sea_level_data",
    "load_temperature_data",
    "plot_correlation_heatmap",
    "plot_time_series",
]
