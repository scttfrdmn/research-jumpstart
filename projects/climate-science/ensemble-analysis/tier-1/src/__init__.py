"""
Climate Science Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing climate datasets.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_temperature_data,
    load_co2_data,
    load_sea_level_data,
    calculate_anomalies,
)
from .analysis import (
    calculate_trend,
    decompose_time_series,
    calculate_correlation_matrix,
)
from .visualization import (
    plot_time_series,
    plot_correlation_heatmap,
    create_interactive_dashboard,
)

__all__ = [
    "load_temperature_data",
    "load_co2_data",
    "load_sea_level_data",
    "calculate_anomalies",
    "calculate_trend",
    "decompose_time_series",
    "calculate_correlation_matrix",
    "plot_time_series",
    "plot_correlation_heatmap",
    "create_interactive_dashboard",
]
