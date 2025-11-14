"""
Precision Agriculture Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing multi-sensor agricultural data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_sentinel2,
    load_landsat8,
    load_modis,
    load_weather_data,
    load_soil_data,
)
from .preprocessing import (
    cloud_mask,
    atmospheric_correction,
    coregister_sensors,
    gap_fill_timeseries,
)
from .features import (
    calculate_ndvi,
    calculate_evi,
    calculate_savi,
    extract_phenology_metrics,
    create_temporal_features,
)
from .models import (
    build_cnn_model,
    build_lstm_model,
    build_ensemble_model,
)
from .visualization import (
    plot_timeseries,
    plot_field_map,
    plot_yield_prediction,
    create_interactive_map,
)

__all__ = [
    "load_sentinel2",
    "load_landsat8",
    "load_modis",
    "load_weather_data",
    "load_soil_data",
    "cloud_mask",
    "atmospheric_correction",
    "coregister_sensors",
    "gap_fill_timeseries",
    "calculate_ndvi",
    "calculate_evi",
    "calculate_savi",
    "extract_phenology_metrics",
    "create_temporal_features",
    "build_cnn_model",
    "build_lstm_model",
    "build_ensemble_model",
    "plot_timeseries",
    "plot_field_map",
    "plot_yield_prediction",
    "create_interactive_map",
]
