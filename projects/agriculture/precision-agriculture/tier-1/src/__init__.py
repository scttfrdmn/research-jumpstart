"""
Precision Agriculture Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing multi-sensor agricultural data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_landsat8,
    load_modis,
    load_sentinel2,
    load_soil_data,
    load_weather_data,
)
from .features import (
    calculate_evi,
    calculate_ndvi,
    calculate_savi,
    create_temporal_features,
    extract_phenology_metrics,
)
from .models import (
    build_cnn_model,
    build_ensemble_model,
    build_lstm_model,
)
from .preprocessing import (
    atmospheric_correction,
    cloud_mask,
    coregister_sensors,
    gap_fill_timeseries,
)
from .visualization import (
    create_interactive_map,
    plot_field_map,
    plot_timeseries,
    plot_yield_prediction,
)

__all__ = [
    "atmospheric_correction",
    "build_cnn_model",
    "build_ensemble_model",
    "build_lstm_model",
    "calculate_evi",
    "calculate_ndvi",
    "calculate_savi",
    "cloud_mask",
    "coregister_sensors",
    "create_interactive_map",
    "create_temporal_features",
    "extract_phenology_metrics",
    "gap_fill_timeseries",
    "load_landsat8",
    "load_modis",
    "load_sentinel2",
    "load_soil_data",
    "load_weather_data",
    "plot_field_map",
    "plot_timeseries",
    "plot_yield_prediction",
]
