"""
Urban Planning Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing urban datasets including
satellite imagery, mobility patterns, and demographic data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    calculate_urban_indices,
    load_demographic_data,
    load_mobility_data,
    load_satellite_imagery,
)
from .mobility_analysis import (
    analyze_transit_accessibility,
    calculate_traffic_metrics,
    compute_commute_patterns,
)
from .urban_models import (
    MobilityPredictor,
    UrbanGrowthCNN,
    train_city_model,
)
from .visualization import (
    create_city_comparison_dashboard,
    plot_mobility_heatmap,
    plot_urban_growth,
)

__all__ = [
    "MobilityPredictor",
    "UrbanGrowthCNN",
    "analyze_transit_accessibility",
    "calculate_traffic_metrics",
    "calculate_urban_indices",
    "compute_commute_patterns",
    "create_city_comparison_dashboard",
    "load_demographic_data",
    "load_mobility_data",
    "load_satellite_imagery",
    "plot_mobility_heatmap",
    "plot_urban_growth",
    "train_city_model",
]
