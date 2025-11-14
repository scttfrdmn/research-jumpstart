"""
Urban Planning Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing urban datasets including
satellite imagery, mobility patterns, and demographic data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_satellite_imagery,
    load_mobility_data,
    load_demographic_data,
    calculate_urban_indices,
)
from .urban_models import (
    UrbanGrowthCNN,
    MobilityPredictor,
    train_city_model,
)
from .mobility_analysis import (
    calculate_traffic_metrics,
    analyze_transit_accessibility,
    compute_commute_patterns,
)
from .visualization import (
    plot_urban_growth,
    plot_mobility_heatmap,
    create_city_comparison_dashboard,
)

__all__ = [
    "load_satellite_imagery",
    "load_mobility_data",
    "load_demographic_data",
    "calculate_urban_indices",
    "UrbanGrowthCNN",
    "MobilityPredictor",
    "train_city_model",
    "calculate_traffic_metrics",
    "analyze_transit_accessibility",
    "compute_commute_patterns",
    "plot_urban_growth",
    "plot_mobility_heatmap",
    "create_city_comparison_dashboard",
]
