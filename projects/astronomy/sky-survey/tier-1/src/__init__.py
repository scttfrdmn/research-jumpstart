"""
Astronomy Sky Survey Cross-Matching Toolkit for SageMaker Studio Lab

Utilities for querying surveys, cross-matching catalogs, and classifying sources.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .crossmatch import (
    build_healpix_index,
    match_catalogs,
    spatial_crossmatch,
)
from .features import (
    build_feature_matrix,
    calculate_colors,
    calculate_proper_motions,
    extract_morphology,
)
from .survey_utils import (
    query_2mass,
    query_gaia,
    query_sdss,
    query_wise,
)
from .visualization import (
    plot_color_color,
    plot_confusion_matrix,
    plot_sky_distribution,
)

__all__ = [
    "build_feature_matrix",
    "build_healpix_index",
    "calculate_colors",
    "calculate_proper_motions",
    "extract_morphology",
    "match_catalogs",
    "plot_color_color",
    "plot_confusion_matrix",
    "plot_sky_distribution",
    "query_2mass",
    "query_gaia",
    "query_sdss",
    "query_wise",
    "spatial_crossmatch",
]
