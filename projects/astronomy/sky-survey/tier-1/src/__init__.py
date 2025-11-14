"""
Astronomy Sky Survey Cross-Matching Toolkit for SageMaker Studio Lab

Utilities for querying surveys, cross-matching catalogs, and classifying sources.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .survey_utils import (
    query_sdss,
    query_gaia,
    query_2mass,
    query_wise,
)
from .crossmatch import (
    spatial_crossmatch,
    build_healpix_index,
    match_catalogs,
)
from .features import (
    calculate_colors,
    calculate_proper_motions,
    extract_morphology,
    build_feature_matrix,
)
from .visualization import (
    plot_sky_distribution,
    plot_color_color,
    plot_confusion_matrix,
)

__all__ = [
    "query_sdss",
    "query_gaia",
    "query_2mass",
    "query_wise",
    "spatial_crossmatch",
    "build_healpix_index",
    "match_catalogs",
    "calculate_colors",
    "calculate_proper_motions",
    "extract_morphology",
    "build_feature_matrix",
    "plot_sky_distribution",
    "plot_color_color",
    "plot_confusion_matrix",
]
