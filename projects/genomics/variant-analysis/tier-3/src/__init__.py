"""Genomic Variant Analysis - Unified Studio Package."""

__version__ = "1.0.0"

from .data_access import GenomicsDataAccess
from .variant_analysis import (
    annotate_variants,
    calculate_allele_frequencies,
    filter_variants_by_quality,
)
from .visualization import plot_manhattan, plot_quality_distribution, plot_variant_density

__all__ = [
    "GenomicsDataAccess",
    "annotate_variants",
    "calculate_allele_frequencies",
    "filter_variants_by_quality",
    "plot_manhattan",
    "plot_quality_distribution",
    "plot_variant_density",
]
