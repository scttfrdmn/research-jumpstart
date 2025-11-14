"""Genomic Variant Analysis - Unified Studio Package."""

__version__ = '1.0.0'

from .data_access import GenomicsDataAccess
from .variant_analysis import (
    calculate_allele_frequencies,
    filter_variants_by_quality,
    annotate_variants
)
from .visualization import (
    plot_manhattan,
    plot_variant_density,
    plot_quality_distribution
)

__all__ = [
    'GenomicsDataAccess',
    'calculate_allele_frequencies',
    'filter_variants_by_quality',
    'annotate_variants',
    'plot_manhattan',
    'plot_variant_density',
    'plot_quality_distribution',
]
