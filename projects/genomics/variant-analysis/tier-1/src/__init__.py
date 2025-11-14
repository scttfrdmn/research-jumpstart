"""
Genomics Variant Analysis Toolkit for SageMaker Studio Lab

Utilities for loading BAM files, generating pileup tensors, variant calling,
and analyzing population-scale genomic data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    download_bam_file,
    load_bam_file,
    load_reference_genome,
    generate_pileup_tensor,
    extract_variant_features,
)
from .analysis import (
    train_variant_caller,
    call_variants,
    ensemble_predict,
    calculate_metrics,
    compare_with_truth_set,
)
from .visualization import (
    plot_read_depth,
    plot_variant_density,
    plot_roc_curve,
    plot_population_comparison,
    create_manhattan_plot,
)

__all__ = [
    "download_bam_file",
    "load_bam_file",
    "load_reference_genome",
    "generate_pileup_tensor",
    "extract_variant_features",
    "train_variant_caller",
    "call_variants",
    "ensemble_predict",
    "calculate_metrics",
    "compare_with_truth_set",
    "plot_read_depth",
    "plot_variant_density",
    "plot_roc_curve",
    "plot_population_comparison",
    "create_manhattan_plot",
]
