"""
Genomics Variant Analysis Toolkit for SageMaker Studio Lab

Utilities for loading BAM files, generating pileup tensors, variant calling,
and analyzing population-scale genomic data.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .analysis import (
    calculate_metrics,
    call_variants,
    compare_with_truth_set,
    ensemble_predict,
    train_variant_caller,
)
from .data_utils import (
    download_bam_file,
    extract_variant_features,
    generate_pileup_tensor,
    load_bam_file,
    load_reference_genome,
)
from .visualization import (
    create_manhattan_plot,
    plot_population_comparison,
    plot_read_depth,
    plot_roc_curve,
    plot_variant_density,
)

__all__ = [
    "calculate_metrics",
    "call_variants",
    "compare_with_truth_set",
    "create_manhattan_plot",
    "download_bam_file",
    "ensemble_predict",
    "extract_variant_features",
    "generate_pileup_tensor",
    "load_bam_file",
    "load_reference_genome",
    "plot_population_comparison",
    "plot_read_depth",
    "plot_roc_curve",
    "plot_variant_density",
    "train_variant_caller",
]
