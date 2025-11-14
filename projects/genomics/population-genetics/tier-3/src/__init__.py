"""
Population Genetics Analysis Package

Large-scale population genetics analysis using 1000 Genomes data on AWS.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_access import load_variants, load_genotypes
from .population_structure import calculate_pca, calculate_fst
from .selection import calculate_ihs, calculate_tajimas_d
from .visualization import plot_pca, plot_manhattan

__all__ = [
    "load_variants",
    "load_genotypes",
    "calculate_pca",
    "calculate_fst",
    "calculate_ihs",
    "calculate_tajimas_d",
    "plot_pca",
    "plot_manhattan",
]
