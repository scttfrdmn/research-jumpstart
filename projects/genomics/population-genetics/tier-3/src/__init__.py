"""
Population Genetics Analysis Package

Large-scale population genetics analysis using 1000 Genomes data on AWS.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_access import load_genotypes, load_variants
from .population_structure import calculate_fst, calculate_pca
from .selection import calculate_ihs, calculate_tajimas_d
from .visualization import plot_manhattan, plot_pca

__all__ = [
    "calculate_fst",
    "calculate_ihs",
    "calculate_pca",
    "calculate_tajimas_d",
    "load_genotypes",
    "load_variants",
    "plot_manhattan",
    "plot_pca",
]
