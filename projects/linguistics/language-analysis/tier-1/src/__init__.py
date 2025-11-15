"""
Linguistics Analysis Utilities for SageMaker Studio Lab

This package provides data loading, feature extraction, analysis, and
visualization tools for cross-linguistic dialectology research.

Modules:
    data_utils: Dataset loading and caching
    feature_extraction: Linguistic feature extraction
    analysis: Statistical analysis and modeling
    visualization: Plotting and visualization functions
"""

__version__ = "1.0.0"

from .analysis import cross_linguistic_comparison, evaluate_model, train_dialect_classifier
from .data_utils import download_corpus, get_data_path, load_dialect_corpus
from .feature_extraction import (
    extract_lexical_features,
    extract_phonetic_features,
    extract_syntactic_features,
)
from .visualization import create_feature_importance_plot, plot_confusion_matrix, plot_dialect_space

__all__ = [
    "create_feature_importance_plot",
    "cross_linguistic_comparison",
    "download_corpus",
    "evaluate_model",
    "extract_lexical_features",
    "extract_phonetic_features",
    "extract_syntactic_features",
    "get_data_path",
    "load_dialect_corpus",
    "plot_confusion_matrix",
    "plot_dialect_space",
    "train_dialect_classifier",
]
