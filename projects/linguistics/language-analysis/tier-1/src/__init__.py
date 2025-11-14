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

__version__ = '1.0.0'

from .data_utils import (
    load_dialect_corpus,
    get_data_path,
    download_corpus
)

from .feature_extraction import (
    extract_phonetic_features,
    extract_lexical_features,
    extract_syntactic_features
)

from .analysis import (
    train_dialect_classifier,
    evaluate_model,
    cross_linguistic_comparison
)

from .visualization import (
    plot_dialect_space,
    plot_confusion_matrix,
    create_feature_importance_plot
)

__all__ = [
    'load_dialect_corpus',
    'get_data_path',
    'download_corpus',
    'extract_phonetic_features',
    'extract_lexical_features',
    'extract_syntactic_features',
    'train_dialect_classifier',
    'evaluate_model',
    'cross_linguistic_comparison',
    'plot_dialect_space',
    'plot_confusion_matrix',
    'create_feature_importance_plot'
]
