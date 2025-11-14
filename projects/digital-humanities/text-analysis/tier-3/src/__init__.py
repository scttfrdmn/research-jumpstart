"""Digital Humanities Text Analysis - Unified Studio Package."""

__version__ = '1.0.0'

from .data_access import TextDataAccess
from .text_analysis import (
    analyze_sentiment,
    extract_entities,
    extract_keywords,
    calculate_readability,
    detect_language,
    perform_topic_modeling
)
from .visualization import (
    plot_word_cloud,
    plot_sentiment_timeline,
    plot_entity_network,
    plot_topic_distribution,
    plot_ngram_frequency
)

__all__ = [
    'TextDataAccess',
    'analyze_sentiment',
    'extract_entities',
    'extract_keywords',
    'calculate_readability',
    'detect_language',
    'perform_topic_modeling',
    'plot_word_cloud',
    'plot_sentiment_timeline',
    'plot_entity_network',
    'plot_topic_distribution',
    'plot_ngram_frequency',
]
