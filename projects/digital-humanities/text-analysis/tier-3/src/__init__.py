"""Digital Humanities Text Analysis - Unified Studio Package."""

__version__ = "1.0.0"

from .data_access import TextDataAccess
from .text_analysis import (
    analyze_sentiment,
    calculate_readability,
    detect_language,
    extract_entities,
    extract_keywords,
    perform_topic_modeling,
)
from .visualization import (
    plot_entity_network,
    plot_ngram_frequency,
    plot_sentiment_timeline,
    plot_topic_distribution,
    plot_word_cloud,
)

__all__ = [
    "TextDataAccess",
    "analyze_sentiment",
    "calculate_readability",
    "detect_language",
    "extract_entities",
    "extract_keywords",
    "perform_topic_modeling",
    "plot_entity_network",
    "plot_ngram_frequency",
    "plot_sentiment_timeline",
    "plot_topic_distribution",
    "plot_word_cloud",
]
