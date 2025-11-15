"""
Social Media Analysis Package

Production-ready modules for social media analysis on AWS.
"""

from .comprehend_client import ComprehendAnalyzer
from .data_access import SocialMediaDataAccess
from .network_analysis import build_interaction_network, calculate_centrality, detect_communities
from .nlp_analysis import (
    analyze_sentiment,
    detect_misinformation_patterns,
    extract_topics,
    preprocess_text,
)
from .visualization import (
    plot_engagement_analysis,
    plot_network_graph,
    plot_sentiment_distribution,
    plot_topic_evolution,
)

__version__ = "1.0.0"
__all__ = [
    # AWS Comprehend
    "ComprehendAnalyzer",
    # Data access
    "SocialMediaDataAccess",
    "analyze_sentiment",
    # Network analysis
    "build_interaction_network",
    "calculate_centrality",
    "detect_communities",
    "detect_misinformation_patterns",
    "extract_topics",
    "plot_engagement_analysis",
    "plot_network_graph",
    # Visualization
    "plot_sentiment_distribution",
    "plot_topic_evolution",
    # NLP analysis
    "preprocess_text",
]
