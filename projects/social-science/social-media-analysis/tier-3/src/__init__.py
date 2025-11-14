"""
Social Media Analysis Package

Production-ready modules for social media analysis on AWS.
"""

from .data_access import SocialMediaDataAccess
from .nlp_analysis import (
    preprocess_text,
    analyze_sentiment,
    extract_topics,
    detect_misinformation_patterns
)
from .network_analysis import (
    build_interaction_network,
    calculate_centrality,
    detect_communities
)
from .comprehend_client import ComprehendAnalyzer
from .visualization import (
    plot_sentiment_distribution,
    plot_topic_evolution,
    plot_network_graph,
    plot_engagement_analysis
)

__version__ = '1.0.0'
__all__ = [
    # Data access
    'SocialMediaDataAccess',
    # NLP analysis
    'preprocess_text',
    'analyze_sentiment',
    'extract_topics',
    'detect_misinformation_patterns',
    # Network analysis
    'build_interaction_network',
    'calculate_centrality',
    'detect_communities',
    # AWS Comprehend
    'ComprehendAnalyzer',
    # Visualization
    'plot_sentiment_distribution',
    'plot_topic_evolution',
    'plot_network_graph',
    'plot_engagement_analysis',
]
