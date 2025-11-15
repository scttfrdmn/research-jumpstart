"""
Social Network Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing multi-platform social networks.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .analysis import (
    analyze_information_diffusion,
    calculate_influence_scores,
    detect_communities,
)
from .data_utils import (
    download_social_data,
    load_facebook_graph,
    load_reddit_graph,
    load_twitter_graph,
)
from .gnn_models import (
    GAT,
    GraphSAGE,
    TemporalGNN,
)
from .graph_utils import (
    build_unified_graph,
    compute_graph_statistics,
    extract_subgraph,
)
from .visualization import (
    create_interactive_dashboard,
    plot_influence_heatmap,
    plot_network_graph,
)

__all__ = [
    "GAT",
    "GraphSAGE",
    "TemporalGNN",
    "analyze_information_diffusion",
    "build_unified_graph",
    "calculate_influence_scores",
    "compute_graph_statistics",
    "create_interactive_dashboard",
    "detect_communities",
    "download_social_data",
    "extract_subgraph",
    "load_facebook_graph",
    "load_reddit_graph",
    "load_twitter_graph",
    "plot_influence_heatmap",
    "plot_network_graph",
]
