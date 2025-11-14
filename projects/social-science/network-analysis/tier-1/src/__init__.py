"""
Social Network Analysis Toolkit for SageMaker Studio Lab

Utilities for loading, processing, and analyzing multi-platform social networks.
"""

__version__ = "1.0.0"
__author__ = "AWS Research Jumpstart"

from .data_utils import (
    load_twitter_graph,
    load_reddit_graph,
    load_facebook_graph,
    download_social_data,
)
from .graph_utils import (
    build_unified_graph,
    extract_subgraph,
    compute_graph_statistics,
)
from .gnn_models import (
    GraphSAGE,
    GAT,
    TemporalGNN,
)
from .analysis import (
    detect_communities,
    calculate_influence_scores,
    analyze_information_diffusion,
)
from .visualization import (
    plot_network_graph,
    plot_influence_heatmap,
    create_interactive_dashboard,
)

__all__ = [
    "load_twitter_graph",
    "load_reddit_graph",
    "load_facebook_graph",
    "download_social_data",
    "build_unified_graph",
    "extract_subgraph",
    "compute_graph_statistics",
    "GraphSAGE",
    "GAT",
    "TemporalGNN",
    "detect_communities",
    "calculate_influence_scores",
    "analyze_information_diffusion",
    "plot_network_graph",
    "plot_influence_heatmap",
    "create_interactive_dashboard",
]
