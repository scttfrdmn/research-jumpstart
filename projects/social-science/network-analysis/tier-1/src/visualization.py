"""
Visualization functions for social network analysis.

Includes network plots, influence heatmaps, and interactive dashboards.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def plot_network_graph(
    G: nx.Graph,
    node_colors: Optional[dict] = None,
    node_sizes: Optional[dict] = None,
    title: str = "Network Graph",
    layout: str = "spring",
    figsize: tuple[int, int] = (14, 10),
    show_labels: bool = True,
    top_k_labels: int = 20,
):
    """
    Plot network graph with customizable node colors and sizes.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    node_colors : dict, optional
        Dictionary mapping nodes to colors
    node_sizes : dict, optional
        Dictionary mapping nodes to sizes
    title : str
        Plot title
    layout : str
        Layout algorithm ('spring', 'kamada_kawai', 'circular')
    figsize : tuple
        Figure size
    show_labels : bool
        Whether to show node labels
    top_k_labels : int
        Number of top nodes to label
    """
    plt.figure(figsize=figsize)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Prepare node colors
    if node_colors is None:
        colors = ["skyblue"] * G.number_of_nodes()
    else:
        colors = [node_colors.get(node, "skyblue") for node in G.nodes()]

    # Prepare node sizes
    if node_sizes is None:
        sizes = [100] * G.number_of_nodes()
    else:
        sizes = [node_sizes.get(node, 100) for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_size=sizes, alpha=0.7, edgecolors="black", linewidths=0.5
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, alpha=0.2, arrows=G.is_directed(), arrowsize=10, edge_color="gray", width=0.5
    )

    # Draw labels for top nodes
    if show_labels and top_k_labels > 0:
        # Label top nodes by size
        if node_sizes:
            top_nodes = sorted(node_sizes.items(), key=lambda x: x[1], reverse=True)[:top_k_labels]
            labels = {node: str(node) for node, _ in top_nodes}
        else:
            labels = {node: str(node) for node in list(G.nodes())[:top_k_labels]}

        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    plt.title(title, fontsize=15, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_degree_distribution(
    G: nx.Graph, log_scale: bool = True, figsize: tuple[int, int] = (12, 5)
):
    """
    Plot degree distribution of the network.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    log_scale : bool
        Whether to use log-log scale
    figsize : tuple
        Figure size
    """
    _fig, axes = plt.subplots(1, 2 if G.is_directed() else 1, figsize=figsize)

    if not G.is_directed():
        axes = [axes]

    # In-degree distribution (or total degree for undirected)
    if G.is_directed():
        degrees = dict(G.in_degree())
        title = "In-Degree Distribution"
    else:
        degrees = dict(G.degree())
        title = "Degree Distribution"

    degree_values = list(degrees.values())
    unique, counts = np.unique(degree_values, return_counts=True)

    if log_scale:
        axes[0].loglog(unique, counts, "bo-", alpha=0.6, markersize=4)
        axes[0].set_xlabel("Degree (log scale)", fontweight="bold")
        axes[0].set_ylabel("Count (log scale)", fontweight="bold")
    else:
        axes[0].plot(unique, counts, "bo-", alpha=0.6, markersize=4)
        axes[0].set_xlabel("Degree", fontweight="bold")
        axes[0].set_ylabel("Count", fontweight="bold")

    axes[0].set_title(title, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Out-degree distribution (for directed graphs)
    if G.is_directed():
        degrees = dict(G.out_degree())
        degree_values = list(degrees.values())
        unique, counts = np.unique(degree_values, return_counts=True)

        if log_scale:
            axes[1].loglog(unique, counts, "ro-", alpha=0.6, markersize=4)
            axes[1].set_xlabel("Degree (log scale)", fontweight="bold")
            axes[1].set_ylabel("Count (log scale)", fontweight="bold")
        else:
            axes[1].plot(unique, counts, "ro-", alpha=0.6, markersize=4)
            axes[1].set_xlabel("Degree", fontweight="bold")
            axes[1].set_ylabel("Count", fontweight="bold")

        axes[1].set_title("Out-Degree Distribution", fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_influence_heatmap(
    influence_scores: pd.DataFrame, top_k: int = 50, figsize: tuple[int, int] = (10, 12)
):
    """
    Plot heatmap of influence scores.

    Parameters:
    -----------
    influence_scores : pd.DataFrame
        DataFrame with influence scores
    top_k : int
        Number of top nodes to show
    figsize : tuple
        Figure size
    """
    # Get top nodes
    if "combined_score" in influence_scores.columns:
        top_nodes = influence_scores.nlargest(top_k, "combined_score")
    else:
        top_nodes = influence_scores.head(top_k)

    # Normalize scores for visualization
    df_norm = top_nodes.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_norm,
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Score"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(f"Top {top_k} Nodes by Influence Metrics", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Influence Metric", fontsize=12, fontweight="bold")
    plt.ylabel("Node", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_community_sizes(communities: list, top_k: int = 20, figsize: tuple[int, int] = (12, 6)):
    """
    Plot community size distribution.

    Parameters:
    -----------
    communities : list
        List of community sets
    top_k : int
        Number of top communities to show
    figsize : tuple
        Figure size
    """
    # Get community sizes
    sizes = [len(comm) for comm in communities]
    sizes_sorted = sorted(sizes, reverse=True)[:top_k]

    _fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of top communities
    axes[0].bar(range(1, len(sizes_sorted) + 1), sizes_sorted, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Community Rank", fontweight="bold")
    axes[0].set_ylabel("Number of Nodes", fontweight="bold")
    axes[0].set_title(f"Top {top_k} Communities by Size", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Distribution of all community sizes
    axes[1].hist(sizes, bins=50, color="coral", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Community Size", fontweight="bold")
    axes[1].set_ylabel("Number of Communities", fontweight="bold")
    axes[1].set_title("Community Size Distribution", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print(f"Total communities: {len(communities)}")
    print(f"Largest community: {max(sizes):,} nodes")
    print(f"Smallest community: {min(sizes):,} nodes")
    print(f"Average size: {np.mean(sizes):.1f} nodes")


def plot_diffusion_cascade(diffusion_stats: dict, figsize: tuple[int, int] = (12, 6)):
    """
    Plot information diffusion cascade over time.

    Parameters:
    -----------
    diffusion_stats : dict
        Diffusion statistics from analyze_information_diffusion
    figsize : tuple
        Figure size
    """
    active_by_step = diffusion_stats["active_by_step"]

    _fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cumulative reach over time
    cumulative = np.cumsum(active_by_step)
    axes[0].plot(range(len(cumulative)), cumulative, "b-", linewidth=2, marker="o")
    axes[0].set_xlabel("Diffusion Step", fontweight="bold")
    axes[0].set_ylabel("Cumulative Nodes Reached", fontweight="bold")
    axes[0].set_title("Information Diffusion Cascade", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # New activations per step
    axes[1].bar(range(len(active_by_step)), active_by_step, color="green", alpha=0.7)
    axes[1].set_xlabel("Diffusion Step", fontweight="bold")
    axes[1].set_ylabel("New Nodes Activated", fontweight="bold")
    axes[1].set_title("Activation Rate Over Time", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    print(f"Total nodes reached: {diffusion_stats['total_reached']:,}")
    print(f"Reach rate: {diffusion_stats['reach_rate'] * 100:.1f}%")
    print(f"Convergence step: {diffusion_stats['convergence_step']}")


def create_interactive_dashboard(
    G: nx.Graph, influence_scores: pd.DataFrame, communities: Optional[list] = None
):
    """
    Create interactive dashboard using Plotly.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    influence_scores : pd.DataFrame
        DataFrame with influence scores
    communities : list, optional
        List of community sets
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Network Graph",
            "Degree Distribution",
            "Top Influencers",
            "Community Sizes",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    # 1. Network graph (sample for performance)
    sample_size = min(500, G.number_of_nodes())
    nodes_sample = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
    G_sample = G.subgraph(nodes_sample)

    pos = nx.spring_layout(G_sample, k=0.5, iterations=20)

    edge_trace = go.Scatter(
        x=[], y=[], line={"width": 0.5, "color": "#888"}, hoverinfo="none", mode="lines"
    )

    for edge in G_sample.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G_sample.nodes()],
        y=[pos[node][1] for node in G_sample.nodes()],
        mode="markers",
        hoverinfo="text",
        marker={"size": 10, "color": "skyblue", "line": {"width": 2}},
    )

    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)

    # 2. Degree distribution
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    unique, counts = np.unique(degree_values, return_counts=True)

    fig.add_trace(
        go.Scatter(x=unique, y=counts, mode="markers", marker={"color": "blue"}), row=1, col=2
    )

    # 3. Top influencers
    if "combined_score" in influence_scores.columns:
        top_20 = influence_scores.nlargest(20, "combined_score")
        scores = top_20["combined_score"].values
    else:
        top_20 = influence_scores.head(20)
        scores = top_20.iloc[:, 0].values

    fig.add_trace(go.Bar(x=list(range(1, 21)), y=scores, marker={"color": "coral"}), row=2, col=1)

    # 4. Community sizes
    if communities:
        sizes = sorted([len(c) for c in communities], reverse=True)[:20]
        fig.add_trace(
            go.Bar(x=list(range(1, len(sizes) + 1)), y=sizes, marker={"color": "green"}),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title_text="Social Network Analysis Dashboard", showlegend=False, height=900, width=1400
    )

    fig.show()

    print("✓ Interactive dashboard created")
