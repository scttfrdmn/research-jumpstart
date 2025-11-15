"""
Analysis functions for social network analysis.

Includes community detection, influence scoring, and diffusion analysis.
"""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


def detect_communities(G: nx.Graph, method: str = "louvain", resolution: float = 1.0) -> list[set]:
    """
    Detect communities in the network.

    Parameters:
    -----------
    G : nx.Graph
        Input graph (will be converted to undirected if needed)
    method : str
        Community detection method ('louvain', 'label_propagation', 'greedy')
    resolution : float
        Resolution parameter for modularity-based methods

    Returns:
    --------
    list : List of sets, each containing nodes in a community
    """
    # Convert to undirected
    G_undirected = G.to_undirected() if G.is_directed() else G

    print(f"Detecting communities using {method} algorithm...")

    if method == "louvain":
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(G_undirected, resolution=resolution)
            # Convert to list of sets
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            communities = list(communities.values())
        except ImportError:
            print("python-louvain not installed, falling back to greedy modularity")
            from networkx.algorithms import community

            communities = list(community.greedy_modularity_communities(G_undirected))

    elif method == "label_propagation":
        from networkx.algorithms import community

        communities = list(community.label_propagation_communities(G_undirected))

    elif method == "greedy":
        from networkx.algorithms import community

        communities = list(community.greedy_modularity_communities(G_undirected))

    else:
        raise ValueError(f"Unknown community detection method: {method}")

    # Sort by size
    communities = sorted(communities, key=len, reverse=True)

    print(f"✓ Detected {len(communities)} communities")
    print(f"  Largest: {len(communities[0]):,} nodes")
    print(f"  Smallest: {len(communities[-1]):,} nodes")

    return communities


def calculate_influence_scores(G: nx.Graph, methods: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Calculate various influence scores for all nodes.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    methods : list, optional
        List of methods to use. Options: 'pagerank', 'betweenness', 'degree',
        'closeness', 'eigenvector'. If None, uses all.

    Returns:
    --------
    pd.DataFrame : DataFrame with influence scores for each node
    """
    if methods is None:
        methods = ["pagerank", "degree", "betweenness"]

    print("Calculating influence scores...")
    scores = {}

    # PageRank
    if "pagerank" in methods:
        print("  Computing PageRank...")
        scores["pagerank"] = nx.pagerank(G, alpha=0.85)

    # Degree centrality
    if "degree" in methods:
        print("  Computing degree centrality...")
        if G.is_directed():
            scores["in_degree"] = dict(G.in_degree())
            scores["out_degree"] = dict(G.out_degree())
            scores["degree_centrality"] = nx.in_degree_centrality(G)
        else:
            scores["degree"] = dict(G.degree())
            scores["degree_centrality"] = nx.degree_centrality(G)

    # Betweenness centrality
    if "betweenness" in methods:
        print("  Computing betweenness centrality (sampling for speed)...")
        k = min(1000, G.number_of_nodes())
        scores["betweenness"] = nx.betweenness_centrality(G, k=k)

    # Closeness centrality
    if "closeness" in methods and G.number_of_nodes() < 10000:
        print("  Computing closeness centrality...")
        scores["closeness"] = nx.closeness_centrality(G)

    # Eigenvector centrality
    if "eigenvector" in methods and G.number_of_nodes() < 10000:
        print("  Computing eigenvector centrality...")
        try:
            scores["eigenvector"] = nx.eigenvector_centrality(G, max_iter=100)
        except:
            print("    Warning: Eigenvector centrality failed to converge")

    # Convert to DataFrame
    df = pd.DataFrame(scores)
    df.index.name = "node"

    print(f"✓ Calculated {len(df.columns)} influence metrics for {len(df):,} nodes")

    return df


def analyze_information_diffusion(
    G: nx.Graph, seed_nodes: list[str], steps: int = 10, threshold: float = 0.5
) -> dict:
    """
    Simulate information diffusion using Linear Threshold Model.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    seed_nodes : list
        Initial nodes that have the information
    steps : int
        Number of diffusion steps to simulate
    threshold : float
        Activation threshold (0-1)

    Returns:
    --------
    dict : Diffusion statistics
    """
    print(f"Simulating information diffusion from {len(seed_nodes)} seed nodes...")

    # Initialize active nodes
    active = set(seed_nodes)
    active_by_step = [set(seed_nodes)]

    # Simulate diffusion
    for step in range(steps):
        newly_active = set()

        # Check each inactive node
        for node in G.nodes():
            if node in active:
                continue

            # Get active neighbors
            neighbors = list(G.predecessors(node)) if G.is_directed() else list(G.neighbors(node))

            if not neighbors:
                continue

            # Calculate activation probability
            active_neighbors = [n for n in neighbors if n in active]
            activation_prob = len(active_neighbors) / len(neighbors)

            # Activate if above threshold
            if activation_prob >= threshold:
                newly_active.add(node)

        # Update active set
        active.update(newly_active)
        active_by_step.append(newly_active)

        if not newly_active:
            print(f"  Diffusion converged at step {step + 1}")
            break

    # Calculate statistics
    total_reached = len(active)
    reach_rate = total_reached / G.number_of_nodes()

    stats = {
        "total_reached": total_reached,
        "reach_rate": reach_rate,
        "convergence_step": len(active_by_step) - 1,
        "active_by_step": [len(s) for s in active_by_step],
        "final_active_nodes": active,
    }

    print(f"✓ Diffusion reached {total_reached:,} nodes ({reach_rate * 100:.1f}% of network)")

    return stats


def identify_key_spreaders(
    G: nx.Graph, influence_scores: pd.DataFrame, top_k: int = 100
) -> pd.DataFrame:
    """
    Identify key information spreaders combining multiple metrics.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    influence_scores : pd.DataFrame
        DataFrame of influence scores
    top_k : int
        Number of top spreaders to return

    Returns:
    --------
    pd.DataFrame : Top spreaders with combined scores
    """
    # Normalize all scores to 0-1
    df_norm = influence_scores.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # Compute combined score (weighted average)
    weights = {"pagerank": 0.4, "betweenness": 0.3, "degree_centrality": 0.2, "in_degree": 0.1}

    combined_score = 0
    for metric, weight in weights.items():
        if metric in df_norm.columns:
            combined_score += weight * df_norm[metric]

    df_norm["combined_score"] = combined_score

    # Get top spreaders
    top_spreaders = df_norm.nlargest(top_k, "combined_score")

    return top_spreaders


def compute_network_metrics(G: nx.Graph) -> dict:
    """
    Compute comprehensive network metrics.

    Returns:
    --------
    dict : Dictionary of network metrics
    """
    print("Computing network metrics...")

    metrics = {}

    # Basic properties
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)

    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    metrics["avg_degree"] = np.mean(degree_values)
    metrics["median_degree"] = np.median(degree_values)
    metrics["max_degree"] = np.max(degree_values)

    # Clustering
    if not G.is_directed() and G.number_of_nodes() < 10000:
        metrics["avg_clustering"] = nx.average_clustering(G)

    # Connected components
    if G.is_directed():
        metrics["num_weakly_connected"] = nx.number_weakly_connected_components(G)
        metrics["num_strongly_connected"] = nx.number_strongly_connected_components(G)
    else:
        metrics["num_connected_components"] = nx.number_connected_components(G)

    # Path length (on sample for large graphs)
    if G.number_of_nodes() < 10000 and not G.is_directed() and nx.is_connected(G):
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G)
        metrics["diameter"] = nx.diameter(G)

    print("✓ Network metrics computed")

    return metrics


def detect_coordinated_behavior(
    G: nx.Graph, time_window: int = 3600, similarity_threshold: float = 0.8
) -> list[set]:
    """
    Detect potentially coordinated behavior (bot networks, manipulation).

    Looks for:
    - Highly synchronized posting patterns
    - Similar content/behavior
    - Tight clustering with high similarity

    Parameters:
    -----------
    G : nx.Graph
        Input graph with temporal edge attributes
    time_window : int
        Time window in seconds for synchronization detection
    similarity_threshold : float
        Threshold for behavioral similarity (0-1)

    Returns:
    --------
    list : List of suspicious node groups
    """
    print("Detecting coordinated behavior...")

    suspicious_groups = []

    # Find tightly connected clusters
    G_undirected = G.to_undirected() if G.is_directed() else G

    # Get k-cores (nodes with at least k connections)
    k_cores = nx.k_core(G_undirected, k=5)

    # Within each core, check for high similarity
    communities = list(nx.connected_components(k_cores))

    for comm in communities:
        if len(comm) < 5:  # Too small to be interesting
            continue

        # Analyze temporal patterns within community
        # (This is simplified - real analysis would look at timestamps)
        subgraph = G.subgraph(comm)

        # Check density
        density = nx.density(subgraph)

        if density > 0.5:  # High internal connectivity
            suspicious_groups.append(comm)

    print(f"✓ Found {len(suspicious_groups)} potentially coordinated groups")

    return suspicious_groups
