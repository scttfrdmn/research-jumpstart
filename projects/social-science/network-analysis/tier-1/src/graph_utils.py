"""
Graph construction and manipulation utilities.

Functions for building unified graphs, extracting subgraphs, and computing statistics.
"""

from typing import Optional

import networkx as nx
import numpy as np


def build_unified_graph(
    platform_graphs: dict[str, nx.Graph],
    cross_platform_links: Optional[list[tuple[str, str, str, str]]] = None,
) -> nx.DiGraph:
    """
    Build unified multi-platform graph.

    Parameters:
    -----------
    platform_graphs : dict
        Dictionary mapping platform names to graphs
    cross_platform_links : list, optional
        List of (platform1, user1, platform2, user2) tuples for cross-platform edges

    Returns:
    --------
    nx.DiGraph : Unified graph with all platforms
    """
    print("Building unified multi-platform graph...")

    G_unified = nx.DiGraph()

    # Add all nodes and edges from each platform
    node_count = 0
    edge_count = 0

    for platform, G in platform_graphs.items():
        print(f"  Adding {platform}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

        # Add nodes with platform prefix to avoid ID collisions
        for node in G.nodes():
            unified_id = f"{platform}:{node}"
            attrs = G.nodes[node].copy()
            attrs["original_id"] = node
            attrs["platform"] = platform
            G_unified.add_node(unified_id, **attrs)
            node_count += 1

        # Add edges
        for u, v in G.edges():
            unified_u = f"{platform}:{u}"
            unified_v = f"{platform}:{v}"
            attrs = G[u][v].copy()
            G_unified.add_edge(unified_u, unified_v, **attrs)
            edge_count += 1

    # Add cross-platform edges if provided
    if cross_platform_links:
        print(f"  Adding {len(cross_platform_links):,} cross-platform links")
        for platform1, user1, platform2, user2 in cross_platform_links:
            unified_u = f"{platform1}:{user1}"
            unified_v = f"{platform2}:{user2}"
            if unified_u in G_unified and unified_v in G_unified:
                G_unified.add_edge(
                    unified_u,
                    unified_v,
                    weight=1.0,
                    edge_type="cross_platform",
                    platform_source=platform1,
                    platform_target=platform2,
                )
                edge_count += 1

    print(f"✓ Unified graph: {node_count:,} nodes, {edge_count:,} edges")
    return G_unified


def extract_subgraph(G: nx.Graph, center_nodes: list[str], k_hops: int = 2) -> nx.Graph:
    """
    Extract k-hop neighborhood subgraph around center nodes.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    center_nodes : list
        List of center node IDs
    k_hops : int
        Number of hops to include

    Returns:
    --------
    nx.Graph : Subgraph
    """
    subgraph_nodes = set(center_nodes)

    # Add k-hop neighbors
    for node in center_nodes:
        if node in G:
            # BFS to find k-hop neighbors
            for _ in range(k_hops):
                neighbors = set()
                for n in subgraph_nodes:
                    if n in G:
                        neighbors.update(G.neighbors(n))
                subgraph_nodes.update(neighbors)

    # Extract subgraph
    H = G.subgraph(subgraph_nodes).copy()
    return H


def compute_graph_statistics(G: nx.Graph) -> dict:
    """
    Compute comprehensive graph statistics.

    Parameters:
    -----------
    G : nx.Graph
        Input graph

    Returns:
    --------
    dict : Dictionary of statistics
    """
    stats = {}

    # Basic properties
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["density"] = nx.density(G)
    stats["is_directed"] = G.is_directed()

    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    stats["avg_degree"] = np.mean(degree_values)
    stats["median_degree"] = np.median(degree_values)
    stats["max_degree"] = np.max(degree_values)
    stats["min_degree"] = np.min(degree_values)

    # Connectivity
    if G.is_directed():
        stats["is_weakly_connected"] = nx.is_weakly_connected(G)
        stats["is_strongly_connected"] = nx.is_strongly_connected(G)
        stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(G)
        stats["num_strongly_connected_components"] = nx.number_strongly_connected_components(G)
    else:
        stats["is_connected"] = nx.is_connected(G)
        stats["num_connected_components"] = nx.number_connected_components(G)

    # Clustering (only for undirected or small graphs)
    if not G.is_directed() and G.number_of_nodes() < 10000:
        stats["avg_clustering"] = nx.average_clustering(G)

    return stats


def compute_degree_distribution(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute degree distribution.

    Returns:
    --------
    tuple : (degrees, counts) arrays
    """
    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    unique_degrees, counts = np.unique(degree_values, return_counts=True)
    return unique_degrees, counts


def find_largest_component(G: nx.Graph) -> nx.Graph:
    """
    Extract largest connected component.

    Parameters:
    -----------
    G : nx.Graph
        Input graph

    Returns:
    --------
    nx.Graph : Largest component subgraph
    """
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def sample_graph(G: nx.Graph, n_nodes: int, method: str = "random") -> nx.Graph:
    """
    Sample a subgraph.

    Parameters:
    -----------
    G : nx.Graph
        Input graph
    n_nodes : int
        Number of nodes to sample
    method : str
        Sampling method ('random', 'top_degree', 'random_walk')

    Returns:
    --------
    nx.Graph : Sampled subgraph
    """
    n_nodes = min(n_nodes, G.number_of_nodes())

    if method == "random":
        # Random node sampling
        nodes = np.random.choice(list(G.nodes()), size=n_nodes, replace=False)

    elif method == "top_degree":
        # Sample highest degree nodes
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        nodes = [node for node, _ in sorted_nodes[:n_nodes]]

    elif method == "random_walk":
        # Random walk sampling
        nodes = set()
        start_node = np.random.choice(list(G.nodes()))
        current = start_node

        while len(nodes) < n_nodes:
            nodes.add(current)
            neighbors = list(G.neighbors(current))
            if not neighbors:
                # Restart from random node
                current = np.random.choice(list(G.nodes()))
            else:
                current = np.random.choice(neighbors)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return G.subgraph(nodes).copy()


def convert_to_pytorch_geometric(G: nx.Graph):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Returns:
    --------
    torch_geometric.data.Data : PyG Data object
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import from_networkx
    except ImportError:
        raise ImportError(
            "PyTorch Geometric not installed. Install with: pip install torch-geometric"
        )

    # Use PyG's built-in converter
    data = from_networkx(G)

    # If node features don't exist, create them
    if not hasattr(data, "x") or data.x is None:
        # Use degree as simple feature
        degrees = torch.tensor([G.degree(node) for node in G.nodes()], dtype=torch.float).unsqueeze(
            1
        )
        data.x = degrees

    return data
