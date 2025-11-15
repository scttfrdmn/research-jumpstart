#!/usr/bin/env python3
"""
Social Network Analysis Utilities

Tools for analyzing social networks: centrality measures, community detection,
influence metrics, and information diffusion.

Usage:
    from network_analysis import compute_centrality, detect_communities

    centrality = compute_centrality(G, method='pagerank')
    communities = detect_communities(G, method='louvain')
"""

from collections import defaultdict

import community as community_louvain
import networkx as nx
import numpy as np


def compute_centrality(G, method="degree", top_k=None):
    """
    Compute node centrality measures.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    method : str
        Centrality measure: 'degree', 'betweenness', 'closeness',
        'eigenvector', 'pagerank', 'katz'
    top_k : int, optional
        Return only top k nodes

    Returns:
    --------
    centrality : dict
        Node -> centrality score mapping
    """
    if method == "degree":
        centrality = dict(G.degree())
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif method == "closeness":
        centrality = nx.closeness_centrality(G)
    elif method == "eigenvector":
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Fallback to eigenvector centrality with numpy
            centrality = nx.eigenvector_centrality_numpy(G)
    elif method == "pagerank":
        centrality = nx.pagerank(G)
    elif method == "katz":
        centrality = nx.katz_centrality(G)
    else:
        raise ValueError(f"Unknown centrality method: {method}")

    # Optionally return top k
    if top_k is not None:
        centrality = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k])

    return centrality


def detect_communities(G, method="louvain", resolution=1.0):
    """
    Detect communities in a network.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    method : str
        Community detection algorithm: 'louvain', 'label_propagation',
        'greedy_modularity'
    resolution : float
        Resolution parameter for Louvain (higher = more communities)

    Returns:
    --------
    communities : dict
        Node -> community ID mapping
    """
    if method == "louvain":
        communities = community_louvain.best_partition(G, resolution=resolution)

    elif method == "label_propagation":
        communities_gen = nx.community.label_propagation_communities(G)
        communities = {}
        for i, comm in enumerate(communities_gen):
            for node in comm:
                communities[node] = i

    elif method == "greedy_modularity":
        communities_gen = nx.community.greedy_modularity_communities(G)
        communities = {}
        for i, comm in enumerate(communities_gen):
            for node in comm:
                communities[node] = i

    else:
        raise ValueError(f"Unknown community detection method: {method}")

    return communities


def compute_modularity(G, communities):
    """
    Compute modularity of community partition.

    Modularity measures the strength of division of a network into communities.
    Values range from -1 to 1, with higher values indicating stronger community structure.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    communities : dict
        Node -> community ID mapping

    Returns:
    --------
    modularity : float
        Modularity score
    """
    # Convert dict to list of sets
    community_sets = defaultdict(set)
    for node, comm_id in communities.items():
        community_sets[comm_id].add(node)

    modularity = nx.community.modularity(G, community_sets.values())
    return modularity


def compute_network_metrics(G):
    """
    Compute comprehensive network statistics.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph

    Returns:
    --------
    metrics : dict
        Network metrics
    """
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
    }

    # Average clustering coefficient
    metrics["avg_clustering"] = nx.average_clustering(G)

    # Diameter (only if connected)
    if metrics["is_connected"]:
        metrics["diameter"] = nx.diameter(G)
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        if G.is_directed():
            largest_cc = max(nx.weakly_connected_components(G), key=len)
        else:
            largest_cc = max(nx.connected_components(G), key=len)

        G_cc = G.subgraph(largest_cc)
        metrics["diameter"] = nx.diameter(G_cc)
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G_cc)
        metrics["largest_cc_size"] = len(largest_cc)

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    metrics["avg_degree"] = np.mean(degrees)
    metrics["max_degree"] = np.max(degrees)
    metrics["min_degree"] = np.min(degrees)

    return metrics


def identify_influencers(G, methods=None, top_k=10):
    """
    Identify influential nodes using multiple metrics.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    methods : list of str
        Centrality methods to use
    top_k : int
        Number of top influencers per method

    Returns:
    --------
    influencers : dict
        Method -> list of (node, score) tuples
    """
    if methods is None:
        methods = ["pagerank", "betweenness", "degree"]
    influencers = {}

    for method in methods:
        centrality = compute_centrality(G, method=method)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        influencers[method] = top_nodes

    return influencers


def simulate_information_diffusion(G, seed_nodes, threshold=0.3, max_iterations=100):
    """
    Simulate information diffusion using threshold model.

    Nodes adopt if threshold fraction of neighbors have adopted.

    Parameters:
    -----------
    G : networkx.Graph
        Social network
    seed_nodes : list
        Initial adopters
    threshold : float
        Adoption threshold (fraction of neighbors)
    max_iterations : int
        Maximum simulation steps

    Returns:
    --------
    adoption_history : list of sets
        Adopters at each time step
    """
    adopters = set(seed_nodes)
    adoption_history = [adopters.copy()]

    for _iteration in range(max_iterations):
        new_adopters = set()

        for node in G.nodes():
            if node in adopters:
                continue

            # Check if threshold is met
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                continue

            adopted_neighbors = sum(1 for n in neighbors if n in adopters)
            fraction = adopted_neighbors / len(neighbors)

            if fraction >= threshold:
                new_adopters.add(node)

        if not new_adopters:
            break

        adopters.update(new_adopters)
        adoption_history.append(adopters.copy())

    return adoption_history


def compute_k_core(G, k=2):
    """
    Find k-core of network.

    The k-core is the maximal subgraph where all nodes have degree >= k.
    Useful for identifying cohesive subgroups.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    k : int
        Minimum degree

    Returns:
    --------
    k_core : networkx.Graph
        K-core subgraph
    """
    return nx.k_core(G, k=k)


def find_bridges(G):
    """
    Find bridge edges in network.

    Bridges are edges whose removal increases the number of connected components.
    Important for understanding network fragility.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph (must be undirected)

    Returns:
    --------
    bridges : list of tuples
        List of bridge edges
    """
    if G.is_directed():
        raise ValueError("Bridge detection requires undirected graph")

    return list(nx.bridges(G))


def compute_assortativity(G, attribute=None):
    """
    Compute degree assortativity coefficient.

    Measures tendency of nodes to connect to others with similar degree.
    Positive = assortative, negative = disassortative.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    attribute : str, optional
        Node attribute for attribute assortativity

    Returns:
    --------
    assortativity : float
        Assortativity coefficient
    """
    if attribute is None:
        # Degree assortativity
        return nx.degree_assortativity_coefficient(G)
    else:
        # Attribute assortativity
        return nx.attribute_assortativity_coefficient(G, attribute)


def detect_cliques(G, min_size=3):
    """
    Find cliques (complete subgraphs) in network.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    min_size : int
        Minimum clique size

    Returns:
    --------
    cliques : list of sets
        List of cliques
    """
    cliques = [c for c in nx.find_cliques(G) if len(c) >= min_size]
    return cliques


def compute_ego_network(G, node, radius=1):
    """
    Extract ego network for a node.

    Ego network includes the node, its neighbors within radius hops,
    and edges among them.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    node : node
        Ego node
    radius : int
        Neighborhood radius

    Returns:
    --------
    ego_graph : networkx.Graph
        Ego network
    """
    return nx.ego_graph(G, node, radius=radius)


def rank_nodes_by_influence(G, seed_nodes, method="independent_cascade", n_simulations=100):
    """
    Rank nodes by their influence in information diffusion.

    Parameters:
    -----------
    G : networkx.Graph
        Social network
    seed_nodes : list
        Candidate seed nodes to evaluate
    method : str
        Diffusion model ('independent_cascade' or 'threshold')
    n_simulations : int
        Number of simulation runs per seed

    Returns:
    --------
    rankings : list of tuples
        (node, avg_influence_spread) sorted by influence
    """
    influence_scores = {}

    for seed in seed_nodes:
        spreads = []

        for _ in range(n_simulations):
            if method == "independent_cascade":
                # Simplified IC model
                adopters = {seed}
                for _ in range(10):  # Max hops
                    new_adopters = set()
                    for node in adopters:
                        for neighbor in G.neighbors(node):
                            if neighbor not in adopters and np.random.rand() < 0.1:
                                new_adopters.add(neighbor)
                    if not new_adopters:
                        break
                    adopters.update(new_adopters)

                spreads.append(len(adopters))

            else:  # threshold model
                history = simulate_information_diffusion(G, [seed], threshold=0.3)
                spreads.append(len(history[-1]))

        influence_scores[seed] = np.mean(spreads)

    rankings = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    return rankings


if __name__ == "__main__":
    # Example usage
    print("Social Network Analysis Utilities")
    print("=" * 60)

    # Create example network
    print("\n1. Creating example social network...")
    G = nx.karate_club_graph()  # Classic social network dataset
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")

    # Compute network metrics
    print("\n2. Computing network metrics...")
    metrics = compute_network_metrics(G)
    print(f"   Density: {metrics['density']:.3f}")
    print(f"   Avg clustering: {metrics['avg_clustering']:.3f}")
    print(f"   Diameter: {metrics['diameter']}")
    print(f"   Avg degree: {metrics['avg_degree']:.2f}")

    # Centrality analysis
    print("\n3. Analyzing centrality...")
    pagerank = compute_centrality(G, method="pagerank", top_k=3)
    print("   Top 3 nodes by PageRank:")
    for node, score in pagerank.items():
        print(f"     Node {node}: {score:.4f}")

    # Community detection
    print("\n4. Detecting communities...")
    communities = detect_communities(G, method="louvain")
    modularity = compute_modularity(G, communities)
    n_communities = len(set(communities.values()))
    print(f"   Communities found: {n_communities}")
    print(f"   Modularity: {modularity:.3f}")

    # Information diffusion
    print("\n5. Simulating information diffusion...")
    seed_nodes = [0]  # Start with node 0
    history = simulate_information_diffusion(G, seed_nodes, threshold=0.25)
    print(f"   Initial adopters: {len(seed_nodes)}")
    print(f"   Final adopters: {len(history[-1])}")
    print(f"   Diffusion steps: {len(history)}")

    print("\n✓ Network analysis utilities ready")
