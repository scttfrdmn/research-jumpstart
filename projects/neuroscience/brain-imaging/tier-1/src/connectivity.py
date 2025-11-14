"""
Functional connectivity analysis utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import networkx as nx


# Define base directory
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def compute_correlation_matrix(timeseries, method='pearson'):
    """
    Compute functional connectivity matrix.

    Parameters
    ----------
    timeseries : ndarray, shape (n_timepoints, n_rois)
        ROI time series
    method : str
        Correlation method ('pearson' or 'spearman')

    Returns
    -------
    conn_matrix : ndarray, shape (n_rois, n_rois)
        Connectivity matrix
    """
    print(f"Computing {method} correlation matrix...")

    if method == 'pearson':
        conn_matrix = np.corrcoef(timeseries.T)
    elif method == 'spearman':
        conn_matrix = stats.spearmanr(timeseries)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"✓ Connectivity matrix: {conn_matrix.shape}")
    return conn_matrix


def fisher_z_transform(conn_matrix):
    """
    Apply Fisher Z-transformation to correlation matrix.

    Parameters
    ----------
    conn_matrix : ndarray
        Correlation matrix

    Returns
    -------
    z_matrix : ndarray
        Fisher Z-transformed matrix
    """
    # Clip to avoid invalid values
    conn_clipped = np.clip(conn_matrix, -0.9999, 0.9999)

    # Apply Fisher Z-transform
    z_matrix = np.arctanh(conn_clipped)

    return z_matrix


def threshold_matrix(conn_matrix, threshold=0.3, absolute=True):
    """
    Threshold connectivity matrix.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    threshold : float
        Threshold value
    absolute : bool
        If True, threshold absolute values

    Returns
    -------
    thresholded : ndarray
        Thresholded matrix
    """
    thresholded = conn_matrix.copy()

    if absolute:
        mask = np.abs(thresholded) < threshold
    else:
        mask = thresholded < threshold

    thresholded[mask] = 0

    n_edges = np.sum(thresholded != 0) / 2  # Divide by 2 for symmetric matrix
    print(f"✓ Thresholded at {threshold}: {int(n_edges)} edges remaining")

    return thresholded


def binarize_matrix(conn_matrix, threshold=0.3):
    """
    Binarize connectivity matrix.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    threshold : float
        Threshold value

    Returns
    -------
    binary : ndarray
        Binary adjacency matrix
    """
    binary = (np.abs(conn_matrix) > threshold).astype(int)
    np.fill_diagonal(binary, 0)  # Remove self-connections

    return binary


def compute_graph_metrics(conn_matrix, threshold=0.3):
    """
    Compute graph theory metrics.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    threshold : float
        Threshold for binarization

    Returns
    -------
    metrics : dict
        Graph metrics
    """
    print("Computing graph metrics...")

    # Binarize for graph analysis
    adj_matrix = binarize_matrix(conn_matrix, threshold)

    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix)

    # Compute metrics
    metrics = {}

    # Basic properties
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Check if connected
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
        metrics['global_efficiency'] = nx.global_efficiency(G)
    else:
        # For disconnected graphs, use largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        metrics['avg_path_length'] = nx.average_shortest_path_length(G_connected)
        metrics['diameter'] = nx.diameter(G_connected)
        metrics['global_efficiency'] = nx.global_efficiency(G)

    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(G)
    metrics['transitivity'] = nx.transitivity(G)

    # Centrality (expensive for large graphs, compute average)
    degree_centrality = nx.degree_centrality(G)
    metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))

    # Small-worldness
    # Compare to random graph
    n_nodes = metrics['n_nodes']
    n_edges = metrics['n_edges']
    G_random = nx.gnm_random_graph(n_nodes, n_edges, seed=42)

    C_rand = nx.average_clustering(G_random)
    L_rand = nx.average_shortest_path_length(G_random) if nx.is_connected(G_random) else np.inf

    gamma = metrics['avg_clustering'] / C_rand if C_rand > 0 else 0
    lambda_val = metrics['avg_path_length'] / L_rand if L_rand > 0 and L_rand < np.inf else 0
    sigma = gamma / lambda_val if lambda_val > 0 else 0

    metrics['small_worldness'] = sigma

    print(f"✓ Graph metrics computed:")
    print(f"  Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"  Clustering: {metrics['avg_clustering']:.3f}")
    print(f"  Path length: {metrics['avg_path_length']:.3f}")
    print(f"  Small-worldness: {metrics['small_worldness']:.3f}")

    return metrics


def save_connectivity_matrix(conn_matrix, subject_id):
    """
    Save connectivity matrix to disk.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    subject_id : str
        Subject identifier
    """
    output_path = PROCESSED_DIR / 'connectivity' / f'{subject_id}_conn.npy'
    np.save(output_path, conn_matrix)
    print(f"✓ Saved connectivity matrix to {output_path}")


def load_connectivity_matrix(subject_id):
    """
    Load connectivity matrix from disk.

    Parameters
    ----------
    subject_id : str
        Subject identifier

    Returns
    -------
    conn_matrix : ndarray
        Connectivity matrix
    """
    input_path = PROCESSED_DIR / 'connectivity' / f'{subject_id}_conn.npy'

    if not input_path.exists():
        raise FileNotFoundError(f"Connectivity matrix not found: {input_path}")

    conn_matrix = np.load(input_path)
    print(f"✓ Loaded connectivity matrix from cache: {conn_matrix.shape}")
    return conn_matrix


def group_average_connectivity(subject_ids):
    """
    Compute group-average connectivity matrix.

    Parameters
    ----------
    subject_ids : list
        List of subject identifiers

    Returns
    -------
    avg_conn : ndarray
        Group-average connectivity matrix
    std_conn : ndarray
        Standard deviation across subjects
    """
    print(f"Computing group average over {len(subject_ids)} subjects...")

    # Load all matrices
    matrices = []
    for subject_id in subject_ids:
        try:
            conn = load_connectivity_matrix(subject_id)
            matrices.append(conn)
        except FileNotFoundError:
            print(f"Warning: Matrix not found for {subject_id}, skipping")

    if not matrices:
        raise ValueError("No connectivity matrices found")

    # Stack and compute statistics
    matrices = np.array(matrices)
    avg_conn = np.mean(matrices, axis=0)
    std_conn = np.std(matrices, axis=0)

    print(f"✓ Group average computed from {len(matrices)} subjects")
    return avg_conn, std_conn


def compute_network_segregation(conn_matrix, network_labels):
    """
    Compute within-network vs between-network connectivity.

    Parameters
    ----------
    conn_matrix : ndarray
        Connectivity matrix
    network_labels : list or ndarray
        Network assignment for each ROI

    Returns
    -------
    segregation : dict
        Within and between network connectivity
    """
    network_labels = np.array(network_labels)
    unique_networks = np.unique(network_labels)

    within_conn = []
    between_conn = []

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(conn_matrix), k=1)

    for i, j in zip(*triu_idx):
        if network_labels[i] == network_labels[j]:
            within_conn.append(conn_matrix[i, j])
        else:
            between_conn.append(conn_matrix[i, j])

    segregation = {
        'within_mean': np.mean(within_conn),
        'within_std': np.std(within_conn),
        'between_mean': np.mean(between_conn),
        'between_std': np.std(between_conn),
        'segregation_index': np.mean(within_conn) - np.mean(between_conn)
    }

    print(f"✓ Network segregation computed:")
    print(f"  Within-network: {segregation['within_mean']:.3f} ± {segregation['within_std']:.3f}")
    print(f"  Between-network: {segregation['between_mean']:.3f} ± {segregation['between_std']:.3f}")
    print(f"  Segregation index: {segregation['segregation_index']:.3f}")

    return segregation


if __name__ == '__main__':
    # Test connectivity functions
    print("Testing connectivity utilities...")

    # Generate synthetic time series
    n_timepoints = 200
    n_rois = 100
    timeseries = np.random.randn(n_timepoints, n_rois)

    # Compute connectivity
    conn_matrix = compute_correlation_matrix(timeseries)
    print(f"\nConnectivity matrix shape: {conn_matrix.shape}")

    # Compute graph metrics
    metrics = compute_graph_metrics(conn_matrix, threshold=0.3)
    print(f"\nGraph metrics: {list(metrics.keys())}")
