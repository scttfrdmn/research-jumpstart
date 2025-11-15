"""
Graph construction utilities for crystal structures.

Functions for converting crystal structures to graph representations.
"""


import numpy as np
import torch
from torch_geometric.data import Data


def create_crystal_graph(
    material_data: dict, radius_cutoff: float = 5.0, max_neighbors: int = 12
) -> Data:
    """
    Create a graph representation of a crystal structure.

    Args:
        material_data: Dictionary with material properties
        radius_cutoff: Maximum distance for edges (Angstroms)
        max_neighbors: Maximum number of neighbors per atom

    Returns:
        PyTorch Geometric Data object
    """
    n_atoms = material_data.get("n_atoms", np.random.randint(4, 20))

    # Node features (atom properties)
    # In production, these would be real atomic properties
    node_features = generate_node_features(n_atoms)

    # Edge list (connectivity)
    edge_index, edge_features = generate_edges(
        n_atoms, radius_cutoff=radius_cutoff, max_neighbors=max_neighbors
    )

    # Convert to PyTorch tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_features)

    # Target properties
    y_band_gap = torch.FloatTensor([material_data.get("band_gap", 0.0)])
    y_formation = torch.FloatTensor([material_data.get("formation_energy", 0.0)])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_band_gap=y_band_gap,
        y_formation=y_formation,
        material_id=material_data.get("material_id", "unknown"),
    )


def generate_node_features(n_atoms: int, feature_dim: int = 16) -> np.ndarray:
    """
    Generate node features for atoms.

    In production, this would extract real atomic properties:
    - Atomic number
    - Electronegativity
    - Atomic radius
    - Valence electrons
    - etc.

    Args:
        n_atoms: Number of atoms
        feature_dim: Dimension of feature vectors

    Returns:
        Array of node features (n_atoms, feature_dim)
    """
    # Simulate diverse atomic features
    features = np.random.randn(n_atoms, feature_dim)

    # Add some structure (e.g., atomic number encoded)
    atomic_numbers = np.random.randint(1, 100, n_atoms)
    features[:, 0] = atomic_numbers / 100.0  # Normalized atomic number

    return features


def generate_edges(
    n_atoms: int, radius_cutoff: float = 5.0, max_neighbors: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate edge list and edge features for crystal graph.

    In production, this would use real crystal structure and
    calculate actual neighbor lists based on atomic positions.

    Args:
        n_atoms: Number of atoms
        radius_cutoff: Maximum distance for edges
        max_neighbors: Maximum neighbors per atom

    Returns:
        Tuple of (edge_index, edge_features)
    """
    edges = []
    edge_features_list = []

    # Create a reasonable crystal-like connectivity
    # Average coordination number in crystals is ~6-12
    avg_neighbors = min(max_neighbors, int(n_atoms * 0.5))

    for i in range(n_atoms):
        # Randomly select neighbors
        n_neighbors = np.random.randint(
            max(2, avg_neighbors - 2), min(n_atoms - 1, avg_neighbors + 2)
        )
        possible_neighbors = [j for j in range(n_atoms) if j != i]
        neighbors = np.random.choice(
            possible_neighbors, size=min(n_neighbors, len(possible_neighbors)), replace=False
        )

        for j in neighbors:
            # Add edge (undirected, so add both directions)
            edges.append([i, j])

            # Edge features (in production: distance, bond type, etc.)
            distance = np.random.uniform(1.5, radius_cutoff)  # Simulate bond length
            edge_feat = np.array(
                [
                    distance / radius_cutoff,  # Normalized distance
                    np.exp(-distance),  # Gaussian basis function
                    np.cos(distance),  # Angular feature
                    1.0 / distance,  # Coulomb-like
                    distance**2,  # Quadratic
                    np.sin(distance),  # Periodic
                    np.random.randn(),  # Random feature 1
                    np.random.randn(),  # Random feature 2
                ]
            )
            edge_features_list.append(edge_feat)

    if len(edges) == 0:  # Ensure at least one edge
        edges = [[0, 1], [1, 0]]
        edge_features_list = [np.random.randn(8) for _ in range(2)]

    edge_index = np.array(edges).T
    edge_features = np.array(edge_features_list)

    return edge_index, edge_features


def batch_create_graphs(
    materials_df, cache_dir: str = "data/processed/graphs", batch_size: int = 1000
) -> list[Data]:
    """
    Create graphs for a batch of materials and cache them.

    Args:
        materials_df: DataFrame with materials data
        cache_dir: Directory to cache graphs
        batch_size: Number of graphs to process at once

    Returns:
        List of PyTorch Geometric Data objects
    """
    from pathlib import Path

    from tqdm.auto import tqdm

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    graphs = []
    print(f"Creating graphs for {len(materials_df)} materials...")

    for _idx, row in tqdm(materials_df.iterrows(), total=len(materials_df)):
        material_dict = row.to_dict()
        graph = create_crystal_graph(material_dict)
        graphs.append(graph)

        # Save checkpoint every batch_size graphs
        if len(graphs) % batch_size == 0:
            checkpoint_file = cache_path / f"graphs_checkpoint_{len(graphs)}.pt"
            torch.save(graphs, checkpoint_file)

    # Save final graphs
    final_file = cache_path / "all_graphs.pt"
    torch.save(graphs, final_file)
    print(f"Saved {len(graphs)} graphs to {final_file}")

    return graphs


def load_cached_graphs(cache_file: str = "data/processed/graphs/all_graphs.pt") -> list[Data]:
    """
    Load cached graphs from disk.

    Args:
        cache_file: Path to cached graphs file

    Returns:
        List of PyTorch Geometric Data objects
    """
    from pathlib import Path

    cache_path = Path(cache_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    print(f"Loading cached graphs from {cache_file}...")
    graphs = torch.load(cache_file)
    print(f"Loaded {len(graphs)} graphs")

    return graphs
