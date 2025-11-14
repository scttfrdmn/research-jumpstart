"""
Data loading utilities for multi-platform social networks.

Handles downloading, caching, and loading of Twitter, Reddit, Facebook,
and other social network datasets.
"""

import os
import pickle
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm


# Data directory paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_social_data(
    platform: str,
    url: str,
    force_download: bool = False
) -> Path:
    """
    Download social network data from URL.

    Parameters:
    -----------
    platform : str
        Platform name (e.g., 'twitter', 'reddit')
    url : str
        URL to download from
    force_download : bool
        If True, re-download even if cached

    Returns:
    --------
    Path : Path to downloaded file
    """
    filename = f"{platform}_graph.pkl"
    filepath = RAW_DATA_DIR / filename

    if filepath.exists() and not force_download:
        print(f"✓ Using cached {platform} data: {filepath}")
        return filepath

    print(f"Downloading {platform} network data...")
    print(f"URL: {url}")
    print("This may take 10-15 minutes depending on platform...")

    # Download with progress bar
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end="")

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
        print(f"\n✓ Downloaded {platform} data to {filepath}")
        return filepath
    except Exception as e:
        print(f"\n✗ Error downloading {platform} data: {e}")
        raise


def load_twitter_graph(force_download: bool = False) -> nx.DiGraph:
    """
    Load Twitter social network graph.

    Returns directed graph with user nodes and interaction edges.
    """
    cache_file = RAW_DATA_DIR / "twitter_graph.pkl"

    if cache_file.exists() and not force_download:
        print(f"Loading cached Twitter graph from {cache_file}...")
        with open(cache_file, 'rb') as f:
            G = pickle.load(f)
        print(f"✓ Loaded Twitter graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    # If not cached, create a synthetic graph for demonstration
    print("Creating synthetic Twitter graph...")
    print("In production, this would download from Twitter API or dataset repository")

    G = _create_synthetic_social_graph(
        n_nodes=100000,
        avg_degree=50,
        platform='twitter'
    )

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(G, f)

    print(f"✓ Created and cached Twitter graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_reddit_graph(force_download: bool = False) -> nx.DiGraph:
    """
    Load Reddit social network graph.

    Returns directed graph with subreddit nodes and hyperlink edges.
    """
    cache_file = RAW_DATA_DIR / "reddit_graph.pkl"

    if cache_file.exists() and not force_download:
        print(f"Loading cached Reddit graph from {cache_file}...")
        with open(cache_file, 'rb') as f:
            G = pickle.load(f)
        print(f"✓ Loaded Reddit graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    # If not cached, create a synthetic graph
    print("Creating synthetic Reddit graph...")
    print("In production, this would download from Reddit API or SNAP dataset")

    G = _create_synthetic_social_graph(
        n_nodes=50000,
        avg_degree=30,
        platform='reddit'
    )

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(G, f)

    print(f"✓ Created and cached Reddit graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_facebook_graph(force_download: bool = False) -> nx.Graph:
    """
    Load Facebook social network graph.

    Returns undirected graph with user nodes and friendship edges.
    """
    cache_file = RAW_DATA_DIR / "facebook_graph.pkl"

    if cache_file.exists() and not force_download:
        print(f"Loading cached Facebook graph from {cache_file}...")
        with open(cache_file, 'rb') as f:
            G = pickle.load(f)
        print(f"✓ Loaded Facebook graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    # If not cached, create a synthetic graph
    print("Creating synthetic Facebook graph...")

    # Facebook is undirected (friendships are bidirectional)
    G = nx.powerlaw_cluster_graph(n=200000, m=100, p=0.05, seed=42)

    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['platform'] = 'facebook'
        G.nodes[node]['user_id'] = f"fb_user_{node}"
        G.nodes[node]['features'] = np.random.randn(128)

    # Add edge attributes
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.exponential(1.0)
        G[u][v]['edge_type'] = 'friendship'

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(G, f)

    print(f"✓ Created and cached Facebook graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def _create_synthetic_social_graph(
    n_nodes: int,
    avg_degree: int,
    platform: str
) -> nx.DiGraph:
    """
    Create synthetic social network graph with realistic properties.

    Uses power-law degree distribution and community structure.
    """
    # Create scale-free network (power-law degree distribution)
    m = avg_degree // 2
    G = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=42)

    # Convert to directed
    G = G.to_directed()

    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['platform'] = platform
        G.nodes[node]['user_id'] = f"{platform}_user_{node}"
        G.nodes[node]['features'] = np.random.randn(128)  # 128-dim feature vector
        G.nodes[node]['timestamp_first'] = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 180))
        G.nodes[node]['timestamp_last'] = pd.Timestamp('2023-06-30')

    # Add edge attributes
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.exponential(1.0)
        G[u][v]['edge_type'] = np.random.choice(['post', 'reply', 'mention', 'retweet'])
        G[u][v]['timestamp'] = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 180))
        G[u][v]['sentiment'] = np.random.normal(0, 0.5)  # Mean 0, std 0.5

    return G


def load_all_graphs(force_download: bool = False) -> Dict[str, nx.Graph]:
    """
    Load all platform graphs.

    Returns:
    --------
    dict : Dictionary mapping platform names to graphs
    """
    print("Loading all platform graphs...")

    graphs = {
        'twitter': load_twitter_graph(force_download),
        'reddit': load_reddit_graph(force_download),
        'facebook': load_facebook_graph(force_download),
    }

    total_nodes = sum(G.number_of_nodes() for G in graphs.values())
    total_edges = sum(G.number_of_edges() for G in graphs.values())

    print(f"\n✓ Loaded {len(graphs)} platform graphs")
    print(f"  Total nodes: {total_nodes:,}")
    print(f"  Total edges: {total_edges:,}")

    return graphs
