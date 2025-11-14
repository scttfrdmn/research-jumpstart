"""
Data Utilities

Common functions for loading, preprocessing, and managing
archaeological datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


def load_site_metadata(metadata_file: Path) -> pd.DataFrame:
    """
    Load site metadata from JSON or CSV file.

    Args:
        metadata_file: Path to metadata file

    Returns:
        DataFrame with site information
    """
    if metadata_file.suffix == '.json':
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif metadata_file.suffix == '.csv':
        return pd.read_csv(metadata_file)
    else:
        raise ValueError(f"Unsupported file format: {metadata_file.suffix}")


def align_coordinate_systems(source_coords: np.ndarray,
                            source_crs: str,
                            target_crs: str) -> np.ndarray:
    """
    Align coordinates from different reference systems.

    Args:
        source_coords: Array of coordinates (N x 2) [x, y]
        source_crs: Source coordinate reference system (e.g., 'EPSG:4326')
        target_crs: Target coordinate reference system

    Returns:
        Transformed coordinates
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    transformed = np.array(transformer.transform(source_coords[:, 0],
                                                 source_coords[:, 1])).T
    return transformed


def cache_processed_data(data: np.ndarray,
                        cache_file: Path,
                        metadata: Optional[Dict] = None) -> None:
    """
    Cache processed data to disk for faster loading.

    Args:
        data: Array to cache
        cache_file: Path to cache file
        metadata: Optional metadata dictionary to store
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.suffix == '.npy':
        np.save(cache_file, data)
        if metadata:
            meta_file = cache_file.with_suffix('.json')
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    elif cache_file.suffix == '.npz':
        if metadata:
            np.savez(cache_file, data=data, metadata=np.array([metadata]))
        else:
            np.savez(cache_file, data=data)


def load_cached_data(cache_file: Path) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load cached processed data.

    Args:
        cache_file: Path to cache file

    Returns:
        Tuple of (data array, metadata dict or None)
    """
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    if cache_file.suffix == '.npy':
        data = np.load(cache_file)
        meta_file = cache_file.with_suffix('.json')
        metadata = None
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        return data, metadata
    elif cache_file.suffix == '.npz':
        npz = np.load(cache_file, allow_pickle=True)
        data = npz['data']
        metadata = npz['metadata'].item() if 'metadata' in npz else None
        return data, metadata


def create_data_splits(n_samples: int,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create train/validation/test splits for data.

    Args:
        n_samples: Number of samples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' index arrays
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(random_seed)
    indices = np.random.permutation(n_samples)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    return {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:]
    }


def normalize_features(features: np.ndarray,
                      method: str = 'standardize',
                      params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Normalize feature arrays.

    Args:
        features: Feature array (N x D)
        method: Normalization method ('standardize' or 'minmax')
        params: Pre-computed normalization parameters (for test data)

    Returns:
        Tuple of (normalized features, normalization parameters)
    """
    if params is None:
        if method == 'standardize':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            normalized = (features - mean) / std
            params = {'mean': mean, 'std': std, 'method': 'standardize'}
        elif method == 'minmax':
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            normalized = (features - min_val) / range_val
            params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        if params['method'] == 'standardize':
            normalized = (features - params['mean']) / params['std']
        elif params['method'] == 'minmax':
            normalized = (features - params['min']) / (params['max'] - params['min'])
        else:
            raise ValueError(f"Unknown normalization method: {params['method']}")

    return normalized, params


def merge_multi_modal_data(artifact_features: np.ndarray,
                           lidar_features: np.ndarray,
                           geophysical_features: np.ndarray,
                           weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Merge features from different data modalities.

    Args:
        artifact_features: Features from artifact analysis (N x D1)
        lidar_features: Features from LiDAR analysis (N x D2)
        geophysical_features: Features from geophysical surveys (N x D3)
        weights: Optional weights for each modality

    Returns:
        Merged feature array (N x (D1 + D2 + D3))
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]

    # Normalize each modality
    artifact_norm, _ = normalize_features(artifact_features)
    lidar_norm, _ = normalize_features(lidar_features)
    geophysical_norm, _ = normalize_features(geophysical_features)

    # Weight and concatenate
    merged = np.concatenate([
        artifact_norm * weights[0],
        lidar_norm * weights[1],
        geophysical_norm * weights[2]
    ], axis=1)

    return merged


def generate_site_report(site_name: str,
                        artifact_stats: Dict,
                        lidar_stats: Dict,
                        geophysical_stats: Dict,
                        output_file: Path) -> None:
    """
    Generate comprehensive site analysis report.

    Args:
        site_name: Name of archaeological site
        artifact_stats: Dictionary of artifact analysis statistics
        lidar_stats: Dictionary of LiDAR analysis statistics
        geophysical_stats: Dictionary of geophysical analysis statistics
        output_file: Path to output report file
    """
    report = {
        'site_name': site_name,
        'artifact_analysis': artifact_stats,
        'terrain_analysis': lidar_stats,
        'geophysical_analysis': geophysical_stats,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Site report saved to: {output_file}")
