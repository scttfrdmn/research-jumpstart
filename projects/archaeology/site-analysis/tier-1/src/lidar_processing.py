"""
LiDAR Data Processing Utilities

Functions for processing LiDAR terrain data, detecting archaeological
features, and analyzing settlement patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import rasterio
from scipy import ndimage
from skimage import feature, filters


def load_lidar_data(lidar_file: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load LiDAR elevation data from file.

    Args:
        lidar_file: Path to LiDAR data file (GeoTIFF or similar)

    Returns:
        Tuple of (elevation array, metadata dict)
    """
    with rasterio.open(lidar_file) as src:
        elevation = src.read(1)
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'resolution': src.res
        }

    return elevation, metadata


def calculate_terrain_metrics(elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
    """
    Calculate terrain analysis metrics from elevation data.

    Args:
        elevation: 2D array of elevation values
        resolution: Spatial resolution in meters

    Returns:
        Dictionary of terrain metric arrays (slope, aspect, curvature, etc.)
    """
    # Calculate gradients
    dy, dx = np.gradient(elevation, resolution)

    # Slope (in degrees)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect (in degrees)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = (aspect + 360) % 360  # Convert to 0-360 range

    # Curvature (second derivatives)
    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)

    # Profile curvature (curvature in direction of maximum slope)
    profile_curvature = -2 * (dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / \
                        ((dx**2 + dy**2) * (1 + dx**2 + dy**2)**1.5)

    # Plan curvature (curvature perpendicular to maximum slope)
    plan_curvature = 2 * (dyy * dx**2 - 2 * dxy * dx * dy + dxx * dy**2) / \
                     ((dx**2 + dy**2)**1.5)

    return {
        'slope': slope,
        'aspect': aspect,
        'profile_curvature': profile_curvature,
        'plan_curvature': plan_curvature,
        'elevation': elevation
    }


def detect_structures(elevation: np.ndarray,
                      min_height: float = 0.5,
                      min_size: int = 10) -> np.ndarray:
    """
    Detect potential archaeological structures from LiDAR data.

    Args:
        elevation: 2D elevation array
        min_height: Minimum height difference for structure detection (meters)
        min_size: Minimum size of detected features (pixels)

    Returns:
        Binary mask of detected structures
    """
    # Remove large-scale topography with high-pass filter
    detrended = elevation - ndimage.gaussian_filter(elevation, sigma=50)

    # Detect positive anomalies (potential structures)
    structures = detrended > min_height

    # Remove small noise
    structures = ndimage.binary_opening(structures, structure=np.ones((3, 3)))
    structures = ndimage.binary_closing(structures, structure=np.ones((3, 3)))

    # Remove small features
    labeled, num_features = ndimage.label(structures)
    sizes = ndimage.sum(structures, labeled, range(num_features + 1))
    mask_size = sizes >= min_size
    structures = mask_size[labeled]

    return structures


def extract_structure_features(elevation: np.ndarray,
                               structure_mask: np.ndarray,
                               resolution: float) -> pd.DataFrame:
    """
    Extract features of detected structures.

    Args:
        elevation: 2D elevation array
        structure_mask: Binary mask of detected structures
        resolution: Spatial resolution in meters

    Returns:
        DataFrame with structure features
    """
    labeled, num_structures = ndimage.label(structure_mask)
    features = []

    for i in range(1, num_structures + 1):
        structure_pixels = labeled == i
        structure_elevation = elevation[structure_pixels]

        # Calculate structure properties
        y_coords, x_coords = np.where(structure_pixels)
        centroid_y = np.mean(y_coords) * resolution
        centroid_x = np.mean(x_coords) * resolution

        features.append({
            'structure_id': i,
            'area_m2': np.sum(structure_pixels) * resolution**2,
            'max_height': np.max(structure_elevation),
            'mean_height': np.mean(structure_elevation),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'perimeter_m': np.sum(ndimage.binary_dilation(structure_pixels) &
                                 ~structure_pixels) * resolution
        })

    return pd.DataFrame(features)


def calculate_visibility_network(elevation: np.ndarray,
                                 structure_locations: List[Tuple[int, int]],
                                 resolution: float,
                                 observer_height: float = 1.6) -> np.ndarray:
    """
    Calculate inter-site visibility network.

    Args:
        elevation: 2D elevation array
        structure_locations: List of (y, x) coordinates of structures/sites
        resolution: Spatial resolution in meters
        observer_height: Height of observer above ground (meters)

    Returns:
        Visibility matrix (1 = visible, 0 = not visible)
    """
    n_sites = len(structure_locations)
    visibility_matrix = np.zeros((n_sites, n_sites), dtype=bool)

    for i, loc_a in enumerate(structure_locations):
        for j, loc_b in enumerate(structure_locations):
            if i == j:
                visibility_matrix[i, j] = True
                continue

            # Check line of sight
            visible = check_line_of_sight(elevation, loc_a, loc_b, observer_height)
            visibility_matrix[i, j] = visible

    return visibility_matrix.astype(int)


def check_line_of_sight(elevation: np.ndarray,
                        loc_a: Tuple[int, int],
                        loc_b: Tuple[int, int],
                        observer_height: float) -> bool:
    """
    Check if there is line of sight between two locations.

    Args:
        elevation: 2D elevation array
        loc_a: (y, x) coordinates of first location
        loc_b: (y, x) coordinates of second location
        observer_height: Height of observer above ground

    Returns:
        True if line of sight exists, False otherwise
    """
    # Sample points along line
    n_samples = int(np.hypot(loc_b[0] - loc_a[0], loc_b[1] - loc_a[1]))
    y_samples = np.linspace(loc_a[0], loc_b[0], n_samples).astype(int)
    x_samples = np.linspace(loc_a[1], loc_b[1], n_samples).astype(int)

    # Clip to array bounds
    y_samples = np.clip(y_samples, 0, elevation.shape[0] - 1)
    x_samples = np.clip(x_samples, 0, elevation.shape[1] - 1)

    # Get elevation profile
    elevation_profile = elevation[y_samples, x_samples]

    # Calculate required height for line of sight
    distances = np.linspace(0, 1, n_samples)
    elev_a = elevation[loc_a[0], loc_a[1]] + observer_height
    elev_b = elevation[loc_b[0], loc_b[1]] + observer_height
    required_height = elev_a + distances * (elev_b - elev_a)

    # Check if terrain blocks line of sight
    return np.all(elevation_profile <= required_height)


def analyze_settlement_pattern(structure_features: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze spatial patterns in settlement structure locations.

    Args:
        structure_features: DataFrame with structure features

    Returns:
        Dictionary of settlement pattern metrics
    """
    from scipy.spatial.distance import pdist

    # Extract coordinates
    coords = structure_features[['centroid_x', 'centroid_y']].values

    # Calculate nearest neighbor distances
    if len(coords) > 1:
        distances = pdist(coords)
        mean_distance = np.mean(distances)
        nearest_neighbor = np.min(pdist(coords, metric='euclidean'))
    else:
        mean_distance = 0
        nearest_neighbor = 0

    # Calculate density
    if len(structure_features) > 0:
        total_area = (structure_features['centroid_x'].max() -
                     structure_features['centroid_x'].min()) * \
                    (structure_features['centroid_y'].max() -
                     structure_features['centroid_y'].min())
        density = len(structure_features) / max(total_area, 1)
    else:
        density = 0

    return {
        'n_structures': len(structure_features),
        'mean_inter_structure_distance': float(mean_distance),
        'nearest_neighbor_distance': float(nearest_neighbor),
        'structure_density': float(density)
    }
