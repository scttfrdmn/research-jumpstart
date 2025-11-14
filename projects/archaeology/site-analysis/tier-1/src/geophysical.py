"""
Geophysical Survey Analysis Utilities

Functions for processing ground-penetrating radar (GPR) and
magnetometry data to detect subsurface archaeological features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import signal, ndimage
from skimage import filters


def load_gpr_data(gpr_file: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load ground-penetrating radar data.

    Args:
        gpr_file: Path to GPR data file

    Returns:
        Tuple of (GPR array [depth x distance], metadata dict)
    """
    # Placeholder - actual implementation would depend on GPR data format
    # Common formats: SEG-Y, DZT (GSSI), DT1 (Sensors & Software)
    data = np.load(gpr_file) if gpr_file.suffix == '.npy' else None
    metadata = {
        'time_window': 100,  # nanoseconds
        'samples_per_trace': 512,
        'traces': 1000,
        'frequency': 400  # MHz
    }
    return data, metadata


def process_gpr_profile(gpr_data: np.ndarray,
                        background_removal: bool = True,
                        gain: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Process GPR profile with standard preprocessing steps.

    Args:
        gpr_data: 2D array of GPR data (depth x distance)
        background_removal: Remove horizontal banding
        gain: Time gain function to apply

    Returns:
        Processed GPR array
    """
    processed = gpr_data.copy()

    # Remove DC offset
    processed = processed - np.mean(processed, axis=1, keepdims=True)

    # Background removal (horizontal features)
    if background_removal:
        background = np.median(processed, axis=1, keepdims=True)
        processed = processed - background

    # Apply time gain
    if gain is None:
        # Default: exponential gain
        depths = np.arange(processed.shape[0])
        gain = np.exp(depths / (processed.shape[0] / 3))
        gain = gain / np.max(gain)

    processed = processed * gain[:, np.newaxis]

    # Bandpass filter
    sos = signal.butter(4, [0.1, 0.4], btype='bandpass', output='sos')
    processed = signal.sosfiltfilt(sos, processed, axis=0)

    return processed


def detect_gpr_anomalies(gpr_data: np.ndarray,
                         threshold: float = 2.0,
                         min_size: int = 5) -> np.ndarray:
    """
    Detect anomalies in GPR data that may indicate subsurface features.

    Args:
        gpr_data: Processed GPR array
        threshold: Number of standard deviations for anomaly detection
        min_size: Minimum size of detected features (pixels)

    Returns:
        Binary mask of detected anomalies
    """
    # Calculate local standard deviation
    local_std = ndimage.generic_filter(gpr_data, np.std, size=5)

    # Identify anomalies
    anomalies = np.abs(gpr_data) > (threshold * local_std)

    # Morphological cleaning
    anomalies = ndimage.binary_opening(anomalies, structure=np.ones((3, 3)))
    anomalies = ndimage.binary_closing(anomalies, structure=np.ones((3, 3)))

    # Remove small features
    labeled, num_features = ndimage.label(anomalies)
    sizes = ndimage.sum(anomalies, labeled, range(num_features + 1))
    mask_size = sizes >= min_size
    anomalies = mask_size[labeled]

    return anomalies


def load_magnetometry_data(mag_file: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load magnetometry survey data.

    Args:
        mag_file: Path to magnetometry data file

    Returns:
        Tuple of (magnetic field array, metadata dict)
    """
    # Placeholder - actual implementation would depend on data format
    data = np.load(mag_file) if mag_file.suffix == '.npy' else None
    metadata = {
        'units': 'nT',  # nanoTesla
        'grid_spacing': 0.5,  # meters
        'sensor_type': 'fluxgate'
    }
    return data, metadata


def process_magnetometry(mag_data: np.ndarray,
                        despike: bool = True,
                        detrend: bool = True) -> np.ndarray:
    """
    Process magnetometry data with standard preprocessing.

    Args:
        mag_data: 2D array of magnetic field measurements
        despike: Remove outliers/spikes
        detrend: Remove linear trends

    Returns:
        Processed magnetometry array
    """
    processed = mag_data.copy()

    # Despike - remove extreme outliers
    if despike:
        median = np.median(processed)
        mad = np.median(np.abs(processed - median))
        threshold = median + 5 * mad
        processed = np.clip(processed, median - threshold, median + threshold)

    # Detrend - remove linear background
    if detrend:
        y_indices, x_indices = np.mgrid[0:processed.shape[0], 0:processed.shape[1]]
        A = np.column_stack([
            np.ones(processed.size),
            x_indices.ravel(),
            y_indices.ravel()
        ])
        coeffs, _, _, _ = np.linalg.lstsq(A, processed.ravel(), rcond=None)
        trend = (coeffs[0] + coeffs[1] * x_indices + coeffs[2] * y_indices)
        processed = processed - trend

    return processed


def detect_magnetic_anomalies(mag_data: np.ndarray,
                              threshold: float = 2.0,
                              min_size: int = 4) -> Dict[str, np.ndarray]:
    """
    Detect positive and negative magnetic anomalies.

    Args:
        mag_data: Processed magnetometry array
        threshold: Number of standard deviations for anomaly detection
        min_size: Minimum size of detected features (pixels)

    Returns:
        Dictionary with 'positive' and 'negative' anomaly masks
    """
    # Calculate statistics
    mean_field = np.mean(mag_data)
    std_field = np.std(mag_data)

    # Detect anomalies
    positive_anomalies = mag_data > (mean_field + threshold * std_field)
    negative_anomalies = mag_data < (mean_field - threshold * std_field)

    # Clean up detections
    for anomalies in [positive_anomalies, negative_anomalies]:
        anomalies = ndimage.binary_opening(anomalies, structure=np.ones((3, 3)))
        labeled, num_features = ndimage.label(anomalies)
        sizes = ndimage.sum(anomalies, labeled, range(num_features + 1))
        mask_size = sizes >= min_size
        anomalies = mask_size[labeled]

    return {
        'positive': positive_anomalies,
        'negative': negative_anomalies
    }


def extract_anomaly_features(data: np.ndarray,
                             anomaly_mask: np.ndarray,
                             grid_spacing: float) -> pd.DataFrame:
    """
    Extract features of detected anomalies.

    Args:
        data: Original geophysical data array
        anomaly_mask: Binary mask of detected anomalies
        grid_spacing: Spatial resolution in meters

    Returns:
        DataFrame with anomaly features
    """
    labeled, num_anomalies = ndimage.label(anomaly_mask)
    features = []

    for i in range(1, num_anomalies + 1):
        anomaly_pixels = labeled == i
        anomaly_values = data[anomaly_pixels]

        # Calculate properties
        y_coords, x_coords = np.where(anomaly_pixels)
        centroid_y = np.mean(y_coords) * grid_spacing
        centroid_x = np.mean(x_coords) * grid_spacing

        features.append({
            'anomaly_id': i,
            'area_m2': np.sum(anomaly_pixels) * grid_spacing**2,
            'max_amplitude': np.max(np.abs(anomaly_values)),
            'mean_amplitude': np.mean(anomaly_values),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'aspect_ratio': (np.max(y_coords) - np.min(y_coords) + 1) /
                          (np.max(x_coords) - np.min(x_coords) + 1)
        })

    return pd.DataFrame(features)


def integrate_geophysical_data(gpr_anomalies: np.ndarray,
                               mag_anomalies: np.ndarray,
                               grid_spacing: float) -> pd.DataFrame:
    """
    Integrate GPR and magnetometry data to identify high-confidence features.

    Args:
        gpr_anomalies: Binary mask of GPR anomalies
        mag_anomalies: Binary mask of magnetic anomalies
        grid_spacing: Spatial resolution in meters

    Returns:
        DataFrame with integrated feature locations and confidence
    """
    # Find co-located anomalies (high confidence)
    coincident = gpr_anomalies & mag_anomalies

    # Label features
    labeled_coincident, n_coincident = ndimage.label(coincident)
    labeled_gpr_only, n_gpr = ndimage.label(gpr_anomalies & ~mag_anomalies)
    labeled_mag_only, n_mag = ndimage.label(mag_anomalies & ~gpr_anomalies)

    features = []

    # Coincident features (high confidence)
    for i in range(1, n_coincident + 1):
        mask = labeled_coincident == i
        y_coords, x_coords = np.where(mask)
        features.append({
            'feature_id': f'COIN_{i}',
            'centroid_x': np.mean(x_coords) * grid_spacing,
            'centroid_y': np.mean(y_coords) * grid_spacing,
            'area_m2': np.sum(mask) * grid_spacing**2,
            'confidence': 'high',
            'detection_methods': 'GPR+Magnetometry'
        })

    # GPR-only features (medium confidence)
    for i in range(1, n_gpr + 1):
        mask = labeled_gpr_only == i
        y_coords, x_coords = np.where(mask)
        features.append({
            'feature_id': f'GPR_{i}',
            'centroid_x': np.mean(x_coords) * grid_spacing,
            'centroid_y': np.mean(y_coords) * grid_spacing,
            'area_m2': np.sum(mask) * grid_spacing**2,
            'confidence': 'medium',
            'detection_methods': 'GPR only'
        })

    # Magnetometry-only features (medium confidence)
    for i in range(1, n_mag + 1):
        mask = labeled_mag_only == i
        y_coords, x_coords = np.where(mask)
        features.append({
            'feature_id': f'MAG_{i}',
            'centroid_x': np.mean(x_coords) * grid_spacing,
            'centroid_y': np.mean(y_coords) * grid_spacing,
            'area_m2': np.sum(mask) * grid_spacing**2,
            'confidence': 'medium',
            'detection_methods': 'Magnetometry only'
        })

    return pd.DataFrame(features)
