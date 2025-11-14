"""
Preprocessing utilities for satellite imagery.

Functions for cloud masking, atmospheric correction, co-registration,
and temporal gap-filling.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional


def cloud_mask(
    image: np.ndarray,
    threshold: float = 0.3,
    band_idx: int = 0
) -> np.ndarray:
    """
    Generate cloud mask from satellite imagery.

    Args:
        image: Satellite image (height, width, bands)
        threshold: Brightness threshold for cloud detection
        band_idx: Band index to use for cloud detection (default: 0 = Blue)

    Returns:
        Boolean mask (True = valid, False = cloud)
    """
    # Simple threshold-based cloud masking
    # In production, use more sophisticated methods (e.g., Sen2Cor, Fmask)
    brightness = image[:, :, band_idx]
    mask = brightness < threshold

    # Morphological operations to clean mask
    from scipy.ndimage import binary_erosion, binary_dilation
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)

    return mask


def atmospheric_correction(
    image: np.ndarray,
    method: str = 'dos'
) -> np.ndarray:
    """
    Apply atmospheric correction to satellite imagery.

    Args:
        image: Satellite image (height, width, bands)
        method: Correction method ('dos' = Dark Object Subtraction)

    Returns:
        Corrected image
    """
    if method == 'dos':
        # Dark Object Subtraction (simple atmospheric correction)
        corrected = image.copy()
        for band in range(image.shape[2]):
            dark_pixel = np.percentile(image[:, :, band], 1)
            corrected[:, :, band] = np.clip(
                (image[:, :, band] - dark_pixel) / (1 - dark_pixel),
                0, 1
            )
        return corrected
    else:
        raise ValueError(f"Unknown method: {method}")


def coregister_sensors(
    reference: np.ndarray,
    target: np.ndarray,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Co-register two satellite images to common grid.

    Args:
        reference: Reference image (height, width, bands)
        target: Target image to be resampled
        method: Resampling method ('nearest', 'bilinear')

    Returns:
        Resampled target image matching reference grid
    """
    from scipy.ndimage import zoom

    # Calculate zoom factors for each dimension
    zoom_factors = [
        reference.shape[0] / target.shape[0],
        reference.shape[1] / target.shape[1],
        1  # Don't zoom bands dimension
    ]

    # Resample
    order = 0 if method == 'nearest' else 1
    resampled = zoom(target, zoom_factors, order=order)

    return resampled


def gap_fill_timeseries(
    data: np.ndarray,
    dates: np.ndarray,
    max_gap: int = 3
) -> np.ndarray:
    """
    Fill gaps in time series using interpolation.

    Args:
        data: Time series data (n_times, height, width, bands)
        dates: Array of dates
        max_gap: Maximum gap size to fill (in time steps)

    Returns:
        Gap-filled data
    """
    filled = data.copy()

    # Find missing data (NaN or zeros)
    missing = np.isnan(filled) | (filled == 0)

    # Linear interpolation for small gaps
    for i in range(1, len(filled) - 1):
        if missing[i].any():
            # Check gap size
            gap_start = i
            gap_end = i
            while gap_end < len(filled) and missing[gap_end].any():
                gap_end += 1

            gap_size = gap_end - gap_start
            if gap_size <= max_gap:
                # Linear interpolation
                for j in range(gap_start, gap_end):
                    weight = (j - gap_start + 1) / (gap_end - gap_start + 1)
                    filled[j] = (1 - weight) * filled[gap_start - 1] + weight * filled[gap_end]

    return filled


def calculate_cloud_free_composite(
    images: np.ndarray,
    cloud_masks: np.ndarray,
    method: str = 'median'
) -> np.ndarray:
    """
    Create cloud-free composite from time series.

    Args:
        images: Time series images (n_times, height, width, bands)
        cloud_masks: Boolean masks (n_times, height, width)
        method: Compositing method ('median', 'mean', 'max')

    Returns:
        Cloud-free composite image
    """
    # Mask clouds
    masked = images.copy()
    for i in range(len(images)):
        masked[i][~cloud_masks[i]] = np.nan

    # Composite
    if method == 'median':
        composite = np.nanmedian(masked, axis=0)
    elif method == 'mean':
        composite = np.nanmean(masked, axis=0)
    elif method == 'max':
        composite = np.nanmax(masked, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return composite


def normalize_bands(
    image: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize band values.

    Args:
        image: Satellite image (height, width, bands)
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Min-max scaling to [0, 1]
        normalized = np.zeros_like(image)
        for band in range(image.shape[2]):
            band_data = image[:, :, band]
            min_val = np.percentile(band_data, 2)
            max_val = np.percentile(band_data, 98)
            normalized[:, :, band] = np.clip(
                (band_data - min_val) / (max_val - min_val + 1e-8),
                0, 1
            )
        return normalized

    elif method == 'zscore':
        # Z-score normalization
        normalized = np.zeros_like(image)
        for band in range(image.shape[2]):
            band_data = image[:, :, band]
            mean = np.mean(band_data)
            std = np.std(band_data)
            normalized[:, :, band] = (band_data - mean) / (std + 1e-8)
        return normalized

    else:
        raise ValueError(f"Unknown method: {method}")


def spatial_smoothing(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply spatial smoothing to reduce noise.

    Args:
        image: Satellite image (height, width, bands)
        sigma: Gaussian kernel standard deviation

    Returns:
        Smoothed image
    """
    smoothed = np.zeros_like(image)
    for band in range(image.shape[2]):
        smoothed[:, :, band] = gaussian_filter(image[:, :, band], sigma=sigma)
    return smoothed
