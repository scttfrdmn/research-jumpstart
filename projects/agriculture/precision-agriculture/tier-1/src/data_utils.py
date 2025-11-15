"""
Data loading and processing utilities for agricultural datasets.

This module provides functions to download, cache, and process multi-sensor
satellite data, weather data, and soil data. Data is cached in Studio Lab's
persistent storage to avoid re-downloading.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Data directory (persistent in Studio Lab)
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
for subdir in ["sentinel2", "landsat8", "modis", "weather", "soil"]:
    (RAW_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

for subdir in ["features", "training", "predictions"]:
    (PROCESSED_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    """
    Download file from URL to destination, with caching.

    Args:
        url: URL to download from
        destination: Local path to save to
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded file
    """
    if destination.exists() and not force:
        print(f"Using cached file: {destination.name}")
        return destination

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        destination.write_bytes(response.content)
        print(f"Saved to {destination}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise

    return destination


def load_sentinel2(
    aoi: str, date_range: str, bands: Optional[list[str]] = None, force_download: bool = False
) -> np.ndarray:
    """
    Load Sentinel-2 imagery for area of interest.

    Args:
        aoi: Area of interest identifier (e.g., 'field_001')
        date_range: Date range in format 'YYYY-MM-DD/YYYY-MM-DD'
        bands: List of bands to load (default: ['B4', 'B8', 'B11'])
        force_download: If True, re-download even if cached

    Returns:
        Array of shape (n_dates, height, width, n_bands)
    """
    if bands is None:
        bands = ["B4", "B8", "B11"]  # Red, NIR, SWIR

    cache_key = f"s2_{aoi}_{date_range.replace('/', '_')}"
    cache_file = PROCESSED_DATA_DIR / "features" / f"{cache_key}.npy"

    if cache_file.exists() and not force_download:
        print(f"Loading Sentinel-2 from cache: {aoi}")
        return np.load(cache_file)

    print(f"Downloading Sentinel-2 imagery for {aoi}...")
    print(f"  Date range: {date_range}")
    print(f"  Bands: {bands}")
    print("  This may take 15-20 minutes...")

    # In production, download from AWS S3: s3://sentinel-s2-l2a/
    # For demo, create synthetic data
    print("  (Creating synthetic data for demo)")

    # Parse date range
    start_date, end_date = date_range.split("/")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate dates (5-day revisit)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=5)

    # Generate synthetic imagery
    n_dates = len(dates)
    height, width = 256, 256
    n_bands = len(bands)

    data = np.random.rand(n_dates, height, width, n_bands).astype(np.float32)

    # Add realistic temporal patterns (growing season)
    for i, date in enumerate(dates):
        # Simulate vegetation growth curve
        days_since_start = (date - start).days
        growth_phase = days_since_start / (end - start).days

        # NDVI increases then decreases (growing season)
        vegetation_signal = np.sin(growth_phase * np.pi)

        # NIR increases, Red decreases for healthy vegetation
        if "B4" in bands:  # Red
            red_idx = bands.index("B4")
            data[i, :, :, red_idx] *= 0.5 - 0.3 * vegetation_signal
        if "B8" in bands:  # NIR
            nir_idx = bands.index("B8")
            data[i, :, :, nir_idx] *= 0.3 + 0.4 * vegetation_signal

    # Cache
    np.save(cache_file, data)
    print(f"Cached Sentinel-2 data ({data.nbytes / 1e9:.2f} GB)")

    return data


def load_landsat8(aoi: str, date_range: str, force_download: bool = False) -> np.ndarray:
    """
    Load Landsat-8 imagery for validation.

    Args:
        aoi: Area of interest identifier
        date_range: Date range in format 'YYYY-MM-DD/YYYY-MM-DD'
        force_download: If True, re-download even if cached

    Returns:
        Array of shape (n_dates, height, width, n_bands)
    """
    cache_key = f"l8_{aoi}_{date_range.replace('/', '_')}"
    cache_file = PROCESSED_DATA_DIR / "features" / f"{cache_key}.npy"

    if cache_file.exists() and not force_download:
        print(f"Loading Landsat-8 from cache: {aoi}")
        return np.load(cache_file)

    print(f"Downloading Landsat-8 imagery for {aoi}...")
    print("  (Creating synthetic data for demo)")

    # Generate synthetic data (30m resolution, 16-day revisit)
    n_dates = 12  # ~6 months
    data = np.random.rand(n_dates, 128, 128, 4).astype(np.float32)

    np.save(cache_file, data)
    print(f"Cached Landsat-8 data ({data.nbytes / 1e6:.1f} MB)")

    return data


def load_modis(aoi: str, date_range: str, force_download: bool = False) -> pd.DataFrame:
    """
    Load MODIS vegetation indices.

    Args:
        aoi: Area of interest identifier
        date_range: Date range in format 'YYYY-MM-DD/YYYY-MM-DD'
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: date, ndvi, evi, quality
    """
    cache_file = PROCESSED_DATA_DIR / "features" / f"modis_{aoi}.csv"

    if cache_file.exists() and not force_download:
        print(f"Loading MODIS from cache: {aoi}")
        return pd.read_csv(cache_file, parse_dates=["date"])

    print(f"Downloading MODIS data for {aoi}...")

    # Parse dates
    start_date, end_date = date_range.split("/")
    dates = pd.date_range(start=start_date, end=end_date, freq="16D")

    # Generate synthetic MODIS data
    df = pd.DataFrame(
        {
            "date": dates,
            "ndvi": np.random.uniform(0.3, 0.8, len(dates)),
            "evi": np.random.uniform(0.2, 0.6, len(dates)),
            "quality": np.random.choice(["good", "marginal"], len(dates)),
        }
    )

    df.to_csv(cache_file, index=False)
    print(f"Cached MODIS data ({len(df)} records)")

    return df


def load_weather_data(aoi: str, date_range: str, force_download: bool = False) -> pd.DataFrame:
    """
    Load weather data (temperature, precipitation, GDD).

    Args:
        aoi: Area of interest identifier
        date_range: Date range in format 'YYYY-MM-DD/YYYY-MM-DD'
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: date, temp_min, temp_max, precip, gdd
    """
    cache_file = PROCESSED_DATA_DIR / "features" / f"weather_{aoi}.csv"

    if cache_file.exists() and not force_download:
        print(f"Loading weather data from cache: {aoi}")
        return pd.read_csv(cache_file, parse_dates=["date"])

    print(f"Downloading weather data for {aoi}...")

    # Parse dates
    start_date, end_date = date_range.split("/")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate synthetic weather data
    temp_base = 20  # Celsius
    df = pd.DataFrame(
        {
            "date": dates,
            "temp_min": temp_base + np.random.normal(0, 3, len(dates)),
            "temp_max": temp_base + 10 + np.random.normal(0, 3, len(dates)),
            "precip": np.random.exponential(2, len(dates)),
            "gdd": np.maximum(0, (temp_base + 5 - 10) * np.random.uniform(0.8, 1.2, len(dates))),
        }
    )

    df.to_csv(cache_file, index=False)
    print(f"Cached weather data ({len(df)} records)")

    return df


def load_soil_data(aoi: str, force_download: bool = False) -> dict:
    """
    Load soil properties for area of interest.

    Args:
        aoi: Area of interest identifier
        force_download: If True, re-download even if cached

    Returns:
        Dictionary with soil properties
    """
    cache_file = RAW_DATA_DIR / "soil" / f"soil_{aoi}.json"

    if cache_file.exists() and not force_download:
        print(f"Loading soil data from cache: {aoi}")
        import json

        with open(cache_file) as f:
            return json.load(f)

    print(f"Loading soil data for {aoi}...")

    # Generate synthetic soil data
    import json

    soil_data = {
        "texture": "loam",
        "organic_matter": 2.5,  # %
        "ph": 6.5,
        "awc": 0.15,  # Available water capacity (cm/cm)
        "bulk_density": 1.3,  # g/cm3
    }

    with open(cache_file, "w") as f:
        json.dump(soil_data, f)

    print("Cached soil data")
    return soil_data


def merge_all_data(aoi: str, date_range: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load and merge all data sources for ML pipeline.

    Args:
        aoi: Area of interest identifier
        date_range: Date range in format 'YYYY-MM-DD/YYYY-MM-DD'

    Returns:
        Tuple of (satellite_data, tabular_data)
    """
    print(f"Loading all data sources for {aoi}...")

    # Load satellite data
    s2 = load_sentinel2(aoi, date_range)

    # Load auxiliary data
    modis = load_modis(aoi, date_range)
    weather = load_weather_data(aoi, date_range)
    soil = load_soil_data(aoi)

    # Merge tabular data
    tabular = weather.merge(modis, on="date", how="left")

    # Add soil properties (constant across time)
    for key, value in soil.items():
        tabular[f"soil_{key}"] = value

    print(f"Data merged: {s2.shape} satellite, {len(tabular)} tabular records")

    return s2, tabular
