"""
Data utilities for urban planning analysis.

Functions for downloading and loading satellite imagery, mobility data,
and demographic datasets.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Optional, Tuple, List
import requests
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_satellite_imagery(
    city: str,
    year: int,
    bands: Optional[List[str]] = None,
    force_download: bool = False
) -> np.ndarray:
    """
    Load Landsat satellite imagery for a specific city and year.

    Parameters:
    -----------
    city : str
        City name (e.g., 'austin', 'denver')
    year : int
        Year of imagery (2000-2024)
    bands : list of str, optional
        Spectral bands to load (default: ['red', 'green', 'blue', 'nir'])
    force_download : bool
        Force re-download even if cached

    Returns:
    --------
    np.ndarray
        Image array of shape (height, width, bands)
    """
    if bands is None:
        bands = ['red', 'green', 'blue', 'nir']

    cache_file = PROCESSED_DIR / f"{city}_{year}_imagery.npy"

    if cache_file.exists() and not force_download:
        print(f"Loading cached imagery for {city} ({year})...")
        return np.load(cache_file)

    print(f"Downloading imagery for {city} ({year})...")
    # TODO: Implement actual download from USGS or Google Earth Engine
    # Placeholder: return synthetic data
    imagery = np.random.rand(1000, 1000, len(bands))

    # Cache for future use
    np.save(cache_file, imagery)
    print(f"Cached imagery to {cache_file}")

    return imagery


def load_mobility_data(
    city: str,
    data_type: str = 'roads',
    force_download: bool = False
) -> gpd.GeoDataFrame:
    """
    Load mobility data (roads, transit, traffic) for a specific city.

    Parameters:
    -----------
    city : str
        City name (e.g., 'austin', 'denver')
    data_type : str
        Type of mobility data ('roads', 'transit', 'traffic')
    force_download : bool
        Force re-download even if cached

    Returns:
    --------
    geopandas.GeoDataFrame
        Mobility data with geometry
    """
    cache_file = PROCESSED_DIR / f"{city}_{data_type}.geojson"

    if cache_file.exists() and not force_download:
        print(f"Loading cached {data_type} data for {city}...")
        return gpd.read_file(cache_file)

    print(f"Downloading {data_type} data for {city}...")
    # TODO: Implement actual download from OpenStreetMap or transit APIs
    # Placeholder: return empty GeoDataFrame
    gdf = gpd.GeoDataFrame()

    # Cache for future use
    gdf.to_file(cache_file, driver='GeoJSON')
    print(f"Cached {data_type} data to {cache_file}")

    return gdf


def load_demographic_data(
    city: str,
    variables: Optional[List[str]] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load demographic data from US Census for a specific city.

    Parameters:
    -----------
    city : str
        City name (e.g., 'austin', 'denver')
    variables : list of str, optional
        Census variables to load (default: population, income, employment)
    force_download : bool
        Force re-download even if cached

    Returns:
    --------
    pandas.DataFrame
        Demographic data by census tract
    """
    if variables is None:
        variables = ['population', 'median_income', 'employment_rate']

    cache_file = PROCESSED_DIR / f"{city}_demographics.csv"

    if cache_file.exists() and not force_download:
        print(f"Loading cached demographic data for {city}...")
        return pd.read_csv(cache_file)

    print(f"Downloading demographic data for {city}...")
    # TODO: Implement actual download from Census Bureau API
    # Placeholder: return synthetic data
    df = pd.DataFrame({
        'tract_id': range(100),
        'population': np.random.randint(1000, 10000, 100),
        'median_income': np.random.randint(30000, 120000, 100),
        'employment_rate': np.random.uniform(0.6, 0.95, 100)
    })

    # Cache for future use
    df.to_csv(cache_file, index=False)
    print(f"Cached demographic data to {cache_file}")

    return df


def calculate_urban_indices(imagery: np.ndarray, bands: dict) -> dict:
    """
    Calculate urban indices from satellite imagery.

    Parameters:
    -----------
    imagery : np.ndarray
        Satellite imagery array
    bands : dict
        Dictionary mapping band names to array indices

    Returns:
    --------
    dict
        Dictionary of computed indices (NDVI, NDBI, etc.)
    """
    indices = {}

    if 'nir' in bands and 'red' in bands:
        # NDVI (Normalized Difference Vegetation Index)
        nir = imagery[:, :, bands['nir']]
        red = imagery[:, :, bands['red']]
        indices['ndvi'] = (nir - red) / (nir + red + 1e-8)

    if 'swir' in bands and 'nir' in bands:
        # NDBI (Normalized Difference Built-up Index)
        swir = imagery[:, :, bands['swir']]
        nir = imagery[:, :, bands['nir']]
        indices['ndbi'] = (swir - nir) / (swir + nir + 1e-8)

    return indices


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with progress bar.

    Parameters:
    -----------
    url : str
        URL to download from
    output_path : Path
        Local path to save file
    desc : str
        Description for progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
