"""
Data loading and processing utilities for climate datasets.

This module provides functions to download, cache, and process climate data
from public sources (NOAA, NASA, etc.). Data is cached in Studio Lab's
persistent storage to avoid re-downloading.
"""

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
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


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
        print(f"✓ Using cached file: {destination.name}")
        return destination

    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()

    destination.write_bytes(response.content)
    print(f"✓ Saved to {destination}")
    return destination


def load_temperature_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load NOAA GISTEMP global temperature anomaly data.

    Data is cached in persistent storage after first download.

    Args:
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: Year, Month, Anomaly
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    cache_file = RAW_DATA_DIR / "gistemp_global.csv"
    processed_file = PROCESSED_DATA_DIR / "temperature_monthly.csv"

    # Check for processed cache
    if processed_file.exists() and not force_download:
        print("✓ Loading processed temperature data from cache")
        return pd.read_csv(processed_file)

    # Download raw data
    download_file(url, cache_file, force=force_download)

    # Process data
    df = pd.read_csv(cache_file, skiprows=1)

    # Reshape to long format (Year, Month, Anomaly)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    records = []
    for _, row in df.iterrows():
        year = row["Year"]
        for month_idx, month in enumerate(months, 1):
            if month in row and row[month] != "***":
                records.append(
                    {"Year": int(year), "Month": month_idx, "Anomaly": float(row[month])}
                )

    result = pd.DataFrame(records)
    result["Date"] = pd.to_datetime(result[["Year", "Month"]].assign(day=1))

    # Cache processed data
    result.to_csv(processed_file, index=False)
    print(f"✓ Processed and cached temperature data ({len(result)} records)")

    return result


def load_co2_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load Mauna Loa atmospheric CO2 concentration data.

    Args:
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: Year, Month, CO2_ppm
    """
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    cache_file = RAW_DATA_DIR / "co2_mlo.txt"
    processed_file = PROCESSED_DATA_DIR / "co2_monthly.csv"

    if processed_file.exists() and not force_download:
        print("✓ Loading processed CO2 data from cache")
        return pd.read_csv(processed_file, parse_dates=["Date"])

    download_file(url, cache_file, force=force_download)

    # Parse fixed-width format (skip header lines)
    df = pd.read_csv(
        cache_file,
        delim_whitespace=True,
        comment="#",
        names=[
            "year",
            "month",
            "decimal_date",
            "average",
            "deseasonalized",
            "ndays",
            "stddev",
            "uncertainty",
        ],
    )

    # Filter valid data
    df = df[df["average"] > 0].copy()
    df["Date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    result = df[["year", "month", "average", "Date"]].copy()
    result.columns = ["Year", "Month", "CO2_ppm", "Date"]

    result.to_csv(processed_file, index=False)
    print(f"✓ Processed and cached CO2 data ({len(result)} records)")

    return result


def load_sea_level_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load NOAA global mean sea level data (satellite altimetry).

    Args:
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: Date, Sea_Level_mm
    """
    # Note: This is a simplified example. Real implementation would use actual NOAA data.
    RAW_DATA_DIR / "sea_level.csv"
    processed_file = PROCESSED_DATA_DIR / "sea_level_monthly.csv"

    if processed_file.exists() and not force_download:
        print("✓ Loading processed sea level data from cache")
        return pd.read_csv(processed_file, parse_dates=["Date"])

    # For demo purposes, create synthetic data based on real trends
    # In production, would download actual data
    print("⚠️  Using synthetic sea level data (real download URL may change)")

    dates = pd.date_range(start="1993-01-01", end="2024-01-01", freq="MS")
    # Real trend: ~3.3mm/year with seasonal cycle
    years = (dates - dates[0]).days / 365.25
    trend = 3.3 * years  # mm/year
    seasonal = 10 * np.sin(2 * np.pi * years)  # Seasonal variation
    noise = np.random.normal(0, 5, len(dates))

    result = pd.DataFrame({"Date": dates, "Sea_Level_mm": trend + seasonal + noise})

    result.to_csv(processed_file, index=False)
    print(f"✓ Processed and cached sea level data ({len(result)} records)")

    return result


def calculate_anomalies(
    data: pd.Series, baseline_start: int, baseline_end: int, date_column: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate anomalies relative to a baseline period.

    Args:
        data: Time series data
        baseline_start: Start year of baseline period
        baseline_end: End year of baseline period
        date_column: Optional date column for filtering baseline

    Returns:
        Series of anomalies
    """
    if date_column is not None:
        baseline_mask = (date_column.dt.year >= baseline_start) & (
            date_column.dt.year <= baseline_end
        )
        baseline_mean = data[baseline_mask].mean()
    else:
        # Assume data is indexed by year
        baseline_mean = data[baseline_start:baseline_end].mean()

    return data - baseline_mean


def merge_climate_datasets(
    temp_df: pd.DataFrame, co2_df: pd.DataFrame, sea_level_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge multiple climate datasets on common date column.

    Args:
        temp_df: Temperature anomaly DataFrame
        co2_df: CO2 concentration DataFrame
        sea_level_df: Optional sea level DataFrame

    Returns:
        Merged DataFrame with all variables
    """
    # Merge on Date column
    merged = temp_df[["Date", "Anomaly"]].merge(co2_df[["Date", "CO2_ppm"]], on="Date", how="inner")

    if sea_level_df is not None:
        merged = merged.merge(sea_level_df[["Date", "Sea_Level_mm"]], on="Date", how="left")

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged
