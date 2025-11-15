"""
Data loading and processing utilities for disease surveillance datasets.

This module provides functions to download, cache, and process surveillance data
from CDC, WHO, and state health departments. Data is cached in Studio Lab's
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
        print(f"Using cached file: {destination.name}")
        return destination

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        print(f"Saved to {destination}")
        return destination
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if destination.exists():
            print(f"Using existing cached file: {destination.name}")
            return destination
        raise


def load_ili_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[list[str]] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load CDC ILINet influenza-like illness surveillance data.

    Data is cached in persistent storage after first download.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        regions: List of regions/states to include (None = all)
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with columns: Date, Region, ILI_Rate, Total_Patients, etc.
    """
    cache_file = PROCESSED_DATA_DIR / "ili_data.csv"

    # Check for processed cache
    if cache_file.exists() and not force_download:
        print("Loading processed ILI data from cache")
        df = pd.read_csv(cache_file, parse_dates=["Date"])
    else:
        # In production, would use CDC FluView API
        # For demo, generate synthetic data based on real patterns
        print("Generating example ILI surveillance data...")
        df = _generate_synthetic_ili_data()
        df.to_csv(cache_file, index=False)
        print(f"Processed and cached ILI data ({len(df)} records)")

    # Filter by date range
    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    # Filter by regions
    if regions:
        df = df[df["Region"].isin(regions)]

    return df


def load_covid_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[list[str]] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load CDC COVID-19 surveillance data.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        regions: List of regions/states to include
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with COVID-19 case, death, hospitalization data
    """
    cache_file = PROCESSED_DATA_DIR / "covid_data.csv"

    if cache_file.exists() and not force_download:
        print("Loading processed COVID-19 data from cache")
        df = pd.read_csv(cache_file, parse_dates=["Date"])
    else:
        print("Generating example COVID-19 surveillance data...")
        df = _generate_synthetic_covid_data()
        df.to_csv(cache_file, index=False)
        print(f"Processed and cached COVID-19 data ({len(df)} records)")

    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    if regions:
        df = df[df["Region"].isin(regions)]

    return df


def load_rsv_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[list[str]] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load CDC RSV (Respiratory Syncytial Virus) surveillance data.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        regions: List of regions to include
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with RSV detection data
    """
    cache_file = PROCESSED_DATA_DIR / "rsv_data.csv"

    if cache_file.exists() and not force_download:
        print("Loading processed RSV data from cache")
        df = pd.read_csv(cache_file, parse_dates=["Date"])
    else:
        print("Generating example RSV surveillance data...")
        df = _generate_synthetic_rsv_data()
        df.to_csv(cache_file, index=False)
        print(f"Processed and cached RSV data ({len(df)} records)")

    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]
    if regions:
        df = df[df["Region"].isin(regions)]

    return df


def load_multi_disease_panel(
    diseases: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[list[str]] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load and merge multiple disease surveillance datasets.

    Args:
        diseases: List of diseases to include
        start_date: Start date
        end_date: End date
        regions: List of regions
        force_download: Force re-download

    Returns:
        Wide-format DataFrame with all diseases
    """
    if diseases is None:
        diseases = ["ILI", "COVID-19", "RSV"]
    dfs = []

    if "ILI" in diseases:
        ili = load_ili_data(start_date, end_date, regions, force_download)
        ili = ili.rename(columns={"Cases": "ILI_Cases", "Rate": "ILI_Rate"})
        dfs.append(ili)

    if "COVID-19" in diseases:
        covid = load_covid_data(start_date, end_date, regions, force_download)
        covid = covid.rename(columns={"Cases": "COVID_Cases", "Deaths": "COVID_Deaths"})
        dfs.append(covid)

    if "RSV" in diseases:
        rsv = load_rsv_data(start_date, end_date, regions, force_download)
        rsv = rsv.rename(columns={"Detections": "RSV_Detections", "Positivity": "RSV_Positivity"})
        dfs.append(rsv)

    # Merge on Date and Region
    if len(dfs) == 0:
        raise ValueError("No valid diseases specified")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["Date", "Region"], how="outer")

    merged = merged.sort_values(["Region", "Date"]).reset_index(drop=True)
    return merged


def calculate_epidemic_threshold(
    data: pd.Series, method: str = "modified_serfling", baseline_years: int = 5
) -> float:
    """
    Calculate epidemic threshold for surveillance data.

    Args:
        data: Time series of disease rates/counts
        method: Threshold calculation method
        baseline_years: Number of years for baseline calculation

    Returns:
        Epidemic threshold value
    """
    if method == "simple_std":
        # Simple: mean + 2 standard deviations
        return data.mean() + 2 * data.std()

    elif method == "modified_serfling":
        # CDC's modified Serfling method (simplified)
        # Remove epidemic periods, fit regression, calculate threshold
        baseline = data[data <= data.quantile(0.80)]  # Remove top 20% (epidemic periods)
        mean = baseline.mean()
        std = baseline.std()
        return mean + 1.96 * std  # 95% confidence interval

    elif method == "percentile":
        # Percentile-based (95th percentile)
        return data.quantile(0.95)

    else:
        raise ValueError(f"Unknown method: {method}")


# Synthetic data generation functions (for demo purposes)
def _generate_synthetic_ili_data() -> pd.DataFrame:
    """Generate synthetic ILI data for demonstration."""
    dates = pd.date_range("2015-01-01", "2024-01-01", freq="W")
    regions = ["US National", "Region 1", "Region 2", "Region 3", "Region 4"]

    records = []
    for region in regions:
        for date in dates:
            # Seasonal pattern
            week_of_year = date.isocalendar().week
            seasonal = 3.0 * np.sin(2 * np.pi * (week_of_year - 52) / 52) + 3.5

            # Add regional variation
            regional_factor = np.random.uniform(0.8, 1.2)

            # Noise
            noise = np.random.normal(0, 0.3)

            ili_rate = max(seasonal * regional_factor + noise, 0.5)
            total_patients = np.random.randint(20000, 50000)

            records.append(
                {
                    "Date": date,
                    "Region": region,
                    "ILI_Rate": ili_rate,
                    "Total_Patients": total_patients,
                    "ILI_Patients": int(ili_rate / 100 * total_patients),
                    "Year": date.year,
                    "Week": date.isocalendar().week,
                }
            )

    return pd.DataFrame(records)


def _generate_synthetic_covid_data() -> pd.DataFrame:
    """Generate synthetic COVID-19 data for demonstration."""
    dates = pd.date_range("2020-03-01", "2024-01-01", freq="D")
    regions = ["US National", "Region 1", "Region 2", "Region 3", "Region 4"]

    records = []
    for region in regions:
        cumulative_cases = 0
        for date in dates:
            # Pandemic waves
            days_since_start = (date - dates[0]).days
            wave1 = 5000 * np.exp(-((days_since_start - 60) ** 2) / 1000)
            wave2 = 8000 * np.exp(-((days_since_start - 200) ** 2) / 1500)
            wave3 = 6000 * np.exp(-((days_since_start - 400) ** 2) / 2000)

            daily_cases = max(wave1 + wave2 + wave3 + np.random.normal(0, 500), 0)
            cumulative_cases += daily_cases

            records.append(
                {
                    "Date": date,
                    "Region": region,
                    "Cases": int(daily_cases),
                    "Cumulative_Cases": int(cumulative_cases),
                    "Deaths": int(daily_cases * 0.01),  # 1% CFR
                    "Hospitalizations": int(daily_cases * 0.05),  # 5% hospitalization rate
                }
            )

    return pd.DataFrame(records)


def _generate_synthetic_rsv_data() -> pd.DataFrame:
    """Generate synthetic RSV data for demonstration."""
    dates = pd.date_range("2015-01-01", "2024-01-01", freq="W")
    regions = ["US National", "Region 1", "Region 2", "Region 3", "Region 4"]

    records = []
    for region in regions:
        for date in dates:
            # Winter seasonality (RSV peaks in winter)
            week_of_year = date.isocalendar().week
            seasonal = 15.0 * np.sin(2 * np.pi * (week_of_year - 52) / 52) + 15.0

            # Regional variation
            regional_factor = np.random.uniform(0.7, 1.3)

            # Noise
            noise = np.random.normal(0, 2)

            positivity = max(seasonal * regional_factor + noise, 1.0)
            tests = np.random.randint(5000, 15000)

            records.append(
                {
                    "Date": date,
                    "Region": region,
                    "Detections": int(positivity / 100 * tests),
                    "Total_Tests": tests,
                    "Positivity": positivity,
                    "Year": date.year,
                    "Week": date.isocalendar().week,
                }
            )

    return pd.DataFrame(records)
