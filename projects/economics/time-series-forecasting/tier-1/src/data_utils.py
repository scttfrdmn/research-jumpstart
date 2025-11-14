"""
Data loading and processing utilities for economic datasets.

This module provides functions to download, cache, and process economic data
from FRED, World Bank, OECD, and other public sources. Data is cached in
Studio Lab's persistent storage to avoid re-downloading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Optional, List, Dict
import time
from datetime import datetime

try:
    import pandas_datareader as pdr
    from fredapi import Fred
    import wbdata
except ImportError:
    print("⚠️  Some data APIs not installed. Run: pip install -r requirements.txt")

# Data directory (persistent in Studio Lab)
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_fred_data(
    series_id: str,
    start_date: str = '1990-01-01',
    end_date: Optional[str] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load data from FRED (Federal Reserve Economic Data).

    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'UNRATE')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date (defaults to today)
        force_download: If True, re-download even if cached

    Returns:
        DataFrame with date index and series values
    """
    cache_file = RAW_DATA_DIR / f"fred_{series_id}.csv"

    if cache_file.exists() and not force_download:
        print(f"✓ Loading {series_id} from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"Downloading {series_id} from FRED...")

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        data = pdr.DataReader(series_id, 'fred', start_date, end_date)
        data.to_csv(cache_file)
        print(f"✓ Cached {series_id} ({len(data)} observations)")
        return data
    except Exception as e:
        print(f"✗ Error downloading {series_id}: {e}")
        return pd.DataFrame()


def load_multi_country_indicators(
    countries: List[str],
    indicators: Dict[str, str],
    start_date: str = '1990-01-01',
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load multiple economic indicators for multiple countries.

    Args:
        countries: List of country codes (e.g., ['US', 'GB', 'DE'])
        indicators: Dict mapping indicator names to FRED series patterns
        start_date: Start date
        force_download: Force re-download

    Returns:
        Multi-level DataFrame with (country, indicator) columns
    """
    cache_file = PROCESSED_DATA_DIR / "multi_country_panel.csv"

    if cache_file.exists() and not force_download:
        print("✓ Loading multi-country data from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"Downloading data for {len(countries)} countries...")

    all_data = {}

    for country in countries:
        print(f"\n  Processing {country}...")
        country_data = {}

        for indicator_name, series_pattern in indicators.items():
            # Construct series ID (pattern may include {country} placeholder)
            series_id = series_pattern.format(country=country)

            try:
                data = load_fred_data(series_id, start_date, force_download=force_download)
                if not data.empty:
                    country_data[indicator_name] = data.iloc[:, 0]
                    print(f"    ✓ {indicator_name}")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"    ✗ {indicator_name}: {e}")

        if country_data:
            all_data[country] = pd.DataFrame(country_data)

    # Combine into multi-level DataFrame
    panel = pd.concat(all_data, axis=1)
    panel.columns.names = ['Country', 'Indicator']

    # Save cache
    panel.to_csv(cache_file)
    print(f"\n✓ Cached panel data ({panel.shape[0]} periods, {panel.shape[1]} series)")

    return panel


def load_world_bank_data(
    countries: List[str],
    indicators: Dict[str, str],
    start_year: int = 1990,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load data from World Bank API.

    Args:
        countries: List of ISO country codes (e.g., ['USA', 'GBR', 'DEU'])
        indicators: Dict mapping names to World Bank indicator codes
        start_year: Start year
        force_download: Force re-download

    Returns:
        Panel DataFrame with (country, indicator) columns
    """
    cache_file = PROCESSED_DATA_DIR / "world_bank_panel.csv"

    if cache_file.exists() and not force_download:
        print("✓ Loading World Bank data from cache")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print(f"Downloading World Bank data for {len(countries)} countries...")

    try:
        data_date = (datetime(start_year, 1, 1), datetime.now())
        all_data = {}

        for ind_name, ind_code in indicators.items():
            print(f"  Downloading {ind_name}...")
            data = wbdata.get_dataframe({ind_code: ind_name}, country=countries, data_date=data_date)
            all_data[ind_name] = data
            time.sleep(0.2)  # Rate limiting

        # Combine and save
        panel = pd.concat(all_data, axis=1)
        panel.to_csv(cache_file)
        print(f"✓ Cached World Bank data ({panel.shape})")

        return panel
    except Exception as e:
        print(f"✗ Error loading World Bank data: {e}")
        return pd.DataFrame()


def preprocess_panel_data(
    panel: pd.DataFrame,
    freq: str = 'Q',
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Preprocess panel data: resample, fill missing values, align dates.

    Args:
        panel: Multi-level panel DataFrame
        freq: Target frequency ('M' for monthly, 'Q' for quarterly)
        fill_method: Method to fill missing values ('interpolate', 'ffill', 'bfill')

    Returns:
        Processed panel DataFrame
    """
    print(f"Preprocessing panel data (freq={freq}, fill={fill_method})...")

    # Resample to target frequency
    panel_resampled = panel.resample(freq).mean()

    # Fill missing values
    if fill_method == 'interpolate':
        panel_filled = panel_resampled.interpolate(method='linear', limit=4)
    elif fill_method == 'ffill':
        panel_filled = panel_resampled.ffill(limit=4)
    elif fill_method == 'bfill':
        panel_filled = panel_resampled.bfill(limit=4)
    else:
        panel_filled = panel_resampled

    # Drop columns with too many missing values (>20%)
    missing_pct = panel_filled.isnull().mean()
    keep_cols = missing_pct[missing_pct < 0.2].index
    panel_clean = panel_filled[keep_cols]

    dropped = len(panel_filled.columns) - len(panel_clean.columns)
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} series with >20% missing data")

    print(f"✓ Processed panel: {panel_clean.shape}")
    return panel_clean


def calculate_growth_rates(
    data: pd.DataFrame,
    periods: int = 4
) -> pd.DataFrame:
    """
    Calculate year-over-year growth rates.

    Args:
        data: DataFrame with levels
        periods: Number of periods for growth rate (4 for quarterly YoY, 12 for monthly YoY)

    Returns:
        DataFrame with growth rates (percentage)
    """
    growth = data.pct_change(periods) * 100
    return growth


def create_lags(
    data: pd.DataFrame,
    lags: int = 4
) -> pd.DataFrame:
    """
    Create lagged features for time series modeling.

    Args:
        data: DataFrame with time series
        lags: Number of lags to create

    Returns:
        DataFrame with original and lagged columns
    """
    lagged = data.copy()

    for col in data.columns:
        for lag in range(1, lags + 1):
            lagged[f"{col}_lag{lag}"] = data[col].shift(lag)

    return lagged


def align_panel_data(
    panels: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Align multiple panel DataFrames on common date index.

    Args:
        panels: List of panel DataFrames

    Returns:
        Combined DataFrame with aligned dates
    """
    # Find common date range
    start = max(p.index.min() for p in panels)
    end = min(p.index.max() for p in panels)

    # Align all panels
    aligned = [p.loc[start:end] for p in panels]

    # Concatenate
    combined = pd.concat(aligned, axis=1)

    print(f"✓ Aligned {len(panels)} panels: {combined.shape} ({start.year}-{end.year})")
    return combined
