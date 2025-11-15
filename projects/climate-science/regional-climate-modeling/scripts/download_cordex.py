#!/usr/bin/env python3
"""
Download CORDEX Regional Climate Data

This script provides helpers for downloading CORDEX data from ESGF nodes
or Copernicus Climate Data Store (CDS).

Usage:
    python download_cordex.py --domain NAM --variable tas --scenario rcp85
    python download_cordex.py --list-domains
    python download_cordex.py --list-variables
"""

import argparse
import sys
from pathlib import Path

# CORDEX domain codes
DOMAINS = {
    "EUR": "Europe",
    "NAM": "North America",
    "SAM": "South America",
    "AFR": "Africa",
    "EAS": "East Asia",
    "SEA": "Southeast Asia",
    "WAS": "West Asia",
    "AUS": "Australasia",
    "ANT": "Antarctica",
    "ARC": "Arctic",
}

# Common variables
VARIABLES = {
    "tas": "Near-surface air temperature",
    "tasmax": "Daily maximum temperature",
    "tasmin": "Daily minimum temperature",
    "pr": "Precipitation",
    "sfcWind": "Near-surface wind speed",
    "hurs": "Near-surface relative humidity",
    "ps": "Surface air pressure",
    "rsds": "Surface downwelling shortwave radiation",
}

# RCP scenarios
SCENARIOS = ["rcp26", "rcp45", "rcp60", "rcp85", "historical"]


def list_domains():
    """Print available CORDEX domains."""
    print("\nAvailable CORDEX Domains:")
    print("-" * 50)
    for code, name in DOMAINS.items():
        print(f"  {code:6s} - {name}")
    print()


def list_variables():
    """Print available variables."""
    print("\nCommon CORDEX Variables:")
    print("-" * 50)
    for var, desc in VARIABLES.items():
        print(f"  {var:12s} - {desc}")
    print()


def get_esgf_download_urls(domain, variable, scenario, model="all"):
    """
    Generate ESGF download URLs for CORDEX data.

    Note: This is a template. Actual ESGF queries require:
    1. ESGF account and credentials
    2. Using the esgf-pyclient library
    3. Or web-based search at https://esgf-node.llnl.gov/search/esgf-llnl/

    Parameters:
    -----------
    domain : str
        CORDEX domain code (e.g., 'NAM')
    variable : str
        Variable name (e.g., 'tas')
    scenario : str
        RCP scenario (e.g., 'rcp85')
    model : str
        Specific model or 'all'

    Returns:
    --------
    list : URLs or instructions
    """

    print("\nSearching for CORDEX data:")
    print(f"  Domain: {domain} ({DOMAINS.get(domain, 'Unknown')})")
    print(f"  Variable: {variable} ({VARIABLES.get(variable, 'Unknown')})")
    print(f"  Scenario: {scenario}")
    print(f"  Model: {model}")
    print()

    print("=" * 70)
    print("ESGF DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("Option 1: Web Interface")
    print("-" * 70)
    print("1. Go to: https://esgf-node.llnl.gov/search/esgf-llnl/")
    print("2. Search criteria:")
    print("   - Project: CORDEX")
    print(f"   - Domain: {domain}")
    print(f"   - Variable: {variable}")
    print(f"   - Experiment: {scenario}")
    print("3. Create account (if needed): https://esgf-node.llnl.gov/user/add/")
    print("4. Add files to cart and download using wget script")
    print()

    print("Option 2: Python esgf-pyclient")
    print("-" * 70)
    print("Install: pip install esgf-pyclient")
    print()
    print("Example code:")
    print(f"""
from pyesgf.search import SearchConnection
conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)

ctx = conn.new_context(
    project='CORDEX',
    domain='{domain}',
    variable='{variable}',
    experiment='{scenario}',
    time_frequency='mon'
)

results = ctx.search()
for result in results:
    print(result.dataset_id)
    files = result.file_context().search()
    for f in files:
        print(f"  {f.download_url}")
""")

    print()
    print("Option 3: Copernicus CDS")
    print("-" * 70)
    print("For EURO-CORDEX data:")
    print("1. Register at: https://cds.climate.copernicus.eu/")
    print("2. Install cdsapi: pip install cdsapi")
    print("3. Setup API key: https://cds.climate.copernicus.eu/api-how-to")
    print()
    print("Example CDS API code:")
    print("""
import cdsapi
c = cdsapi.Client()

c.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'domain': 'europe',
        'experiment': 'rcp_8_5',
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'monthly',
        'variable': '2m_temperature',
        'gcm_model': 'mohc_hadgem2_es',
        'rcm_model': 'clmcom_cclm4_8_17',
        'ensemble_member': 'r1i1p1',
        'start_year': '2071',
        'end_year': '2100',
    },
    'download.nc')
""")
    print()
    print("=" * 70)
    print()

    return []


def download_sample_data(output_dir="../data/cordex_sample"):
    """
    Download a small sample dataset for testing.

    This creates synthetic data that mimics CORDEX structure.
    """
    from datetime import datetime

    import numpy as np
    import xarray as xr

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating sample CORDEX-like data in {output_path}")

    # Create a small synthetic dataset
    lats = np.arange(30, 50, 0.5)
    lons = np.arange(-100, -60, 0.5)
    time = pd.date_range("1990-01-01", "2005-12-31", freq="MS")

    # Simple temperature field with trend
    tas_data = np.random.randn(len(time), len(lats), len(lons)) * 2 + 15
    # Add warming trend
    for i in range(len(time)):
        tas_data[i] += i * 0.002  # Small warming

    ds = xr.Dataset(
        {"tas": (["time", "lat", "lon"], tas_data)}, coords={"time": time, "lat": lats, "lon": lons}
    )

    # Add CF-compliant attributes
    ds["tas"].attrs = {
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "units": "K",
        "cell_methods": "time: mean",
    }

    ds.attrs = {
        "Conventions": "CF-1.6",
        "project_id": "CORDEX",
        "CORDEX_domain": "NAM-44",
        "experiment": "historical",
        "driving_model_id": "SAMPLE-GCM",
        "model_id": "SAMPLE-RCM",
        "frequency": "mon",
        "creation_date": datetime.now().isoformat(),
    }

    output_file = output_path / "tas_NAM-44_sample_historical_199001-200512.nc"
    ds.to_netcdf(output_file, encoding={"tas": {"zlib": True, "complevel": 5}})

    print(f"✓ Created sample file: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Download CORDEX regional climate data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--domain", type=str, help="CORDEX domain (e.g., NAM, EUR)")
    parser.add_argument("--variable", type=str, help="Variable name (e.g., tas, pr)")
    parser.add_argument("--scenario", type=str, choices=SCENARIOS, help="RCP scenario")
    parser.add_argument("--model", type=str, default="all", help="Specific model or all")
    parser.add_argument("--list-domains", action="store_true", help="List available domains")
    parser.add_argument("--list-variables", action="store_true", help="List available variables")
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample data for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/cordex_sample",
        help="Output directory for downloaded data",
    )

    args = parser.parse_args()

    if args.list_domains:
        list_domains()
        return

    if args.list_variables:
        list_variables()
        return

    if args.create_sample:
        sample_file = download_sample_data(args.output_dir)
        print("\nSample data ready! Load it with:")
        print("  import xarray as xr")
        print(f"  ds = xr.open_dataset('{sample_file}')")
        return

    if not all([args.domain, args.variable, args.scenario]):
        parser.print_help()
        print("\nError: --domain, --variable, and --scenario are required")
        print("Or use --list-domains, --list-variables, or --create-sample")
        sys.exit(1)

    get_esgf_download_urls(args.domain, args.variable, args.scenario, args.model)


if __name__ == "__main__":
    main()
