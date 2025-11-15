#!/usr/bin/env python3
"""
AWS Open Data Access for Climate Science

Access climate datasets from AWS Open Data Registry:
- CMIP6 (Coupled Model Intercomparison Project Phase 6)
- ERA5 Reanalysis
- NOAA GFS (Global Forecast System)
- Sentinel-2 (satellite imagery)

Usage:
    from aws_data_access import list_cmip6_data, download_era5_sample

    # List available CMIP6 data
    files = list_cmip6_data(variable='tas', experiment='historical')

    # Download ERA5 sample
    download_era5_sample(variable='2m_temperature', year=2020, month=1)

AWS Setup:
    # Configure AWS credentials (if accessing requester-pays buckets)
    aws configure

    # Or set environment variables
    export AWS_ACCESS_KEY_ID=your_key
    export AWS_SECRET_ACCESS_KEY=your_secret
    export AWS_DEFAULT_REGION=us-east-1
"""

import os

import boto3
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config

# AWS Open Data Registry S3 buckets
BUCKETS = {
    "cmip6": "cmip6-pds",  # Public, no credentials needed
    "era5": "era5-pds",  # Public, no credentials needed
    "noaa_gfs": "noaa-gfs-bdp-pds",  # Public
    "sentinel2": "sentinel-s2-l2a",  # Public but large
}


def get_s3_client(anonymous=True):
    """
    Create S3 client for accessing AWS Open Data.

    Parameters:
    -----------
    anonymous : bool
        If True, use unsigned requests (for public data)
        If False, use AWS credentials

    Returns:
    --------
    s3_client : boto3.client
        S3 client
    """
    if anonymous:
        # Unsigned requests for public buckets
        config = Config(signature_version=UNSIGNED)
        return boto3.client("s3", config=config)
    else:
        # Use AWS credentials
        return boto3.client("s3")


def list_cmip6_data(
    variable="tas",
    experiment="historical",
    model="CESM2",
    frequency="mon",
    max_results=20,
    anonymous=True,
):
    """
    List CMIP6 data files on AWS.

    CMIP6 data structure:
    s3://cmip6-pds/CMIP6/[activity]/[institution]/[model]/[experiment]/...

    Parameters:
    -----------
    variable : str
        Variable name (e.g., 'tas', 'pr', 'tasmax')
    experiment : str
        Experiment ID (e.g., 'historical', 'ssp585')
    model : str
        Model name (e.g., 'CESM2', 'GFDL-ESM4')
    frequency : str
        Temporal frequency ('mon', 'day')
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["cmip6"]

    # Construct prefix (simplified search)
    prefix = "CMIP6/"

    print(f"Searching CMIP6 data on s3://{bucket}/")
    print(f"  Variable: {variable}")
    print(f"  Experiment: {experiment}")
    print(f"  Model: {model}")

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1000)

        files = []
        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Filter by criteria
                if all(
                    [
                        variable in key,
                        experiment in key,
                        model in key,
                        frequency in key,
                        key.endswith(".nc"),
                    ]
                ):
                    files.append(key)

                    if len(files) >= max_results:
                        break

            if len(files) >= max_results:
                break

        print(f"\nFound {len(files)} matching files")
        return files

    except Exception as e:
        print(f"Error listing CMIP6 data: {e}")
        print("\nExample CMIP6 data access:")
        print("  aws s3 ls s3://cmip6-pds/CMIP6/ --no-sign-request")
        return []


def download_cmip6_file(s3_key, output_path=None, anonymous=True):
    """
    Download a CMIP6 file from AWS.

    Parameters:
    -----------
    s3_key : str
        S3 key of file to download
    output_path : str, optional
        Local output path (default: basename of s3_key)
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    local_path : str
        Path to downloaded file
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["cmip6"]

    if output_path is None:
        output_path = os.path.basename(s3_key)

    print(f"Downloading: s3://{bucket}/{s3_key}")
    print(f"         to: {output_path}")

    try:
        s3.download_file(bucket, s3_key, output_path)
        print("Download complete!")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def open_cmip6_from_s3(s3_key, anonymous=True):
    """
    Open CMIP6 NetCDF file directly from S3 using xarray.

    Requires s3fs: pip install s3fs

    Parameters:
    -----------
    s3_key : str
        S3 key of file
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    ds : xarray.Dataset
        Opened dataset
    """
    try:
        import s3fs
    except ImportError:
        raise ImportError("s3fs required: pip install s3fs")

    bucket = BUCKETS["cmip6"]
    s3_path = f"s3://{bucket}/{s3_key}"

    # Create S3 filesystem
    fs = s3fs.S3FileSystem(anon=anonymous)

    print(f"Opening: {s3_path}")

    try:
        with fs.open(s3_path, "rb") as f:
            ds = xr.open_dataset(f)
        return ds
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None


def list_era5_data(
    variable="2m_temperature", year=2020, month=None, max_results=10, anonymous=True
):
    """
    List ERA5 reanalysis data on AWS.

    ERA5 structure:
    s3://era5-pds/[year]/[month]/data/[variable].nc

    Parameters:
    -----------
    variable : str
        Variable name (e.g., '2m_temperature', 'precipitation_amount_1hour_Accumulation')
    year : int
        Year
    month : int, optional
        Month (1-12), if None list all months
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["era5"]

    files = []

    months = [month] if month is not None else range(1, 13)

    print(f"Searching ERA5 data on s3://{bucket}/")
    print(f"  Variable: {variable}")
    print(f"  Year: {year}")

    try:
        for m in months:
            prefix = f"{year}/{m:02d}/data/"

            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    if variable in key and key.endswith(".nc"):
                        files.append(key)

                        if len(files) >= max_results:
                            break

            if len(files) >= max_results:
                break

        print(f"\nFound {len(files)} matching files")
        return files

    except Exception as e:
        print(f"Error listing ERA5 data: {e}")
        print("\nExample ERA5 data access:")
        print("  aws s3 ls s3://era5-pds/2020/01/data/ --no-sign-request")
        return []


def download_era5_sample(
    variable="2m_temperature", year=2020, month=1, output_dir="data", anonymous=True
):
    """
    Download a sample ERA5 file.

    Parameters:
    -----------
    variable : str
        Variable name
    year : int
        Year
    month : int
        Month
    output_dir : str
        Output directory
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    local_path : str
        Path to downloaded file
    """
    files = list_era5_data(variable, year, month, max_results=1, anonymous=anonymous)

    if not files:
        print("No files found")
        return None

    s3_key = files[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(s3_key))

    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["era5"]

    print(f"\nDownloading: s3://{bucket}/{s3_key}")
    print(f"         to: {output_path}")

    try:
        s3.download_file(bucket, s3_key, output_path)
        print("Download complete!")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def list_noaa_gfs_data(date="20240101", hour="00", max_results=10, anonymous=True):
    """
    List NOAA GFS forecast data on AWS.

    GFS structure:
    s3://noaa-gfs-bdp-pds/gfs.[date]/[hour]/atmos/gfs.t[hour]z.pgrb2.0p25.f[forecast_hour]

    Parameters:
    -----------
    date : str
        Date in YYYYMMDD format
    hour : str
        Hour in HH format ('00', '06', '12', '18')
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["noaa_gfs"]

    prefix = f"gfs.{date}/{hour}/atmos/"

    print(f"Searching NOAA GFS data on s3://{bucket}/")
    print(f"  Date: {date}")
    print(f"  Hour: {hour}")

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_results)

        files = []
        if "Contents" in response:
            files = [obj["Key"] for obj in response["Contents"]]

        print(f"\nFound {len(files)} files")
        return files

    except Exception as e:
        print(f"Error listing GFS data: {e}")
        print("\nExample GFS data access:")
        print("  aws s3 ls s3://noaa-gfs-bdp-pds/gfs.20240101/00/atmos/ --no-sign-request")
        return []


def get_bucket_info():
    """
    Print information about AWS Open Data buckets for climate science.
    """
    print("AWS Open Data Registry - Climate Science Datasets")
    print("=" * 70)

    datasets = {
        "CMIP6": {
            "bucket": "s3://cmip6-pds",
            "description": "Coupled Model Intercomparison Project Phase 6",
            "size": "~400 TB",
            "variables": "Temperature, precipitation, sea level, etc.",
            "access": "Public, no credentials required",
            "docs": "https://registry.opendata.aws/cmip6/",
        },
        "ERA5": {
            "bucket": "s3://era5-pds",
            "description": "ECMWF Reanalysis v5",
            "size": "~500 TB",
            "variables": "Temperature, pressure, wind, humidity, etc.",
            "access": "Public, no credentials required",
            "docs": "https://registry.opendata.aws/ecmwf-era5/",
        },
        "NOAA GFS": {
            "bucket": "s3://noaa-gfs-bdp-pds",
            "description": "Global Forecast System",
            "size": "Updated 4x daily",
            "variables": "Operational weather forecasts",
            "access": "Public, no credentials required",
            "docs": "https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        },
        "Sentinel-2": {
            "bucket": "s3://sentinel-s2-l2a",
            "description": "Sentinel-2 Level-2A (atmospherically corrected)",
            "size": "~2 PB",
            "variables": "Multispectral satellite imagery",
            "access": "Public, no credentials required",
            "docs": "https://registry.opendata.aws/sentinel-2/",
        },
    }

    for name, info in datasets.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key:15s}: {value}")

    print("\n" + "=" * 70)
    print("\nGetting Started:")
    print("  1. Install AWS CLI: pip install awscli")
    print("  2. Optional credentials: aws configure")
    print("  3. List data: aws s3 ls s3://cmip6-pds/ --no-sign-request")
    print("  4. Download: aws s3 cp s3://bucket/key local_file --no-sign-request")
    print("\nPython Access:")
    print("  pip install boto3 s3fs xarray")


if __name__ == "__main__":
    print("AWS Open Data Access for Climate Science")
    print("=" * 70)

    # Show available datasets
    print("\n1. Available Datasets")
    print("-" * 70)
    get_bucket_info()

    # Example: List CMIP6 data
    print("\n\n2. Example: Searching CMIP6 Data")
    print("-" * 70)
    cmip6_files = list_cmip6_data(
        variable="tas", experiment="historical", model="CESM2", max_results=5
    )
    if cmip6_files:
        print("\nSample files:")
        for f in cmip6_files[:3]:
            print(f"  {f}")

    # Example: List ERA5 data
    print("\n\n3. Example: Searching ERA5 Data")
    print("-" * 70)
    era5_files = list_era5_data(variable="2m_temperature", year=2020, month=1, max_results=3)
    if era5_files:
        print("\nSample files:")
        for f in era5_files:
            print(f"  {f}")

    # Example: List GFS data
    print("\n\n4. Example: Searching NOAA GFS Data")
    print("-" * 70)
    gfs_files = list_noaa_gfs_data(date="20240101", hour="00", max_results=5)
    if gfs_files:
        print("\nSample files:")
        for f in gfs_files[:3]:
            print(f"  {f}")

    print("\n" + "=" * 70)
    print("\n✓ AWS Open Data access ready")
    print("\nNext steps:")
    print("  - Run: python aws_data_access.py")
    print("  - Uncomment download functions to fetch data")
    print("  - See AWS Open Data Registry: https://registry.opendata.aws/")
