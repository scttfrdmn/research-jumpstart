"""
Upload economic time series data to S3.

This script:
1. Loads economic data from local CSV files or downloads from FRED
2. Validates data format (date, value columns)
3. Uploads to S3 organized by indicator and country
4. Displays progress and summary statistics

Usage:
    python upload_to_s3.py --bucket economic-data-12345 --data-dir sample_data/
    python upload_to_s3.py --bucket economic-data-12345 --download-fred GDP,UNRATE,CPIAUCSL
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
from datetime import datetime


class EconomicDataUploader:
    """Upload economic time series data to S3."""

    def __init__(self, bucket_name: str):
        """Initialize uploader with S3 bucket name."""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.uploaded_files = []
        self.failed_files = []

    def validate_bucket(self) -> bool:
        """Verify S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ S3 bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"✗ Bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                print(f"✗ Access denied to bucket '{self.bucket_name}'")
            else:
                print(f"✗ Error accessing bucket: {e}")
            return False
        except NoCredentialsError:
            print("✗ AWS credentials not configured. Run 'aws configure'")
            return False

    def validate_csv(self, file_path: str) -> bool:
        """Validate CSV has required columns (date, value)."""
        try:
            df = pd.read_csv(file_path, nrows=5)

            # Check for required columns
            required_cols = ['date', 'value']
            if not all(col in df.columns for col in required_cols):
                print(f"  ✗ Missing required columns: {required_cols}")
                return False

            # Check date format
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                print(f"  ✗ Invalid date format: {e}")
                return False

            # Check value is numeric
            if not pd.api.types.is_numeric_dtype(df['value']):
                print(f"  ✗ 'value' column must be numeric")
                return False

            return True

        except Exception as e:
            print(f"  ✗ Error validating CSV: {e}")
            return False

    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """Upload single file to S3."""
        try:
            # Read file size for progress
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)

            print(f"  Uploading: {file_path} ({file_size_mb:.2f} MB)")

            # Upload file
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'text/csv'}
            )

            print(f"  ✓ Uploaded to s3://{self.bucket_name}/{s3_key}")
            self.uploaded_files.append(s3_key)
            return True

        except FileNotFoundError:
            print(f"  ✗ File not found: {file_path}")
            self.failed_files.append(file_path)
            return False
        except ClientError as e:
            print(f"  ✗ Upload failed: {e}")
            self.failed_files.append(file_path)
            return False

    def upload_directory(self, data_dir: str, prefix: str = "raw/") -> int:
        """Upload all CSV files from directory."""
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"✗ Directory not found: {data_dir}")
            return 0

        # Find all CSV files
        csv_files = list(data_path.rglob("*.csv"))

        if not csv_files:
            print(f"✗ No CSV files found in {data_dir}")
            return 0

        print(f"\nFound {len(csv_files)} CSV files to upload")
        print("-" * 60)

        uploaded_count = 0

        for csv_file in csv_files:
            print(f"\nProcessing: {csv_file.name}")

            # Validate CSV format
            if not self.validate_csv(str(csv_file)):
                print(f"  ⚠ Skipping invalid file")
                continue

            # Determine S3 key based on file structure
            # Example: sample_data/gdp/usa_gdp_quarterly.csv → raw/gdp/usa_gdp_quarterly.csv
            relative_path = csv_file.relative_to(data_path)
            s3_key = prefix + str(relative_path).replace('\\', '/')

            # Upload file
            if self.upload_file(str(csv_file), s3_key):
                uploaded_count += 1

        return uploaded_count

    def print_summary(self):
        """Print upload summary."""
        print("\n" + "=" * 60)
        print("Upload Summary")
        print("=" * 60)
        print(f"Successfully uploaded: {len(self.uploaded_files)} files")
        print(f"Failed uploads: {len(self.failed_files)} files")

        if self.uploaded_files:
            print("\n✓ Uploaded files:")
            for s3_key in self.uploaded_files:
                print(f"  - s3://{self.bucket_name}/{s3_key}")

        if self.failed_files:
            print("\n✗ Failed files:")
            for file_path in self.failed_files:
                print(f"  - {file_path}")

        print("=" * 60)


def download_fred_data(indicators: List[str], output_dir: str = "sample_data/fred/"):
    """
    Download economic data from FRED API.

    Requires: pip install fredapi
    Set FRED_API_KEY environment variable
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("✗ fredapi not installed. Install with: pip install fredapi")
        return

    # Get API key from environment
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        print("✗ FRED_API_KEY environment variable not set")
        print("  Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    fred = Fred(api_key=api_key)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDownloading {len(indicators)} indicators from FRED...")

    for indicator in indicators:
        try:
            print(f"  Downloading {indicator}...")
            series = fred.get_series(indicator)

            # Convert to DataFrame
            df = pd.DataFrame({
                'date': series.index,
                'value': series.values
            })

            # Save to CSV
            output_file = os.path.join(output_dir, f"{indicator.lower()}.csv")
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved to {output_file}")

        except Exception as e:
            print(f"  ✗ Failed to download {indicator}: {e}")

    print(f"\n✓ Downloaded data saved to {output_dir}")


def create_sample_data(output_dir: str = "sample_data/"):
    """Create sample economic data for testing."""
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating sample economic data...")

    # Sample GDP data (quarterly)
    gdp_dir = os.path.join(output_dir, "gdp")
    os.makedirs(gdp_dir, exist_ok=True)

    dates_quarterly = pd.date_range(start='2018-01-01', end='2023-10-01', freq='Q')
    gdp_values = [20000 + i * 100 + (i % 4) * 50 for i in range(len(dates_quarterly))]

    gdp_df = pd.DataFrame({'date': dates_quarterly, 'value': gdp_values})
    gdp_file = os.path.join(gdp_dir, "usa_gdp_quarterly.csv")
    gdp_df.to_csv(gdp_file, index=False)
    print(f"  ✓ Created {gdp_file}")

    # Sample unemployment data (monthly)
    unemp_dir = os.path.join(output_dir, "unemployment")
    os.makedirs(unemp_dir, exist_ok=True)

    dates_monthly = pd.date_range(start='2018-01-01', end='2023-12-01', freq='M')
    unemp_values = [4.0 + (i % 12) * 0.1 + (i / 12) * 0.05 for i in range(len(dates_monthly))]

    unemp_df = pd.DataFrame({'date': dates_monthly, 'value': unemp_values})
    unemp_file = os.path.join(unemp_dir, "usa_unemployment_monthly.csv")
    unemp_df.to_csv(unemp_file, index=False)
    print(f"  ✓ Created {unemp_file}")

    # Sample inflation data (monthly)
    infl_dir = os.path.join(output_dir, "inflation")
    os.makedirs(infl_dir, exist_ok=True)

    infl_values = [250 + i * 0.5 + (i % 12) * 0.2 for i in range(len(dates_monthly))]

    infl_df = pd.DataFrame({'date': dates_monthly, 'value': infl_values})
    infl_file = os.path.join(infl_dir, "usa_cpi_monthly.csv")
    infl_df.to_csv(infl_file, index=False)
    print(f"  ✓ Created {infl_file}")

    print(f"\n✓ Sample data created in {output_dir}")
    return output_dir


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Upload economic time series data to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from local directory
  python upload_to_s3.py --bucket economic-data-12345 --data-dir sample_data/

  # Create and upload sample data
  python upload_to_s3.py --bucket economic-data-12345 --create-sample

  # Download from FRED and upload
  export FRED_API_KEY=your_api_key
  python upload_to_s3.py --bucket economic-data-12345 --download-fred GDP,UNRATE,CPIAUCSL
        """
    )

    parser.add_argument(
        '--bucket',
        type=str,
        required=True,
        help='S3 bucket name (e.g., economic-data-12345)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing CSV files to upload'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='raw/',
        help='S3 key prefix (default: raw/)'
    )

    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample data and upload'
    )

    parser.add_argument(
        '--download-fred',
        type=str,
        help='Download indicators from FRED (comma-separated, e.g., GDP,UNRATE)'
    )

    args = parser.parse_args()

    # Create uploader
    uploader = EconomicDataUploader(args.bucket)

    # Validate bucket access
    if not uploader.validate_bucket():
        sys.exit(1)

    # Handle create sample data
    if args.create_sample:
        data_dir = create_sample_data()
        args.data_dir = data_dir

    # Handle FRED download
    if args.download_fred:
        indicators = [ind.strip() for ind in args.download_fred.split(',')]
        fred_dir = "sample_data/fred/"
        download_fred_data(indicators, fred_dir)
        if not args.data_dir:
            args.data_dir = fred_dir

    # Validate data directory
    if not args.data_dir:
        print("✗ No data source specified. Use --data-dir, --create-sample, or --download-fred")
        sys.exit(1)

    # Upload data
    uploaded_count = uploader.upload_directory(args.data_dir, args.prefix)

    # Print summary
    uploader.print_summary()

    if uploaded_count > 0:
        print(f"\n✓ Successfully uploaded {uploaded_count} files")
        print(f"\nNext steps:")
        print(f"  1. Monitor Lambda execution in CloudWatch")
        print(f"  2. Query results: python scripts/query_results.py --table EconomicForecasts")
        print(f"  3. Analyze: jupyter notebook notebooks/economic_analysis.ipynb")
    else:
        print("\n✗ No files uploaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
