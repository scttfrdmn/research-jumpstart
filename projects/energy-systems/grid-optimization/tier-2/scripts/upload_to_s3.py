#!/usr/bin/env python3
"""
Upload smart grid data to S3 bucket.

This script uploads grid sensor data files to S3, where they will be automatically
processed by the Lambda function for optimization analysis.

Usage:
    python upload_to_s3.py --bucket energy-grid-XXXXX --file path/to/data.csv
    python upload_to_s3.py --bucket energy-grid-XXXXX --input directory/
    python upload_to_s3.py --bucket energy-grid-XXXXX --generate
"""

import argparse
import boto3
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import csv
import numpy as np
import pandas as pd

# AWS S3 client
s3_client = boto3.client('s3')


def generate_grid_data(days=7, interval_minutes=15, output_dir='./sample_data'):
    """
    Generate synthetic smart grid data for testing.

    Creates realistic load profiles, generation data, voltage, frequency,
    and renewable energy generation.

    Args:
        days (int): Number of days of data to generate
        interval_minutes (int): Time interval between measurements
        output_dir (str): Directory to save generated files

    Returns:
        list: Paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {days} days of smart grid data...")
    print(f"Interval: {interval_minutes} minutes")

    # Time range
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = start_time - timedelta(days=days)
    intervals = days * 24 * (60 // interval_minutes)

    timestamps = [start_time + timedelta(minutes=i*interval_minutes)
                  for i in range(intervals)]

    # Generate data for multiple substations
    locations = ['substation_001', 'substation_002', 'substation_003']
    files_created = []

    for location in locations:
        data = []

        # Use location-specific seed for reproducibility
        np.random.seed(hash(location) % 2**32)

        for i, ts in enumerate(timestamps):
            # Hour of day for daily patterns
            hour = ts.hour
            day_of_week = ts.weekday()

            # Base load with daily and weekly patterns
            base_load = 100 + 30 * np.sin(2 * np.pi * hour / 24)

            # Weekday vs weekend
            if day_of_week >= 5:  # Weekend
                base_load *= 0.85

            # Add noise
            load_mw = base_load + np.random.normal(0, 5)

            # Generation slightly exceeds load (grid stability)
            generation_mw = load_mw * 1.05 + np.random.normal(0, 2)

            # Voltage (nominal 13.8 kV, ±5%)
            voltage_kv = 13.8 + np.random.normal(0, 0.3)

            # Frequency (nominal 60 Hz, ±0.05 Hz)
            frequency_hz = 60.0 + np.random.normal(0, 0.02)

            # Solar generation (peak at noon, zero at night)
            if 6 <= hour <= 18:
                solar_factor = np.sin(np.pi * (hour - 6) / 12)
                solar_mw = 20 * solar_factor + np.random.normal(0, 2)
                solar_mw = max(0, solar_mw)
            else:
                solar_mw = 0

            # Wind generation (more variable, less predictable)
            wind_base = 15 + 10 * np.sin(2 * np.pi * hour / 24)
            wind_mw = max(0, wind_base + np.random.normal(0, 5))

            # Power factor (typical range 0.85-0.98)
            power_factor = 0.90 + np.random.uniform(-0.05, 0.08)
            power_factor = np.clip(power_factor, 0.85, 0.98)

            data.append({
                'timestamp': ts.isoformat(),
                'location': location,
                'load_mw': round(load_mw, 2),
                'generation_mw': round(generation_mw, 2),
                'voltage_kv': round(voltage_kv, 3),
                'frequency_hz': round(frequency_hz, 4),
                'solar_mw': round(solar_mw, 2),
                'wind_mw': round(wind_mw, 2),
                'power_factor': round(power_factor, 3)
            })

        # Save to CSV
        filename = f"grid_data_{location}_{start_time.strftime('%Y%m%d')}.csv"
        filepath = output_path / filename

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Created {filename} ({file_size_mb:.2f}MB, {len(data)} records)")

        files_created.append(filepath)

    print(f"\n✓ Generated {len(files_created)} files in {output_dir}/")
    return files_created


def upload_file_to_s3(bucket_name, file_path, prefix='raw/'):
    """
    Upload a single file to S3.

    Args:
        bucket_name (str): Target S3 bucket name
        file_path (str): Local file path
        prefix (str): S3 prefix/folder ('raw/', 'test/', etc.)

    Returns:
        str: S3 object key or None on error
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    if file_size_mb > 100:
        print(f"Warning: Large file ({file_size_mb:.1f}MB). Upload may take time.")

    # Generate S3 key
    s3_key = f"{prefix}{file_path.name}"

    print(f"Uploading: {file_path.name} ({file_size_mb:.2f}MB)")
    print(f"  Destination: s3://{bucket_name}/{s3_key}")

    try:
        # Upload file with progress tracking
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'text/csv'},
            Callback=UploadProgressCallback(file_path)
        )

        print(f"  ✓ Upload successful")
        return s3_key

    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return None


def upload_directory_to_s3(bucket_name, directory, file_pattern='*.csv', prefix='raw/'):
    """
    Upload all matching files from a directory to S3.

    Args:
        bucket_name (str): Target S3 bucket name
        directory (str): Local directory path
        file_pattern (str): Glob pattern for files to upload
        prefix (str): S3 prefix/folder

    Returns:
        list: Uploaded file information
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: Directory not found: {directory}")
        return []

    files = list(dir_path.glob(file_pattern))

    if not files:
        print(f"No files matching '{file_pattern}' found in {directory}")
        return []

    print(f"Found {len(files)} files to upload")
    print("=" * 60)

    uploaded = []
    failed = []

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")

        s3_key = upload_file_to_s3(bucket_name, file_path, prefix)

        if s3_key:
            uploaded.append({
                'file': file_path.name,
                's3_key': s3_key,
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })
        else:
            failed.append(file_path.name)

    # Summary
    print("\n" + "=" * 60)
    print(f"Upload Summary")
    print("=" * 60)
    print(f"Successful: {len(uploaded)}/{len(files)}")
    print(f"Failed: {len(failed)}/{len(files)}")

    if uploaded:
        total_size = sum(item['size_mb'] for item in uploaded)
        print(f"Total uploaded: {total_size:.2f}MB")

    if failed:
        print(f"\nFailed files:")
        for filename in failed:
            print(f"  - {filename}")

    return uploaded


class UploadProgressCallback:
    """Display upload progress bar."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.file_size = self.file_path.stat().st_size
        self.uploaded = 0

    def __call__(self, bytes_amount):
        self.uploaded += bytes_amount
        percent = (self.uploaded / self.file_size) * 100
        bar_length = 30
        filled = int(bar_length * self.uploaded / self.file_size)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  [{bar}] {percent:.1f}%", end='\r')

        # Print newline when complete
        if self.uploaded >= self.file_size:
            print()


def verify_bucket_access(bucket_name):
    """Verify S3 bucket exists and is accessible."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ S3 bucket accessible: {bucket_name}")
        return True
    except Exception as e:
        print(f"✗ Cannot access bucket '{bucket_name}': {e}")
        print("\nPlease ensure:")
        print("  1. Bucket exists")
        print("  2. AWS credentials are configured (aws configure)")
        print("  3. IAM permissions allow S3 access")
        return False


def list_bucket_contents(bucket_name, prefix='raw/'):
    """List contents of bucket prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=100)

        if 'Contents' not in response:
            print(f"No objects in s3://{bucket_name}/{prefix}")
            return []

        objects = response['Contents']
        print(f"\nObjects in s3://{bucket_name}/{prefix}:")
        print(f"{'Key':<60} {'Size':<15} {'Last Modified'}")
        print("-" * 90)

        for obj in objects:
            size_mb = obj['Size'] / (1024 * 1024)
            modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M')
            print(f"{obj['Key']:<60} {size_mb:>10.2f} MB  {modified}")

        print(f"\nTotal: {len(objects)} objects")
        return objects

    except Exception as e:
        print(f"Error listing bucket contents: {e}")
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Upload smart grid data to S3 for optimization processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate and upload sample data
  python upload_to_s3.py --bucket energy-grid-12345 --generate

  # Upload single file
  python upload_to_s3.py --bucket energy-grid-12345 --file data.csv

  # Upload directory of files
  python upload_to_s3.py --bucket energy-grid-12345 --input ./grid_data/

  # List uploaded files
  python upload_to_s3.py --bucket energy-grid-12345 --list
        '''
    )

    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name (e.g., energy-grid-12345)'
    )
    parser.add_argument(
        '--file',
        help='Single file to upload'
    )
    parser.add_argument(
        '--input',
        help='Directory containing files to upload'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate sample grid data before uploading'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Days of data to generate (default: 7)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List files in bucket'
    )
    parser.add_argument(
        '--pattern',
        default='*.csv',
        help='File pattern for directory upload (default: *.csv)'
    )
    parser.add_argument(
        '--prefix',
        default='raw/',
        help='S3 prefix/folder for uploads (default: raw/)'
    )

    args = parser.parse_args()

    print("Smart Grid Data Upload Tool")
    print("=" * 60)

    # Verify bucket access
    if not verify_bucket_access(args.bucket):
        sys.exit(1)

    # Handle different operations
    if args.list:
        list_bucket_contents(args.bucket, args.prefix)

    elif args.generate:
        # Generate sample data
        print("\nGenerating sample grid data...")
        files = generate_grid_data(days=args.days)

        # Upload generated files
        if files:
            print(f"\nUploading {len(files)} generated files...")
            upload_directory_to_s3(args.bucket, './sample_data', '*.csv', args.prefix)

    elif args.file:
        # Upload single file
        upload_file_to_s3(args.bucket, args.file, args.prefix)

    elif args.input:
        # Upload directory
        upload_directory_to_s3(args.bucket, args.input, args.pattern, args.prefix)

    else:
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("\nNext steps:")
    print(f"  1. Lambda will process files automatically (if trigger configured)")
    print(f"  2. Or manually invoke: python scripts/lambda_function.py --test")
    print(f"  3. Query results: python scripts/query_results.py --bucket {args.bucket}")


if __name__ == '__main__':
    main()
