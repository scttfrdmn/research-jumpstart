#!/usr/bin/env python3
"""
Upload oceanographic data to S3 for Lambda processing.
Supports CSV and NetCDF formats.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError


def generate_sample_ocean_data(num_profiles=10, depths=20):
    """
    Generate sample oceanographic data (CTD profiles).

    Parameters:
    - num_profiles: Number of oceanographic profiles to generate
    - depths: Number of depth levels per profile

    Returns:
    - pandas DataFrame with ocean observations
    """
    print(f"Generating {num_profiles} ocean profiles with {depths} depth levels each...")

    data = []
    locations = [
        {"name": "Gulf Stream", "lat": 40.5, "lon": -70.2},
        {"name": "Sargasso Sea", "lat": 32.0, "lon": -64.0},
        {"name": "Labrador Sea", "lat": 58.0, "lon": -52.0},
        {"name": "Station ALOHA", "lat": 22.75, "lon": -158.0},
        {"name": "Drake Passage", "lat": -60.0, "lon": -65.0},
    ]

    base_time = datetime.utcnow() - timedelta(days=30)

    for i in range(num_profiles):
        location = locations[i % len(locations)]
        profile_time = base_time + timedelta(days=i)

        # Create depth profile
        depth_levels = np.linspace(0, 500, depths)

        for depth in depth_levels:
            # Temperature decreases with depth (thermocline model)
            surface_temp = 20.0 + np.random.normal(0, 2)
            deep_temp = 4.0 + np.random.normal(0, 0.5)
            temperature = surface_temp * np.exp(-depth / 100) + deep_temp * (1 - np.exp(-depth / 100))

            # Add temperature anomaly (simulate marine heatwave)
            if location["name"] == "Gulf Stream" and depth < 50:
                temperature += 3.5  # Marine heatwave!

            # Salinity increases slightly with depth
            salinity = 35.0 + depth / 200 + np.random.normal(0, 0.2)

            # pH decreases slightly with depth (ocean acidification)
            ph = 8.1 - depth / 1000 + np.random.normal(0, 0.05)
            if location["name"] == "Sargasso Sea":
                ph -= 0.3  # Acidification event

            # Dissolved oxygen decreases with depth (oxygen minimum zone)
            do = 8.0 * np.exp(-depth / 150) + 2.0 + np.random.normal(0, 0.3)
            if location["name"] == "Station ALOHA" and 200 < depth < 400:
                do = max(1.5, do - 2.0)  # Oxygen minimum zone

            # Chlorophyll-a highest near surface, decreases with depth
            chlorophyll = max(0.1, 5.0 * np.exp(-depth / 50) + np.random.normal(0, 0.5))
            if location["name"] == "Labrador Sea" and depth < 30:
                chlorophyll = 25.0  # Spring bloom

            data.append({
                "timestamp": profile_time.isoformat() + "Z",
                "location_name": location["name"],
                "latitude": location["lat"],
                "longitude": location["lon"],
                "depth": round(depth, 1),
                "temperature": round(temperature, 2),
                "salinity": round(salinity, 2),
                "ph": round(ph, 3),
                "dissolved_oxygen": round(max(0.1, do), 2),
                "chlorophyll": round(max(0.1, chlorophyll), 2),
            })

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} ocean observations")
    print(f"\nSample data:")
    print(df.head(10))
    print(f"\nStatistics:")
    print(df.describe())

    return df


def save_to_csv(df, filename):
    """Save DataFrame to CSV file."""
    df.to_csv(filename, index=False)
    print(f"\nSaved data to {filename}")
    file_size = os.path.getsize(filename) / 1024  # KB
    print(f"File size: {file_size:.2f} KB")


def upload_to_s3(bucket_name, local_file, s3_key):
    """
    Upload file to S3 bucket.

    Parameters:
    - bucket_name: S3 bucket name
    - local_file: Local file path
    - s3_key: S3 object key (path in bucket)
    """
    s3_client = boto3.client('s3')

    try:
        print(f"\nUploading {local_file} to s3://{bucket_name}/{s3_key}")

        # Upload with progress
        file_size = os.path.getsize(local_file)

        def upload_progress(bytes_transferred):
            percent = (bytes_transferred / file_size) * 100
            print(f"\rProgress: {percent:.1f}% ({bytes_transferred}/{file_size} bytes)", end='')

        s3_client.upload_file(
            local_file,
            bucket_name,
            s3_key,
            Callback=upload_progress
        )

        print(f"\n✓ Successfully uploaded to s3://{bucket_name}/{s3_key}")

        # Verify upload
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"  Size: {response['ContentLength']} bytes")
        print(f"  Last Modified: {response['LastModified']}")

        return True

    except ClientError as e:
        print(f"\n✗ Error uploading to S3: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def verify_bucket_exists(bucket_name):
    """Verify S3 bucket exists and is accessible."""
    s3_client = boto3.client('s3')

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket '{bucket_name}' exists and is accessible")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"✗ Bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            print(f"✗ Access denied to bucket '{bucket_name}'")
        else:
            print(f"✗ Error accessing bucket: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Upload oceanographic data to S3 for Lambda processing'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        help='S3 bucket name (or set BUCKET_NAME env variable)',
        default=os.environ.get('BUCKET_NAME')
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample ocean data instead of uploading existing file'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Local CSV file to upload (if not generating sample data)'
    )
    parser.add_argument(
        '--s3-key',
        type=str,
        help='S3 key (path in bucket). Default: raw/<filename>',
        default=None
    )
    parser.add_argument(
        '--profiles',
        type=int,
        default=10,
        help='Number of ocean profiles to generate (default: 10)'
    )
    parser.add_argument(
        '--depths',
        type=int,
        default=20,
        help='Number of depth levels per profile (default: 20)'
    )

    args = parser.parse_args()

    # Validate bucket name
    if not args.bucket:
        print("Error: Bucket name required. Use --bucket or set BUCKET_NAME environment variable")
        print("\nExample:")
        print("  export BUCKET_NAME=ocean-data-myname-12345")
        print("  python upload_to_s3.py --generate-sample")
        sys.exit(1)

    # Verify bucket exists
    if not verify_bucket_exists(args.bucket):
        print("\nPlease create the bucket first:")
        print(f"  aws s3 mb s3://{args.bucket}")
        sys.exit(1)

    # Generate or use existing data
    if args.generate_sample:
        print("\n=== Generating Sample Ocean Data ===")
        df = generate_sample_ocean_data(
            num_profiles=args.profiles,
            depths=args.depths
        )

        # Save to local file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        local_file = f"ocean_data_{timestamp}.csv"
        save_to_csv(df, local_file)

        # Set S3 key
        s3_key = args.s3_key or f"raw/{local_file}"

    else:
        if not args.file:
            print("Error: Must specify --file or use --generate-sample")
            sys.exit(1)

        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)

        local_file = args.file
        filename = os.path.basename(local_file)
        s3_key = args.s3_key or f"raw/{filename}"

    # Upload to S3
    print("\n=== Uploading to S3 ===")
    success = upload_to_s3(args.bucket, local_file, s3_key)

    if success:
        print("\n=== Upload Complete ===")
        print(f"✓ File uploaded successfully")
        print(f"✓ S3 URI: s3://{args.bucket}/{s3_key}")
        print(f"\nLambda will automatically process this file.")
        print(f"Check CloudWatch logs for processing status:")
        print(f"  aws logs tail /aws/lambda/analyze-ocean-data --follow")

        # Clean up local file if it was generated
        if args.generate_sample:
            print(f"\nLocal file saved at: {local_file}")
    else:
        print("\n✗ Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
