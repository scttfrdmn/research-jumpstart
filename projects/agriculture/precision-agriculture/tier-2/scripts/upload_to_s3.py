#!/usr/bin/env python3
"""
Upload Sentinel-2 imagery to S3 bucket.

This script uploads satellite imagery files to S3, where they will be automatically
processed by the Lambda function for NDVI calculation.

Usage:
    python upload_to_s3.py --bucket satellite-imagery-XXXXX --file path/to/image.tif
    python upload_to_s3.py --bucket satellite-imagery-XXXXX --input directory/
    python upload_to_s3.py --bucket satellite-imagery-XXXXX --file image.tif --field-id field_001
"""

import argparse
import sys
from pathlib import Path

import boto3

# AWS S3 client
s3_client = boto3.client("s3")


def upload_file_to_s3(bucket_name, file_path, field_id=None, prefix="raw/"):
    """
    Upload a single file to S3.

    Args:
        bucket_name (str): Target S3 bucket name
        file_path (str): Local file path
        field_id (str, optional): Field identifier for metadata
        prefix (str): S3 prefix/folder ('raw/', 'test/', etc.)

    Returns:
        str: S3 object key
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    if file_size_mb > 500:
        print(f"Warning: Large file ({file_size_mb:.1f}MB). Upload may take time.")

    # Generate S3 key
    if field_id:
        # Extract date from filename if available
        filename = file_path.stem
        s3_key = f"{prefix}{field_id}_{filename}.tif"
    else:
        s3_key = f"{prefix}{file_path.name}"

    print(f"Uploading: {file_path.name} ({file_size_mb:.2f}MB)")
    print(f"  Destination: s3://{bucket_name}/{s3_key}")

    try:
        # Upload file with progress tracking
        s3_client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "image/tiff"},
            Callback=UploadProgressCallback(file_path),
        )

        print("  ✓ Upload successful")
        return s3_key

    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return None


def upload_directory_to_s3(bucket_name, directory, file_pattern="*.tif", prefix="raw/"):
    """
    Upload all matching files from a directory to S3.

    Args:
        bucket_name (str): Target S3 bucket name
        directory (str): Local directory path
        file_pattern (str): Glob pattern for files to upload
        prefix (str): S3 prefix/folder
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: Directory not found: {directory}")
        return

    files = list(dir_path.glob(file_pattern))

    if not files:
        print(f"No files matching '{file_pattern}' found in {directory}")
        return

    print(f"Found {len(files)} files to upload")
    print("=" * 60)

    uploaded = []
    failed = []

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")

        # Extract field_id from filename if possible
        # Expected format: field_XXX_YYYYMMDD.tif
        field_id = None
        parts = file_path.stem.split("_")
        if len(parts) >= 2:
            field_id = "_".join(parts[:2])

        s3_key = upload_file_to_s3(bucket_name, file_path, field_id, prefix)

        if s3_key:
            uploaded.append(
                {
                    "file": file_path.name,
                    "s3_key": s3_key,
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                }
            )
        else:
            failed.append(file_path.name)

    # Summary
    print("\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"Successful: {len(uploaded)}/{len(files)}")
    print(f"Failed: {len(failed)}/{len(files)}")

    if uploaded:
        total_size = sum(item["size_mb"] for item in uploaded)
        print(f"Total uploaded: {total_size:.2f}MB")

    if failed:
        print("\nFailed files:")
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
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {percent:.1f}%", end="\r")


def create_sample_data(directory):
    """Create sample GeoTIFF file for testing."""

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Create simple sample data
    sample_file = dir_path / "field_001_20240615.tif"

    if sample_file.exists():
        print(f"Sample file already exists: {sample_file}")
        return

    print(f"Creating sample file: {sample_file}")

    # Create sample geotiff data (simplified for testing)
    with open(sample_file, "wb") as f:
        # Write a minimal GeoTIFF header + test data
        # For production, use rasterio to create proper GeoTIFFs
        f.write(b"Sample GeoTIFF data for testing")

    print(f"  ✓ Sample file created ({sample_file.stat().st_size} bytes)")
    return sample_file


def verify_bucket_access(bucket_name):
    """Verify S3 bucket exists and is accessible."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ S3 bucket accessible: {bucket_name}")
        return True
    except Exception as e:
        print(f"✗ Cannot access bucket '{bucket_name}': {e}")
        return False


def list_bucket_contents(bucket_name, prefix="raw/"):
    """List contents of bucket prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if "Contents" not in response:
            print(f"No objects in {prefix}")
            return []

        objects = response["Contents"]
        print(f"\nObjects in s3://{bucket_name}/{prefix}:")
        for obj in objects:
            size_mb = obj["Size"] / (1024 * 1024)
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - {obj['Key']:<50} {size_mb:>10.2f}MB  {modified}")

        return objects

    except Exception as e:
        print(f"Error listing bucket contents: {e}")
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Upload Sentinel-2 imagery to S3 for NDVI processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload single file
  python upload_to_s3.py --bucket satellite-imagery-12345 --file image.tif

  # Upload directory of files
  python upload_to_s3.py --bucket satellite-imagery-12345 --input ./images/

  # Create and upload sample data
  python upload_to_s3.py --bucket satellite-imagery-12345 --sample

  # List uploaded files
  python upload_to_s3.py --bucket satellite-imagery-12345 --list
        """,
    )

    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name (e.g., satellite-imagery-12345)"
    )
    parser.add_argument("--file", help="Single file to upload")
    parser.add_argument("--input", help="Directory containing files to upload")
    parser.add_argument("--field-id", help="Field identifier for metadata (e.g., field_001)")
    parser.add_argument(
        "--sample", action="store_true", help="Create sample GeoTIFF file for testing"
    )
    parser.add_argument("--list", action="store_true", help="List files in bucket")
    parser.add_argument(
        "--pattern", default="*.tif", help="File pattern for directory upload (default: *.tif)"
    )
    parser.add_argument(
        "--prefix", default="raw/", help="S3 prefix/folder for uploads (default: raw/)"
    )

    args = parser.parse_args()

    print("Sentinel-2 Upload Tool")
    print("=" * 60)

    # Verify bucket access
    if not verify_bucket_access(args.bucket):
        sys.exit(1)

    # Handle different operations
    if args.list:
        list_bucket_contents(args.bucket, args.prefix)

    elif args.sample:
        # Create and upload sample data
        sample_file = create_sample_data("./sample_data")
        if sample_file:
            upload_file_to_s3(args.bucket, sample_file, "field_001", args.prefix)
            print(f"\n✓ Sample data uploaded to s3://{args.bucket}/{args.prefix}")

    elif args.file:
        # Upload single file
        upload_file_to_s3(args.bucket, args.file, args.field_id, args.prefix)

    elif args.input:
        # Upload directory
        upload_directory_to_s3(args.bucket, args.input, args.pattern, args.prefix)

    else:
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Upload complete!")


if __name__ == "__main__":
    main()
