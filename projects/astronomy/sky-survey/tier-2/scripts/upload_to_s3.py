#!/usr/bin/env python3
"""
Upload FITS images to S3 bucket.

Reads all FITS files from data/raw directory and uploads to S3.
"""

import os
import sys
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_s3_bucket_name():
    """Get S3 bucket name from environment."""
    bucket = os.environ.get('BUCKET_RAW')
    if not bucket:
        print("Error: BUCKET_RAW environment variable not set")
        print("Set it with: export BUCKET_RAW=your-bucket-name")
        sys.exit(1)
    return bucket


def upload_fits_to_s3(bucket_name):
    """Upload FITS files to S3."""

    # Initialize S3 client
    try:
        s3 = boto3.client('s3')
    except NoCredentialsError:
        print("Error: AWS credentials not configured")
        print("Run: aws configure")
        return False

    # Find FITS files
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    fits_files = list(data_dir.glob("*.fits"))

    if not fits_files:
        print(f"No FITS files found in {data_dir}")
        print("Run: python scripts/download_sample_fits.py")
        return False

    print("=" * 70)
    print("Upload FITS to S3")
    print("=" * 70)
    print(f"\nBucket: s3://{bucket_name}")
    print(f"Files to upload: {len(fits_files)}")
    print(f"Source directory: {data_dir}\n")

    # Verify bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket exists: s3://{bucket_name}\n")
    except ClientError as e:
        print(f"✗ Error: Cannot access bucket s3://{bucket_name}")
        print(f"  {e}")
        print("\nMake sure the bucket exists. Create it with:")
        print(f"  aws s3 mb s3://{bucket_name} --region us-east-1")
        return False

    # Upload files
    uploaded = []
    failed = []
    total_size = 0

    with tqdm(total=len(fits_files), desc="Uploading files", unit="file") as pbar:
        for filepath in fits_files:
            try:
                file_size = filepath.stat().st_size
                s3_key = f"images/{filepath.name}"

                # Upload to S3
                s3.upload_file(
                    str(filepath),
                    bucket_name,
                    s3_key,
                    ExtraArgs={'Metadata': {
                        'original_path': str(filepath),
                        'source': 'tier-2-astronomy-project'
                    }}
                )

                uploaded.append({
                    'file': filepath.name,
                    'key': s3_key,
                    'size': file_size
                })
                total_size += file_size
                pbar.update(1)

            except Exception as e:
                failed.append({
                    'file': filepath.name,
                    'error': str(e)
                })
                pbar.update(1)

    # Summary
    print("\n" + "=" * 70)
    print("Upload Summary")
    print("=" * 70)

    if uploaded:
        print(f"\n✓ Successfully uploaded {len(uploaded)} files:")
        for item in uploaded:
            size_mb = item['size'] / 1024 / 1024
            print(f"  • {item['file']} ({size_mb:.1f} MB)")
            print(f"    → s3://{bucket_name}/{item['key']}")

    if failed:
        print(f"\n✗ Failed to upload {len(failed)} files:")
        for item in failed:
            print(f"  • {item['file']}: {item['error']}")

    print(f"\nTotal uploaded: {len(uploaded)} files")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    # Verify upload
    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix='images/'
        )
        count = response.get('KeyCount', 0)
        print(f"\nVerification: {count} files in s3://{bucket_name}/images/")
    except Exception as e:
        print(f"\nError verifying upload: {e}")

    print("\n✓ Ready for source detection!")
    print(f"  Run: python scripts/invoke_lambda.py")
    print()

    return len(failed) == 0


def list_uploaded_files(bucket_name):
    """List files uploaded to S3."""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix='images/'
        )

        if 'Contents' not in response:
            print(f"No files in s3://{bucket_name}/images/")
            return

        print(f"\nFiles in s3://{bucket_name}/images/:")
        for obj in response['Contents']:
            size_mb = obj['Size'] / 1024 / 1024
            print(f"  • {obj['Key']} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"Error listing files: {e}")


def main():
    """Main function."""
    print("\nAstronomical Image Processing - FITS Upload to S3\n")

    # Get bucket name
    bucket_name = get_s3_bucket_name()

    # Upload files
    success = upload_fits_to_s3(bucket_name)

    # List uploaded files
    if success:
        list_uploaded_files(bucket_name)

    return 0 if success else 1


if __name__ == "__main__":
    # Check for environment file
    env_file = Path.home() / ".astronomy_env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("export "):
                    key_value = line.replace("export ", "").strip()
                    if "=" in key_value:
                        key, value = key_value.split("=", 1)
                        os.environ[key] = value

    sys.exit(main())
