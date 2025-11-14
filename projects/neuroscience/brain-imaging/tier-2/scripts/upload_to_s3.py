#!/usr/bin/env python3
"""
Upload fMRI data to S3 bucket.

This script uploads NIfTI-format fMRI files from local storage to an S3 bucket
with progress tracking and error handling.

Usage:
    python upload_to_s3.py --bucket fmri-input-myname --local-path sample_data/
"""

import boto3
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 client
s3 = boto3.client('s3')


def upload_file_to_s3(local_path: str, bucket: str, s3_key: str) -> bool:
    """
    Upload a single file to S3.

    Args:
        local_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key (path in bucket)

    Returns:
        True if successful, False otherwise
    """
    try:
        file_size = os.path.getsize(local_path)
        logger.info(f"Uploading {local_path} ({file_size / 1e9:.2f}GB) to s3://{bucket}/{s3_key}")

        # Use progress callback for large files
        s3.upload_file(
            local_path,
            bucket,
            s3_key,
            Callback=ProgressPercentage(local_path)
        )

        logger.info(f"Successfully uploaded to s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {str(e)}")
        return False


def upload_directory_to_s3(local_dir: str, bucket: str, s3_prefix: str = "") -> dict:
    """
    Upload all NIfTI files from a directory to S3.

    Args:
        local_dir: Local directory path
        bucket: S3 bucket name
        s3_prefix: Optional S3 prefix (path in bucket)

    Returns:
        Dictionary with upload statistics
    """
    local_path = Path(local_dir)

    if not local_path.exists():
        logger.error(f"Directory not found: {local_dir}")
        return {'uploaded': 0, 'failed': 0, 'files': []}

    # Find all NIfTI files
    nifti_files = list(local_path.glob('**/*.nii.gz')) + list(local_path.glob('**/*.nii'))

    if not nifti_files:
        logger.warning(f"No NIfTI files found in {local_dir}")
        return {'uploaded': 0, 'failed': 0, 'files': []}

    logger.info(f"Found {len(nifti_files)} NIfTI files to upload")

    uploaded_files = []
    failed_files = []

    # Upload each file
    for file_path in tqdm(nifti_files, desc="Uploading files"):
        # Construct S3 key
        relative_path = file_path.relative_to(local_path)
        s3_key = str(relative_path) if not s3_prefix else f"{s3_prefix}/{relative_path}"

        # Upload file
        success = upload_file_to_s3(str(file_path), bucket, s3_key)

        if success:
            uploaded_files.append(s3_key)
        else:
            failed_files.append(str(file_path))

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Upload Summary")
    logger.info("="*60)
    logger.info(f"Total files: {len(nifti_files)}")
    logger.info(f"Uploaded: {len(uploaded_files)}")
    logger.info(f"Failed: {len(failed_files)}")

    if failed_files:
        logger.warning("Failed files:")
        for f in failed_files:
            logger.warning(f"  - {f}")

    return {
        'uploaded': len(uploaded_files),
        'failed': len(failed_files),
        'files': uploaded_files,
        'failed_files': failed_files
    }


class ProgressPercentage:
    """
    Callback for S3 upload progress tracking.

    Usage: s3.upload_file(..., Callback=ProgressPercentage(filename))
    """

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        print(f"\r{self._filename}: {percentage:.1f}%", end="")


def verify_upload(bucket: str, s3_key: str, local_path: str) -> bool:
    """
    Verify uploaded file by comparing file size.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key
        local_path: Local file path

    Returns:
        True if sizes match, False otherwise
    """
    try:
        local_size = os.path.getsize(local_path)
        response = s3.head_object(Bucket=bucket, Key=s3_key)
        s3_size = response['ContentLength']

        if local_size == s3_size:
            logger.info(f"Verification passed: {s3_key} ({local_size} bytes)")
            return True
        else:
            logger.error(f"Verification failed: Size mismatch for {s3_key}")
            logger.error(f"  Local: {local_size} bytes, S3: {s3_size} bytes")
            return False

    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return False


def list_s3_contents(bucket: str, prefix: str = "") -> list:
    """
    List all objects in S3 bucket (with optional prefix).

    Args:
        bucket: S3 bucket name
        prefix: Optional prefix to filter

    Returns:
        List of object keys
    """
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            logger.warning(f"No objects found in s3://{bucket}/{prefix}")
            return []

        objects = [obj['Key'] for obj in response['Contents']]
        logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
        return objects

    except Exception as e:
        logger.error(f"Error listing S3 contents: {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Upload fMRI data to S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload single file
  python upload_to_s3.py --bucket fmri-input --local-path sample_fmri.nii.gz

  # Upload directory
  python upload_to_s3.py --bucket fmri-input --local-path sample_data/ --prefix data/

  # List S3 contents
  python upload_to_s3.py --bucket fmri-input --list
        """
    )

    parser.add_argument(
        '--bucket',
        required=False,
        help='S3 bucket name'
    )
    parser.add_argument(
        '--local-path',
        help='Local file or directory path'
    )
    parser.add_argument(
        '--prefix',
        default="",
        help='S3 key prefix (optional)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify upload by checking file size'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List contents of S3 bucket'
    )

    args = parser.parse_args()

    # List S3 contents
    if args.list:
        if not args.bucket:
            logger.error("--bucket required for --list")
            return

        logger.info(f"Listing contents of s3://{args.bucket}/")
        objects = list_s3_contents(args.bucket, args.prefix)

        if objects:
            print("\nObjects in bucket:")
            for obj in objects:
                print(f"  - {obj}")
        return

    # Upload files
    if not args.bucket or not args.local_path:
        parser.print_help()
        logger.error("--bucket and --local-path are required")
        return

    local_path = args.local_path

    # Determine if file or directory
    if os.path.isfile(local_path):
        # Single file upload
        s3_key = args.prefix if args.prefix else os.path.basename(local_path)

        logger.info(f"Uploading single file: {local_path}")
        success = upload_file_to_s3(local_path, args.bucket, s3_key)

        if success and args.verify:
            verify_upload(args.bucket, s3_key, local_path)

    elif os.path.isdir(local_path):
        # Directory upload
        logger.info(f"Uploading directory: {local_path}")
        result = upload_directory_to_s3(local_path, args.bucket, args.prefix)

        if args.verify and result['files']:
            logger.info("Verifying uploads...")
            for s3_key in result['files'][:5]:  # Verify first 5 files as sample
                # Reconstruct local path for verification
                relative_key = s3_key.replace(args.prefix + "/", "") if args.prefix else s3_key
                local_file = os.path.join(local_path, relative_key)

                if os.path.exists(local_file):
                    verify_upload(args.bucket, s3_key, local_file)

    else:
        logger.error(f"Path not found: {local_path}")


if __name__ == '__main__':
    main()
