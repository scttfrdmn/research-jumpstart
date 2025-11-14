#!/usr/bin/env python3
"""
Download and query processed fMRI results from S3.

This script downloads processed fMRI data from S3 and loads it for analysis
in a local Jupyter notebook or Python environment.

Usage:
    python query_results.py --output-bucket fmri-output-myname --local-path results/
"""

import boto3
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 client
s3 = boto3.client('s3')


def download_file_from_s3(bucket: str, s3_key: str, local_path: str) -> bool:
    """
    Download a single file from S3.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key
        local_path: Local destination path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get file size
        response = s3.head_object(Bucket=bucket, Key=s3_key)
        file_size = response['ContentLength']

        logger.info(f"Downloading s3://{bucket}/{s3_key} ({file_size / 1e9:.2f}GB)")

        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download with progress
        s3.download_file(
            bucket,
            s3_key,
            local_path,
            Callback=ProgressPercentage(s3_key, file_size)
        )

        print()  # New line after progress bar
        logger.info(f"Successfully downloaded to {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {s3_key}: {str(e)}")
        return False


def download_directory_from_s3(bucket: str, prefix: str, local_dir: str) -> dict:
    """
    Download all files from S3 prefix to local directory.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to download
        local_dir: Local destination directory

    Returns:
        Dictionary with download statistics
    """
    try:
        # List all objects with prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        objects = []
        for page in pages:
            if 'Contents' in page:
                objects.extend([obj['Key'] for obj in page['Contents']])

        if not objects:
            logger.warning(f"No objects found with prefix: {prefix}")
            return {'downloaded': 0, 'failed': 0, 'files': []}

        logger.info(f"Found {len(objects)} objects to download")

        downloaded_files = []
        failed_files = []

        # Download each file
        for s3_key in tqdm(objects, desc="Downloading files"):
            # Construct local path
            relative_path = s3_key[len(prefix):].lstrip('/')
            local_path = os.path.join(local_dir, relative_path)

            # Download file
            success = download_file_from_s3(bucket, s3_key, local_path)

            if success:
                downloaded_files.append(local_path)
            else:
                failed_files.append(s3_key)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Download Summary")
        logger.info("="*60)
        logger.info(f"Total files: {len(objects)}")
        logger.info(f"Downloaded: {len(downloaded_files)}")
        logger.info(f"Failed: {len(failed_files)}")

        if failed_files:
            logger.warning("Failed files:")
            for f in failed_files:
                logger.warning(f"  - {f}")

        return {
            'downloaded': len(downloaded_files),
            'failed': len(failed_files),
            'files': downloaded_files,
            'failed_files': failed_files
        }

    except Exception as e:
        logger.error(f"Error downloading directory: {str(e)}")
        return {'downloaded': 0, 'failed': 0, 'files': []}


class ProgressPercentage:
    """Callback for S3 download progress tracking."""

    def __init__(self, filename, size):
        self._filename = Path(filename).name
        self._size = float(size)
        self._seen_so_far = 0

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        print(f"\r{self._filename}: {percentage:.1f}%", end="")


def list_s3_results(bucket: str, prefix: str = "") -> dict:
    """
    List all processed results in S3 bucket.

    Args:
        bucket: S3 bucket name
        prefix: Optional prefix to filter

    Returns:
        Dictionary with organized results by type
    """
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        motion_corrected = []
        smoothed = []
        other = []

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                size = obj['Size']

                if 'motion_corrected' in key or 'motion-corrected' in key:
                    motion_corrected.append({'key': key, 'size': size})
                elif 'smoothed' in key:
                    smoothed.append({'key': key, 'size': size})
                else:
                    other.append({'key': key, 'size': size})

        logger.info(f"\nResults in s3://{bucket}/{prefix}:")
        logger.info(f"  Motion-corrected: {len(motion_corrected)}")
        logger.info(f"  Smoothed: {len(smoothed)}")
        logger.info(f"  Other: {len(other)}")

        return {
            'motion_corrected': motion_corrected,
            'smoothed': smoothed,
            'other': other
        }

    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        return {'motion_corrected': [], 'smoothed': [], 'other': []}


def get_result_metadata(bucket: str, s3_key: str) -> dict:
    """
    Get metadata about an S3 object.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key

    Returns:
        Dictionary with metadata
    """
    try:
        response = s3.head_object(Bucket=bucket, Key=s3_key)

        return {
            'key': s3_key,
            'size': response['ContentLength'],
            'last_modified': str(response['LastModified']),
            'content_type': response.get('ContentType', 'unknown'),
            'etag': response.get('ETag', 'unknown')
        }

    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return {}


def verify_download(local_path: str, bucket: str, s3_key: str) -> bool:
    """
    Verify downloaded file by comparing file size.

    Args:
        local_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key

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
            logger.error(f"  S3: {s3_size} bytes, Local: {local_size} bytes")
            return False

    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return False


def load_fmri_data(local_path: str):
    """
    Load fMRI NIfTI file for analysis.

    Args:
        local_path: Path to NIfTI file

    Returns:
        Tuple of (data, affine) or None if error
    """
    try:
        import nibabel as nib
        import numpy as np

        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return None

        logger.info(f"Loading fMRI data from {local_path}")

        img = nib.load(local_path)
        data = img.get_fdata()
        affine = img.affine

        logger.info(f"Loaded shape: {data.shape}, dtype: {data.dtype}")
        logger.info(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")

        return {'data': data, 'affine': affine, 'header': img.header}

    except ImportError:
        logger.error("nibabel not installed. Install with: pip install nibabel")
        return None
    except Exception as e:
        logger.error(f"Error loading fMRI data: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and query processed fMRI results from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all results
  python query_results.py --output-bucket fmri-output --local-path results/

  # Download specific file type (smoothed)
  python query_results.py --output-bucket fmri-output --local-path results/ --filter smoothed

  # List results without downloading
  python query_results.py --output-bucket fmri-output --list

  # Verify downloaded files
  python query_results.py --output-bucket fmri-output --local-path results/ --verify
        """
    )

    parser.add_argument(
        '--output-bucket',
        required=False,
        help='S3 output bucket name'
    )
    parser.add_argument(
        '--local-path',
        default='results',
        help='Local directory to download to (default: results/)'
    )
    parser.add_argument(
        '--prefix',
        default="",
        help='S3 prefix to filter downloads'
    )
    parser.add_argument(
        '--filter',
        choices=['motion_corrected', 'smoothed', 'all'],
        default='all',
        help='Filter by result type'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List results without downloading'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify downloaded files'
    )
    parser.add_argument(
        '--load',
        help='Load and inspect a specific NIfTI file'
    )

    args = parser.parse_args()

    # Load and inspect file
    if args.load:
        logger.info(f"Loading fMRI file: {args.load}")
        fmri_data = load_fmri_data(args.load)

        if fmri_data:
            print(json.dumps({
                'path': args.load,
                'shape': str(fmri_data['data'].shape),
                'dtype': str(fmri_data['data'].dtype),
                'affine': fmri_data['affine'].tolist()
            }, indent=2))
        return

    # List results
    if args.list:
        if not args.output_bucket:
            logger.error("--output-bucket required for --list")
            return

        logger.info(f"Listing results in s3://{args.output_bucket}/")
        results = list_s3_results(args.output_bucket, args.prefix)

        print(json.dumps({
            'motion_corrected': len(results['motion_corrected']),
            'smoothed': len(results['smoothed']),
            'other': len(results['other'])
        }, indent=2))
        return

    # Download results
    if not args.output_bucket:
        parser.print_help()
        logger.error("--output-bucket is required")
        return

    logger.info(f"Downloading results from s3://{args.output_bucket}/")
    result = download_directory_from_s3(args.output_bucket, args.prefix, args.local_path)

    if args.verify and result['files']:
        logger.info("Verifying downloads...")
        for local_path in result['files'][:5]:  # Verify first 5 files
            logger.info(f"Verifying {local_path}")


if __name__ == '__main__':
    main()
