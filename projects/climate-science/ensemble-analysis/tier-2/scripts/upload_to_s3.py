#!/usr/bin/env python3
"""
Upload CMIP6 climate data to S3 bucket.

This script handles uploading netCDF files to AWS S3 for processing.
Supports resumable uploads and progress tracking.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3Uploader:
    """Upload files to S3 with progress tracking."""

    def __init__(self, bucket_name, region="us-east-1", profile=None):
        """
        Initialize S3 uploader.

        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
            profile (str): AWS profile name
        """
        self.bucket_name = bucket_name
        self.region = region

        # Create session and S3 client
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()

        self.s3 = session.client("s3", region_name=region)

        # Verify bucket exists
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"✓ Connected to bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"✗ Cannot access bucket: {bucket_name}")
            logger.error(f"  Error: {e}")
            raise

    def upload_file(self, file_path, s3_key, multipart_threshold=100 * 1024 * 1024):
        """
        Upload file to S3 with multipart upload for large files.

        Args:
            file_path (str): Local file path
            s3_key (str): S3 object key (path in bucket)
            multipart_threshold (int): Use multipart for files > threshold

        Returns:
            bool: Success status
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"✗ File not found: {file_path}")
            return False

        file_size = file_path.stat().st_size
        logger.info(f"Uploading: {file_path.name} ({file_size / 1e9:.2f}GB)")

        try:
            # Use multipart upload for large files
            if file_size > multipart_threshold:
                self._multipart_upload(file_path, s3_key, file_size)
            else:
                self._simple_upload(file_path, s3_key)

            logger.info(f"✓ Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"✗ Upload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False

    def _simple_upload(self, file_path, s3_key):
        """Simple upload for small files."""
        self.s3.upload_file(str(file_path), self.bucket_name, s3_key)

    def _multipart_upload(self, file_path, s3_key, file_size):
        """Multipart upload for large files with progress bar."""
        part_size = 50 * 1024 * 1024  # 50MB parts

        # Initiate multipart upload
        mpu = self.s3.create_multipart_upload(Bucket=self.bucket_name, Key=s3_key)
        upload_id = mpu["UploadId"]

        try:
            parts = []
            with open(file_path, "rb") as f:
                part_num = 1
                with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                    while True:
                        data = f.read(part_size)
                        if not data:
                            break

                        response = self.s3.upload_part(
                            Body=data,
                            Bucket=self.bucket_name,
                            Key=s3_key,
                            PartNumber=part_num,
                            UploadId=upload_id,
                        )

                        parts.append({"ETag": response["ETag"], "PartNumber": part_num})

                        pbar.update(len(data))
                        part_num += 1

            # Complete multipart upload
            self.s3.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

        except Exception:
            # Abort upload on error
            self.s3.abort_multipart_upload(Bucket=self.bucket_name, Key=s3_key, UploadId=upload_id)
            raise

    def upload_directory(self, local_dir, s3_prefix="raw/"):
        """
        Upload all files in directory to S3.

        Args:
            local_dir (str): Local directory path
            s3_prefix (str): S3 folder prefix

        Returns:
            tuple: (successful_uploads, failed_uploads)
        """
        local_dir = Path(local_dir)

        if not local_dir.exists():
            logger.error(f"✗ Directory not found: {local_dir}")
            return 0, 0

        # Find all netCDF and data files
        files = (
            list(local_dir.glob("**/*.nc"))
            + list(local_dir.glob("**/*.nc4"))
            + list(local_dir.glob("**/*.zarr"))
        )

        if not files:
            logger.warning(f"✗ No data files found in {local_dir}")
            return 0, 0

        logger.info(f"Found {len(files)} files to upload")

        successful = 0
        failed = 0

        for file_path in files:
            # Create S3 key maintaining directory structure
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}".replace(os.sep, "/")

            if self.upload_file(str(file_path), s3_key):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def list_uploaded_files(self, prefix="raw/"):
        """List all uploaded files in S3."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" not in response:
                logger.info(f"No files found in s3://{self.bucket_name}/{prefix}")
                return []

            files = []
            for obj in response["Contents"]:
                size_gb = obj["Size"] / 1e9
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "size_gb": size_gb,
                        "modified": obj["LastModified"],
                    }
                )

            logger.info(f"\nUploaded files in {prefix}:")
            total_size = 0
            for f in files:
                logger.info(f"  {f['key']} ({f['size_gb']:.3f}GB)")
                total_size += f["size"]

            logger.info(f"Total: {total_size / 1e9:.3f}GB")
            return files

        except ClientError as e:
            logger.error(f"✗ Failed to list files: {e}")
            return []


def upload_climate_data(bucket_name, data_dir="sample_data", region="us-east-1"):
    """
    Upload climate data from local directory to S3.

    Args:
        bucket_name (str): S3 bucket name
        data_dir (str): Local directory with climate data
        region (str): AWS region

    Returns:
        bool: Success status
    """
    uploader = S3Uploader(bucket_name, region)
    successful, failed = uploader.upload_directory(data_dir)

    logger.info(f"\n✓ Upload complete: {successful} successful, {failed} failed")
    uploader.list_uploaded_files()

    return failed == 0


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Upload climate data to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--data-dir", default="sample_data", help="Local directory with data files")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--file", help="Upload single file instead of directory")
    parser.add_argument("--s3-key", default="raw/", help="S3 key/prefix for uploaded files")
    parser.add_argument(
        "--list-only", action="store_true", help="Only list files without uploading"
    )

    args = parser.parse_args()

    try:
        uploader = S3Uploader(args.bucket, args.region, args.profile)

        if args.list_only:
            uploader.list_uploaded_files(args.s3_key)
        elif args.file:
            # Upload single file
            uploader.upload_file(args.file, args.s3_key)
            uploader.list_uploaded_files(args.s3_key)
        else:
            # Upload directory
            successful, failed = uploader.upload_directory(args.data_dir, args.s3_key)
            logger.info(f"\n✓ Upload complete: {successful} successful, {failed} failed")
            uploader.list_uploaded_files(args.s3_key)

            sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
