#!/usr/bin/env python3
"""
Upload chest X-ray images to AWS S3 bucket.

This script handles:
- Batch upload of image files to S3
- Progress tracking with tqdm
- Error handling and retry logic
- Support for multiple image formats (PNG, JPG, DICOM)

Usage:
    python scripts/upload_to_s3.py \
        --input-dir ./sample_data/sample_xrays \
        --s3-bucket medical-images-{user-id} \
        --prefix raw-images/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".dcm", ".nii"}
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


class S3ImageUploader:
    """Handle uploading medical images to AWS S3."""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: Name of S3 bucket
            region: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client("s3", region_name=region)
        self.upload_log = []

    def validate_bucket_exists(self) -> bool:
        """Check if S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"Bucket '{self.bucket_name}' does not exist")
            elif error_code == "403":
                logger.error(f"Access denied to bucket '{self.bucket_name}'")
            else:
                logger.error(f"Error accessing bucket: {error_code}")
            return False

    def get_image_files(self, input_dir: str) -> list[Path]:
        """
        Get list of supported image files from directory.

        Args:
            input_dir: Path to directory containing images

        Returns:
            List of image file paths
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            logger.warning(f"Directory '{input_dir}' does not exist")
            return []

        if not input_path.is_dir():
            logger.warning(f"Path '{input_dir}' is not a directory")
            return []

        # Find all supported image files
        image_files = []
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                image_files.append(file_path)

        logger.info(f"Found {len(image_files)} image files in '{input_dir}'")
        return sorted(image_files)

    def upload_file(
        self, file_path: Path, s3_key: str, extra_args: Optional[dict] = None
    ) -> tuple[bool, str]:
        """
        Upload single file to S3 with retry logic.

        Args:
            file_path: Path to local file
            s3_key: S3 object key (path in bucket)
            extra_args: Additional arguments for upload

        Returns:
            Tuple of (success: bool, message: str)
        """
        if extra_args is None:
            extra_args = {
                "ContentType": self._get_content_type(file_path),
                "Metadata": {
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "original_path": str(file_path),
                },
            }

        for attempt in range(MAX_RETRIES):
            try:
                file_size = file_path.stat().st_size
                self.s3_client.upload_file(
                    str(file_path), self.bucket_name, s3_key, ExtraArgs=extra_args
                )

                message = f"✓ Uploaded {s3_key} ({file_size} bytes)"
                logger.info(message)

                return True, message

            except (ClientError, BotoCoreError) as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {s3_key}: {e!s}")
                if attempt == MAX_RETRIES - 1:
                    error_msg = f"✗ Failed to upload {s3_key} after {MAX_RETRIES} attempts"
                    logger.error(error_msg)
                    return False, error_msg

        return False, f"✗ Unknown error uploading {s3_key}"

    @staticmethod
    def _get_content_type(file_path: Path) -> str:
        """Get MIME type for file."""
        suffix = file_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".dcm": "application/octet-stream",
            ".nii": "application/octet-stream",
        }
        return mime_types.get(suffix, "application/octet-stream")

    def upload_directory(self, input_dir: str, s3_prefix: str = "") -> dict:
        """
        Upload all images from directory to S3.

        Args:
            input_dir: Local directory containing images
            s3_prefix: Prefix for S3 keys (e.g., 'raw-images/')

        Returns:
            Dictionary with upload statistics
        """
        # Validate bucket
        if not self.validate_bucket_exists():
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_bytes": 0,
                "errors": ["Bucket validation failed"],
            }

        # Get image files
        image_files = self.get_image_files(input_dir)
        if not image_files:
            logger.warning("No image files found to upload")
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_bytes": 0,
                "errors": ["No image files found"],
            }

        # Upload with progress bar
        successful = 0
        failed = 0
        total_bytes = 0
        errors = []

        logger.info(
            f"Starting upload of {len(image_files)} images to s3://{self.bucket_name}/{s3_prefix}"
        )

        with tqdm(total=len(image_files), desc="Uploading images") as pbar:
            for file_path in image_files:
                # Build S3 key
                relative_path = file_path.relative_to(Path(input_dir))
                s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")

                # Upload file
                success, message = self.upload_file(file_path, s3_key)

                if success:
                    successful += 1
                    total_bytes += file_path.stat().st_size
                    self.upload_log.append(
                        {
                            "local_path": str(file_path),
                            "s3_key": s3_key,
                            "size": file_path.stat().st_size,
                            "status": "success",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                else:
                    failed += 1
                    errors.append(message)
                    self.upload_log.append(
                        {
                            "local_path": str(file_path),
                            "s3_key": s3_key,
                            "status": "failed",
                            "error": message,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                pbar.update(1)

        # Print summary
        logger.info("=" * 70)
        logger.info("UPLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total files:     {len(image_files)}")
        logger.info(f"Successful:      {successful}")
        logger.info(f"Failed:          {failed}")
        logger.info(f"Total bytes:     {total_bytes:,} ({total_bytes / 1e6:.2f} MB)")
        logger.info(f"S3 Bucket:       s3://{self.bucket_name}/")
        logger.info(f"S3 Prefix:       {s3_prefix}")
        logger.info("=" * 70)

        if errors:
            logger.warning(f"Errors encountered: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")

        return {
            "total": len(image_files),
            "successful": successful,
            "failed": failed,
            "total_bytes": total_bytes,
            "errors": errors,
        }

    def save_upload_log(self, output_file: str = "upload_log.json"):
        """Save upload log to JSON file."""
        try:
            with open(output_file, "w") as f:
                json.dump(self.upload_log, f, indent=2)
            logger.info(f"Upload log saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save upload log: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload medical images to AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from local directory
  python upload_to_s3.py --input-dir ./sample_data/xrays \\
                         --s3-bucket medical-images-user1 \\
                         --prefix raw-images/

  # Use environment variables
  python upload_to_s3.py --input-dir ./data
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.getenv("INPUT_DIR", "./data"),
        help="Directory containing medical images (default: ./data)",
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("S3_BUCKET_NAME"),
        required=not os.getenv("S3_BUCKET_NAME"),
        help="S3 bucket name (required if S3_BUCKET_NAME env var not set)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=os.getenv("S3_RAW_PREFIX", "raw-images/"),
        help="S3 prefix for uploads (default: raw-images/)",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="upload_log.json",
        help="Output file for upload log (default: upload_log.json)",
    )

    args = parser.parse_args()

    # Create uploader
    uploader = S3ImageUploader(args.s3_bucket, region=args.region)

    # Upload directory
    results = uploader.upload_directory(args.input_dir, s3_prefix=args.prefix)

    # Save log
    uploader.save_upload_log(args.log_file)

    # Return exit code based on failures
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
