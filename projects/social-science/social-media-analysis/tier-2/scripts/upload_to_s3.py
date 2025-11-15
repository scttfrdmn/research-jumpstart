#!/usr/bin/env python3
"""
Upload social media data (CSV/JSON) to AWS S3 bucket.

This script handles:
- Batch upload of social media posts to S3
- Support for CSV and JSON formats
- Progress tracking with tqdm
- Error handling and retry logic
- Validation of post structure

Usage:
    python scripts/upload_to_s3.py \
        --input-file ./data/tweets.json \
        --s3-bucket social-media-data-{user-id} \
        --prefix raw/
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
MAX_RETRIES = 3
BATCH_SIZE = 100  # Number of posts per file upload


class SocialMediaUploader:
    """Handle uploading social media data to AWS S3."""

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
        self.upload_stats = {
            "total_posts": 0,
            "total_files": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
        }

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

    def load_data(self, file_path: str) -> list[dict[str, Any]]:
        """
        Load social media data from JSON or CSV file.

        Args:
            file_path: Path to input file

        Returns:
            List of post dictionaries
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        logger.info(f"Loading data from: {file_path}")

        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle both single object and array
                    if isinstance(data, dict):
                        data = [data]
                    return data

            elif file_path.suffix.lower() == ".csv":
                posts = []
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        posts.append(row)
                return posts

            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return []

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []

    def validate_post(self, post: dict[str, Any]) -> bool:
        """
        Validate that post has required fields.

        Args:
            post: Post dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["post_id", "text"]
        for field in required_fields:
            if field not in post:
                logger.warning(f"Post missing required field '{field}': {post}")
                return False

        # Ensure text is not empty
        if not post["text"] or not post["text"].strip():
            logger.warning(f"Post has empty text: {post.get('post_id')}")
            return False

        return True

    def upload_batch(self, posts: list[dict[str, Any]], batch_num: int, s3_prefix: str) -> bool:
        """
        Upload batch of posts to S3.

        Args:
            posts: List of post dictionaries
            batch_num: Batch number for naming
            s3_prefix: S3 prefix for uploads

        Returns:
            True if successful, False otherwise
        """
        # Create batch JSON
        batch_data = json.dumps(posts, ensure_ascii=False, indent=2)

        # Generate S3 key
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{s3_prefix}posts_batch_{batch_num:04d}_{timestamp}.json"

        # Upload with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=batch_data.encode("utf-8"),
                    ContentType="application/json",
                    Metadata={
                        "uploaded_at": datetime.utcnow().isoformat(),
                        "batch_number": str(batch_num),
                        "post_count": str(len(posts)),
                    },
                )

                logger.info(f"Uploaded batch {batch_num}: {s3_key} ({len(posts)} posts)")
                self.upload_stats["successful_uploads"] += 1
                return True

            except (ClientError, BotoCoreError) as e:
                logger.warning(f"Upload attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to upload batch {batch_num} after {MAX_RETRIES} attempts")
                    self.upload_stats["failed_uploads"] += 1
                    return False

        return False

    def upload_data(self, file_path: str, s3_prefix: str = "raw/") -> dict[str, Any]:
        """
        Upload social media data to S3 in batches.

        Args:
            file_path: Path to input file
            s3_prefix: S3 prefix for uploads (default: 'raw/')

        Returns:
            Dictionary with upload statistics
        """
        # Validate bucket
        if not self.validate_bucket_exists():
            return self.upload_stats

        # Load data
        posts = self.load_data(file_path)
        if not posts:
            logger.error("No data to upload")
            return self.upload_stats

        logger.info(f"Loaded {len(posts)} posts from {file_path}")

        # Validate posts
        valid_posts = [post for post in posts if self.validate_post(post)]
        invalid_count = len(posts) - len(valid_posts)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid posts")

        if not valid_posts:
            logger.error("No valid posts to upload")
            return self.upload_stats

        self.upload_stats["total_posts"] = len(valid_posts)

        # Upload in batches
        batch_num = 1
        with tqdm(total=len(valid_posts), desc="Uploading posts") as pbar:
            for i in range(0, len(valid_posts), BATCH_SIZE):
                batch = valid_posts[i : i + BATCH_SIZE]
                self.upload_batch(batch, batch_num, s3_prefix)
                self.upload_stats["total_files"] += 1
                batch_num += 1
                pbar.update(len(batch))

        # Print summary
        self._print_summary()

        return self.upload_stats

    def _print_summary(self):
        """Print upload summary statistics."""
        logger.info("=" * 70)
        logger.info("UPLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total posts:        {self.upload_stats['total_posts']}")
        logger.info(f"Total files:        {self.upload_stats['total_files']}")
        logger.info(f"Successful uploads: {self.upload_stats['successful_uploads']}")
        logger.info(f"Failed uploads:     {self.upload_stats['failed_uploads']}")
        logger.info(f"S3 Bucket:          s3://{self.bucket_name}/")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload social media data to AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload JSON file
  python upload_to_s3.py --input-file tweets.json \\
                         --s3-bucket social-media-data-user1

  # Upload CSV file with custom prefix
  python upload_to_s3.py --input-file posts.csv \\
                         --s3-bucket social-media-data-user1 \\
                         --prefix raw-data/
        """,
    )

    parser.add_argument(
        "--input-file", type=str, required=True, help="Input file (JSON or CSV format)"
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("BUCKET_NAME"),
        required=not os.getenv("BUCKET_NAME"),
        help="S3 bucket name (required if BUCKET_NAME env var not set)",
    )

    parser.add_argument(
        "--prefix", type=str, default="raw/", help="S3 prefix for uploads (default: raw/)"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    args = parser.parse_args()

    # Create uploader
    uploader = SocialMediaUploader(args.s3_bucket, region=args.region)

    # Upload data
    stats = uploader.upload_data(args.input_file, s3_prefix=args.prefix)

    # Return exit code based on failures
    sys.exit(0 if stats["failed_uploads"] == 0 else 1)


if __name__ == "__main__":
    main()
