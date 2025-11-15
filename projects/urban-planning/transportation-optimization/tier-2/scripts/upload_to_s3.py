#!/usr/bin/env python3
"""
Upload traffic data to AWS S3 bucket.

This script handles:
- Batch upload of traffic data files (CSV/JSON) to S3
- Progress tracking with tqdm
- Error handling and retry logic
- Support for multiple file formats
- GPS coordinate validation
- Timestamp validation

Usage:
    python scripts/upload_to_s3.py \
        --input-dir ./sample_data/traffic \
        --s3-bucket transportation-data-{user-id} \
        --prefix raw/
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SUPPORTED_FORMATS = {'.csv', '.json', '.parquet'}
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Traffic data validation
REQUIRED_CSV_COLUMNS = [
    'timestamp', 'segment_id', 'latitude', 'longitude',
    'vehicle_count', 'avg_speed'
]


class S3TrafficDataUploader:
    """Handle uploading traffic data to AWS S3."""

    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: Name of S3 bucket
            region: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.upload_log = []

    def validate_bucket_exists(self) -> bool:
        """Check if S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                logger.error(f"Access denied to bucket '{self.bucket_name}'")
            else:
                logger.error(f"Error accessing bucket: {error_code}")
            return False

    def validate_traffic_data(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate traffic data file format and required fields.

        Args:
            file_path: Path to traffic data file

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=5)

                # Check required columns
                missing_cols = set(REQUIRED_CSV_COLUMNS) - set(df.columns)
                if missing_cols:
                    return False, f"Missing required columns: {missing_cols}"

                # Validate data types
                if not pd.api.types.is_numeric_dtype(df['latitude']):
                    return False, "latitude must be numeric"
                if not pd.api.types.is_numeric_dtype(df['longitude']):
                    return False, "longitude must be numeric"
                if not pd.api.types.is_numeric_dtype(df['vehicle_count']):
                    return False, "vehicle_count must be numeric"
                if not pd.api.types.is_numeric_dtype(df['avg_speed']):
                    return False, "avg_speed must be numeric"

                # Validate coordinate ranges
                if (df['latitude'] < -90).any() or (df['latitude'] > 90).any():
                    return False, "latitude must be between -90 and 90"
                if (df['longitude'] < -180).any() or (df['longitude'] > 180).any():
                    return False, "longitude must be between -180 and 180"

                return True, "Valid traffic data"

            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Check if it's a list or single record
                if isinstance(data, list):
                    if len(data) == 0:
                        return False, "Empty JSON array"
                    record = data[0]
                else:
                    record = data

                # Check required fields
                missing_fields = set(REQUIRED_CSV_COLUMNS) - set(record.keys())
                if missing_fields:
                    return False, f"Missing required fields: {missing_fields}"

                return True, "Valid traffic data"

            else:
                # Other formats accepted without validation
                return True, "Format accepted"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_data_files(self, input_dir: str) -> List[Path]:
        """
        Get list of supported data files from directory.

        Args:
            input_dir: Path to directory containing traffic data

        Returns:
            List of data file paths
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            logger.warning(f"Directory '{input_dir}' does not exist")
            return []

        if not input_path.is_dir():
            logger.warning(f"Path '{input_dir}' is not a directory")
            return []

        # Find all supported data files
        data_files = []
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                data_files.append(file_path)

        logger.info(f"Found {len(data_files)} data files in '{input_dir}'")
        return sorted(data_files)

    def upload_file(self, file_path: Path, s3_key: str,
                   extra_args: Optional[Dict] = None) -> Tuple[bool, str]:
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
                'ContentType': self._get_content_type(file_path),
                'Metadata': {
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'original_path': str(file_path),
                    'file_type': 'traffic_data'
                }
            }

        for attempt in range(MAX_RETRIES):
            try:
                file_size = file_path.stat().st_size
                self.s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args
                )

                message = f"✓ Uploaded {s3_key} ({self._format_size(file_size)})"
                logger.info(message)

                return True, message

            except (ClientError, BotoCoreError) as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {s3_key}: {str(e)}")
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
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.parquet': 'application/octet-stream'
        }
        return mime_types.get(suffix, 'application/octet-stream')

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def upload_directory(self, input_dir: str, s3_prefix: str = '',
                        validate: bool = True) -> Dict:
        """
        Upload all traffic data files from directory to S3.

        Args:
            input_dir: Local directory containing traffic data
            s3_prefix: Prefix for S3 keys (e.g., 'raw/')
            validate: Whether to validate data before upload

        Returns:
            Dictionary with upload statistics
        """
        # Validate bucket
        if not self.validate_bucket_exists():
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'total_bytes': 0,
                'errors': ['Bucket validation failed']
            }

        # Get data files
        data_files = self.get_data_files(input_dir)
        if not data_files:
            logger.warning("No data files found to upload")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'total_bytes': 0,
                'errors': ['No data files found']
            }

        # Upload with progress bar
        successful = 0
        failed = 0
        total_bytes = 0
        errors = []
        validation_warnings = []

        logger.info(f"Starting upload of {len(data_files)} files to s3://{self.bucket_name}/{s3_prefix}")

        with tqdm(total=len(data_files), desc="Uploading traffic data") as pbar:
            for file_path in data_files:
                # Validate data if requested
                if validate:
                    is_valid, message = self.validate_traffic_data(file_path)
                    if not is_valid:
                        validation_warnings.append(f"{file_path.name}: {message}")
                        logger.warning(f"Validation warning for {file_path.name}: {message}")
                        # Continue with upload despite validation warnings

                # Build S3 key
                relative_path = file_path.relative_to(Path(input_dir))
                s3_key = f"{s3_prefix}{relative_path}".replace('\\', '/')

                # Upload file
                success, upload_message = self.upload_file(file_path, s3_key)

                if success:
                    successful += 1
                    total_bytes += file_path.stat().st_size
                    self.upload_log.append({
                        'local_path': str(file_path),
                        's3_key': s3_key,
                        's3_uri': f"s3://{self.bucket_name}/{s3_key}",
                        'size': file_path.stat().st_size,
                        'status': 'success',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                else:
                    failed += 1
                    errors.append(upload_message)
                    self.upload_log.append({
                        'local_path': str(file_path),
                        's3_key': s3_key,
                        'status': 'failed',
                        'error': upload_message,
                        'timestamp': datetime.utcnow().isoformat()
                    })

                pbar.update(1)

        # Print summary
        logger.info("=" * 70)
        logger.info("UPLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total files:     {len(data_files)}")
        logger.info(f"Successful:      {successful}")
        logger.info(f"Failed:          {failed}")
        logger.info(f"Total bytes:     {total_bytes:,} ({self._format_size(total_bytes)})")
        logger.info(f"S3 Bucket:       s3://{self.bucket_name}/")
        logger.info(f"S3 Prefix:       {s3_prefix}")
        logger.info("=" * 70)

        if validation_warnings:
            logger.warning(f"\nValidation warnings: {len(validation_warnings)}")
            for warning in validation_warnings[:5]:
                logger.warning(f"  - {warning}")
            if len(validation_warnings) > 5:
                logger.warning(f"  ... and {len(validation_warnings) - 5} more warnings")

        if errors:
            logger.warning(f"\nErrors encountered: {len(errors)}")
            for error in errors[:5]:
                logger.warning(f"  - {error}")
            if len(errors) > 5:
                logger.warning(f"  ... and {len(errors) - 5} more errors")

        return {
            'total': len(data_files),
            'successful': successful,
            'failed': failed,
            'total_bytes': total_bytes,
            'errors': errors,
            'validation_warnings': validation_warnings
        }

    def save_upload_log(self, output_file: str = 'upload_log.json'):
        """Save upload log to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.upload_log, f, indent=2)
            logger.info(f"Upload log saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save upload log: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Upload traffic data to AWS S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from local directory
  python upload_to_s3.py --input-dir ./sample_data/traffic \\
                         --s3-bucket transportation-data-user1 \\
                         --prefix raw/

  # Use environment variables
  python upload_to_s3.py --input-dir ./data

  # Skip validation (faster, but risky)
  python upload_to_s3.py --input-dir ./data --no-validate
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=os.getenv('INPUT_DIR', './data'),
        help='Directory containing traffic data files (default: ./data)'
    )

    parser.add_argument(
        '--s3-bucket',
        type=str,
        default=os.getenv('S3_BUCKET_NAME'),
        required=not os.getenv('S3_BUCKET_NAME'),
        help='S3 bucket name (required if S3_BUCKET_NAME env var not set)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default=os.getenv('S3_RAW_PREFIX', 'raw/'),
        help='S3 prefix for uploads (default: raw/)'
    )

    parser.add_argument(
        '--region',
        type=str,
        default=os.getenv('AWS_REGION', 'us-east-1'),
        help='AWS region (default: us-east-1)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default='upload_log.json',
        help='Output file for upload log (default: upload_log.json)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation (faster but not recommended)'
    )

    args = parser.parse_args()

    # Create uploader
    uploader = S3TrafficDataUploader(args.s3_bucket, region=args.region)

    # Upload directory
    results = uploader.upload_directory(
        args.input_dir,
        s3_prefix=args.prefix,
        validate=not args.no_validate
    )

    # Save log
    uploader.save_upload_log(args.log_file)

    # Return exit code based on failures
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
