"""
Upload disease case report data to AWS S3.

This script:
1. Validates case report data format and quality
2. Anonymizes sensitive information (PII)
3. Uploads data to S3 bucket
4. Triggers Lambda processing via S3 event notification
5. Tracks upload progress and handles errors

Usage:
    python upload_to_s3.py --input-file case_reports.csv --s3-bucket epidemiology-data-123456

Author: Research Jumpstart
"""

import argparse
import os
import sys
import csv
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
s3_client = boto3.client('s3')

# Required columns for case reports
REQUIRED_COLUMNS = [
    'case_id', 'disease', 'report_date', 'region',
    'age_group', 'sex', 'outcome'
]

# Valid values for categorical fields
VALID_VALUES = {
    'age_group': ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+', 'unknown'],
    'sex': ['M', 'F', 'other', 'unknown'],
    'outcome': ['recovered', 'hospitalized', 'icu', 'fatal', 'active', 'unknown']
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload disease case report data to S3'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input CSV file with case data'
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        required=True,
        help='S3 bucket name for upload'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='case-data/',
        help='S3 prefix (folder) for uploaded files (default: case-data/)'
    )
    parser.add_argument(
        '--anonymize',
        action='store_true',
        help='Anonymize sensitive case IDs before upload'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without uploading'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Save processed data to local file before upload'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate case report data.

    Args:
        df: DataFrame with case data

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("No data rows found")
        return False, errors

    # Validate case_id uniqueness
    if df['case_id'].duplicated().any():
        dup_count = df['case_id'].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate case IDs")

    # Validate date format
    try:
        pd.to_datetime(df['report_date'], errors='coerce')
        invalid_dates = df['report_date'].isna().sum()
        if invalid_dates > 0:
            errors.append(f"Found {invalid_dates} invalid date formats (expected YYYY-MM-DD)")
    except Exception as e:
        errors.append(f"Date validation error: {str(e)}")

    # Validate categorical fields
    for field, valid_values in VALID_VALUES.items():
        if field in df.columns:
            invalid = ~df[field].isin(valid_values)
            invalid_count = invalid.sum()
            if invalid_count > 0:
                errors.append(
                    f"Found {invalid_count} invalid values in '{field}'. "
                    f"Valid values: {valid_values}"
                )

    # Check for missing required fields
    for col in REQUIRED_COLUMNS:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            errors.append(f"Found {missing_count} missing values in required column '{col}'")

    # Validate disease names (should not be empty)
    if df['disease'].str.strip().eq('').any():
        errors.append("Found empty disease names")

    # Validate region format (should not be empty)
    if df['region'].str.strip().eq('').any():
        errors.append("Found empty region values")

    is_valid = len(errors) == 0
    return is_valid, errors


def anonymize_case_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymize case IDs using SHA-256 hashing.

    Args:
        df: DataFrame with case data

    Returns:
        DataFrame with anonymized case IDs
    """
    logger.info("Anonymizing case IDs...")

    df = df.copy()

    # Hash case IDs
    df['case_id'] = df['case_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )

    # Remove any PII columns if present (common PII fields)
    pii_columns = ['name', 'ssn', 'address', 'phone', 'email', 'dob']
    existing_pii = [col for col in pii_columns if col in df.columns]

    if existing_pii:
        logger.warning(f"Removing PII columns: {existing_pii}")
        df = df.drop(columns=existing_pii)

    # Anonymize zip codes (keep only first 3 digits)
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str[:3] + 'XX'

    logger.info(f"Anonymized {len(df)} case records")
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add metadata columns for tracking.

    Args:
        df: DataFrame with case data

    Returns:
        DataFrame with metadata columns added
    """
    df = df.copy()

    # Add upload timestamp
    df['upload_timestamp'] = datetime.utcnow().isoformat()

    # Add data version
    df['data_version'] = '1.0'

    return df


def check_bucket_exists(bucket_name: str) -> bool:
    """
    Check if S3 bucket exists and is accessible.

    Args:
        bucket_name: Name of S3 bucket

    Returns:
        True if bucket exists and is accessible
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error(f"Bucket '{bucket_name}' does not exist")
        elif error_code == '403':
            logger.error(f"Access denied to bucket '{bucket_name}'")
        else:
            logger.error(f"Error checking bucket: {str(e)}")
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found. Run 'aws configure' first.")
        return False


def upload_to_s3(
    df: pd.DataFrame,
    bucket_name: str,
    prefix: str,
    output_filename: Optional[str] = None
) -> bool:
    """
    Upload data to S3 bucket.

    Args:
        df: DataFrame to upload
        bucket_name: S3 bucket name
        prefix: S3 key prefix
        output_filename: Optional local filename to save before upload

    Returns:
        True if upload successful
    """
    try:
        # Generate S3 key
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{prefix}case_reports_{timestamp}.csv"

        # Convert to CSV
        csv_buffer = df.to_csv(index=False)

        # Save locally if requested
        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(csv_buffer)
            logger.info(f"Saved processed data to: {output_filename}")

        # Upload to S3
        logger.info(f"Uploading to s3://{bucket_name}/{s3_key}")

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.encode('utf-8'),
            ContentType='text/csv',
            Metadata={
                'upload-timestamp': datetime.utcnow().isoformat(),
                'record-count': str(len(df)),
                'data-type': 'case-reports'
            }
        )

        logger.info(f"✓ Upload successful: s3://{bucket_name}/{s3_key}")
        logger.info(f"✓ Uploaded {len(df)} case records")

        return True

    except ClientError as e:
        logger.error(f"S3 upload error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        return False


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=== Disease Case Data Upload ===")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"S3 bucket: {args.s3_bucket}")
    logger.info(f"S3 prefix: {args.prefix}")

    # Check input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Load data
    logger.info("Loading case data...")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} records from {args.input_file}")
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        sys.exit(1)

    # Validate data
    logger.info("Validating data...")
    is_valid, errors = validate_data(df)

    if not is_valid:
        logger.error("❌ Data validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("✓ Data validation passed")

    # Anonymize if requested
    if args.anonymize:
        df = anonymize_case_ids(df)

    # Add metadata
    df = add_metadata(df)

    # If validate-only mode, stop here
    if args.validate_only:
        logger.info("✓ Validation complete (--validate-only mode)")
        sys.exit(0)

    # Check bucket exists
    logger.info("Checking S3 bucket access...")
    if not check_bucket_exists(args.s3_bucket):
        logger.error("Cannot access S3 bucket. Upload aborted.")
        sys.exit(1)

    logger.info("✓ S3 bucket accessible")

    # Upload to S3
    success = upload_to_s3(
        df=df,
        bucket_name=args.s3_bucket,
        prefix=args.prefix,
        output_filename=args.output_file
    )

    if success:
        logger.info("=== Upload Complete ===")
        logger.info("Lambda function will automatically process the uploaded data.")
        logger.info("Check CloudWatch logs for processing status:")
        logger.info("  aws logs tail /aws/lambda/analyze-disease-surveillance --follow")
        sys.exit(0)
    else:
        logger.error("❌ Upload failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
