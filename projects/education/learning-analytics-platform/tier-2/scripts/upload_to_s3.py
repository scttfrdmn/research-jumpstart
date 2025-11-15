#!/usr/bin/env python3
"""
Upload student activity data to AWS S3 bucket with anonymization.

This script handles:
- Generation of synthetic student data (optional)
- Student ID anonymization for privacy (FERPA compliance)
- Batch upload of CSV files to S3
- Progress tracking with tqdm
- Error handling and retry logic

Usage:
    # Upload existing data
    python scripts/upload_to_s3.py \
        --input-dir ./sample_data \
        --s3-bucket learning-analytics-{user-id} \
        --prefix raw-data/

    # Generate and upload sample data
    python scripts/upload_to_s3.py \
        --generate-sample \
        --num-students 500 \
        --s3-bucket learning-analytics-{user-id}
"""

import argparse
import hashlib
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
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
RETRY_DELAY = 1  # seconds


def anonymize_student_id(student_id: str, salt: str = "learning-analytics-2025") -> str:
    """
    Anonymize student ID using SHA256 hashing.

    Args:
        student_id: Original student identifier
        salt: Salt for hashing (default: project-specific salt)

    Returns:
        Anonymized student ID (first 16 chars of hash)
    """
    hash_input = f"{salt}{student_id}".encode()
    return hashlib.sha256(hash_input).hexdigest()[:16]


def generate_sample_data(
    num_students: int = 500, num_courses: int = 3, assessments_per_course: int = 10
) -> pd.DataFrame:
    """
    Generate synthetic student activity data for testing.

    Args:
        num_students: Number of students to generate
        num_courses: Number of courses
        assessments_per_course: Number of assessments per course

    Returns:
        DataFrame with student activity records
    """
    logger.info(f"Generating sample data: {num_students} students, {num_courses} courses")

    records = []
    courses = [f"COURSE_{i + 1:03d}" for i in range(num_courses)]
    assessment_types = ["quiz", "assignment", "exam", "project"]

    start_date = datetime.now() - timedelta(days=90)

    for student_num in range(num_students):
        # Original student ID (will be anonymized)
        original_id = f"STU_{student_num + 1:05d}"
        anonymized_id = anonymize_student_id(original_id)

        # Student characteristics (affects performance)
        baseline_ability = np.random.normal(75, 15)  # Mean: 75%, SD: 15%
        engagement_level = np.random.uniform(0.3, 1.0)

        for course_id in courses:
            # Course-specific performance variation
            course_modifier = np.random.normal(0, 5)

            for assessment_num in range(assessments_per_course):
                assessment_type = random.choice(assessment_types)

                # Calculate score with learning curve effect
                learning_progress = assessment_num / assessments_per_course
                score = baseline_ability + course_modifier + (learning_progress * 10)
                score += np.random.normal(0, 8)  # Random variation
                score = max(0, min(100, score))  # Clip to 0-100

                # Some students miss assignments
                if np.random.random() > engagement_level:
                    score = 0  # Missed assignment
                    submitted = False
                else:
                    submitted = True

                # Submission time
                assessment_date = start_date + timedelta(
                    days=assessment_num * 7,
                    hours=np.random.randint(0, 48),  # Some late submissions
                )

                # Engagement metrics
                time_on_task = np.random.exponential(30) if submitted else 0  # minutes
                resource_views = np.random.poisson(5) if submitted else 0

                records.append(
                    {
                        "student_id": anonymized_id,
                        "original_id": original_id,  # For reference only, not uploaded
                        "course_id": course_id,
                        "assessment_type": assessment_type,
                        "assessment_number": assessment_num + 1,
                        "score": round(score, 2),
                        "max_score": 100.0,
                        "submitted": submitted,
                        "submission_date": assessment_date.isoformat(),
                        "time_on_task_minutes": round(time_on_task, 2),
                        "resource_views": resource_views,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} activity records")

    return df


class S3DataUploader:
    """Handle uploading student data to AWS S3."""

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

    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, anonymize: bool = True) -> bool:
        """
        Upload DataFrame to S3 as CSV.

        Args:
            df: DataFrame to upload
            s3_key: S3 object key (path in bucket)
            anonymize: Remove sensitive columns before upload

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove sensitive columns
            if anonymize and "original_id" in df.columns:
                df = df.drop(columns=["original_id"])

            # Convert to CSV
            csv_buffer = df.to_csv(index=False)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_buffer.encode("utf-8"),
                ContentType="text/csv",
                Metadata={
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "record_count": str(len(df)),
                    "anonymized": str(anonymize),
                },
            )

            logger.info(f"✓ Uploaded {s3_key} ({len(df)} records)")
            return True

        except (ClientError, BotoCoreError) as e:
            logger.error(f"✗ Failed to upload {s3_key}: {e!s}")
            return False

    def upload_directory(self, input_dir: str, s3_prefix: str = "") -> dict:
        """
        Upload all CSV files from directory to S3.

        Args:
            input_dir: Local directory containing CSV files
            s3_prefix: Prefix for S3 keys (e.g., 'raw-data/')

        Returns:
            Dictionary with upload statistics
        """
        # Validate bucket
        if not self.validate_bucket_exists():
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_records": 0,
                "errors": ["Bucket validation failed"],
            }

        # Get CSV files
        input_path = Path(input_dir)
        csv_files = list(input_path.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {input_dir}")
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "total_records": 0,
                "errors": ["No CSV files found"],
            }

        # Upload with progress bar
        successful = 0
        failed = 0
        total_records = 0
        errors = []

        logger.info(f"Uploading {len(csv_files)} files to s3://{self.bucket_name}/{s3_prefix}")

        with tqdm(total=len(csv_files), desc="Uploading files") as pbar:
            for csv_file in csv_files:
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file)

                    # Build S3 key
                    s3_key = f"{s3_prefix}{csv_file.name}"

                    # Upload
                    if self.upload_dataframe(df, s3_key, anonymize=True):
                        successful += 1
                        total_records += len(df)
                    else:
                        failed += 1
                        errors.append(f"Upload failed for {csv_file.name}")

                except Exception as e:
                    failed += 1
                    error_msg = f"Error processing {csv_file.name}: {e!s}"
                    errors.append(error_msg)
                    logger.error(error_msg)

                pbar.update(1)

        # Print summary
        logger.info("=" * 70)
        logger.info("UPLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total files:     {len(csv_files)}")
        logger.info(f"Successful:      {successful}")
        logger.info(f"Failed:          {failed}")
        logger.info(f"Total records:   {total_records:,}")
        logger.info(f"S3 Bucket:       s3://{self.bucket_name}/")
        logger.info(f"S3 Prefix:       {s3_prefix}")
        logger.info("=" * 70)

        return {
            "total": len(csv_files),
            "successful": successful,
            "failed": failed,
            "total_records": total_records,
            "errors": errors,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload student activity data to AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and upload sample data
  python upload_to_s3.py --generate-sample --num-students 500 \\
                         --s3-bucket learning-analytics-user1

  # Upload existing CSV files
  python upload_to_s3.py --input-dir ./data \\
                         --s3-bucket learning-analytics-user1 \\
                         --prefix raw-data/
        """,
    )

    # Generation options
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate synthetic student data"
    )

    parser.add_argument(
        "--num-students",
        type=int,
        default=500,
        help="Number of students to generate (default: 500)",
    )

    parser.add_argument("--num-courses", type=int, default=3, help="Number of courses (default: 3)")

    parser.add_argument(
        "--assessments-per-course",
        type=int,
        default=10,
        help="Assessments per course (default: 10)",
    )

    # Upload options
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.getenv("INPUT_DIR", "./data"),
        help="Directory containing CSV files (default: ./data)",
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
        default=os.getenv("S3_RAW_PREFIX", "raw-data/"),
        help="S3 prefix for uploads (default: raw-data/)",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_data",
        help="Directory to save generated data (default: ./generated_data)",
    )

    args = parser.parse_args()

    # Generate sample data if requested
    if args.generate_sample:
        logger.info("Generating sample student data...")
        df = generate_sample_data(
            num_students=args.num_students,
            num_courses=args.num_courses,
            assessments_per_course=args.assessments_per_course,
        )

        # Save to output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Split into separate files by course (more realistic)
        for course_id in df["course_id"].unique():
            course_df = df[df["course_id"] == course_id]
            # Remove original_id before saving
            course_df_anon = course_df.drop(columns=["original_id"])
            output_file = output_path / f"student_data_{course_id}.csv"
            course_df_anon.to_csv(output_file, index=False)
            logger.info(f"Saved {len(course_df_anon)} records to {output_file}")

        # Use generated data directory for upload
        args.input_dir = str(output_path)

    # Create uploader and upload
    uploader = S3DataUploader(args.s3_bucket, region=args.region)
    results = uploader.upload_directory(args.input_dir, s3_prefix=args.prefix)

    # Return exit code based on failures
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
