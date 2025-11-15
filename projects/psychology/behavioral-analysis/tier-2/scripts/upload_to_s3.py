#!/usr/bin/env python3
"""
Upload Behavioral Data to S3

This script uploads behavioral experiment data (CSV files) to S3.
Can also generate sample data for testing.

Usage:
    python upload_to_s3.py --bucket behavioral-data-12345 --generate-sample
    python upload_to_s3.py --bucket behavioral-data-12345 --data-dir ./my_data
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError


def generate_stroop_data(participant_id, n_trials=100):
    """
    Generate sample Stroop task data.

    Stroop task: Participants identify color of word.
    Congruent: color matches word (e.g., "RED" in red)
    Incongruent: color doesn't match word (e.g., "RED" in blue)
    """
    np.random.seed(hash(participant_id) % 2**32)

    data = []
    for trial in range(1, n_trials + 1):
        condition = "congruent" if np.random.random() > 0.5 else "incongruent"

        # Congruent trials are faster and more accurate
        if condition == "congruent":
            rt = np.random.normal(500, 80)  # mean 500ms, SD 80ms
            accuracy = 1 if np.random.random() > 0.05 else 0  # 95% accuracy
        else:
            rt = np.random.normal(650, 100)  # mean 650ms, SD 100ms
            accuracy = 1 if np.random.random() > 0.15 else 0  # 85% accuracy

        rt = max(200, rt)  # Minimum RT 200ms

        data.append(
            {
                "participant_id": participant_id,
                "trial": trial,
                "task_type": "stroop",
                "stimulus": condition,
                "response": np.random.choice(["left", "right"]),
                "rt": round(rt, 2),
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(data)


def generate_decision_data(participant_id, n_trials=100):
    """
    Generate sample two-alternative forced choice decision making data.

    Participants choose between two options with varying difficulty.
    """
    np.random.seed(hash(participant_id) % 2**32 + 1)

    data = []
    difficulties = ["easy", "medium", "hard"]

    for trial in range(1, n_trials + 1):
        difficulty = np.random.choice(difficulties)

        # Difficulty affects RT and accuracy
        if difficulty == "easy":
            rt = np.random.normal(400, 60)
            accuracy = 1 if np.random.random() > 0.05 else 0  # 95% accuracy
        elif difficulty == "medium":
            rt = np.random.normal(550, 80)
            accuracy = 1 if np.random.random() > 0.20 else 0  # 80% accuracy
        else:  # hard
            rt = np.random.normal(700, 120)
            accuracy = 1 if np.random.random() > 0.35 else 0  # 65% accuracy

        rt = max(200, rt)

        data.append(
            {
                "participant_id": participant_id,
                "trial": trial,
                "task_type": "decision",
                "stimulus": difficulty,
                "response": np.random.choice(["option_a", "option_b"]),
                "rt": round(rt, 2),
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(data)


def generate_learning_data(participant_id, n_trials=150):
    """
    Generate sample reinforcement learning task data.

    Participants learn to choose better option through feedback.
    """
    np.random.seed(hash(participant_id) % 2**32 + 2)

    # Simulate learning with increasing accuracy over trials
    learning_rate = 0.02
    initial_accuracy = 0.5

    data = []
    for trial in range(1, n_trials + 1):
        # Accuracy improves over time (learning curve)
        expected_accuracy = min(0.9, initial_accuracy + learning_rate * trial)
        accuracy = 1 if np.random.random() < expected_accuracy else 0

        # RT decreases as learning progresses (practice effects)
        rt = np.random.normal(600 - trial * 1.5, 100)
        rt = max(200, rt)

        data.append(
            {
                "participant_id": participant_id,
                "trial": trial,
                "task_type": "learning",
                "stimulus": np.random.choice(["stim_a", "stim_b", "stim_c"]),
                "response": np.random.choice(["choose", "avoid"]),
                "rt": round(rt, 2),
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(data)


def generate_sample_data(n_participants=10, output_dir="sample_data"):
    """
    Generate sample behavioral data for multiple participants.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating sample data for {n_participants} participants...")

    files = []
    for i in range(1, n_participants + 1):
        participant_id = f"sub{i:03d}"

        # Each participant does 1-2 tasks
        tasks = np.random.choice(
            ["stroop", "decision", "learning"], size=np.random.randint(1, 3), replace=False
        )

        for task in tasks:
            if task == "stroop":
                df = generate_stroop_data(participant_id)
            elif task == "decision":
                df = generate_decision_data(participant_id)
            else:  # learning
                df = generate_learning_data(participant_id)

            filename = f"{participant_id}_{task}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            files.append(filepath)
            print(f"  Generated: {filename}")

    print(f"\nGenerated {len(files)} data files in {output_dir}/")
    return files


def upload_to_s3(file_path, bucket_name, s3_key=None, s3_client=None):
    """
    Upload a single file to S3.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    if s3_key is None:
        s3_key = f"raw/{os.path.basename(file_path)}"

    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"  Uploaded: {file_path} -> s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"  Error uploading {file_path}: {e}")
        return False


def upload_directory(directory, bucket_name, s3_prefix="raw/"):
    """
    Upload all CSV files in a directory to S3.
    """
    s3_client = boto3.client("s3")

    # Find all CSV files
    csv_files = list(Path(directory).glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Found {len(csv_files)} CSV files to upload...")

    success_count = 0
    for file_path in csv_files:
        s3_key = f"{s3_prefix}{file_path.name}"
        if upload_to_s3(str(file_path), bucket_name, s3_key, s3_client):
            success_count += 1

    print(f"\nSuccessfully uploaded {success_count}/{len(csv_files)} files")


def verify_bucket_exists(bucket_name):
    """
    Verify that S3 bucket exists and is accessible.
    """
    s3_client = boto3.client("s3")

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket {bucket_name} exists and is accessible")
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"✗ Bucket {bucket_name} does not exist")
        elif error_code == "403":
            print(f"✗ Access denied to bucket {bucket_name}")
        else:
            print(f"✗ Error accessing bucket {bucket_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload behavioral data to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data and upload
  python upload_to_s3.py --bucket behavioral-data-12345 --generate-sample

  # Upload existing data
  python upload_to_s3.py --bucket behavioral-data-12345 --data-dir ./my_data

  # Specify number of sample participants
  python upload_to_s3.py --bucket behavioral-data-12345 --generate-sample --n-participants 20
        """,
    )

    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--data-dir", default=None, help="Directory containing CSV files to upload")
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate sample behavioral data"
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=10,
        help="Number of participants for sample data (default: 10)",
    )
    parser.add_argument(
        "--s3-prefix", default="raw/", help="S3 prefix (folder) for uploaded files (default: raw/)"
    )

    args = parser.parse_args()

    # Verify bucket exists
    print("Verifying S3 bucket...")
    if not verify_bucket_exists(args.bucket):
        print("\nPlease create the bucket first:")
        print(f"  aws s3 mb s3://{args.bucket}")
        sys.exit(1)

    # Generate sample data if requested
    if args.generate_sample:
        print("\n" + "=" * 60)
        print("Generating Sample Data")
        print("=" * 60)

        output_dir = "sample_data"
        generate_sample_data(args.n_participants, output_dir)

        print("\n" + "=" * 60)
        print("Uploading to S3")
        print("=" * 60)
        upload_directory(output_dir, args.bucket, args.s3_prefix)

    # Upload existing data if directory specified
    elif args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"Error: Directory {args.data_dir} does not exist")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Uploading to S3")
        print("=" * 60)
        upload_directory(args.data_dir, args.bucket, args.s3_prefix)

    else:
        print("Error: Must specify --generate-sample or --data-dir")
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print("\nView files in S3:")
    print(f"  aws s3 ls s3://{args.bucket}/{args.s3_prefix}")
    print("\nNext steps:")
    print("  1. Lambda function will process files automatically (if configured)")
    print("  2. Or invoke Lambda manually with scripts/invoke_lambda.py")
    print("  3. Query results with scripts/query_results.py")


if __name__ == "__main__":
    main()
