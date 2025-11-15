#!/usr/bin/env python3
"""
Download and query processed climate data results from S3.

This script:
1. Downloads processed results from S3
2. Analyzes statistics across multiple files
3. Generates summary reports
4. (Optional) Query using Athena for SQL-based analysis
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3ResultsQuerier:
    """Query and analyze results from S3."""

    def __init__(self, bucket_name, region="us-east-1", profile=None):
        """
        Initialize results querier.

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
        except ClientError:
            logger.error(f"✗ Cannot access bucket: {bucket_name}")
            raise

    def list_results(self, prefix="results/"):
        """
        List all processed results in S3.

        Args:
            prefix (str): S3 prefix to search

        Returns:
            list: List of result files
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" not in response:
                logger.info(f"No results found in {prefix}")
                return []

            files = []
            for obj in response["Contents"]:
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"],
                        "filename": os.path.basename(obj["Key"]),
                    }
                )

            logger.info(f"Found {len(files)} result files")
            for f in files:
                logger.info(f"  {f['filename']} ({f['size']} bytes)")

            return files

        except ClientError as e:
            logger.error(f"✗ Failed to list results: {e}")
            return []

    def download_results(self, output_dir="./results", prefix="results/"):
        """
        Download all results from S3.

        Args:
            output_dir (str): Local directory to save results
            prefix (str): S3 prefix to download

        Returns:
            list: List of downloaded files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.list_results(prefix)
        downloaded = []

        for result in results:
            try:
                output_path = output_dir / result["filename"]
                logger.info(f"Downloading: {result['key']}")

                self.s3.download_file(self.bucket_name, result["key"], str(output_path))

                logger.info(f"✓ Saved to: {output_path}")
                downloaded.append(str(output_path))

            except ClientError as e:
                logger.error(f"✗ Failed to download {result['key']}: {e}")

        logger.info(f"\n✓ Downloaded {len(downloaded)} files")
        return downloaded

    def read_result_file(self, file_path):
        """
        Read a result JSON file.

        Args:
            file_path (str): Path to result file

        Returns:
            dict: Parsed JSON data
        """
        try:
            with open(file_path) as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"✗ Failed to read {file_path}: {e}")
            return None

    def analyze_results(self, result_files):
        """
        Analyze multiple result files.

        Args:
            result_files (list): List of result file paths

        Returns:
            dict: Summary statistics
        """
        logger.info("\nAnalyzing results...")

        data_list = []
        for file_path in result_files:
            data = self.read_result_file(file_path)
            if data:
                data_list.append(data)

        if not data_list:
            logger.warning("No valid result files found")
            return {}

        # Extract statistics
        summary = {
            "total_files": len(data_list),
            "timestamp": datetime.utcnow().isoformat(),
            "files": [],
            "temperature_statistics": {},
            "precipitation_statistics": {},
        }

        temp_means = []
        precip_means = []

        for data in data_list:
            file_info = {
                "file": data.get("file", "unknown"),
                "processed_at": data.get("timestamp", "unknown"),
            }

            # Extract temperature stats
            if "statistics" in data and "temperature" in data["statistics"]:
                temp_stats = data["statistics"]["temperature"]
                file_info["temperature_mean"] = temp_stats.get("mean")
                temp_means.append(temp_stats.get("mean"))

            # Extract precipitation stats
            if "statistics" in data and "precipitation" in data["statistics"]:
                precip_stats = data["statistics"]["precipitation"]
                file_info["precipitation_mean"] = precip_stats.get("mean")
                precip_means.append(precip_stats.get("mean"))

            summary["files"].append(file_info)

        # Calculate aggregate statistics
        if temp_means:
            summary["temperature_statistics"] = {
                "mean_of_means": float(sum(temp_means) / len(temp_means)),
                "min": float(min(temp_means)),
                "max": float(max(temp_means)),
                "std": float(
                    (
                        sum((x - sum(temp_means) / len(temp_means)) ** 2 for x in temp_means)
                        / len(temp_means)
                    )
                    ** 0.5
                )
                if len(temp_means) > 1
                else 0,
            }

        if precip_means:
            summary["precipitation_statistics"] = {
                "mean_of_means": float(sum(precip_means) / len(precip_means)),
                "min": float(min(precip_means)),
                "max": float(max(precip_means)),
                "std": float(
                    (
                        sum((x - sum(precip_means) / len(precip_means)) ** 2 for x in precip_means)
                        / len(precip_means)
                    )
                    ** 0.5
                )
                if len(precip_means) > 1
                else 0,
            }

        return summary

    def export_summary(self, summary, output_file="results_summary.json"):
        """
        Export summary to file.

        Args:
            summary (dict): Summary data
            output_file (str): Output file path
        """
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Summary exported to: {output_file}")

    def create_dataframe(self, result_files):
        """
        Create pandas DataFrame from results.

        Args:
            result_files (list): List of result file paths

        Returns:
            pd.DataFrame: Results as DataFrame
        """
        records = []

        for file_path in result_files:
            data = self.read_result_file(file_path)
            if not data:
                continue

            # Create record
            record = {
                "file": data.get("file", "unknown"),
                "timestamp": data.get("timestamp", "unknown"),
            }

            # Extract temperature
            if "statistics" in data and "temperature" in data["statistics"]:
                temp = data["statistics"]["temperature"]
                record["temperature_mean"] = temp.get("mean")
                record["temperature_std"] = temp.get("std")
                record["temperature_min"] = temp.get("min")
                record["temperature_max"] = temp.get("max")

            # Extract precipitation
            if "statistics" in data and "precipitation" in data["statistics"]:
                precip = data["statistics"]["precipitation"]
                record["precipitation_mean"] = precip.get("mean")
                record["precipitation_std"] = precip.get("std")
                record["precipitation_min"] = precip.get("min")
                record["precipitation_max"] = precip.get("max")

            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"\n✓ Created DataFrame with {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    def print_summary(self, summary):
        """
        Print summary to console.

        Args:
            summary (dict): Summary data
        """
        print("\n" + "=" * 60)
        print("CLIMATE DATA ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal files analyzed: {summary.get('total_files', 0)}")
        print(f"Analysis timestamp: {summary.get('timestamp', 'unknown')}")

        if summary.get("temperature_statistics"):
            print("\nTemperature Statistics:")
            temp_stats = summary["temperature_statistics"]
            for key, value in temp_stats.items():
                print(f"  {key}: {value:.2f}")

        if summary.get("precipitation_statistics"):
            print("\nPrecipitation Statistics:")
            precip_stats = summary["precipitation_statistics"]
            for key, value in precip_stats.items():
                print(f"  {key}: {value:.2f}")

        if summary.get("files"):
            print("\nProcessed Files:")
            for f in summary["files"][:5]:  # Show first 5
                print(f"  {f.get('file', 'unknown')}")
            if len(summary["files"]) > 5:
                print(f"  ... and {len(summary['files']) - 5} more")

        print("=" * 60 + "\n")


def query_s3_results(bucket_name, output_dir="./results", region="us-east-1"):
    """
    Main function to query S3 results.

    Args:
        bucket_name (str): S3 bucket name
        output_dir (str): Output directory for downloaded results
        region (str): AWS region

    Returns:
        dict: Summary data
    """
    try:
        # Initialize querier
        querier = S3ResultsQuerier(bucket_name, region)

        # Download results
        logger.info("\nStep 1: Downloading results from S3...")
        result_files = querier.download_results(output_dir)

        if not result_files:
            logger.warning("No result files downloaded")
            return {}

        # Analyze results
        logger.info("\nStep 2: Analyzing results...")
        summary = querier.analyze_results(result_files)

        # Create DataFrame
        logger.info("\nStep 3: Creating DataFrame...")
        df = querier.create_dataframe(result_files)

        # Export summary
        summary_file = Path(output_dir) / "analysis_summary.json"
        querier.export_summary(summary, str(summary_file))

        # Export DataFrame
        csv_file = Path(output_dir) / "results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"✓ CSV exported to: {csv_file}")

        # Print summary
        querier.print_summary(summary)

        return summary

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return {}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Query and analyze climate results from S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for downloaded files"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument(
        "--list-only", action="store_true", help="Only list results without downloading"
    )
    parser.add_argument("--prefix", default="results/", help="S3 prefix to search")

    args = parser.parse_args()

    try:
        querier = S3ResultsQuerier(args.bucket, args.region, args.profile)

        if args.list_only:
            files = querier.list_results(args.prefix)
            logger.info(f"\nFound {len(files)} files")
        else:
            summary = query_s3_results(args.bucket, args.output_dir, args.region)
            sys.exit(0 if summary else 1)

    except Exception as e:
        logger.error(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
