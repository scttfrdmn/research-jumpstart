#!/usr/bin/env python3
"""
Query prediction results from DynamoDB and download processed images from S3.

This script:
- Queries DynamoDB for prediction metadata
- Retrieves processing statistics
- Downloads sample processed images from S3
- Generates analysis reports
- Creates visualizations

Usage:
    python scripts/query_results.py \
        --table-name medical-predictions \
        --limit 100 \
        --output-dir ./results/
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_TABLE = "medical-predictions"
DEFAULT_BUCKET = os.getenv("S3_BUCKET_NAME")


class ResultsAnalyzer:
    """Analyze prediction results from DynamoDB and S3."""

    def __init__(
        self, table_name: str, bucket_name: Optional[str] = None, region: str = "us-east-1"
    ):
        """
        Initialize results analyzer.

        Args:
            table_name: DynamoDB table name
            bucket_name: S3 bucket name (optional)
            region: AWS region
        """
        self.table_name = table_name
        self.bucket_name = bucket_name
        self.region = region

        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)
        self.table = self.dynamodb.Table(table_name)

    def query_all_results(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Query all prediction results from DynamoDB.

        Args:
            limit: Maximum number of items to retrieve

        Returns:
            List of prediction records
        """
        try:
            logger.info(f"Querying DynamoDB table: {self.table_name}")

            response = self.table.scan(Limit=limit)
            items = response.get("Items", [])

            logger.info(f"Retrieved {len(items)} items from DynamoDB")

            # Parse metadata JSON strings
            for item in items:
                if "metadata" in item and isinstance(item["metadata"], str):
                    try:
                        item["metadata"] = json.loads(item["metadata"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata for {item.get('image_id')}")

            return items

        except ClientError as e:
            logger.error(f"DynamoDB query error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error querying DynamoDB: {e}")
            return []

    def get_statistics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calculate statistics from results.

        Args:
            results: List of prediction records

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {"error": "No results to analyze"}

        # Extract processing times and image statistics
        processing_times = []
        image_means = []
        image_stds = []
        file_sizes = []

        for result in results:
            metadata = result.get("metadata", {})

            if isinstance(metadata, dict):
                if "processing_time_ms" in metadata:
                    processing_times.append(metadata["processing_time_ms"])
                if "mean_value" in metadata:
                    image_means.append(metadata["mean_value"])
                if "std_value" in metadata:
                    image_stds.append(metadata["std_value"])
                if "source_size" in metadata:
                    file_sizes.append(metadata["source_size"])

        # Calculate statistics
        stats = {
            "total_images": len(results),
            "processing_times": {
                "count": len(processing_times),
                "min": min(processing_times) if processing_times else None,
                "max": max(processing_times) if processing_times else None,
                "mean": sum(processing_times) / len(processing_times) if processing_times else None,
                "unit": "milliseconds",
            },
            "image_intensity": {
                "count": len(image_means),
                "mean_intensity_avg": sum(image_means) / len(image_means) if image_means else None,
                "std_intensity_avg": sum(image_stds) / len(image_stds) if image_stds else None,
                "range": "[0, 1]",
            },
            "file_sizes": {
                "count": len(file_sizes),
                "total_bytes": sum(file_sizes) if file_sizes else 0,
                "mean_bytes": sum(file_sizes) / len(file_sizes) if file_sizes else None,
                "total_mb": sum(file_sizes) / (1024**2) if file_sizes else None,
            },
        }

        return stats

    def save_results_csv(self, results: list[dict[str, Any]], output_file: str):
        """
        Save results to CSV file.

        Args:
            results: List of prediction records
            output_file: Output CSV file path
        """
        try:
            if not results:
                logger.warning("No results to save")
                return

            # Extract flattened data
            rows = []
            for result in results:
                metadata = result.get("metadata", {})

                if isinstance(metadata, dict):
                    row = {
                        "image_id": result.get("image_id"),
                        "timestamp": result.get("timestamp"),
                        "source_key": metadata.get("source_key"),
                        "source_size_bytes": metadata.get("source_size"),
                        "output_key": metadata.get("output_key"),
                        "output_size_bytes": metadata.get("output_size"),
                        "processing_time_ms": metadata.get("processing_time_ms"),
                        "min_intensity": metadata.get("min_value"),
                        "max_intensity": metadata.get("max_value"),
                        "mean_intensity": metadata.get("mean_value"),
                        "std_intensity": metadata.get("std_value"),
                    }
                    rows.append(row)

            # Write CSV
            if rows:
                keys = rows[0].keys()
                with open(output_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)

                logger.info(f"Results saved to {output_file}")
                return len(rows)
            else:
                logger.warning("No rows to write")
                return 0

        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

    def download_sample_images(
        self, results: list[dict[str, Any]], output_dir: str, count: int = 5
    ) -> int:
        """
        Download sample processed images from S3.

        Args:
            results: List of prediction records
            output_dir: Directory to save images
            count: Number of sample images to download

        Returns:
            Number of images downloaded
        """
        if not self.bucket_name:
            logger.warning("Bucket name not provided, skipping image download")
            return 0

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for i, result in enumerate(results[:count]):
            try:
                metadata = result.get("metadata", {})
                output_key = metadata.get("output_key")

                if not output_key:
                    logger.warning(f"No output key for result {i}")
                    continue

                # Download from S3
                local_file = output_path / Path(output_key).name
                logger.info(f"Downloading {output_key} to {local_file}")

                self.s3_client.download_file(self.bucket_name, output_key, str(local_file))
                downloaded += 1

            except ClientError as e:
                logger.error(f"Error downloading image {i}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error downloading image {i}: {e}")

        logger.info(f"Downloaded {downloaded}/{min(count, len(results))} sample images")
        return downloaded

    def generate_report(self, results: list[dict[str, Any]], output_file: str):
        """
        Generate text report with analysis.

        Args:
            results: List of prediction records
            output_file: Output report file path
        """
        try:
            stats = self.get_statistics(results)

            report_lines = [
                "=" * 70,
                "MEDICAL IMAGE PROCESSING ANALYSIS REPORT",
                "=" * 70,
                "",
                f"Report generated: {datetime.utcnow().isoformat()}",
                f"DynamoDB Table: {self.table_name}",
                "",
                "SUMMARY",
                "-" * 70,
                f"Total images processed: {stats.get('total_images', 0)}",
                "",
                "PROCESSING PERFORMANCE",
                "-" * 70,
            ]

            processing = stats.get("processing_times", {})
            if processing.get("count", 0) > 0:
                report_lines.extend(
                    [
                        "Processing time statistics (milliseconds):",
                        f"  - Count: {processing.get('count')}",
                        f"  - Min: {processing.get('min'):.2f}",
                        f"  - Max: {processing.get('max'):.2f}",
                        f"  - Mean: {processing.get('mean'):.2f}",
                    ]
                )

            report_lines.append("")
            report_lines.append("IMAGE INTENSITY STATISTICS")
            report_lines.append("-" * 70)

            image_intensity = stats.get("image_intensity", {})
            if image_intensity.get("count", 0) > 0:
                report_lines.extend(
                    [
                        f"Image intensity (normalized to {image_intensity.get('range')}):",
                        f"  - Average mean intensity: {image_intensity.get('mean_intensity_avg'):.3f}",
                        f"  - Average std intensity: {image_intensity.get('std_intensity_avg'):.3f}",
                    ]
                )

            report_lines.append("")
            report_lines.append("DATA SIZES")
            report_lines.append("-" * 70)

            file_sizes = stats.get("file_sizes", {})
            if file_sizes.get("count", 0) > 0:
                report_lines.extend(
                    [
                        "File size statistics:",
                        f"  - Total processed: {file_sizes.get('total_mb'):.2f} MB",
                        f"  - Average file size: {file_sizes.get('mean_bytes'):.0f} bytes",
                    ]
                )

            report_lines.extend(
                [
                    "",
                    "COST ESTIMATE",
                    "-" * 70,
                    "Based on AWS pricing (us-east-1):",
                    f"  - DynamoDB writes: {stats.get('total_images', 0)} × $0.0000012 = ${stats.get('total_images', 0) * 0.0000012:.4f}",
                    "  - Data transfer: ~$0.12 per GB",
                    "  - Lambda execution: ~$0.0000002 per 100ms",
                    "",
                    "=" * 70,
                ]
            )

            report_text = "\n".join(report_lines)

            # Save report
            with open(output_file, "w") as f:
                f.write(report_text)

            logger.info(f"Report saved to {output_file}")
            print(report_text)

        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def print_results_summary(self, results: list[dict[str, Any]]):
        """Print summary of results to console."""
        if not results:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("PREDICTION RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total images: {len(results)}\n")

        # Show first 5 results
        for i, result in enumerate(results[:5], 1):
            metadata = result.get("metadata", {})
            print(f"Image {i}:")
            print(f"  ID: {result.get('image_id')}")
            print(f"  Source: {metadata.get('source_key')}")
            print(f"  Output: {metadata.get('output_key')}")
            print(f"  Processing time: {metadata.get('processing_time_ms'):.2f} ms")
            print(f"  Mean intensity: {metadata.get('mean_value'):.3f}")
            print()

        if len(results) > 5:
            print(f"... and {len(results) - 5} more results")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query medical image processing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query results and save to CSV
  python query_results.py --table-name medical-predictions \\
                          --output-dir ./results/

  # Download sample images
  python query_results.py --table-name medical-predictions \\
                          --download-images \\
                          --image-count 10
        """,
    )

    parser.add_argument(
        "--table-name",
        type=str,
        default=os.getenv("DYNAMODB_TABLE_NAME", DEFAULT_TABLE),
        help=f"DynamoDB table name (default: {DEFAULT_TABLE})",
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("S3_BUCKET_NAME", DEFAULT_BUCKET),
        help="S3 bucket name (required for downloading images)",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results to retrieve (default: 100)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/",
        help="Output directory for results (default: ./results/)",
    )

    parser.add_argument(
        "--download-images", action="store_true", help="Download sample processed images from S3"
    )

    parser.add_argument(
        "--image-count",
        type=int,
        default=5,
        help="Number of sample images to download (default: 5)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create analyzer
    analyzer = ResultsAnalyzer(args.table_name, bucket_name=args.s3_bucket, region=args.region)

    # Query results
    results = analyzer.query_all_results(limit=args.limit)

    if not results:
        logger.error("No results found")
        sys.exit(1)

    # Display summary
    analyzer.print_results_summary(results)

    # Save to CSV
    csv_file = output_path / "results.csv"
    analyzer.save_results_csv(results, str(csv_file))

    # Generate report
    report_file = output_path / "analysis_report.txt"
    analyzer.generate_report(results, str(report_file))

    # Download sample images if requested
    if args.download_images:
        images_dir = output_path / "sample_images"
        analyzer.download_sample_images(results, str(images_dir), count=args.image_count)

    logger.info(f"Analysis complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
