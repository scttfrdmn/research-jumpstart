#!/usr/bin/env python3
"""
Query and analyze grid optimization results from DynamoDB and S3.

This script retrieves analysis results from AWS services and displays them
in a formatted table or exports to CSV/JSON.

Usage:
    python query_results.py --bucket energy-grid-XXXXX
    python query_results.py --bucket energy-grid-XXXXX --location substation_001
    python query_results.py --bucket energy-grid-XXXXX --export results.csv
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import boto3

# AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def query_dynamodb(table_name, location=None, limit=100):
    """
    Query DynamoDB for grid analysis results.

    Args:
        table_name (str): DynamoDB table name
        location (str, optional): Filter by location
        limit (int): Maximum number of results

    Returns:
        list: Query results
    """
    try:
        table = dynamodb.Table(table_name)

        if location:
            # Query specific location
            response = table.query(
                KeyConditionExpression="location = :loc",
                ExpressionAttributeValues={":loc": location},
                Limit=limit,
                ScanIndexForward=False,  # Most recent first
            )
        else:
            # Scan all records (expensive for large tables)
            response = table.scan(Limit=limit)

        items = response.get("Items", [])
        print(f"✓ Retrieved {len(items)} records from DynamoDB")

        return items

    except Exception as e:
        print(f"Error querying DynamoDB: {e}")
        return []


def query_s3_results(bucket_name, prefix="results/", max_files=50):
    """
    List and download result files from S3.

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix for results
        max_files (int): Maximum files to process

    Returns:
        list: Downloaded result objects
    """
    try:
        # List objects in results folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_files)

        if "Contents" not in response:
            print(f"No results found in s3://{bucket_name}/{prefix}")
            return []

        results = []
        for obj in response["Contents"]:
            key = obj["Key"]

            # Skip folder objects
            if key.endswith("/"):
                continue

            # Download and parse JSON
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response["Body"].read().decode("utf-8")
                data = json.loads(content)
                data["s3_key"] = key
                results.append(data)
            except Exception as e:
                print(f"Error downloading {key}: {e}")

        print(f"✓ Retrieved {len(results)} result files from S3")
        return results

    except Exception as e:
        print(f"Error querying S3: {e}")
        return []


def display_results_table(results):
    """
    Display results in formatted table.

    Args:
        results (list): Analysis results
    """
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 120)
    print("Grid Analysis Results")
    print("=" * 120)

    # Table header
    print(
        f"{'Location':<20} {'Timestamp':<20} {'Load (MW)':<15} {'Renewable %':<15} {'Stability':<12} {'Status':<10}"
    )
    print("-" * 120)

    # Sort by timestamp
    sorted_results = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)

    for result in sorted_results:
        location = result.get("location", "N/A")
        timestamp = result.get("timestamp", "N/A")[:19]  # Trim to datetime

        # Extract metrics
        load_metrics = result.get("load_metrics", {})
        renewable_metrics = result.get("renewable_metrics", {})
        power_quality = result.get("power_quality", {})

        load_avg = load_metrics.get("avg_mw", 0)
        renewable_pct = renewable_metrics.get("renewable_percentage", 0)
        stability = power_quality.get("stability_score", 0)
        status = result.get("alert_status", "N/A")

        print(
            f"{location:<20} {timestamp:<20} {load_avg:>10.2f} MW  {renewable_pct:>10.2f}%  {stability:>10.3f}  {status:<10}"
        )

    print("-" * 120)
    print(f"Total records: {len(results)}")
    print("=" * 120)


def display_detailed_summary(results):
    """
    Display detailed statistical summary.

    Args:
        results (list): Analysis results
    """
    if not results:
        return

    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)

    # Aggregate metrics
    loads = [r["load_metrics"]["avg_mw"] for r in results if "load_metrics" in r]
    renewable_pcts = [
        r["renewable_metrics"]["renewable_percentage"] for r in results if "renewable_metrics" in r
    ]
    stability_scores = [
        r["power_quality"]["stability_score"] for r in results if "power_quality" in r
    ]

    if loads:
        print("\nLoad (MW):")
        print(f"  Average: {sum(loads) / len(loads):.2f} MW")
        print(f"  Min: {min(loads):.2f} MW")
        print(f"  Max: {max(loads):.2f} MW")

    if renewable_pcts:
        print("\nRenewable Penetration:")
        print(f"  Average: {sum(renewable_pcts) / len(renewable_pcts):.2f}%")
        print(f"  Min: {min(renewable_pcts):.2f}%")
        print(f"  Max: {max(renewable_pcts):.2f}%")

    if stability_scores:
        print("\nGrid Stability:")
        print(f"  Average: {sum(stability_scores) / len(stability_scores):.3f}")
        print(f"  Min: {min(stability_scores):.3f}")
        print(f"  Max: {max(stability_scores):.3f}")

    # Alert summary
    alert_counts = {}
    for result in results:
        status = result.get("alert_status", "unknown")
        alert_counts[status] = alert_counts.get(status, 0) + 1

    print("\nAlert Status Distribution:")
    for status, count in sorted(alert_counts.items()):
        print(f"  {status.capitalize()}: {count}")

    print("=" * 80)


def export_results(results, output_path, format="csv"):
    """
    Export results to file.

    Args:
        results (list): Analysis results
        output_path (str): Output file path
        format (str): Export format ('csv' or 'json')
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Export as JSON
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, cls=DecimalEncoder)
            print(f"✓ Exported {len(results)} records to {output_file} (JSON)")

        elif format == "csv":
            # Export as CSV
            import csv

            with open(output_file, "w", newline="") as f:
                if not results:
                    print("No results to export")
                    return

                # Flatten nested structure for CSV
                fieldnames = [
                    "location",
                    "timestamp",
                    "alert_status",
                    "load_avg_mw",
                    "load_min_mw",
                    "load_max_mw",
                    "generation_avg_mw",
                    "renewable_penetration",
                    "renewable_percentage",
                    "voltage_avg_kv",
                    "voltage_min_kv",
                    "voltage_max_kv",
                    "frequency_avg_hz",
                    "frequency_min_hz",
                    "frequency_max_hz",
                    "power_factor_avg",
                    "stability_score",
                    "efficiency_score",
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    row = {
                        "location": result.get("location", ""),
                        "timestamp": result.get("timestamp", ""),
                        "alert_status": result.get("alert_status", ""),
                        "load_avg_mw": result.get("load_metrics", {}).get("avg_mw", 0),
                        "load_min_mw": result.get("load_metrics", {}).get("min_mw", 0),
                        "load_max_mw": result.get("load_metrics", {}).get("max_mw", 0),
                        "generation_avg_mw": result.get("generation_metrics", {}).get("avg_mw", 0),
                        "renewable_penetration": result.get("renewable_metrics", {}).get(
                            "renewable_penetration", 0
                        ),
                        "renewable_percentage": result.get("renewable_metrics", {}).get(
                            "renewable_percentage", 0
                        ),
                        "voltage_avg_kv": result.get("voltage_metrics", {}).get("avg_kv", 0),
                        "voltage_min_kv": result.get("voltage_metrics", {}).get("min_kv", 0),
                        "voltage_max_kv": result.get("voltage_metrics", {}).get("max_kv", 0),
                        "frequency_avg_hz": result.get("frequency_metrics", {}).get("avg_hz", 0),
                        "frequency_min_hz": result.get("frequency_metrics", {}).get("min_hz", 0),
                        "frequency_max_hz": result.get("frequency_metrics", {}).get("max_hz", 0),
                        "power_factor_avg": result.get("power_quality", {}).get(
                            "power_factor_avg", 0
                        ),
                        "stability_score": result.get("power_quality", {}).get(
                            "stability_score", 0
                        ),
                        "efficiency_score": result.get("power_quality", {}).get(
                            "efficiency_score", 0
                        ),
                    }
                    writer.writerow(row)

            print(f"✓ Exported {len(results)} records to {output_file} (CSV)")

    except Exception as e:
        print(f"Error exporting results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Query and analyze grid optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all results from DynamoDB
  python query_results.py --table GridAnalysis

  # Query specific location
  python query_results.py --table GridAnalysis --location substation_001

  # Query S3 results
  python query_results.py --bucket energy-grid-12345

  # Export results to CSV
  python query_results.py --table GridAnalysis --export results.csv

  # Export results to JSON
  python query_results.py --table GridAnalysis --export results.json --format json
        """,
    )

    parser.add_argument(
        "--table", default="GridAnalysis", help="DynamoDB table name (default: GridAnalysis)"
    )
    parser.add_argument("--bucket", help="S3 bucket name for S3 results")
    parser.add_argument("--location", help="Filter by location (e.g., substation_001)")
    parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of results (default: 100)"
    )
    parser.add_argument("--export", help="Export results to file (e.g., results.csv)")
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Export format (default: csv)"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Display detailed statistical summary"
    )

    args = parser.parse_args()

    print("Grid Optimization Results Query Tool")
    print("=" * 80)

    results = []

    # Query from DynamoDB
    if args.table:
        print(f"\nQuerying DynamoDB table: {args.table}")
        results = query_dynamodb(args.table, args.location, args.limit)

    # Query from S3 (if specified)
    if args.bucket:
        print(f"\nQuerying S3 bucket: {args.bucket}")
        s3_results = query_s3_results(args.bucket)
        if s3_results and not results:
            results = s3_results

    # Display results
    if results:
        display_results_table(results)

        if args.summary:
            display_detailed_summary(results)

        # Export if requested
        if args.export:
            export_results(results, args.export, args.format)
    else:
        print("\nNo results found.")
        print("\nTroubleshooting:")
        print(
            "  1. Verify DynamoDB table exists: aws dynamodb describe-table --table-name GridAnalysis"
        )
        print("  2. Verify S3 bucket has results: aws s3 ls s3://BUCKET/results/")
        print("  3. Check AWS credentials: aws sts get-caller-identity")

    print("\n" + "=" * 80)
    print("Query complete!")


if __name__ == "__main__":
    main()
