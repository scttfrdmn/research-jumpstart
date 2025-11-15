"""
Query archaeological artifact data from DynamoDB.

This script:
1. Queries DynamoDB artifact catalog
2. Filters by type, period, site, location
3. Displays results in formatted tables
4. Exports results to CSV

Usage:
    python query_results.py
    python query_results.py --type pottery
    python query_results.py --period "Bronze Age"
    python query_results.py --site SITE_A --export results.csv
"""

import argparse
import os
import sys
from typing import Optional

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr, Key
from dotenv import load_dotenv
from tabulate import tabulate


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()

    config = {
        "table_name": os.getenv("TABLE_NAME", "ArtifactCatalog"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
    }

    return config


def query_all_artifacts(table_name: str, limit: Optional[int] = None) -> list[dict]:
    """Query all artifacts from DynamoDB."""

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    print(f"Querying all artifacts from {table_name}...")

    try:
        response = table.scan(Limit=limit) if limit else table.scan()

        items = response["Items"]

        # Handle pagination if needed
        while "LastEvaluatedKey" in response and (not limit or len(items) < limit):
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response["Items"])
            if limit and len(items) >= limit:
                items = items[:limit]
                break

        print(f"✓ Retrieved {len(items)} artifacts")
        return items

    except Exception as e:
        print(f"✗ Query failed: {e!s}")
        return []


def query_by_type(table_name: str, artifact_type: str) -> list[dict]:
    """Query artifacts by type using GSI."""

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    print(f"Querying artifacts of type: {artifact_type}")

    try:
        # Try using GSI if available
        try:
            response = table.query(
                IndexName="type-index",
                KeyConditionExpression=Key("artifact_type").eq(artifact_type),
            )
            items = response["Items"]
        except Exception:
            # Fall back to scan with filter
            print("  (using scan with filter - consider creating type-index GSI)")
            response = table.scan(FilterExpression=Attr("artifact_type").eq(artifact_type))
            items = response["Items"]

        print(f"✓ Found {len(items)} {artifact_type} artifacts")
        return items

    except Exception as e:
        print(f"✗ Query failed: {e!s}")
        return []


def query_by_site_and_period(
    table_name: str, site_id: str, period: Optional[str] = None
) -> list[dict]:
    """Query artifacts by site and optionally by period using GSI."""

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    print(f"Querying artifacts from site: {site_id}")
    if period:
        print(f"  Period: {period}")

    try:
        # Try using site-period GSI if available
        if period:
            try:
                response = table.query(
                    IndexName="site-period-index",
                    KeyConditionExpression=Key("site_id").eq(site_id) & Key("period").eq(period),
                )
                items = response["Items"]
            except Exception:
                # Fall back to scan with filter
                print("  (using scan with filter - consider creating site-period-index GSI)")
                response = table.scan(
                    FilterExpression=Attr("site_id").eq(site_id) & Attr("period").eq(period)
                )
                items = response["Items"]
        else:
            response = table.scan(FilterExpression=Attr("site_id").eq(site_id))
            items = response["Items"]

        print(f"✓ Found {len(items)} artifacts")
        return items

    except Exception as e:
        print(f"✗ Query failed: {e!s}")
        return []


def query_by_location(
    table_name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> list[dict]:
    """Query artifacts within geographic bounding box."""

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    print("Querying artifacts in bounding box:")
    print(f"  Latitude: {lat_min} to {lat_max}")
    print(f"  Longitude: {lon_min} to {lon_max}")

    try:
        response = table.scan(
            FilterExpression=Attr("gps_lat").between(lat_min, lat_max)
            & Attr("gps_lon").between(lon_min, lon_max)
        )
        items = response["Items"]

        print(f"✓ Found {len(items)} artifacts in region")
        return items

    except Exception as e:
        print(f"✗ Query failed: {e!s}")
        return []


def display_results(items: list[dict], max_rows: int = 20) -> None:
    """Display query results in formatted table."""

    if not items:
        print("\nNo results found.")
        return

    # Convert to DataFrame for better display
    df = pd.DataFrame(items)

    # Select key columns to display
    display_cols = [
        "artifact_id",
        "site_id",
        "artifact_type",
        "artifact_subtype",
        "material",
        "period",
        "length",
        "width",
        "weight",
        "gps_lat",
        "gps_lon",
        "classification_confidence",
    ]

    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    df_display = df[available_cols]

    # Limit rows for display
    if len(df_display) > max_rows:
        print(f"\nShowing first {max_rows} of {len(df_display)} results:")
        df_display = df_display.head(max_rows)
    else:
        print(f"\nShowing all {len(df_display)} results:")

    # Display as formatted table
    print("\n" + tabulate(df_display, headers="keys", tablefmt="grid", showindex=False))

    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    full_df = pd.DataFrame(items)

    if "artifact_type" in full_df.columns:
        print("\nArtifact Types:")
        type_counts = full_df["artifact_type"].value_counts()
        for artifact_type, count in type_counts.items():
            print(f"  {artifact_type}: {count} ({count / len(full_df) * 100:.1f}%)")

    if "period" in full_df.columns:
        print("\nPeriods:")
        period_counts = full_df["period"].value_counts()
        for period, count in period_counts.items():
            print(f"  {period}: {count} ({count / len(full_df) * 100:.1f}%)")

    if "material" in full_df.columns:
        print("\nMaterials:")
        material_counts = full_df["material"].value_counts()
        for material, count in material_counts.items():
            print(f"  {material}: {count} ({count / len(full_df) * 100:.1f}%)")

    # Numeric statistics
    numeric_cols = ["length", "width", "thickness", "weight"]
    available_numeric = [col for col in numeric_cols if col in full_df.columns]

    if available_numeric:
        print("\nMeasurements (mean ± std):")
        for col in available_numeric:
            mean = full_df[col].mean()
            std = full_df[col].std()
            print(f"  {col}: {mean:.2f} ± {std:.2f}")

    print()


def export_to_csv(items: list[dict], output_path: str) -> None:
    """Export query results to CSV file."""

    if not items:
        print("No data to export.")
        return

    df = pd.DataFrame(items)
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} artifacts to {output_path}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Query archaeological artifact data from DynamoDB")
    parser.add_argument("--table", help="DynamoDB table name (overrides .env)")
    parser.add_argument(
        "--type", help="Filter by artifact type (pottery, lithic, bone, coin, architecture)"
    )
    parser.add_argument("--period", help="Filter by period (Neolithic, Bronze Age, etc.)")
    parser.add_argument("--site", help="Filter by site ID")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Filter by geographic bounding box",
    )
    parser.add_argument("--limit", type=int, help="Limit number of results")
    parser.add_argument("--export", help="Export results to CSV file")
    parser.add_argument(
        "--max-display", type=int, default=20, help="Maximum rows to display (default: 20)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    if args.table:
        config["table_name"] = args.table

    print("=" * 80)
    print("Archaeological Artifact Query - DynamoDB")
    print("=" * 80)
    print(f"Table: {config['table_name']}")
    print(f"Region: {config['region']}")
    print()

    try:
        # Execute query based on filters
        items = []

        if args.type:
            # Query by artifact type
            items = query_by_type(config["table_name"], args.type)

        elif args.site:
            # Query by site (and optionally period)
            items = query_by_site_and_period(config["table_name"], args.site, args.period)

        elif args.bbox:
            # Query by geographic bounding box
            lat_min, lat_max, lon_min, lon_max = args.bbox
            items = query_by_location(config["table_name"], lat_min, lat_max, lon_min, lon_max)

        else:
            # Query all artifacts
            items = query_all_artifacts(config["table_name"], args.limit)

        # Apply additional filters if needed
        if items and args.period and not args.site:
            # Filter by period if not already filtered
            items = [item for item in items if item.get("period") == args.period]
            print(f"Filtered to period '{args.period}': {len(items)} artifacts")

        # Display results
        display_results(items, args.max_display)

        # Export if requested
        if args.export and items:
            export_to_csv(items, args.export)

        print("\n" + "=" * 80)
        print(f"Query complete - {len(items)} artifacts found")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\n✗ Error: {e!s}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check if tabulate is available, otherwise provide simple fallback
    try:
        from tabulate import tabulate
    except ImportError:
        print("Warning: 'tabulate' package not found, using simple display")
        print("Install with: pip install tabulate")

        def tabulate(data, headers, tablefmt, showindex):
            return str(data)

    main()
