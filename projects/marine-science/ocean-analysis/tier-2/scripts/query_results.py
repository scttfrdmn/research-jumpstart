#!/usr/bin/env python3
"""
Query and analyze ocean observations from DynamoDB.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd


def decimal_to_float(obj):
    """Convert DynamoDB Decimal to float for pandas."""
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def query_observations(table_name, location=None, days=7, anomaly_status=None):
    """
    Query ocean observations from DynamoDB.

    Parameters:
    - table_name: DynamoDB table name
    - location: Filter by location name (None for all locations)
    - days: Number of days to look back
    - anomaly_status: Filter by anomaly status ('normal', 'warning', 'critical')

    Returns:
    - List of observations
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    print(f"Querying observations from {start_time.isoformat()} to {end_time.isoformat()}")

    try:
        # Scan table (in production, use Query with proper indexes)
        response = table.scan()
        items = response['Items']

        # Continue scanning if there are more items
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response['Items'])

        print(f"Retrieved {len(items)} total observations from DynamoDB")

        # Filter by location if specified
        if location and location.lower() != 'all':
            items = [item for item in items if location.lower() in item.get('location_name', '').lower()]
            print(f"Filtered to {len(items)} observations for location: {location}")

        # Filter by time range
        items = [
            item for item in items
            if start_time.isoformat() <= item.get('timestamp', '') <= end_time.isoformat()
        ]
        print(f"Filtered to {len(items)} observations within {days} days")

        # Filter by anomaly status if specified
        if anomaly_status:
            items = [item for item in items if item.get('anomaly_status') == anomaly_status]
            print(f"Filtered to {len(items)} observations with status: {anomaly_status}")

        return items

    except Exception as e:
        print(f"Error querying DynamoDB: {str(e)}")
        return []


def observations_to_dataframe(observations):
    """Convert DynamoDB observations to pandas DataFrame."""
    if not observations:
        return pd.DataFrame()

    # Convert Decimal to float
    for obs in observations:
        for key, value in obs.items():
            obs[key] = decimal_to_float(value)

    df = pd.DataFrame(observations)

    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')

    return df


def print_summary(df):
    """Print summary statistics of observations."""
    if df.empty:
        print("\nNo observations found.")
        return

    print(f"\n{'='*80}")
    print(f"OCEAN OBSERVATIONS SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal Observations: {len(df)}")

    if 'location_name' in df.columns:
        print(f"\nLocations:")
        for location, count in df['location_name'].value_counts().items():
            print(f"  - {location}: {count} observations")

    if 'anomaly_status' in df.columns:
        print(f"\nAnomaly Status:")
        for status, count in df['anomaly_status'].value_counts().items():
            print(f"  - {status}: {count} observations")

    if 'anomaly_type' in df.columns:
        anomaly_types = df[df['anomaly_type'] != 'none']['anomaly_type'].value_counts()
        if not anomaly_types.empty:
            print(f"\nAnomaly Types:")
            for atype, count in anomaly_types.items():
                print(f"  - {atype}: {count} occurrences")

    # Parameter statistics
    params = ['temperature', 'salinity', 'ph', 'dissolved_oxygen', 'chlorophyll']
    available_params = [p for p in params if p in df.columns]

    if available_params:
        print(f"\n{'Parameter':<25} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std':<12}")
        print(f"{'-'*73}")

        for param in available_params:
            mean_val = df[param].mean()
            min_val = df[param].min()
            max_val = df[param].max()
            std_val = df[param].std()

            units = {
                'temperature': '°C',
                'salinity': 'PSU',
                'ph': '',
                'dissolved_oxygen': 'mg/L',
                'chlorophyll': 'mg/m³'
            }
            unit = units.get(param, '')

            print(f"{param:<25} {mean_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f} {std_val:<12.2f} {unit}")

    # Anomaly metrics
    anomaly_params = ['temperature_anomaly', 'aragonite_saturation', 'primary_production']
    available_anomaly = [p for p in anomaly_params if p in df.columns]

    if available_anomaly:
        print(f"\n{'Metric':<25} {'Mean':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-'*61}")

        for param in available_anomaly:
            mean_val = df[param].mean()
            min_val = df[param].min()
            max_val = df[param].max()

            print(f"{param:<25} {mean_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")


def print_alerts(df):
    """Print observations with alerts."""
    if df.empty:
        return

    alerts = df[df['anomaly_status'].isin(['warning', 'critical'])]

    if alerts.empty:
        print(f"\n{'='*80}")
        print("No alerts found.")
        print(f"{'='*80}")
        return

    print(f"\n{'='*80}")
    print(f"MARINE ANOMALY ALERTS ({len(alerts)} found)")
    print(f"{'='*80}")

    for idx, row in alerts.iterrows():
        print(f"\nLocation: {row['location_name']}")
        print(f"Time: {row['timestamp']}")
        print(f"Depth: {row['depth']:.0f}m")
        print(f"Status: {row['anomaly_status'].upper()}")
        print(f"Type: {row['anomaly_type']}")
        print(f"Temperature: {row['temperature']:.2f}°C (anomaly: {row['temperature_anomaly']:.2f}°C)")
        print(f"pH: {row['ph']:.3f}")
        print(f"Dissolved Oxygen: {row['dissolved_oxygen']:.2f} mg/L")
        print(f"Chlorophyll: {row['chlorophyll']:.2f} mg/m³")
        print(f"{'-'*80}")


def export_to_csv(df, filename):
    """Export observations to CSV file."""
    if df.empty:
        print("No data to export.")
        return

    df.to_csv(filename, index=False)
    print(f"\nExported {len(df)} observations to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Query and analyze ocean observations from DynamoDB'
    )
    parser.add_argument(
        '--table',
        type=str,
        default=os.environ.get('DYNAMODB_TABLE', 'OceanObservations'),
        help='DynamoDB table name'
    )
    parser.add_argument(
        '--location',
        type=str,
        default='all',
        help='Filter by location name (default: all)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back (default: 7)'
    )
    parser.add_argument(
        '--anomaly-status',
        type=str,
        choices=['normal', 'warning', 'critical'],
        help='Filter by anomaly status'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    parser.add_argument(
        '--alerts-only',
        action='store_true',
        help='Show only observations with alerts'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"OCEAN OBSERVATIONS QUERY")
    print(f"{'='*80}")
    print(f"Table: {args.table}")
    print(f"Location: {args.location}")
    print(f"Days: {args.days}")
    if args.anomaly_status:
        print(f"Anomaly Status: {args.anomaly_status}")

    # Query observations
    observations = query_observations(
        args.table,
        location=args.location,
        days=args.days,
        anomaly_status=args.anomaly_status
    )

    if not observations:
        print("\nNo observations found matching criteria.")
        sys.exit(0)

    # Convert to DataFrame
    df = observations_to_dataframe(observations)

    # Print summary
    if not args.alerts_only:
        print_summary(df)

    # Print alerts
    print_alerts(df)

    # Export if requested
    if args.export:
        export_to_csv(df, args.export)

    print(f"\n{'='*80}")
    print("Query complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
