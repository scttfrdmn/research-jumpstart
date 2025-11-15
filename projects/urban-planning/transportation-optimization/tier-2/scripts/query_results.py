#!/usr/bin/env python3
"""
Query traffic analysis results from DynamoDB.

This script allows querying and analyzing traffic data stored in DynamoDB:
- Query by segment ID
- Query by time range
- Query by congestion level
- Aggregate statistics by segment
- Export results to CSV

Usage:
    # Query specific segment
    python scripts/query_results.py --segment-id SEG-001

    # Query time range
    python scripts/query_results.py --start-time 2025-01-15T08:00:00 --end-time 2025-01-15T09:00:00

    # Query congested segments only
    python scripts/query_results.py --congested-only

    # Export to CSV
    python scripts/query_results.py --output results.csv
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

import boto3
from boto3.dynamodb.conditions import Key, Attr
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate

# Load environment variables
load_dotenv()

# AWS configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE_NAME', 'TrafficAnalysis')


class TrafficDataQuery:
    """Query and analyze traffic data from DynamoDB."""

    def __init__(self, table_name: str = DYNAMODB_TABLE, region: str = AWS_REGION):
        """
        Initialize DynamoDB query client.

        Args:
            table_name: DynamoDB table name
            region: AWS region
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.table_name = table_name

    def query_by_segment(self, segment_id: str, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> List[Dict]:
        """
        Query traffic data for a specific segment.

        Args:
            segment_id: Segment ID to query
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)

        Returns:
            List of traffic records
        """
        print(f"Querying segment: {segment_id}")

        # Build query
        if start_time and end_time:
            key_condition = Key('segment_id').eq(segment_id) & \
                           Key('timestamp').between(start_time, end_time)
        elif start_time:
            key_condition = Key('segment_id').eq(segment_id) & \
                           Key('timestamp').gte(start_time)
        elif end_time:
            key_condition = Key('segment_id').eq(segment_id) & \
                           Key('timestamp').lte(end_time)
        else:
            key_condition = Key('segment_id').eq(segment_id)

        # Execute query
        response = self.table.query(KeyConditionExpression=key_condition)

        items = response.get('Items', [])
        print(f"Found {len(items)} records for segment {segment_id}")

        return items

    def scan_all(self, filter_expression: Optional[Attr] = None,
                 limit: Optional[int] = None) -> List[Dict]:
        """
        Scan entire table with optional filter.

        Args:
            filter_expression: DynamoDB filter expression
            limit: Maximum number of items to return

        Returns:
            List of traffic records
        """
        print(f"Scanning table: {self.table_name}")

        scan_kwargs = {}
        if filter_expression:
            scan_kwargs['FilterExpression'] = filter_expression
        if limit:
            scan_kwargs['Limit'] = limit

        # Execute scan
        response = self.table.scan(**scan_kwargs)
        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response and (not limit or len(items) < limit):
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = self.table.scan(**scan_kwargs)
            items.extend(response.get('Items', []))

            if limit and len(items) >= limit:
                items = items[:limit]
                break

        print(f"Found {len(items)} total records")
        return items

    def query_congested_segments(self, threshold: float = 0.8) -> List[Dict]:
        """
        Query segments with high congestion (V/C ratio above threshold).

        Args:
            threshold: V/C ratio threshold for congestion

        Returns:
            List of congested segment records
        """
        print(f"Querying congested segments (V/C > {threshold})")

        filter_expr = Attr('vc_ratio').gt(threshold)
        items = self.scan_all(filter_expression=filter_expr, limit=1000)

        return items

    def query_by_los(self, los_grades: List[str]) -> List[Dict]:
        """
        Query segments by Level of Service grade.

        Args:
            los_grades: List of LOS grades to query (e.g., ['E', 'F'])

        Returns:
            List of records matching LOS grades
        """
        print(f"Querying segments with LOS: {', '.join(los_grades)}")

        filter_expr = Attr('los').is_in(los_grades)
        items = self.scan_all(filter_expression=filter_expr, limit=1000)

        return items

    def get_segment_statistics(self, segment_id: str) -> Dict:
        """
        Calculate aggregate statistics for a segment.

        Args:
            segment_id: Segment ID

        Returns:
            Dictionary of statistics
        """
        records = self.query_by_segment(segment_id)

        if not records:
            return {}

        df = pd.DataFrame(records)

        stats = {
            'segment_id': segment_id,
            'total_records': len(df),
            'avg_speed': df['avg_speed'].mean(),
            'min_speed': df['avg_speed'].min(),
            'max_speed': df['avg_speed'].max(),
            'avg_vc_ratio': df['vc_ratio'].mean(),
            'max_vc_ratio': df['vc_ratio'].max(),
            'avg_travel_time_index': df['travel_time_index'].mean(),
            'congestion_rate': df['is_congested'].sum() / len(df),
            'most_common_los': df['los'].mode()[0] if len(df) > 0 else None,
            'los_distribution': df['los'].value_counts().to_dict()
        }

        return stats

    def get_all_segments(self) -> List[str]:
        """
        Get list of all unique segment IDs in the table.

        Returns:
            List of segment IDs
        """
        print("Retrieving all segment IDs...")

        items = self.scan_all(limit=10000)
        segment_ids = list(set(item['segment_id'] for item in items))

        print(f"Found {len(segment_ids)} unique segments")
        return sorted(segment_ids)

    def get_time_range(self) -> tuple:
        """
        Get the time range of data in the table.

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp)
        """
        items = self.scan_all(limit=1000)

        if not items:
            return None, None

        timestamps = [item['timestamp'] for item in items]
        return min(timestamps), max(timestamps)


def format_results_table(records: List[Dict], max_rows: int = 20) -> str:
    """
    Format records as a readable table.

    Args:
        records: List of traffic records
        max_rows: Maximum rows to display

    Returns:
        Formatted table string
    """
    if not records:
        return "No records found."

    df = pd.DataFrame(records)

    # Select key columns
    columns = [
        'segment_id', 'timestamp_iso', 'avg_speed', 'vehicle_count',
        'vc_ratio', 'los', 'is_congested'
    ]
    display_cols = [col for col in columns if col in df.columns]

    df_display = df[display_cols].head(max_rows)

    # Format numeric columns
    if 'vc_ratio' in df_display.columns:
        df_display['vc_ratio'] = df_display['vc_ratio'].map('{:.3f}'.format)
    if 'avg_speed' in df_display.columns:
        df_display['avg_speed'] = df_display['avg_speed'].map('{:.1f}'.format)

    table = tabulate(df_display, headers='keys', tablefmt='grid', showindex=False)

    if len(df) > max_rows:
        table += f"\n\n... and {len(df) - max_rows} more rows"

    return table


def print_summary_statistics(records: List[Dict]):
    """
    Print summary statistics for records.

    Args:
        records: List of traffic records
    """
    if not records:
        print("No records to summarize.")
        return

    df = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"\nSpeed Statistics:")
    print(f"  Average speed: {df['avg_speed'].mean():.2f} mph")
    print(f"  Min speed: {df['avg_speed'].min():.2f} mph")
    print(f"  Max speed: {df['avg_speed'].max():.2f} mph")
    print(f"\nCongestion Metrics:")
    print(f"  Average V/C ratio: {df['vc_ratio'].mean():.3f}")
    print(f"  Max V/C ratio: {df['vc_ratio'].max():.3f}")
    print(f"  Congested segments: {df['is_congested'].sum()} ({df['is_congested'].sum()/len(df)*100:.1f}%)")
    print(f"\nLevel of Service Distribution:")
    los_dist = df['los'].value_counts().sort_index()
    for los_grade, count in los_dist.items():
        print(f"  LOS {los_grade}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nTravel Time:")
    print(f"  Average TTI: {df['travel_time_index'].mean():.3f}")
    print(f"  Max TTI: {df['travel_time_index'].max():.3f}")
    print("=" * 70 + "\n")


def export_to_csv(records: List[Dict], output_file: str):
    """
    Export records to CSV file.

    Args:
        records: List of traffic records
        output_file: Output CSV file path
    """
    if not records:
        print("No records to export.")
        return

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} records to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Query traffic analysis results from DynamoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query specific segment
  python query_results.py --segment-id SEG-001

  # Query time range (Unix timestamps)
  python query_results.py --segment-id SEG-001 --start-time 1705320000 --end-time 1705330000

  # Query all congested segments
  python query_results.py --congested-only

  # Query by Level of Service
  python query_results.py --los E F

  # Get statistics for all segments
  python query_results.py --segment-stats

  # Export results to CSV
  python query_results.py --segment-id SEG-001 --output results.csv
        """
    )

    parser.add_argument(
        '--segment-id',
        type=str,
        help='Segment ID to query'
    )

    parser.add_argument(
        '--start-time',
        type=int,
        help='Start time (Unix timestamp)'
    )

    parser.add_argument(
        '--end-time',
        type=int,
        help='End time (Unix timestamp)'
    )

    parser.add_argument(
        '--congested-only',
        action='store_true',
        help='Query only congested segments (V/C > 0.8)'
    )

    parser.add_argument(
        '--los',
        nargs='+',
        choices=['A', 'B', 'C', 'D', 'E', 'F'],
        help='Query by Level of Service grade(s)'
    )

    parser.add_argument(
        '--segment-stats',
        action='store_true',
        help='Show statistics for all segments'
    )

    parser.add_argument(
        '--list-segments',
        action='store_true',
        help='List all segment IDs'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Export results to CSV file'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of records to display (default: 100)'
    )

    parser.add_argument(
        '--table',
        type=str,
        default=DYNAMODB_TABLE,
        help=f'DynamoDB table name (default: {DYNAMODB_TABLE})'
    )

    args = parser.parse_args()

    # Initialize query client
    query_client = TrafficDataQuery(table_name=args.table)

    try:
        # Execute query based on arguments
        records = []

        if args.list_segments:
            segments = query_client.get_all_segments()
            print("\nSegment IDs:")
            for seg in segments:
                print(f"  - {seg}")
            return

        if args.segment_stats:
            segments = query_client.get_all_segments()
            print(f"\nCalculating statistics for {len(segments)} segments...")
            for segment_id in segments[:10]:  # Limit to first 10 for demo
                stats = query_client.get_segment_statistics(segment_id)
                print(f"\n{segment_id}:")
                print(f"  Records: {stats['total_records']}")
                print(f"  Avg Speed: {stats['avg_speed']:.2f} mph")
                print(f"  Avg V/C: {stats['avg_vc_ratio']:.3f}")
                print(f"  Congestion Rate: {stats['congestion_rate']*100:.1f}%")
                print(f"  Most Common LOS: {stats['most_common_los']}")
            return

        if args.congested_only:
            records = query_client.query_congested_segments()

        elif args.los:
            records = query_client.query_by_los(args.los)

        elif args.segment_id:
            records = query_client.query_by_segment(
                args.segment_id,
                start_time=args.start_time,
                end_time=args.end_time
            )

        else:
            # Default: scan recent records
            print("No specific query provided. Showing recent records...")
            records = query_client.scan_all(limit=args.limit)

        # Display results
        if records:
            print("\n" + format_results_table(records, max_rows=20))
            print_summary_statistics(records)

            # Export if requested
            if args.output:
                export_to_csv(records, args.output)
        else:
            print("No records found matching the query criteria.")

    except Exception as e:
        print(f"Error querying DynamoDB: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
