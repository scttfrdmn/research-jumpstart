#!/usr/bin/env python3
"""
Query NDVI calculation results from S3 and Athena.

This script:
1. Lists processed results in S3
2. Downloads metrics JSON files
3. Queries results with Athena (SQL)
4. Generates CSV summaries

Usage:
    python query_results.py --bucket satellite-imagery-XXXXX
    python query_results.py --bucket satellite-imagery-XXXXX --output ./results/
    python query_results.py --bucket satellite-imagery-XXXXX --athena --query "SELECT * FROM field_metrics"
"""

import argparse
import boto3
import json
import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# AWS clients
s3_client = boto3.client('s3')
athena_client = boto3.client('athena')


def list_results_in_s3(bucket_name, prefix='results/'):
    """
    List all processed results in S3.

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix to search

    Returns:
        list: List of result objects
    """
    print(f"Listing results in s3://{bucket_name}/{prefix}")

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        results = []
        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                if obj['Key'].endswith('_metrics.json'):
                    results.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'modified': obj['LastModified']
                    })

        return results

    except Exception as e:
        print(f"Error listing results: {e}")
        return []


def download_metrics_from_s3(bucket_name, results: List[Dict], output_dir='./results'):
    """
    Download metrics JSON files from S3.

    Args:
        bucket_name (str): S3 bucket name
        results (list): List of result objects from list_results_in_s3
        output_dir (str): Local directory to save files

    Returns:
        list: Downloaded metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_list = []

    print(f"\nDownloading {len(results)} result files...")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        key = result['key']
        filename = key.split('/')[-1]

        print(f"[{i}/{len(results)}] {filename}")

        try:
            # Download file
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')

            metrics = json.loads(content)

            # Save locally
            local_path = output_path / filename
            with open(local_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            metrics_list.append(metrics)
            print(f"  ✓ Saved to {local_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return metrics_list


def create_csv_summary(metrics_list, output_path='./metrics_summary.csv'):
    """
    Create CSV summary of all metrics.

    Args:
        metrics_list (list): List of metrics dictionaries
        output_path (str): Path to save CSV file

    Returns:
        pd.DataFrame: Metrics as DataFrame
    """
    if not metrics_list:
        print("No metrics to summarize")
        return None

    print(f"\nCreating CSV summary: {output_path}")

    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)

    # Select and order columns
    columns = [
        'field_id', 'date', 'timestamp',
        'avg_ndvi', 'min_ndvi', 'max_ndvi', 'std_ndvi',
        'vegetation_coverage', 'health_status'
    ]

    # Keep only available columns
    available_cols = [col for col in columns if col in df.columns]
    df = df[available_cols]

    # Sort by field_id and date
    if 'field_id' in df.columns and 'date' in df.columns:
        df = df.sort_values(['field_id', 'date'])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ CSV saved: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    return df


def print_metrics_summary(metrics_list):
    """
    Print summary statistics of metrics.

    Args:
        metrics_list (list): List of metrics dictionaries
    """
    if not metrics_list:
        print("No metrics available")
        return

    df = pd.DataFrame(metrics_list)

    print("\nMetrics Summary Statistics")
    print("=" * 60)

    # NDVI statistics
    if 'avg_ndvi' in df.columns:
        print(f"\nNDVI Statistics:")
        print(f"  Average NDVI - Mean: {df['avg_ndvi'].mean():.4f}")
        print(f"  Average NDVI - Median: {df['avg_ndvi'].median():.4f}")
        print(f"  Average NDVI - Range: {df['avg_ndvi'].min():.4f} to {df['avg_ndvi'].max():.4f}")

        # NDVI by health status
        if 'health_status' in df.columns:
            print(f"\nHealth Status Distribution:")
            for status in df['health_status'].unique():
                count = len(df[df['health_status'] == status])
                avg = df[df['health_status'] == status]['avg_ndvi'].mean()
                print(f"  {status}: {count} fields (avg NDVI: {avg:.4f})")

    # Field statistics
    if 'field_id' in df.columns:
        print(f"\nField Statistics:")
        print(f"  Total fields: {df['field_id'].nunique()}")
        print(f"  Total observations: {len(df)}")

        top_fields = df.groupby('field_id')['avg_ndvi'].mean().sort_values(ascending=False)
        print(f"  Top 5 healthiest fields:")
        for field_id, ndvi in top_fields.head(5).items():
            print(f"    - {field_id}: {ndvi:.4f}")


def execute_athena_query(query_string, bucket_name, output_location='athena-results/'):
    """
    Execute SQL query on Athena.

    Args:
        query_string (str): SQL query to execute
        bucket_name (str): S3 bucket for results
        output_location (str): S3 prefix for Athena output

    Returns:
        dict: Query execution details
    """
    print(f"\nExecuting Athena query...")
    print(f"Query: {query_string}")

    try:
        response = athena_client.start_query_execution(
            QueryString=query_string,
            QueryExecutionContext={'Database': 'default'},
            ResultConfiguration={
                'OutputLocation': f's3://{bucket_name}/{output_location}'
            }
        )

        query_id = response['QueryExecutionId']
        print(f"Query ID: {query_id}")

        # Wait for query to complete
        print("Waiting for query execution...")
        max_retries = 30
        retry = 0

        while retry < max_retries:
            response = athena_client.get_query_execution(QueryExecutionId=query_id)
            status = response['QueryExecution']['Status']['State']

            if status == 'SUCCEEDED':
                print(f"✓ Query succeeded")
                return response
            elif status == 'FAILED':
                error = response['QueryExecution']['Status']['StateChangeReason']
                print(f"✗ Query failed: {error}")
                return None
            else:
                print(f"  Status: {status}... ", end='', flush=True)
                time.sleep(2)
                retry += 1

        print("✗ Query timeout")
        return None

    except Exception as e:
        print(f"Error executing Athena query: {e}")
        return None


def get_athena_results(query_id):
    """
    Get results from completed Athena query.

    Args:
        query_id (str): Athena query execution ID

    Returns:
        list: Query results as list of dictionaries
    """
    try:
        # Get query results
        response = athena_client.get_query_results(QueryExecutionId=query_id)

        rows = response['ResultSet']['Rows']
        if len(rows) == 0:
            return []

        # Parse header
        headers = [col['VarCharValue'] for col in rows[0]['Data']]

        # Parse data rows
        results = []
        for row in rows[1:]:
            data = {}
            for i, header in enumerate(headers):
                if i < len(row['Data']):
                    data[header] = row['Data'][i].get('VarCharValue', None)
            results.append(data)

        return results

    except Exception as e:
        print(f"Error getting Athena results: {e}")
        return []


def create_sample_data():
    """Create sample metrics JSON files for testing."""
    import random

    results_dir = Path('./sample_results')
    results_dir.mkdir(exist_ok=True)

    print("Creating sample result files...")

    fields = ['field_001', 'field_002', 'field_003']
    dates = ['20240601', '20240615', '20240630']

    for field in fields:
        for date in dates:
            metrics = {
                'field_id': field,
                'date': date,
                'timestamp': datetime.now().isoformat(),
                'avg_ndvi': round(random.uniform(0.4, 0.8), 4),
                'min_ndvi': round(random.uniform(0.2, 0.4), 4),
                'max_ndvi': round(random.uniform(0.75, 0.95), 4),
                'std_ndvi': round(random.uniform(0.05, 0.2), 4),
                'vegetation_coverage': round(random.uniform(0.6, 0.95), 4),
                'health_status': random.choice(['Healthy', 'Moderate', 'Stressed'])
            }

            filename = f"{field}_{date}_metrics.json"
            filepath = results_dir / filename

            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  Created: {filename}")

    return list(results_dir.glob('*.json'))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Query NDVI calculation results from S3 and Athena',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # List results in S3
  python query_results.py --bucket satellite-imagery-12345 --list

  # Download all results to CSV
  python query_results.py --bucket satellite-imagery-12345 --output ./results/

  # Query with Athena
  python query_results.py --bucket satellite-imagery-12345 --athena \
    --query "SELECT * FROM field_metrics WHERE avg_ndvi < 0.4"

  # Create sample data (for testing)
  python query_results.py --sample
        '''
    )

    parser.add_argument(
        '--bucket',
        help='S3 bucket name'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List results in S3'
    )
    parser.add_argument(
        '--output',
        default='./results',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--athena',
        action='store_true',
        help='Use Athena for querying'
    )
    parser.add_argument(
        '--query',
        help='Athena SQL query to execute'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Create sample result files'
    )
    parser.add_argument(
        '--prefix',
        default='results/',
        help='S3 prefix for results'
    )

    args = parser.parse_args()

    print("NDVI Results Query Tool")
    print("=" * 60)

    # Handle sample data creation
    if args.sample:
        files = create_sample_data()
        print(f"✓ Created {len(files)} sample files in ./sample_results/")

        # Load and display
        all_metrics = []
        for filepath in files:
            with open(filepath) as f:
                all_metrics.append(json.load(f))

        print_metrics_summary(all_metrics)
        return

    # Require bucket for other operations
    if not args.bucket:
        parser.print_help()
        sys.exit(1)

    # List S3 results
    if args.list or (not args.athena and not args.query):
        results = list_results_in_s3(args.bucket, args.prefix)
        print(f"\nFound {len(results)} result files")

        for result in results:
            size_kb = result['size'] / 1024
            print(f"  - {result['key']:<50} {size_kb:>10.2f}KB")

    # Download and process results
    if not args.athena:
        results = list_results_in_s3(args.bucket, args.prefix)

        if results:
            metrics_list = download_metrics_from_s3(args.bucket, results, args.output)

            if metrics_list:
                # Create CSV summary
                csv_path = os.path.join(args.output, 'metrics_summary.csv')
                create_csv_summary(metrics_list, csv_path)

                # Print summary
                print_metrics_summary(metrics_list)

    # Execute Athena query
    if args.athena and args.query:
        response = execute_athena_query(args.query, args.bucket)

        if response:
            query_id = response['QueryExecution']['QueryExecutionId']
            results = get_athena_results(query_id)

            if results:
                print(f"\nQuery Results ({len(results)} rows):")
                for row in results[:10]:
                    print(f"  {row}")

                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more rows")

    print("\n" + "=" * 60)
    print("Query complete!")


if __name__ == '__main__':
    main()
