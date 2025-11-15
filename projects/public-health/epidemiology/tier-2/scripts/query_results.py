"""
Query disease surveillance results from DynamoDB.

This script:
1. Connects to DynamoDB table
2. Queries case data by disease, region, or date range
3. Aggregates statistics
4. Displays results in formatted tables
5. Optionally exports to CSV

Usage:
    # Query by disease
    python query_results.py --disease influenza --limit 100

    # Query by region
    python query_results.py --region northeast --start-date 2024-01-01

    # Query summaries only
    python query_results.py --summaries-only

    # Export to CSV
    python query_results.py --disease influenza --output results.csv

Author: Research Jumpstart
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import boto3
from boto3.dynamodb.conditions import Key, Attr
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
dynamodb = boto3.resource('dynamodb')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Query disease surveillance results from DynamoDB'
    )
    parser.add_argument(
        '--table-name',
        type=str,
        default='DiseaseReports',
        help='DynamoDB table name (default: DiseaseReports)'
    )
    parser.add_argument(
        '--disease',
        type=str,
        help='Filter by disease name'
    )
    parser.add_argument(
        '--region',
        type=str,
        help='Filter by region'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for query (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for query (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of records to retrieve (default: 100)'
    )
    parser.add_argument(
        '--summaries-only',
        action='store_true',
        help='Only retrieve summary records'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--show-outbreak-alerts',
        action='store_true',
        help='Show only records with outbreak alerts'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def query_by_disease(
    table,
    disease: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """
    Query cases by disease using GSI.

    Args:
        table: DynamoDB table resource
        disease: Disease name
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum results

    Returns:
        List of case records
    """
    try:
        logger.info(f"Querying disease: {disease}")

        # Use disease-date-index GSI
        query_params = {
            'IndexName': 'disease-date-index',
            'KeyConditionExpression': Key('disease').eq(disease),
            'Limit': limit
        }

        # Add date range if specified
        if start_date:
            if end_date:
                query_params['KeyConditionExpression'] = (
                    query_params['KeyConditionExpression'] &
                    Key('report_date').between(start_date, end_date)
                )
            else:
                query_params['KeyConditionExpression'] = (
                    query_params['KeyConditionExpression'] &
                    Key('report_date').gte(start_date)
                )

        response = table.query(**query_params)
        items = response.get('Items', [])

        logger.info(f"Retrieved {len(items)} records for disease: {disease}")
        return items

    except ClientError as e:
        logger.error(f"Error querying by disease: {str(e)}")
        return []


def query_by_region(
    table,
    region: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """
    Query cases by region using GSI.

    Args:
        table: DynamoDB table resource
        region: Region name
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum results

    Returns:
        List of case records
    """
    try:
        logger.info(f"Querying region: {region}")

        # Use region-date-index GSI
        query_params = {
            'IndexName': 'region-date-index',
            'KeyConditionExpression': Key('region').eq(region),
            'Limit': limit
        }

        # Add date range if specified
        if start_date:
            if end_date:
                query_params['KeyConditionExpression'] = (
                    query_params['KeyConditionExpression'] &
                    Key('report_date').between(start_date, end_date)
                )
            else:
                query_params['KeyConditionExpression'] = (
                    query_params['KeyConditionExpression'] &
                    Key('report_date').gte(start_date)
                )

        response = table.query(**query_params)
        items = response.get('Items', [])

        logger.info(f"Retrieved {len(items)} records for region: {region}")
        return items

    except ClientError as e:
        logger.error(f"Error querying by region: {str(e)}")
        return []


def scan_summaries(table, limit: int = 100) -> List[Dict]:
    """
    Scan for summary records.

    Args:
        table: DynamoDB table resource
        limit: Maximum results

    Returns:
        List of summary records
    """
    try:
        logger.info("Scanning for summary records...")

        response = table.scan(
            FilterExpression=Attr('record_type').eq('summary'),
            Limit=limit
        )

        items = response.get('Items', [])
        logger.info(f"Retrieved {len(items)} summary records")
        return items

    except ClientError as e:
        logger.error(f"Error scanning summaries: {str(e)}")
        return []


def scan_with_filters(
    table,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    outbreak_only: bool = False,
    limit: int = 100
) -> List[Dict]:
    """
    Scan table with filters.

    Args:
        table: DynamoDB table resource
        start_date: Optional start date filter
        end_date: Optional end date filter
        outbreak_only: Only return outbreak alerts
        limit: Maximum results

    Returns:
        List of case records
    """
    try:
        logger.info("Scanning table with filters...")

        scan_params = {'Limit': limit}

        # Build filter expression
        filter_expressions = []

        if start_date:
            filter_expressions.append(Attr('report_date').gte(start_date))

        if end_date:
            filter_expressions.append(Attr('report_date').lte(end_date))

        if outbreak_only:
            filter_expressions.append(Attr('outbreak_detected').eq(True))

        # Combine filters
        if filter_expressions:
            filter_expr = filter_expressions[0]
            for expr in filter_expressions[1:]:
                filter_expr = filter_expr & expr
            scan_params['FilterExpression'] = filter_expr

        response = table.scan(**scan_params)
        items = response.get('Items', [])

        logger.info(f"Retrieved {len(items)} records")
        return items

    except ClientError as e:
        logger.error(f"Error scanning table: {str(e)}")
        return []


def parse_and_display_results(items: List[Dict], summaries_only: bool = False):
    """
    Parse and display results in formatted tables.

    Args:
        items: List of DynamoDB items
        summaries_only: Whether results are summary records only
    """
    if not items:
        print("\nNo results found.")
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(items)

    print(f"\n{'=' * 80}")
    print(f"QUERY RESULTS: {len(items)} records")
    print(f"{'=' * 80}\n")

    if summaries_only:
        # Display summary statistics
        for idx, row in df.iterrows():
            print(f"\n--- Summary Report {idx + 1} ---")
            print(f"Report Date: {row.get('report_date', 'N/A')}")
            print(f"Total Cases: {row.get('total_cases', 'N/A')}")
            print(f"Incidence Rate: {row.get('incidence_rate', 'N/A'):.2f} per 100,000")

            if 'outbreak_detected' in row:
                outbreak_status = "YES" if row['outbreak_detected'] else "NO"
                confidence = row.get('outbreak_confidence', 'N/A')
                print(f"Outbreak Detected: {outbreak_status} (confidence: {confidence})")

            if 'r0_estimate' in row:
                print(f"R0 Estimate: {row['r0_estimate']:.2f}")

            # Parse metrics JSON if present
            if 'metrics' in row:
                try:
                    metrics = json.loads(row['metrics'])
                    print("\nKey Metrics:")
                    if 'case_fatality_rate' in metrics:
                        print(f"  Case Fatality Rate: {metrics['case_fatality_rate']:.2f}%")
                    if 'hospitalization_rate' in metrics:
                        print(f"  Hospitalization Rate: {metrics['hospitalization_rate']:.2f}%")
                    if 'attack_rate_by_region' in metrics:
                        print("  Attack Rates by Region:")
                        for region, rate in metrics['attack_rate_by_region'].items():
                            print(f"    {region}: {rate:.2f}%")
                except json.JSONDecodeError:
                    pass

            # Parse outbreak signals JSON if present
            if 'outbreak_signals' in row:
                try:
                    signals = json.loads(row['outbreak_signals'])
                    if signals.get('reasons'):
                        print("\nOutbreak Detection Reasons:")
                        for reason in signals['reasons']:
                            print(f"  • {reason}")
                except json.JSONDecodeError:
                    pass

            print(f"{'-' * 60}")

    else:
        # Display case records in table format
        display_cols = [
            'case_id', 'report_date', 'disease', 'region',
            'age_group', 'sex', 'outcome'
        ]

        # Filter to available columns
        available_cols = [col for col in display_cols if col in df.columns]

        if available_cols:
            print(df[available_cols].to_string(index=False))
        else:
            print(df.to_string(index=False))

        # Summary statistics
        print(f"\n{'=' * 80}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 80}\n")

        if 'disease' in df.columns:
            print("Cases by Disease:")
            print(df['disease'].value_counts().to_string())
            print()

        if 'region' in df.columns:
            print("Cases by Region:")
            print(df['region'].value_counts().to_string())
            print()

        if 'outcome' in df.columns:
            print("Cases by Outcome:")
            print(df['outcome'].value_counts().to_string())
            print()

        if 'age_group' in df.columns:
            print("Cases by Age Group:")
            print(df['age_group'].value_counts().to_string())
            print()


def export_to_csv(items: List[Dict], output_file: str):
    """
    Export results to CSV file.

    Args:
        items: List of DynamoDB items
        output_file: Output file path
    """
    try:
        df = pd.DataFrame(items)

        # Expand JSON columns if present
        json_cols = ['metrics', 'outbreak_signals', 'r0_details', 'epi_curve']
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )

        df.to_csv(output_file, index=False)
        logger.info(f"✓ Exported {len(items)} records to: {output_file}")
        print(f"\n✓ Results exported to: {output_file}")

    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=== Disease Surveillance Query ===")
    logger.info(f"Table: {args.table_name}")

    # Connect to DynamoDB table
    try:
        table = dynamodb.Table(args.table_name)

        # Verify table exists
        table.load()
        logger.info(f"✓ Connected to table: {args.table_name}")

    except ClientError as e:
        logger.error(f"Error connecting to DynamoDB: {str(e)}")
        sys.exit(1)
    except NoCredentialsError:
        logger.error("AWS credentials not found. Run 'aws configure' first.")
        sys.exit(1)

    # Query based on parameters
    items = []

    if args.summaries_only:
        items = scan_summaries(table, limit=args.limit)

    elif args.disease:
        items = query_by_disease(
            table,
            disease=args.disease,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )

    elif args.region:
        items = query_by_region(
            table,
            region=args.region,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )

    else:
        items = scan_with_filters(
            table,
            start_date=args.start_date,
            end_date=args.end_date,
            outbreak_only=args.show_outbreak_alerts,
            limit=args.limit
        )

    # Display results
    parse_and_display_results(items, summaries_only=args.summaries_only)

    # Export if requested
    if args.output and items:
        export_to_csv(items, args.output)

    logger.info("=== Query Complete ===")


if __name__ == '__main__':
    main()
