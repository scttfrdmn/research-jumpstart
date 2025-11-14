#!/usr/bin/env python3
"""
Query source detection results using Athena.

Demonstrates SQL queries on the astronomical catalog stored in S3.
"""

import os
import sys
import time
import json
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_environment_variables():
    """Get required environment variables."""
    bucket_catalog = os.environ.get('BUCKET_CATALOG')
    workgroup = os.environ.get('ATHENA_WORKGROUP', 'astronomy-workgroup')

    if not bucket_catalog:
        print("Error: BUCKET_CATALOG environment variable not set")
        print("Set it with: export BUCKET_CATALOG=your-catalog-bucket")
        sys.exit(1)

    return bucket_catalog, workgroup


def execute_athena_query(athena, database, query, workgroup):
    """Execute an Athena query and return results."""
    print(f"\nQuery:\n{query}\n")

    try:
        # Start query execution
        response = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': f's3://{bucket_catalog}/athena-results/'},
            WorkGroup=workgroup
        )

        query_id = response['QueryExecutionId']
        print(f"Query ID: {query_id}")

        # Wait for query completion
        max_attempts = 30
        attempt = 0

        while attempt < max_attempts:
            result = athena.get_query_execution(QueryExecutionId=query_id)
            status = result['QueryExecution']['Status']['State']

            if status == 'SUCCEEDED':
                print("Query succeeded!")
                break
            elif status == 'FAILED':
                error_message = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                print(f"Query failed: {error_message}")
                return None
            elif status == 'CANCELLED':
                print("Query cancelled")
                return None

            print(f"  Status: {status}...", end='\r')
            time.sleep(1)
            attempt += 1

        if attempt >= max_attempts:
            print("Query timeout")
            return None

        # Get results
        results = athena.get_query_results(QueryExecutionId=query_id)
        return results

    except ClientError as e:
        print(f"Error: {e}")
        return None


def print_query_results(results):
    """Print Athena query results."""
    if not results or 'ResultSet' not in results:
        print("No results")
        return

    rows = results['ResultSet']['Rows']
    if not rows:
        print("No data returned")
        return

    # Print header
    header = rows[0]
    header_values = [cell.get('VarCharValue', '') for cell in header['Data']]
    print("\nResults:")
    print(" | ".join(header_values))
    print("-" * 80)

    # Print data rows
    for row in rows[1:]:
        values = [cell.get('VarCharValue', '') for cell in row['Data']]
        print(" | ".join(values))


def main():
    """Main function."""
    print("=" * 70)
    print("Athena Query - Astronomical Catalog")
    print("=" * 70)

    # Get environment variables
    bucket_catalog, workgroup = get_environment_variables()
    print(f"\nConfiguration:")
    print(f"  Catalog bucket: s3://{bucket_catalog}")
    print(f"  Athena workgroup: {workgroup}\n")

    # Initialize Athena client
    try:
        athena = boto3.client('athena')
    except NoCredentialsError:
        print("Error: AWS credentials not configured")
        print("Run: aws configure")
        return 1

    # Define queries
    queries = [
        {
            'name': 'Total Sources',
            'query': 'SELECT COUNT(*) as total_sources FROM astronomy.sources;'
        },
        {
            'name': 'Source Statistics',
            'query': '''
            SELECT
              COUNT(*) as total_sources,
              AVG(flux) as mean_flux,
              MAX(flux) as max_flux,
              MIN(flux) as min_flux,
              AVG(snr) as mean_snr,
              MAX(snr) as max_snr
            FROM astronomy.sources;
            '''
        },
        {
            'name': 'Bright Sources',
            'query': '''
            SELECT ra, dec, flux, snr, x, y
            FROM astronomy.sources
            WHERE snr > 20
            ORDER BY flux DESC
            LIMIT 10;
            '''
        },
        {
            'name': 'Sources by SNR',
            'query': '''
            SELECT
              CASE
                WHEN snr > 50 THEN 'Very High'
                WHEN snr > 20 THEN 'High'
                WHEN snr > 10 THEN 'Medium'
                WHEN snr > 5 THEN 'Low'
                ELSE 'Very Low'
              END as snr_class,
              COUNT(*) as count
            FROM astronomy.sources
            GROUP BY
              CASE
                WHEN snr > 50 THEN 'Very High'
                WHEN snr > 20 THEN 'High'
                WHEN snr > 10 THEN 'Medium'
                WHEN snr > 5 THEN 'Low'
                ELSE 'Very Low'
              END
            ORDER BY snr_class;
            '''
        }
    ]

    # Execute queries
    print("Executing queries...\n")

    for query_info in queries:
        print("=" * 70)
        print(f"Query: {query_info['name']}")
        print("=" * 70)

        results = execute_athena_query(
            athena,
            'astronomy',
            query_info['query'],
            workgroup
        )

        if results:
            print_query_results(results)

        print()

    # Custom query option
    print("=" * 70)
    print("Custom Query")
    print("=" * 70)
    print("\nEnter your own SQL query (or press Enter to skip):")
    print("Available tables: astronomy.sources")
    print("Available columns: image_id, source_id, ra, dec, x, y, flux, peak, fwhm, a, b, theta, snr\n")

    try:
        custom_query = input("Query: ").strip()
        if custom_query:
            if not custom_query.endswith(';'):
                custom_query += ';'

            results = execute_athena_query(
                athena,
                'astronomy',
                custom_query,
                workgroup
            )

            if results:
                print_query_results(results)
    except EOFError:
        pass  # Non-interactive mode

    print("\n✓ Query execution complete!")
    print("\nNext steps:")
    print("  1. Run: jupyter notebook notebooks/sky_analysis.ipynb")
    print("  2. See cleanup_guide.md to delete resources when done")
    print()

    return 0


if __name__ == "__main__":
    # Check for environment file
    env_file = Path.home() / ".astronomy_env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("export "):
                    key_value = line.replace("export ", "").strip()
                    if "=" in key_value:
                        key, value = key_value.split("=", 1)
                        os.environ[key] = value

    sys.exit(main())
