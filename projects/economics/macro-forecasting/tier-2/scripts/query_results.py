"""
Query economic forecasts from DynamoDB.

This script:
1. Queries DynamoDB for forecast predictions
2. Filters by indicator, country, date range
3. Compares multiple forecasting models
4. Displays results in formatted tables
5. Exports to CSV for further analysis

Usage:
    python query_results.py --table EconomicForecasts --indicator GDP --country USA
    python query_results.py --table EconomicForecasts --list-all
    python query_results.py --table EconomicForecasts --indicator GDP --export forecasts.csv
"""

import argparse
import sys
from typing import List, Dict, Optional
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate


class ForecastQueryClient:
    """Query economic forecasts from DynamoDB."""

    def __init__(self, table_name: str):
        """Initialize query client with DynamoDB table name."""
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb')
        self.table = None

    def validate_table(self) -> bool:
        """Verify DynamoDB table exists and is accessible."""
        try:
            self.table = self.dynamodb.Table(self.table_name)
            self.table.load()
            print(f"✓ Connected to DynamoDB table '{self.table_name}'")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                print(f"✗ Table '{self.table_name}' does not exist")
            else:
                print(f"✗ Error accessing table: {e}")
            return False
        except NoCredentialsError:
            print("✗ AWS credentials not configured. Run 'aws configure'")
            return False

    def query_forecasts(
        self,
        indicator: Optional[str] = None,
        country: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        model_type: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Query forecasts from DynamoDB.

        Args:
            indicator: Filter by indicator (GDP, unemployment, etc.)
            country: Filter by country (USA, CHN, etc.)
            start_date: Filter forecasts after this date
            end_date: Filter forecasts before this date
            model_type: Filter by model type (ARIMA, ExponentialSmoothing)
            limit: Maximum number of results

        Returns:
            DataFrame with forecast results
        """
        try:
            # Build query parameters
            if indicator and country:
                # Query with partition key
                key_condition = Key('indicator_country').eq(f"{indicator}_{country}")

                # Add date range if specified
                if start_date:
                    start_ts = int(start_date.timestamp())
                    key_condition = key_condition & Key('forecast_date').gte(start_ts)

                response = self.table.query(
                    KeyConditionExpression=key_condition,
                    Limit=limit
                )
            else:
                # Scan entire table (slower, but works without partition key)
                filter_expression = None

                if indicator:
                    filter_expression = Attr('indicator').eq(indicator)

                if country:
                    country_filter = Attr('country').eq(country)
                    filter_expression = country_filter if filter_expression is None else filter_expression & country_filter

                if model_type:
                    model_filter = Attr('model_type').eq(model_type)
                    filter_expression = model_filter if filter_expression is None else filter_expression & model_filter

                scan_kwargs = {'Limit': limit}
                if filter_expression:
                    scan_kwargs['FilterExpression'] = filter_expression

                response = self.table.scan(**scan_kwargs)

            items = response.get('Items', [])

            if not items:
                print("No forecasts found matching criteria")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(items)

            # Convert timestamp to datetime
            df['forecast_date'] = pd.to_datetime(df['forecast_date'], unit='s')

            # Sort by date
            df = df.sort_values('forecast_date')

            return df

        except Exception as e:
            print(f"✗ Query failed: {e}")
            return pd.DataFrame()

    def list_available_forecasts(self) -> pd.DataFrame:
        """List all unique indicator-country combinations."""
        try:
            # Scan table for unique combinations
            response = self.table.scan(
                ProjectionExpression='indicator_country, indicator, country, model_type',
                Limit=1000
            )

            items = response.get('Items', [])
            df = pd.DataFrame(items)

            if df.empty:
                print("No forecasts found in table")
                return df

            # Get unique combinations
            summary = df.groupby(['indicator', 'country', 'model_type']).size().reset_index(name='forecast_count')

            return summary

        except Exception as e:
            print(f"✗ List failed: {e}")
            return pd.DataFrame()

    def get_latest_forecasts(
        self,
        indicator: str,
        country: str,
        n: int = 10
    ) -> pd.DataFrame:
        """Get the most recent forecasts for an indicator-country pair."""
        key_condition = Key('indicator_country').eq(f"{indicator}_{country}")

        response = self.table.query(
            KeyConditionExpression=key_condition,
            ScanIndexForward=False,  # Sort descending (newest first)
            Limit=n
        )

        items = response.get('Items', [])
        df = pd.DataFrame(items)

        if not df.empty:
            df['forecast_date'] = pd.to_datetime(df['forecast_date'], unit='s')

        return df

    def compare_models(
        self,
        indicator: str,
        country: str
    ) -> pd.DataFrame:
        """Compare forecasts from different models."""
        df = self.query_forecasts(indicator=indicator, country=country)

        if df.empty:
            return df

        # Pivot to compare models side-by-side
        comparison = df.pivot_table(
            index='forecast_date',
            columns='model_type',
            values=['forecast_value', 'confidence_95_lower', 'confidence_95_upper'],
            aggfunc='first'
        )

        return comparison


def print_forecast_table(df: pd.DataFrame, title: str = "Economic Forecasts"):
    """Print forecasts in formatted table."""
    if df.empty:
        print("No data to display")
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # Select columns for display
    display_cols = [
        'indicator',
        'country',
        'forecast_date',
        'forecast_value',
        'confidence_95_lower',
        'confidence_95_upper',
        'model_type'
    ]

    # Filter to available columns
    display_cols = [col for col in display_cols if col in df.columns]

    # Format for display
    display_df = df[display_cols].copy()

    # Format numbers
    if 'forecast_value' in display_df.columns:
        display_df['forecast_value'] = display_df['forecast_value'].apply(lambda x: f"{x:.2f}")

    if 'confidence_95_lower' in display_df.columns:
        display_df['confidence_95_lower'] = display_df['confidence_95_lower'].apply(lambda x: f"{x:.2f}")

    if 'confidence_95_upper' in display_df.columns:
        display_df['confidence_95_upper'] = display_df['confidence_95_upper'].apply(lambda x: f"{x:.2f}")

    # Format dates
    if 'forecast_date' in display_df.columns:
        display_df['forecast_date'] = display_df['forecast_date'].dt.strftime('%Y-%m-%d')

    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Query economic forecasts from DynamoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query specific indicator and country
  python query_results.py --table EconomicForecasts --indicator GDP --country USA

  # List all available forecasts
  python query_results.py --table EconomicForecasts --list-all

  # Compare models
  python query_results.py --table EconomicForecasts --indicator GDP --country USA --compare-models

  # Export to CSV
  python query_results.py --table EconomicForecasts --indicator GDP --export gdp_forecasts.csv
        """
    )

    parser.add_argument(
        '--table',
        type=str,
        required=True,
        help='DynamoDB table name (e.g., EconomicForecasts)'
    )

    parser.add_argument(
        '--indicator',
        type=str,
        help='Filter by indicator (GDP, UNEMPLOYMENT, INFLATION)'
    )

    parser.add_argument(
        '--country',
        type=str,
        help='Filter by country (USA, CHN, DEU)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['ARIMA', 'ExponentialSmoothing'],
        help='Filter by model type'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of results (default: 100)'
    )

    parser.add_argument(
        '--list-all',
        action='store_true',
        help='List all available indicator-country combinations'
    )

    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare forecasts from different models'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )

    args = parser.parse_args()

    # Create query client
    client = ForecastQueryClient(args.table)

    # Validate table
    if not client.validate_table():
        sys.exit(1)

    # Handle list all
    if args.list_all:
        print("\nAvailable Forecasts:")
        df = client.list_available_forecasts()
        if not df.empty:
            print_forecast_table(df, "Available Indicator-Country Combinations")
        sys.exit(0)

    # Handle compare models
    if args.compare_models:
        if not args.indicator or not args.country:
            print("✗ --compare-models requires --indicator and --country")
            sys.exit(1)

        print(f"\nComparing models for {args.indicator} in {args.country}...")
        df = client.compare_models(args.indicator, args.country)

        if not df.empty:
            print("\nModel Comparison:")
            print(df.to_string())
        sys.exit(0)

    # Query forecasts
    print("\nQuerying forecasts...")
    df = client.query_forecasts(
        indicator=args.indicator,
        country=args.country,
        model_type=args.model_type,
        limit=args.limit
    )

    if df.empty:
        print("✗ No forecasts found")
        sys.exit(1)

    # Display results
    print(f"\n✓ Found {len(df)} forecasts")
    print_forecast_table(df)

    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\n✓ Exported to {args.export}")

    # Print summary statistics
    if 'forecast_value' in df.columns:
        print("\nSummary Statistics:")
        print(f"  Mean forecast: {df['forecast_value'].mean():.2f}")
        print(f"  Std deviation: {df['forecast_value'].std():.2f}")
        print(f"  Min forecast: {df['forecast_value'].min():.2f}")
        print(f"  Max forecast: {df['forecast_value'].max():.2f}")

    if 'model_type' in df.columns:
        print("\nForecasts by Model:")
        print(df['model_type'].value_counts().to_string())


if __name__ == "__main__":
    main()
