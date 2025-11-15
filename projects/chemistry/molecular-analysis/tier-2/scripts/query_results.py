#!/usr/bin/env python3
"""
Query molecular properties from DynamoDB.

This script queries the DynamoDB table for molecular properties,
filters by criteria, and exports results to CSV/JSON.
"""

import boto3
import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MolecularPropertyQuery:
    """Query and analyze molecular properties from DynamoDB."""

    def __init__(self, table_name='MolecularProperties', region='us-east-1', profile=None):
        """
        Initialize DynamoDB query client.

        Args:
            table_name (str): DynamoDB table name
            region (str): AWS region
            profile (str): AWS profile name
        """
        self.table_name = table_name
        self.region = region

        # Create session and DynamoDB resource
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)

        self.dynamodb = session.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

        logger.info(f"Connected to table: {table_name}")

    def scan_all(self, limit=None):
        """
        Scan all items from DynamoDB table.

        Args:
            limit (int): Maximum number of items to retrieve

        Returns:
            list: List of molecule property dictionaries
        """
        logger.info("Scanning DynamoDB table...")

        items = []
        scan_kwargs = {}
        if limit:
            scan_kwargs['Limit'] = limit

        try:
            response = self.table.scan(**scan_kwargs)
            items.extend(response['Items'])

            # Handle pagination
            while 'LastEvaluatedKey' in response and (not limit or len(items) < limit):
                scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = self.table.scan(**scan_kwargs)
                items.extend(response['Items'])

                if limit and len(items) >= limit:
                    items = items[:limit]
                    break

            logger.info(f"Retrieved {len(items)} items")
            return items

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            raise

    def query_by_class(self, compound_class):
        """
        Query molecules by compound class.

        Args:
            compound_class (str): Compound class (drug, natural_product, etc.)

        Returns:
            list: List of molecule property dictionaries
        """
        logger.info(f"Querying molecules with compound_class: {compound_class}")

        items = []
        try:
            response = self.table.scan(
                FilterExpression='compound_class = :class',
                ExpressionAttributeValues={':class': compound_class}
            )
            items = response['Items']

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression='compound_class = :class',
                    ExpressionAttributeValues={':class': compound_class},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response['Items'])

            logger.info(f"Retrieved {len(items)} items")
            return items

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def filter_by_properties(self, mw_max=None, logp_max=None, lipinski_only=False):
        """
        Filter molecules by property ranges.

        Args:
            mw_max (float): Maximum molecular weight
            logp_max (float): Maximum LogP
            lipinski_only (bool): Only Lipinski-compliant molecules

        Returns:
            list: Filtered molecule list
        """
        logger.info("Filtering molecules by properties...")

        items = self.scan_all()
        filtered = []

        for item in items:
            # Check molecular weight
            if mw_max and float(item.get('molecular_weight', 9999)) > mw_max:
                continue

            # Check LogP
            if logp_max and float(item.get('logp', 9999)) > logp_max:
                continue

            # Check Lipinski compliance
            if lipinski_only and not item.get('lipinski_compliant', False):
                continue

            filtered.append(item)

        logger.info(f"Filtered to {len(filtered)} molecules")
        return filtered

    def find_drug_like(self):
        """
        Find drug-like molecules (Lipinski's Rule of Five).

        Returns:
            list: Drug-like molecules
        """
        logger.info("Finding drug-like molecules (Lipinski compliant)...")

        items = self.scan_all()
        drug_like = [item for item in items if item.get('lipinski_compliant', False)]

        logger.info(f"Found {len(drug_like)} drug-like molecules")
        return drug_like

    def get_statistics(self, items=None):
        """
        Calculate statistics for molecular properties.

        Args:
            items (list): List of molecule items (or scan all if None)

        Returns:
            dict: Statistics dictionary
        """
        if items is None:
            items = self.scan_all()

        if not items:
            logger.warning("No items to analyze")
            return {}

        # Convert to DataFrame for easy statistics
        df = self.to_dataframe(items)

        stats = {
            'total_molecules': len(df),
            'lipinski_compliant': int(df['lipinski_compliant'].sum()),
            'lipinski_percentage': float(df['lipinski_compliant'].sum() / len(df) * 100),
            'molecular_weight': {
                'mean': float(df['molecular_weight'].mean()),
                'std': float(df['molecular_weight'].std()),
                'min': float(df['molecular_weight'].min()),
                'max': float(df['molecular_weight'].max())
            },
            'logp': {
                'mean': float(df['logp'].mean()),
                'std': float(df['logp'].std()),
                'min': float(df['logp'].min()),
                'max': float(df['logp'].max())
            },
            'tpsa': {
                'mean': float(df['tpsa'].mean()),
                'std': float(df['tpsa'].std()),
                'min': float(df['tpsa'].min()),
                'max': float(df['tpsa'].max())
            },
            'compound_classes': df['compound_class'].value_counts().to_dict()
        }

        return stats

    def to_dataframe(self, items):
        """
        Convert DynamoDB items to pandas DataFrame.

        Args:
            items (list): List of DynamoDB items

        Returns:
            pd.DataFrame: DataFrame with molecular properties
        """
        # Convert Decimal to float for pandas
        cleaned_items = []
        for item in items:
            cleaned = {}
            for key, value in item.items():
                if isinstance(value, Decimal):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value
            cleaned_items.append(cleaned)

        df = pd.DataFrame(cleaned_items)

        # Ensure numeric columns
        numeric_cols = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def export_csv(self, items, output_file):
        """
        Export results to CSV file.

        Args:
            items (list): List of molecule items
            output_file (str): Output CSV file path
        """
        df = self.to_dataframe(items)
        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(df)} molecules to {output_file}")

    def export_json(self, items, output_file):
        """
        Export results to JSON file.

        Args:
            items (list): List of molecule items
            output_file (str): Output JSON file path
        """
        # Convert Decimal to float for JSON serialization
        cleaned_items = []
        for item in items:
            cleaned = {}
            for key, value in item.items():
                if isinstance(value, Decimal):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = value
            cleaned_items.append(cleaned)

        with open(output_file, 'w') as f:
            json.dump(cleaned_items, f, indent=2)

        logger.info(f"Exported {len(items)} molecules to {output_file}")

    def display_summary(self, items):
        """
        Display summary of molecular properties.

        Args:
            items (list): List of molecule items
        """
        if not items:
            print("No molecules found")
            return

        df = self.to_dataframe(items)

        print(f"\n{'='*70}")
        print(f"Molecular Property Summary")
        print(f"{'='*70}")
        print(f"\nTotal molecules: {len(df)}")

        # Lipinski compliance
        lipinski_count = df['lipinski_compliant'].sum()
        lipinski_pct = (lipinski_count / len(df)) * 100
        print(f"Lipinski compliant: {lipinski_count} ({lipinski_pct:.1f}%)")

        # Compound classes
        print(f"\nCompound classes:")
        for class_name, count in df['compound_class'].value_counts().items():
            print(f"  {class_name}: {count}")

        # Property statistics
        print(f"\nProperty Statistics:")
        print(f"{'Property':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'-'*70}")

        for prop in ['molecular_weight', 'logp', 'tpsa']:
            if prop in df.columns:
                mean = df[prop].mean()
                std = df[prop].std()
                min_val = df[prop].min()
                max_val = df[prop].max()
                print(f"{prop:<20} {mean:<10.2f} {std:<10.2f} {min_val:<10.2f} {max_val:<10.2f}")

        print(f"\n{'='*70}\n")

        # Top 10 molecules by molecular weight
        print("Top 10 molecules by molecular weight:")
        top_mw = df.nlargest(10, 'molecular_weight')[['name', 'molecular_weight', 'logp', 'lipinski_compliant']]
        print(top_mw.to_string(index=False))
        print()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Query molecular properties from DynamoDB'
    )
    parser.add_argument(
        '--table',
        default='MolecularProperties',
        help='DynamoDB table name'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region'
    )
    parser.add_argument(
        '--profile',
        help='AWS profile name'
    )
    parser.add_argument(
        '--compound-class',
        help='Filter by compound class'
    )
    parser.add_argument(
        '--mw-max',
        type=float,
        help='Maximum molecular weight'
    )
    parser.add_argument(
        '--logp-max',
        type=float,
        help='Maximum LogP'
    )
    parser.add_argument(
        '--lipinski-only',
        action='store_true',
        help='Only Lipinski-compliant molecules'
    )
    parser.add_argument(
        '--drug-like',
        action='store_true',
        help='Find drug-like molecules'
    )
    parser.add_argument(
        '--output',
        help='Output file (CSV or JSON)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json'],
        default='csv',
        help='Output format'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Display statistics only'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of results'
    )

    args = parser.parse_args()

    try:
        # Initialize query client
        query = MolecularPropertyQuery(args.table, args.region, args.profile)

        # Query molecules
        if args.compound_class:
            items = query.query_by_class(args.compound_class)
        elif args.drug_like:
            items = query.find_drug_like()
        elif args.mw_max or args.logp_max or args.lipinski_only:
            items = query.filter_by_properties(
                mw_max=args.mw_max,
                logp_max=args.logp_max,
                lipinski_only=args.lipinski_only
            )
        else:
            items = query.scan_all(limit=args.limit)

        # Display summary
        query.display_summary(items)

        # Display statistics
        if args.stats:
            stats = query.get_statistics(items)
            print(json.dumps(stats, indent=2))

        # Export results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.format == 'csv':
                query.export_csv(items, args.output)
            else:
                query.export_json(items, args.output)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
