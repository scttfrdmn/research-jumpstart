#!/usr/bin/env python3
"""
Query materials properties from DynamoDB.

This script queries the DynamoDB table for materials matching
specific criteria and displays results in a formatted table.
"""

import boto3
import argparse
import json
from decimal import Decimal
from boto3.dynamodb.conditions import Key, Attr
import pandas as pd
from datetime import datetime
import sys

# Configure boto3
dynamodb = boto3.resource('dynamodb')


class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal to float for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def query_all_materials(table_name):
    """
    Query all materials from DynamoDB.

    Args:
        table_name (str): DynamoDB table name

    Returns:
        list: List of material dictionaries
    """
    table = dynamodb.Table(table_name)

    print(f"Scanning table: {table_name}")

    response = table.scan()
    items = response['Items']

    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    print(f"Found {len(items)} materials")
    return items


def query_by_property(table_name, property_name, min_value=None, max_value=None, exact_value=None):
    """
    Query materials by property value.

    Args:
        table_name (str): DynamoDB table name
        property_name (str): Property to filter by
        min_value (float): Minimum value
        max_value (float): Maximum value
        exact_value: Exact value to match

    Returns:
        list: Filtered materials
    """
    table = dynamodb.Table(table_name)

    # Build filter expression
    if exact_value is not None:
        if isinstance(exact_value, str):
            filter_expr = Attr(property_name).eq(exact_value)
        else:
            filter_expr = Attr(property_name).eq(Decimal(str(exact_value)))
    elif min_value is not None and max_value is not None:
        filter_expr = Attr(property_name).between(
            Decimal(str(min_value)),
            Decimal(str(max_value))
        )
    elif min_value is not None:
        filter_expr = Attr(property_name).gte(Decimal(str(min_value)))
    elif max_value is not None:
        filter_expr = Attr(property_name).lte(Decimal(str(max_value)))
    else:
        # No filter
        return query_all_materials(table_name)

    print(f"Filtering by {property_name}: ", end="")
    if exact_value is not None:
        print(f"= {exact_value}")
    elif min_value is not None and max_value is not None:
        print(f"{min_value} to {max_value}")
    elif min_value is not None:
        print(f">= {min_value}")
    else:
        print(f"<= {max_value}")

    response = table.scan(FilterExpression=filter_expr)
    items = response['Items']

    # Handle pagination
    while 'LastEvaluatedKey' in response:
        response = table.scan(
            FilterExpression=filter_expr,
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        items.extend(response['Items'])

    print(f"Found {len(items)} matching materials")
    return items


def display_materials(materials, format='table'):
    """
    Display materials in formatted output.

    Args:
        materials (list): List of material dictionaries
        format (str): Output format ('table', 'json', 'csv')
    """
    if not materials:
        print("No materials to display")
        return

    # Convert Decimals to floats for pandas
    materials_converted = []
    for mat in materials:
        mat_dict = {}
        for key, value in mat.items():
            if isinstance(value, Decimal):
                mat_dict[key] = float(value)
            else:
                mat_dict[key] = value
        materials_converted.append(mat_dict)

    df = pd.DataFrame(materials_converted)

    # Reorder columns for better display
    preferred_cols = ['material_id', 'formula', 'density', 'volume',
                     'num_atoms', 'space_group', 'crystal_system',
                     'lattice_a', 'lattice_b', 'lattice_c']
    cols = [c for c in preferred_cols if c in df.columns] + \
           [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    if format == 'json':
        print(json.dumps(materials_converted, indent=2, cls=DecimalEncoder))
    elif format == 'csv':
        print(df.to_csv(index=False))
    else:
        # Table format
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        print("\n" + "="*100)
        print(df.to_string(index=False))
        print("="*100)


def export_to_csv(materials, filename):
    """
    Export materials to CSV file.

    Args:
        materials (list): List of material dictionaries
        filename (str): Output filename
    """
    if not materials:
        print("No materials to export")
        return

    # Convert Decimals to floats
    materials_converted = []
    for mat in materials:
        mat_dict = {}
        for key, value in mat.items():
            if isinstance(value, Decimal):
                mat_dict[key] = float(value)
            else:
                mat_dict[key] = value
        materials_converted.append(mat_dict)

    df = pd.DataFrame(materials_converted)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Exported {len(materials)} materials to {filename}")


def get_statistics(materials):
    """
    Calculate statistics for material properties.

    Args:
        materials (list): List of material dictionaries

    Returns:
        dict: Statistics
    """
    if not materials:
        return {}

    # Convert to DataFrame
    materials_converted = []
    for mat in materials:
        mat_dict = {}
        for key, value in mat.items():
            if isinstance(value, Decimal):
                mat_dict[key] = float(value)
            else:
                mat_dict[key] = value
        materials_converted.append(mat_dict)

    df = pd.DataFrame(materials_converted)

    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }

    # Crystal system distribution
    if 'crystal_system' in df.columns:
        stats['crystal_system_counts'] = df['crystal_system'].value_counts().to_dict()

    # Space group distribution
    if 'space_group' in df.columns:
        stats['space_group_counts'] = df['space_group'].value_counts().to_dict()

    return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Query materials properties from DynamoDB'
    )
    parser.add_argument(
        '--table',
        default='MaterialsProperties',
        help='DynamoDB table name'
    )
    parser.add_argument(
        '--property',
        help='Property to filter by (e.g., density, volume, space_group)'
    )
    parser.add_argument(
        '--min',
        type=float,
        help='Minimum value for numeric property'
    )
    parser.add_argument(
        '--max',
        type=float,
        help='Maximum value for numeric property'
    )
    parser.add_argument(
        '--value',
        help='Exact value to match'
    )
    parser.add_argument(
        '--format',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format'
    )
    parser.add_argument(
        '--export',
        help='Export to CSV file'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region'
    )

    args = parser.parse_args()

    try:
        # Override region if specified
        global dynamodb
        dynamodb = boto3.resource('dynamodb', region_name=args.region)

        # Query materials
        if args.property:
            materials = query_by_property(
                args.table,
                args.property,
                args.min,
                args.max,
                args.value
            )
        else:
            materials = query_all_materials(args.table)

        # Display results
        if materials:
            display_materials(materials, args.format)

            # Export if requested
            if args.export:
                export_to_csv(materials, args.export)

            # Show statistics if requested
            if args.stats:
                print("\n" + "="*100)
                print("STATISTICS")
                print("="*100)
                stats = get_statistics(materials)
                print(json.dumps(stats, indent=2, cls=DecimalEncoder))
        else:
            print("No materials found matching criteria")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
