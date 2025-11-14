"""
Query variant results from DynamoDB and S3.

This script:
1. Queries DynamoDB for variant metadata
2. Downloads VCF files from S3
3. Generates summary statistics
4. Creates visualizations

Usage:
    python query_results.py --sample NA12878 --region chr20
    python query_results.py --quality-filter 30 --output results.csv
"""

import os
import sys
import argparse
import boto3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()

    config = {
        'bucket_results': os.getenv('BUCKET_RESULTS'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'table_name': os.getenv('TABLE_NAME', 'variant-metadata'),
    }

    if not config['bucket_results']:
        raise ValueError("BUCKET_RESULTS not set in .env file")

    return config


def query_dynamodb_by_sample(
    table_name: str,
    sample_id: str,
    region: str = 'us-east-1',
    min_quality: Optional[int] = None
) -> pd.DataFrame:
    """
    Query DynamoDB for variants from a specific sample.

    Args:
        table_name: DynamoDB table name
        sample_id: Sample identifier to query
        region: AWS region
        min_quality: Optional minimum quality threshold

    Returns:
        DataFrame with variant data
    """

    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)

    print(f"Querying DynamoDB table '{table_name}' for sample '{sample_id}'")

    try:
        # Use Scan with filter expression
        response = table.scan(
            FilterExpression='sample_id = :sample',
            ExpressionAttributeValues={':sample': sample_id}
        )

        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='sample_id = :sample',
                ExpressionAttributeValues={':sample': sample_id},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        # Convert to DataFrame
        if items:
            df = pd.DataFrame(items)
            print(f"✓ Retrieved {len(df)} variants")

            # Apply quality filter if specified
            if min_quality:
                df = df[df['quality'] >= min_quality]
                print(f"✓ Filtered to {len(df)} variants with quality >= {min_quality}")

            return df
        else:
            print("✗ No variants found for this sample")
            return pd.DataFrame()

    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        raise


def query_dynamodb_by_region(
    table_name: str,
    region_str: str,
    aws_region: str = 'us-east-1'
) -> pd.DataFrame:
    """
    Query DynamoDB for variants in a specific genomic region.

    Args:
        table_name: DynamoDB table name
        region_str: Genomic region (e.g., "chr20:1000000-1010000")
        aws_region: AWS region

    Returns:
        DataFrame with variant data
    """

    dynamodb = boto3.resource('dynamodb', region_name=aws_region)
    table = dynamodb.Table(table_name)

    print(f"Querying DynamoDB table '{table_name}' for region '{region_str}'")

    try:
        # Parse region
        if ':' in region_str:
            chrom, pos_range = region_str.split(':')
            start, end = map(int, pos_range.split('-'))
        else:
            chrom = region_str
            start = None
            end = None

        # Scan with filters
        response = table.scan(
            FilterExpression='chromosome = :chrom',
            ExpressionAttributeValues={':chrom': chrom}
        )

        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='chromosome = :chrom',
                ExpressionAttributeValues={':chrom': chrom},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))

        # Filter by position if specified
        if start and end:
            items = [
                item for item in items
                if start <= item.get('position', 0) <= end
            ]

        if items:
            df = pd.DataFrame(items)
            print(f"✓ Retrieved {len(df)} variants in {region_str}")
            return df
        else:
            print(f"✗ No variants found in {region_str}")
            return pd.DataFrame()

    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        raise


def download_vcf_from_s3(
    bucket: str,
    vcf_key: str,
    local_path: str,
    region: str = 'us-east-1'
) -> None:
    """
    Download VCF file from S3.

    Args:
        bucket: S3 bucket name
        vcf_key: S3 object key for VCF file
        local_path: Local path to save VCF
        region: AWS region
    """

    s3 = boto3.client('s3', region_name=region)

    print(f"Downloading VCF from s3://{bucket}/{vcf_key}")

    try:
        s3.download_file(bucket, vcf_key, local_path)
        file_size = os.path.getsize(local_path) / 1024
        print(f"✓ Downloaded: {local_path} ({file_size:.2f} KB)")
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        raise


def list_vcf_files(
    bucket: str,
    prefix: str = 'results/',
    region: str = 'us-east-1'
) -> List[str]:
    """
    List all VCF files in S3 bucket.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to search
        region: AWS region

    Returns:
        List of S3 keys for VCF files
    """

    s3 = boto3.client('s3', region_name=region)

    print(f"Listing VCF files in s3://{bucket}/{prefix}")

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        vcf_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.vcf') or obj['Key'].endswith('.vcf.gz'):
                    vcf_files.append(obj['Key'])

        print(f"✓ Found {len(vcf_files)} VCF files")
        for vcf in vcf_files:
            print(f"  - {vcf}")

        return vcf_files

    except Exception as e:
        print(f"✗ Listing failed: {str(e)}")
        raise


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for variants.

    Args:
        df: DataFrame with variant data

    Returns:
        Dictionary with summary statistics
    """

    if df.empty:
        return {}

    stats = {
        'total_variants': len(df),
        'chromosomes': df['chromosome'].nunique() if 'chromosome' in df else 0,
        'avg_quality': float(df['quality'].mean()) if 'quality' in df else 0,
        'avg_depth': float(df['depth'].mean()) if 'depth' in df else 0,
        'avg_allele_freq': float(df['allele_freq'].mean()) if 'allele_freq' in df else 0,
    }

    # Count variant types
    if 'ref' in df and 'alt' in df:
        df['var_type'] = df.apply(
            lambda x: 'SNP' if len(x['ref']) == len(x['alt']) else 'INDEL',
            axis=1
        )
        stats['snp_count'] = int((df['var_type'] == 'SNP').sum())
        stats['indel_count'] = int((df['var_type'] == 'INDEL').sum())

    return stats


def print_summary_statistics(stats: Dict) -> None:
    """Print summary statistics in human-readable format."""

    print("\n" + "=" * 60)
    print("Variant Summary Statistics")
    print("=" * 60)

    print(f"Total Variants: {stats.get('total_variants', 'N/A')}")
    print(f"Chromosomes: {stats.get('chromosomes', 'N/A')}")
    print(f"SNPs: {stats.get('snp_count', 'N/A')}")
    print(f"Indels: {stats.get('indel_count', 'N/A')}")
    print(f"Avg Quality: {stats.get('avg_quality', 'N/A'):.2f}")
    print(f"Avg Depth: {stats.get('avg_depth', 'N/A'):.2f}")
    print(f"Avg Allele Frequency: {stats.get('avg_allele_freq', 'N/A'):.4f}")
    print("=" * 60)


def export_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Export variant data to CSV."""

    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} variants to {output_path}")


def export_to_json(df: pd.DataFrame, output_path: str) -> None:
    """Export variant data to JSON."""

    df.to_json(output_path, orient='records', indent=2)
    print(f"✓ Exported {len(df)} variants to {output_path}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Query variant results from DynamoDB and S3'
    )
    parser.add_argument(
        '--sample',
        help='Sample ID to query'
    )
    parser.add_argument(
        '--region',
        help='Genomic region (e.g., chr20:1000000-1010000)'
    )
    parser.add_argument(
        '--quality-filter',
        type=int,
        help='Minimum quality threshold'
    )
    parser.add_argument(
        '--list-vcf',
        action='store_true',
        help='List all VCF files in results bucket'
    )
    parser.add_argument(
        '--download-vcf',
        help='Download VCF file (specify S3 key)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (CSV or JSON based on extension)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    print("=" * 60)
    print("Genomic Variant Analysis - Query Results")
    print("=" * 60)
    print(f"DynamoDB Table: {config['table_name']}")
    print(f"Results Bucket: {config['bucket_results']}")
    print()

    try:
        # List VCF files
        if args.list_vcf:
            vcf_files = list_vcf_files(
                config['bucket_results'],
                region=config['region']
            )
            print()

        # Download VCF
        if args.download_vcf:
            output_path = args.output or Path(args.download_vcf).name
            download_vcf_from_s3(
                config['bucket_results'],
                args.download_vcf,
                output_path,
                region=config['region']
            )
            print()

        # Query DynamoDB
        df = None

        if args.sample:
            df = query_dynamodb_by_sample(
                config['table_name'],
                args.sample,
                region=config['region'],
                min_quality=args.quality_filter
            )
        elif args.region:
            df = query_dynamodb_by_region(
                config['table_name'],
                args.region,
                aws_region=config['region']
            )

        # Generate and print statistics
        if df is not None and not df.empty:
            stats = generate_summary_statistics(df)
            print_summary_statistics(stats)

            # Export results
            if args.output:
                if args.output.endswith('.json'):
                    export_to_json(df, args.output)
                else:
                    export_to_csv(df, args.output)
                print()

        print("✓ Query complete!")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
