"""
Upload archaeological artifact data to S3.

This script:
1. Generates sample artifact dataset (or uses your own data)
2. Uploads CSV files with artifact measurements to S3
3. Optionally uploads artifact images
4. Creates metadata files

Artifact types supported:
- Pottery (ceramics)
- Lithics (stone tools)
- Bones (faunal remains)
- Coins (numismatics)
- Architecture (building remains)

Usage:
    python upload_to_s3.py
    python upload_to_s3.py --bucket archaeology-data-xxxx --site-id SITE_A
    python upload_to_s3.py --use-local-data --data-path ./my_artifacts.csv
"""

import os
import sys
import argparse
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()

    config = {
        'bucket': os.getenv('BUCKET_NAME'),
        'region': os.getenv('AWS_REGION', 'us-east-1'),
        'table_name': os.getenv('TABLE_NAME', 'ArtifactCatalog'),
    }

    if not config['bucket']:
        print("Warning: BUCKET_NAME not set in .env file")
        print("You can specify it with --bucket flag")

    return config


def generate_sample_artifacts(num_artifacts: int = 100, site_id: str = "SITE_A") -> pd.DataFrame:
    """
    Generate sample archaeological artifact dataset.

    Creates realistic artifact data including:
    - Multiple artifact types
    - Realistic measurements
    - GPS coordinates
    - Stratigraphic context
    - Dating information
    """
    np.random.seed(42)

    # Define artifact types and materials
    artifact_types = {
        'pottery': ['ceramic', 'terracotta', 'glazed ceramic'],
        'lithic': ['flint', 'obsidian', 'chert', 'quartzite', 'basalt'],
        'bone': ['mammal bone', 'bird bone', 'fish bone'],
        'coin': ['bronze', 'silver', 'gold', 'copper'],
        'architecture': ['brick', 'stone', 'mortar', 'tile']
    }

    # Archaeological periods
    periods = ['Neolithic', 'Bronze Age', 'Iron Age', 'Classical', 'Medieval', 'Post-Medieval']

    # Generate data
    artifacts = []

    for i in range(num_artifacts):
        # Select artifact type
        artifact_type = np.random.choice(list(artifact_types.keys()))
        material = np.random.choice(artifact_types[artifact_type])

        # Generate measurements based on artifact type
        if artifact_type == 'pottery':
            length = np.random.normal(150, 40)  # mm
            width = np.random.normal(120, 30)
            thickness = np.random.normal(8, 2)
            weight = np.random.normal(300, 100)  # grams
        elif artifact_type == 'lithic':
            length = np.random.normal(50, 20)
            width = np.random.normal(35, 15)
            thickness = np.random.normal(10, 5)
            weight = np.random.normal(40, 20)
        elif artifact_type == 'bone':
            length = np.random.normal(80, 30)
            width = np.random.normal(25, 10)
            thickness = np.random.normal(15, 5)
            weight = np.random.normal(50, 25)
        elif artifact_type == 'coin':
            length = np.random.normal(20, 3)
            width = np.random.normal(20, 3)
            thickness = np.random.normal(2, 0.5)
            weight = np.random.normal(5, 2)
        else:  # architecture
            length = np.random.normal(250, 50)
            width = np.random.normal(150, 30)
            thickness = np.random.normal(50, 10)
            weight = np.random.normal(2000, 500)

        # GPS coordinates (example: Mediterranean region)
        gps_lat = np.random.normal(40.0, 0.05)
        gps_lon = np.random.normal(20.0, 0.05)

        # Stratigraphic unit (deeper = older)
        strat_unit = f"Layer_{np.random.randint(1, 6)}"

        # Period (correlate with stratigraphic depth)
        layer_num = int(strat_unit.split('_')[1])
        period = periods[min(layer_num - 1, len(periods) - 1)]

        # Dating (varies by period)
        if period == 'Neolithic':
            dating_value = -np.random.randint(8000, 4000)
        elif period == 'Bronze Age':
            dating_value = -np.random.randint(3300, 1200)
        elif period == 'Iron Age':
            dating_value = -np.random.randint(1200, 500)
        elif period == 'Classical':
            dating_value = -np.random.randint(800, 300)
        elif period == 'Medieval':
            dating_value = np.random.randint(500, 1500)
        else:  # Post-Medieval
            dating_value = np.random.randint(1500, 1900)

        # Excavation date
        days_ago = np.random.randint(0, 180)
        excavation_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        artifact = {
            'artifact_id': f"ART_{site_id}_{i+1:04d}",
            'site_id': site_id,
            'artifact_type': artifact_type,
            'material': material,
            'length': round(max(1, length), 2),
            'width': round(max(1, width), 2),
            'thickness': round(max(0.5, thickness), 2),
            'weight': round(max(0.1, weight), 2),
            'gps_lat': round(gps_lat, 6),
            'gps_lon': round(gps_lon, 6),
            'stratigraphic_unit': strat_unit,
            'period': period,
            'dating_method': 'relative' if layer_num > 3 else 'radiocarbon',
            'dating_value': dating_value,
            'excavation_date': excavation_date,
            'notes': f'{artifact_type.capitalize()} fragment, {material}'
        }

        artifacts.append(artifact)

    df = pd.DataFrame(artifacts)

    print(f"Generated {len(df)} sample artifacts")
    print(f"\nArtifact type distribution:")
    print(df['artifact_type'].value_counts())
    print(f"\nPeriod distribution:")
    print(df['period'].value_counts())

    return df


def upload_to_s3(
    local_path: str,
    bucket: str,
    s3_key: str,
    region: str,
    show_progress: bool = True
) -> None:
    """Upload file to S3 with progress bar."""

    s3 = boto3.client('s3', region_name=region)

    file_size = os.path.getsize(local_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
    print(f"File size: {file_size_mb:.2f} MB")

    try:
        if show_progress and file_size > 1024 * 1024:  # Show progress for files > 1MB
            # Upload with progress tracking
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
                s3.upload_file(
                    local_path,
                    bucket,
                    s3_key,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                )
        else:
            s3.upload_file(local_path, bucket, s3_key)

        print(f"✓ Uploaded successfully")
    except Exception as e:
        print(f"✗ Upload failed: {str(e)}")
        raise


def create_site_metadata(site_id: str, df: pd.DataFrame) -> Dict:
    """Create metadata JSON for the archaeological site."""

    metadata = {
        'site_id': site_id,
        'site_name': f'Archaeological Site {site_id}',
        'location': {
            'center_lat': float(df['gps_lat'].mean()),
            'center_lon': float(df['gps_lon'].mean()),
            'bounding_box': {
                'north': float(df['gps_lat'].max()),
                'south': float(df['gps_lat'].min()),
                'east': float(df['gps_lon'].max()),
                'west': float(df['gps_lon'].min())
            }
        },
        'statistics': {
            'total_artifacts': len(df),
            'artifact_types': df['artifact_type'].value_counts().to_dict(),
            'periods': df['period'].value_counts().to_dict(),
            'date_range': {
                'earliest': int(df['dating_value'].min()),
                'latest': int(df['dating_value'].max())
            }
        },
        'excavation': {
            'start_date': df['excavation_date'].min(),
            'end_date': df['excavation_date'].max(),
            'stratigraphic_units': sorted(df['stratigraphic_unit'].unique().tolist())
        },
        'created': datetime.now().isoformat(),
        'project': 'Archaeological Site Analysis Tier 2'
    }

    return metadata


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Upload archaeological artifact data to S3'
    )
    parser.add_argument(
        '--bucket',
        help='S3 bucket name (overrides .env)'
    )
    parser.add_argument(
        '--site-id',
        default='SITE_A',
        help='Site identifier (default: SITE_A)'
    )
    parser.add_argument(
        '--num-artifacts',
        type=int,
        default=100,
        help='Number of artifacts to generate (default: 100)'
    )
    parser.add_argument(
        '--use-local-data',
        action='store_true',
        help='Use local CSV file instead of generating data'
    )
    parser.add_argument(
        '--data-path',
        help='Path to local artifact CSV file'
    )
    parser.add_argument(
        '--skip-upload',
        action='store_true',
        help='Generate data but skip S3 upload'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    if args.bucket:
        config['bucket'] = args.bucket

    if not config['bucket'] and not args.skip_upload:
        print("Error: No S3 bucket specified")
        print("Either set BUCKET_NAME in .env or use --bucket flag")
        sys.exit(1)

    print("=" * 60)
    print("Archaeological Artifact Data Upload to S3")
    print("=" * 60)
    print(f"Bucket: {config['bucket']}")
    print(f"Region: {config['region']}")
    print(f"Site ID: {args.site_id}")
    print()

    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Generate or load artifact data
        if args.use_local_data:
            if not args.data_path:
                print("Error: --data-path required when using --use-local-data")
                sys.exit(1)

            print(f"Loading artifact data from {args.data_path}")
            df = pd.read_csv(args.data_path)
            print(f"Loaded {len(df)} artifacts")
        else:
            print(f"Generating {args.num_artifacts} sample artifacts...")
            df = generate_sample_artifacts(args.num_artifacts, args.site_id)

        # Save locally
        csv_path = data_dir / f"{args.site_id}_artifacts.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved artifact data: {csv_path}")

        if args.skip_upload:
            print("\nSkipping S3 upload (--skip-upload flag set)")
            print(f"Data saved locally at: {csv_path}")
            return

        # Step 2: Upload artifact CSV to S3
        print(f"\nUploading artifact data to S3...")
        upload_to_s3(
            str(csv_path),
            config['bucket'],
            f"raw/{csv_path.name}",
            config['region']
        )

        # Step 3: Create and upload site metadata
        print(f"\nCreating site metadata...")
        metadata = create_site_metadata(args.site_id, df)

        metadata_path = data_dir / f"{args.site_id}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Created metadata: {metadata_path}")

        print(f"\nUploading site metadata to S3...")
        upload_to_s3(
            str(metadata_path),
            config['bucket'],
            f"raw/{metadata_path.name}",
            config['region'],
            show_progress=False
        )

        # Step 4: Upload summary statistics
        print(f"\nCreating summary statistics...")
        summary = {
            'site_id': args.site_id,
            'total_artifacts': len(df),
            'upload_date': datetime.now().isoformat(),
            'artifact_summary': {
                'by_type': df.groupby('artifact_type').agg({
                    'artifact_id': 'count',
                    'length': ['mean', 'std'],
                    'width': ['mean', 'std'],
                    'weight': ['mean', 'std']
                }).to_dict(),
                'by_period': df.groupby('period').agg({
                    'artifact_id': 'count',
                    'dating_value': ['min', 'max', 'mean']
                }).to_dict()
            }
        }

        summary_path = data_dir / f"{args.site_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        upload_to_s3(
            str(summary_path),
            config['bucket'],
            f"raw/{summary_path.name}",
            config['region'],
            show_progress=False
        )

        print()
        print("=" * 60)
        print("✓ Upload Complete!")
        print("=" * 60)
        print()
        print(f"Uploaded files:")
        print(f"  - s3://{config['bucket']}/raw/{csv_path.name}")
        print(f"  - s3://{config['bucket']}/raw/{metadata_path.name}")
        print(f"  - s3://{config['bucket']}/raw/{summary_path.name}")
        print()
        print("Next steps:")
        print("1. If S3 trigger is configured, Lambda will process automatically")
        print("2. Or manually invoke: python scripts/invoke_lambda.py")
        print("3. Query results: python scripts/query_results.py")
        print("4. Analyze: jupyter notebook notebooks/archaeology_analysis.ipynb")
        print("5. Cleanup when done: see cleanup_guide.md")
        print()

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
