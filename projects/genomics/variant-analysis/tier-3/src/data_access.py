"""Data access for genomic variant data from S3 and local sources."""

import boto3
import pandas as pd
import logging
from typing import Optional, List, Dict
import io

logger = logging.getLogger(__name__)


class GenomicsDataAccess:
    """Handle loading and saving genomic variant data."""

    def __init__(self, use_anon: bool = False, region: str = 'us-east-1'):
        """
        Initialize data access client.

        Args:
            use_anon: Use anonymous access (for public datasets)
            region: AWS region
        """
        if use_anon:
            self.s3_client = boto3.client('s3', region_name=region,
                                        config=boto3.session.Config(signature_version=boto3.session.UNSIGNED))
        else:
            self.s3_client = boto3.client('s3', region_name=region)

        self.use_anon = use_anon
        logger.info(f"Initialized GenomicsDataAccess (anon={use_anon})")

    def load_vcf_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """
        Load VCF file from S3 and parse into DataFrame.

        Args:
            bucket: S3 bucket name
            key: S3 object key (path to VCF file)

        Returns:
            DataFrame with variant information
        """
        logger.info(f"Loading VCF from s3://{bucket}/{key}")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            vcf_content = response['Body'].read().decode('utf-8')

            # Parse VCF
            variants = []
            for line in vcf_content.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue

                # Parse INFO field
                info_dict = {}
                for item in fields[7].split(';'):
                    if '=' in item:
                        key_info, value = item.split('=', 1)
                        info_dict[key_info] = value

                variant = {
                    'CHROM': fields[0],
                    'POS': int(fields[1]),
                    'ID': fields[2] if fields[2] != '.' else None,
                    'REF': fields[3],
                    'ALT': fields[4],
                    'QUAL': float(fields[5]) if fields[5] != '.' else None,
                    'FILTER': fields[6],
                    'DP': int(info_dict.get('DP', 0)),
                    'AF': float(info_dict.get('AF', 0)),
                    'TYPE': info_dict.get('TYPE', 'UNKNOWN')
                }
                variants.append(variant)

            df = pd.DataFrame(variants)
            logger.info(f"Loaded {len(df)} variants")
            return df

        except Exception as e:
            logger.error(f"Error loading VCF: {e}")
            raise

    def load_vcf_from_local(self, file_path: str) -> pd.DataFrame:
        """
        Load VCF file from local filesystem.

        Args:
            file_path: Path to local VCF file

        Returns:
            DataFrame with variant information
        """
        logger.info(f"Loading VCF from {file_path}")

        variants = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 8:
                    continue

                info_dict = {}
                for item in fields[7].split(';'):
                    if '=' in item:
                        key_info, value = item.split('=', 1)
                        info_dict[key_info] = value

                variant = {
                    'CHROM': fields[0],
                    'POS': int(fields[1]),
                    'ID': fields[2] if fields[2] != '.' else None,
                    'REF': fields[3],
                    'ALT': fields[4],
                    'QUAL': float(fields[5]) if fields[5] != '.' else None,
                    'FILTER': fields[6],
                    'DP': int(info_dict.get('DP', 0)),
                    'AF': float(info_dict.get('AF', 0)),
                    'TYPE': info_dict.get('TYPE', 'UNKNOWN')
                }
                variants.append(variant)

        df = pd.DataFrame(variants)
        logger.info(f"Loaded {len(df)} variants")
        return df

    def save_results(self, df: pd.DataFrame, bucket: str, key: str):
        """
        Save analysis results to S3.

        Args:
            df: DataFrame to save
            bucket: S3 bucket name
            key: S3 object key
        """
        logger.info(f"Saving results to s3://{bucket}/{key}")

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )

        logger.info("Results saved successfully")

    def list_vcf_files(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List VCF files in S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix to filter

        Returns:
            List of S3 keys for VCF files
        """
        logger.info(f"Listing VCF files in s3://{bucket}/{prefix}")

        vcf_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith(('.vcf', '.vcf.gz')):
                        vcf_files.append(obj['Key'])

        logger.info(f"Found {len(vcf_files)} VCF files")
        return vcf_files
