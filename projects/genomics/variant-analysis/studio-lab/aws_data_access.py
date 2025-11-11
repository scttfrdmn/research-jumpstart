#!/usr/bin/env python3
"""
AWS Open Data Access for Genomics

Access genomics datasets from AWS Open Data Registry:
- 1000 Genomes Project
- TCGA (The Cancer Genome Atlas)
- gnomAD (Genome Aggregation Database)
- NIH NCBI SRA (Sequence Read Archive)

Usage:
    from aws_data_access import list_1000genomes, download_sample_vcf

    # List 1000 Genomes VCF files
    files = list_1000genomes(chromosome='chr22', phase='phase3')

    # Download sample VCF
    download_sample_vcf(chromosome='chr22', output_dir='data/')

AWS Setup:
    # For public datasets, no credentials needed
    # Optional: configure for faster access
    aws configure
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config


# AWS Open Data Registry S3 buckets
BUCKETS = {
    '1000genomes': '1000genomes',  # Public
    'tcga': 'tcga-2-open',  # Public
    'gnomad': 'gnomad-public-us-east-1',  # Public
    'ncbi_sra': 'sra-pub-run-odp',  # Public
    'gatk': 'gatk-test-data'  # GATK test datasets
}


def get_s3_client(anonymous=True):
    """
    Create S3 client for accessing AWS Open Data.

    Parameters:
    -----------
    anonymous : bool
        If True, use unsigned requests (for public data)

    Returns:
    --------
    s3_client : boto3.client
    """
    if anonymous:
        config = Config(signature_version=UNSIGNED)
        return boto3.client('s3', config=config)
    else:
        return boto3.client('s3')


def list_1000genomes(phase='phase3', chromosome='chr22', data_type='vcf',
                     max_results=20, anonymous=True):
    """
    List 1000 Genomes Project files on AWS.

    The 1000 Genomes Project provides genetic variation data for
    2,504 individuals from 26 populations.

    Bucket structure:
    s3://1000genomes/phase3/data/...
    s3://1000genomes/phase3/integrated_sv_map/...

    Parameters:
    -----------
    phase : str
        Project phase ('phase1', 'phase3')
    chromosome : str
        Chromosome (e.g., 'chr22', 'chrX')
    data_type : str
        Data type ('vcf', 'bam', 'cram')
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS['1000genomes']

    print(f"Listing 1000 Genomes {phase} data on s3://{bucket}/")
    print(f"  Chromosome: {chromosome}")
    print(f"  Data type: {data_type}")

    try:
        files = []

        # Different prefixes for different data types
        if data_type == 'vcf':
            if phase == 'phase3':
                # Integrated variant calls
                prefix = f'{phase}/integrated_sv_map/supporting/genotypes/'
            else:
                prefix = f'{phase}/data/'
        elif data_type in ['bam', 'cram']:
            prefix = f'{phase}/data/'
        else:
            prefix = f'{phase}/'

        print(f"  Searching in: s3://{bucket}/{prefix}")

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1000)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                # Filter by chromosome and data type
                if chromosome.lower() in key.lower() and key.endswith(f'.{data_type}.gz'):
                    files.append(key)

                if len(files) >= max_results:
                    break

            if len(files) >= max_results:
                break

        if len(files) == 0:
            print(f"\nNo files found. Listing first 10 files in prefix:")
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
            if 'Contents' in response:
                for obj in response['Contents']:
                    print(f"  {obj['Key']}")

        print(f"\nFound {len(files)} matching files")
        return files

    except Exception as e:
        print(f"Error listing 1000 Genomes data: {e}")
        print("\nExample access:")
        print("  aws s3 ls s3://1000genomes/phase3/ --no-sign-request")
        return []


def download_sample_vcf(chromosome='chr22', output_dir='data',
                       phase='phase3', anonymous=True):
    """
    Download a sample VCF file from 1000 Genomes.

    Parameters:
    -----------
    chromosome : str
        Chromosome to download
    output_dir : str
        Output directory
    phase : str
        Project phase
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    local_path : str
        Path to downloaded file
    """
    # Find files
    files = list_1000genomes(
        phase=phase,
        chromosome=chromosome,
        data_type='vcf',
        max_results=1,
        anonymous=anonymous
    )

    if not files:
        print("No VCF files found")
        return None

    s3_key = files[0]
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS['1000genomes']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(s3_key))

    print(f"\nDownloading: s3://{bucket}/{s3_key}")
    print(f"         to: {output_path}")
    print(f"Note: VCF files are large, this may take several minutes...")

    try:
        s3.download_file(bucket, s3_key, output_path)
        print("Download complete!")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def list_tcga_data(project='TCGA-BRCA', data_category='Transcriptome Profiling',
                  max_results=20, anonymous=True):
    """
    List TCGA (The Cancer Genome Atlas) data on AWS.

    TCGA provides multi-omic cancer data including:
    - WGS/WXS (whole genome/exome sequencing)
    - RNA-Seq
    - miRNA-Seq
    - Methylation arrays
    - Clinical data

    Parameters:
    -----------
    project : str
        TCGA project code (e.g., 'TCGA-BRCA' for breast cancer)
    data_category : str
        Data category (e.g., 'Transcriptome Profiling', 'Simple Nucleotide Variation')
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS['tcga']

    print(f"Listing TCGA data on s3://{bucket}/")
    print(f"  Project: {project}")
    print(f"  Data category: {data_category}")

    try:
        files = []
        prefix = ''  # TCGA bucket has flat structure

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1000)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                # Filter by project code
                if project in key:
                    files.append(key)

                if len(files) >= max_results:
                    break

            if len(files) >= max_results:
                break

        print(f"\nFound {len(files)} matching files")
        return files

    except Exception as e:
        print(f"Error listing TCGA data: {e}")
        print("\nExample access:")
        print("  aws s3 ls s3://tcga-2-open/ --no-sign-request")
        return []


def list_gnomad_data(version='v3', data_type='vcf', max_results=20,
                    anonymous=True):
    """
    List gnomAD (Genome Aggregation Database) data on AWS.

    gnomAD provides allele frequencies from 125,748 exomes and
    71,702 genomes from diverse populations.

    Parameters:
    -----------
    version : str
        gnomAD version ('v2', 'v3')
    data_type : str
        Data type ('vcf', 'coverage')
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for matching files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS['gnomad']

    print(f"Listing gnomAD {version} data on s3://{bucket}/")
    print(f"  Data type: {data_type}")

    try:
        files = []
        prefix = f'release/{version}/'

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1000)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                if data_type in key:
                    files.append(key)

                if len(files) >= max_results:
                    break

            if len(files) >= max_results:
                break

        print(f"\nFound {len(files)} matching files")
        return files

    except Exception as e:
        print(f"Error listing gnomAD data: {e}")
        print("\nExample access:")
        print("  aws s3 ls s3://gnomad-public-us-east-1/release/ --no-sign-request")
        return []


def get_bucket_info():
    """
    Print information about AWS Open Data buckets for genomics.
    """
    print("AWS Open Data Registry - Genomics Datasets")
    print("=" * 70)

    datasets = {
        '1000 Genomes': {
            'bucket': 's3://1000genomes',
            'description': 'Genetic variation from 2,504 individuals',
            'size': '~200 TB',
            'data_types': 'VCF, BAM, CRAM',
            'populations': '26 populations, 5 super-populations',
            'access': 'Public, no credentials required',
            'docs': 'https://registry.opendata.aws/1000-genomes/',
            'paper': '1000 Genomes Project Consortium, Nature (2015)'
        },
        'TCGA': {
            'bucket': 's3://tcga-2-open',
            'description': 'The Cancer Genome Atlas',
            'size': '~2.5 PB',
            'data_types': 'WGS, WXS, RNA-Seq, Methylation',
            'samples': '33 cancer types, 11,000+ patients',
            'access': 'Public, no credentials required',
            'docs': 'https://registry.opendata.aws/tcga/',
            'paper': 'TCGA Research Network, Nature (various)'
        },
        'gnomAD': {
            'bucket': 's3://gnomad-public-us-east-1',
            'description': 'Genome Aggregation Database',
            'size': '~20 TB',
            'data_types': 'VCF (variants), coverage data',
            'samples': '125,748 exomes + 71,702 genomes',
            'access': 'Public, no credentials required',
            'docs': 'https://gnomad.broadinstitute.org/',
            'paper': 'Karczewski et al., Nature (2020)'
        },
        'NCBI SRA': {
            'bucket': 's3://sra-pub-run-odp',
            'description': 'Sequence Read Archive',
            'size': '~30 PB',
            'data_types': 'FASTQ, SRA format',
            'samples': 'Millions of sequencing runs',
            'access': 'Public, no credentials required',
            'docs': 'https://registry.opendata.aws/ncbi-sra/',
            'note': 'Use SRA Toolkit for efficient access'
        }
    }

    for name, info in datasets.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key:15s}: {value}")

    print("\n" + "=" * 70)
    print("\nGetting Started:")
    print("  1. Install AWS CLI: pip install awscli boto3")
    print("  2. List data: aws s3 ls s3://1000genomes/ --no-sign-request")
    print("  3. Download: aws s3 cp s3://bucket/key local_file --no-sign-request")
    print("\nPython Access:")
    print("  from aws_data_access import list_1000genomes, download_sample_vcf")
    print("  files = list_1000genomes(chromosome='chr22')")


if __name__ == '__main__':
    print("AWS Open Data Access for Genomics")
    print("=" * 70)

    # Show available datasets
    print("\n1. Available Datasets")
    print("-" * 70)
    get_bucket_info()

    # Example: 1000 Genomes
    print("\n\n2. Example: 1000 Genomes Project")
    print("-" * 70)
    genomes_files = list_1000genomes(
        phase='phase3',
        chromosome='chr22',
        data_type='vcf',
        max_results=5
    )
    if genomes_files:
        print("\nSample VCF files:")
        for f in genomes_files[:3]:
            print(f"  {f}")

    # Example: TCGA
    print("\n\n3. Example: TCGA (The Cancer Genome Atlas)")
    print("-" * 70)
    tcga_files = list_tcga_data(
        project='TCGA-BRCA',
        max_results=5
    )
    if tcga_files:
        print("\nSample TCGA files:")
        for f in tcga_files[:3]:
            print(f"  {f}")

    # Example: gnomAD
    print("\n\n4. Example: gnomAD")
    print("-" * 70)
    gnomad_files = list_gnomad_data(
        version='v3',
        data_type='vcf',
        max_results=5
    )
    if gnomad_files:
        print("\nSample gnomAD files:")
        for f in gnomad_files[:3]:
            print(f"  {f}")

    print("\n" + "=" * 70)
    print("\n✓ AWS Open Data access ready")
    print("\nNext steps:")
    print("  - Run: python aws_data_access.py")
    print("  - Uncomment download functions to fetch data")
    print("  - See AWS Open Data Registry: https://registry.opendata.aws/")
    print("\nNote: Genomics files can be very large (10+ GB)")
    print("      Consider using AWS EC2/SageMaker for processing")
