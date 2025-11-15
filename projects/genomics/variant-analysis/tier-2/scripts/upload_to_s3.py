"""
Upload BAM files and reference data to S3.

This script:
1. Downloads sample BAM files from public repositories
2. Uploads to S3 bucket configured in .env
3. Creates sample metadata and uploads it

Usage:
    python upload_to_s3.py
    python upload_to_s3.py --bucket my-bucket --region chr22:1-10000000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import boto3
from dotenv import load_dotenv


def load_config() -> dict:
    """Load configuration from .env file."""
    load_dotenv()

    config = {
        "bucket": os.getenv("BUCKET_INPUT"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "table_name": os.getenv("TABLE_NAME", "variant-metadata"),
    }

    if not config["bucket"]:
        raise ValueError("BUCKET_INPUT not set in .env file. Run setup_guide.md first.")

    return config


def create_sample_bam(output_path: str) -> None:
    """
    Create a minimal sample BAM file for testing.

    Since downloading real BAM files can be large (1GB+), this creates
    a small synthetic BAM file for demonstration.
    """

    import struct

    print(f"Creating synthetic BAM file: {output_path}")

    # BAM file format basics
    # Magic: "BAM\1"
    # Then SAM header
    # Then aligned segments

    with open(output_path, "wb") as f:
        # Write BAM magic
        f.write(b"BAM\1")

        # Write minimal SAM header
        # Header length + header text
        header_text = "@HD\tVN:1.0\tSO:coordinate\n@SQ\tSN:chr20\tLN:64000000\n"
        header_bytes = header_text.encode("utf-8")

        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)

        # Write reference sequences
        f.write(struct.pack("<I", 1))  # 1 reference sequence
        f.write(struct.pack("<I", len("chr20")))
        f.write(b"chr20")
        f.write(struct.pack("<I", 64000000))  # chr20 length

        # Write a few aligned segments (simplified)
        # This is a minimal representation - real BAM files are more complex
        for i in range(10):
            # QNAME
            qname = f"read_{i}".encode()
            f.write(struct.pack("<I", len(qname)))
            f.write(qname)

            # Flags and positions
            flag = 0  # Forward strand, properly paired
            pos = 1000000 + (i * 1000)
            qual = 60
            f.write(struct.pack("<i", flag))
            f.write(struct.pack("<i", pos))
            f.write(struct.pack("<I", qual))

            # Sequence and qualities
            seq = "ACGTACGTACGTACGTACGTACGTACGTACGT"
            qualities = "I" * len(seq)
            f.write(struct.pack("<I", len(seq)))
            f.write(seq.encode("utf-8"))
            f.write(qualities.encode("utf-8"))

    print(f"✓ Created synthetic BAM file: {output_path}")


def download_1000genomes_bam(
    output_path: str, sample: str = "NA12878", chromosome: str = "20"
) -> None:
    """
    Download BAM file from 1000 Genomes Project.

    Note: Files are typically 1-2GB, may take 5-10 minutes.
    """

    url = f"https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/{sample}/alignment/{sample}.chrom{chromosome}.ILLUMINA.bwa.YRI.low_coverage.20130415.bam"

    print("Downloading sample BAM file from 1000 Genomes...")
    print(f"Sample: {sample}, Chromosome: {chromosome}")
    print(f"URL: {url}")
    print("Note: This file is ~100-200MB and may take a few minutes to download.")

    try:
        urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Downloaded: {output_path} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"✗ Download failed: {e!s}")
        print("Falling back to synthetic BAM file for demonstration")
        create_sample_bam(output_path)


def upload_to_s3(local_path: str, bucket: str, s3_key: str, region: str) -> None:
    """Upload file to S3."""

    s3 = boto3.client("s3", region_name=region)

    file_size = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")

    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"✓ Uploaded: {file_size:.2f} MB")
    except Exception as e:
        print(f"✗ Upload failed: {e!s}")
        raise


def create_sample_metadata(output_path: str) -> None:
    """Create sample metadata JSON file."""

    metadata = {
        "samples": [
            {
                "sample_id": "NA12878",
                "population": "CEU",
                "sex": "Female",
                "bam_file": "samples/NA12878.chr20.bam",
                "coverage": "30x",
                "sequencing_platform": "Illumina",
            },
            {
                "sample_id": "NA19238",
                "population": "YRI",
                "sex": "Male",
                "bam_file": "samples/NA19238.chr20.bam",
                "coverage": "30x",
                "sequencing_platform": "Illumina",
            },
        ],
        "reference": {
            "genome": "GRCh37/hg19",
            "fasta": "reference/chr20.fa",
            "fai": "reference/chr20.fa.fai",
        },
        "metadata": {
            "created": "2025-01-01",
            "project": "Genomic Variant Analysis Tier 2",
            "region": "chr20:1000000-64000000",
        },
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Created metadata: {output_path}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="Upload genomic data to S3")
    parser.add_argument("--bucket", help="S3 bucket name (overrides .env)")
    parser.add_argument(
        "--sample", default="NA12878", help="Sample ID to download (default: NA12878)"
    )
    parser.add_argument("--chromosome", default="20", help="Chromosome to download (default: 20)")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic BAM file instead of downloading"
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip BAM file download")

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    if args.bucket:
        config["bucket"] = args.bucket

    print("=" * 60)
    print("Uploading Genomic Data to S3")
    print("=" * 60)
    print(f"Bucket: {config['bucket']}")
    print(f"Region: {config['region']}")
    print()

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Create or download BAM file
        if not args.skip_download:
            bam_path = data_dir / f"{args.sample}.chr{args.chromosome}.bam"

            if args.synthetic:
                create_sample_bam(str(bam_path))
            else:
                download_1000genomes_bam(
                    str(bam_path), sample=args.sample, chromosome=args.chromosome
                )

            # Upload BAM to S3
            upload_to_s3(
                str(bam_path), config["bucket"], f"samples/{bam_path.name}", config["region"]
            )

        # Step 2: Create and upload metadata
        metadata_path = data_dir / "sample_metadata.json"
        create_sample_metadata(str(metadata_path))
        upload_to_s3(
            str(metadata_path), config["bucket"], "metadata/sample_metadata.json", config["region"]
        )

        # Step 3: Create and upload reference data
        ref_dir = data_dir / "reference"
        ref_dir.mkdir(exist_ok=True)

        # Create minimal reference sequence
        ref_fasta = ref_dir / "chr20.fa"
        with open(ref_fasta, "w") as f:
            f.write(">chr20\n")
            # Write 100 lines of 50bp each (5000bp total)
            for _i in range(100):
                f.write("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")

        upload_to_s3(str(ref_fasta), config["bucket"], "reference/chr20.fa", config["region"])

        print()
        print("=" * 60)
        print("✓ Upload Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run: jupyter notebook notebooks/variant_analysis.ipynb")
        print("2. Follow the notebook to invoke Lambda and query results")
        print("3. Don't forget to run cleanup_guide.md when finished!")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
