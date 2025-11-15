#!/usr/bin/env python3
"""
Upload molecular structures to S3 bucket.

This script handles uploading SMILES, SDF, and MOL2 files to AWS S3
for processing. Supports resumable uploads and progress tracking.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MolecularS3Uploader:
    """Upload molecular structure files to S3 with progress tracking."""

    def __init__(self, bucket_name, region="us-east-1", profile=None):
        """
        Initialize S3 uploader for molecular data.

        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
            profile (str): AWS profile name
        """
        self.bucket_name = bucket_name
        self.region = region

        # Create session and S3 client
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()

        self.s3 = session.client("s3", region_name=region)

        # Verify bucket exists
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"Connected to bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"Cannot access bucket: {bucket_name}")
            logger.error(f"Error: {e}")
            raise

    def upload_file(self, file_path, s3_key, metadata=None):
        """
        Upload molecular structure file to S3.

        Args:
            file_path (str): Local file path
            s3_key (str): S3 object key (path in bucket)
            metadata (dict): Optional metadata tags

        Returns:
            bool: Success status
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        file_size = file_path.stat().st_size
        logger.info(f"Uploading: {file_path.name} ({file_size / 1024:.2f}KB)")

        try:
            # Determine content type
            content_type = self._get_content_type(file_path.suffix)

            # Upload with metadata
            extra_args = {"ContentType": content_type}
            if metadata:
                extra_args["Metadata"] = metadata

            self.s3.upload_file(str(file_path), self.bucket_name, s3_key, ExtraArgs=extra_args)

            logger.info(f"Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def _get_content_type(self, suffix):
        """Get content type based on file extension."""
        content_types = {
            ".smi": "chemical/x-daylight-smiles",
            ".smiles": "chemical/x-daylight-smiles",
            ".sdf": "chemical/x-mdl-sdfile",
            ".mol": "chemical/x-mdl-molfile",
            ".mol2": "chemical/x-mol2",
            ".pdb": "chemical/x-pdb",
            ".xyz": "chemical/x-xyz",
            ".txt": "text/plain",
            ".csv": "text/csv",
        }
        return content_types.get(suffix.lower(), "application/octet-stream")

    def upload_directory(self, local_dir, s3_prefix="molecules/", organize_by_class=True):
        """
        Upload all molecular structure files in directory to S3.

        Args:
            local_dir (str): Local directory path
            s3_prefix (str): S3 folder prefix
            organize_by_class (bool): Organize by compound class subdirectories

        Returns:
            tuple: (successful_uploads, failed_uploads)
        """
        local_dir = Path(local_dir)

        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return 0, 0

        # Find all molecular structure files
        extensions = ["*.smi", "*.smiles", "*.sdf", "*.mol", "*.mol2", "*.pdb"]
        files = []
        for ext in extensions:
            files.extend(list(local_dir.glob(f"**/{ext}")))

        if not files:
            logger.warning(f"No molecular structure files found in {local_dir}")
            return 0, 0

        logger.info(f"Found {len(files)} files to upload")

        successful = 0
        failed = 0

        for file_path in tqdm(files, desc="Uploading molecules"):
            # Determine compound class from directory structure
            if organize_by_class:
                relative_path = file_path.relative_to(local_dir)

                # Extract compound class from path
                if len(relative_path.parts) > 1:
                    relative_path.parts[0]
                else:
                    pass

                s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}".replace(os.sep, "/")
            else:
                s3_key = f"{s3_prefix.rstrip('/')}/{file_path.name}".replace(os.sep, "/")

            # Add metadata
            metadata = {
                "original-filename": file_path.name,
                "file-type": file_path.suffix.lstrip("."),
                "upload-source": "molecular-analysis-tier2",
            }

            if self.upload_file(str(file_path), s3_key, metadata):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def list_uploaded_files(self, prefix="molecules/"):
        """List all uploaded molecular structure files in S3."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

            if "Contents" not in response:
                logger.info(f"No files found in s3://{self.bucket_name}/{prefix}")
                return []

            files = []
            for obj in response["Contents"]:
                size_kb = obj["Size"] / 1024
                files.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "size_kb": size_kb,
                        "modified": obj["LastModified"],
                    }
                )

            logger.info(f"\nUploaded files in {prefix}:")
            total_size = 0
            for f in files:
                logger.info(f"  {f['key']} ({f['size_kb']:.1f}KB)")
                total_size += f["size"]

            logger.info(f"Total: {len(files)} files, {total_size / 1024:.1f}KB")
            return files

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def validate_smiles_file(self, file_path):
        """
        Validate SMILES file format before upload.

        Args:
            file_path (str): Path to SMILES file

        Returns:
            tuple: (valid, num_molecules, errors)
        """
        errors = []
        num_molecules = 0

        try:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 1:
                        errors.append(f"Line {line_num}: Empty SMILES")
                        continue

                    smiles = parts[0]
                    if not smiles:
                        errors.append(f"Line {line_num}: Empty SMILES string")
                        continue

                    # Basic SMILES syntax check (simplified)
                    if not self._is_valid_smiles_syntax(smiles):
                        errors.append(f"Line {line_num}: Invalid SMILES syntax: {smiles}")
                        continue

                    num_molecules += 1

            valid = len(errors) == 0
            return valid, num_molecules, errors

        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return False, 0, errors

    def _is_valid_smiles_syntax(self, smiles):
        """Basic SMILES syntax validation (simplified check)."""
        # Check for balanced parentheses and brackets
        paren_count = 0
        bracket_count = 0

        for char in smiles:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1

            if paren_count < 0 or bracket_count < 0:
                return False

        return paren_count == 0 and bracket_count == 0


def create_sample_molecules(output_dir="sample_data"):
    """
    Create sample molecular structure files for testing.

    Args:
        output_dir (str): Output directory for sample files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "drugs").mkdir(exist_ok=True)
    (output_dir / "natural_products").mkdir(exist_ok=True)

    # Sample drugs
    drugs = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ]

    # Sample natural products
    natural_products = [
        ("morphine", "CN1CC[C@]23[C@@H]4Oc5c3c(ccc5O)C[C@@H]1[C@@H]2C=C[C@@H]4O"),
        ("quinine", "COc1ccc2nccc([C@H](O)[C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1"),
        ("nicotine", "CN1CCC[C@H]1c2cccnc2"),
        ("menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"),
        ("camphor", "CC1(C)C2CCC1(C)C(=O)C2"),
    ]

    # Write drugs
    drugs_file = output_dir / "drugs" / "drugs.smi"
    with open(drugs_file, "w") as f:
        f.write("# Sample drug molecules\n")
        for name, smiles in drugs:
            f.write(f"{smiles} {name}\n")
    logger.info(f"Created: {drugs_file} ({len(drugs)} molecules)")

    # Write natural products
    np_file = output_dir / "natural_products" / "natural_products.smi"
    with open(np_file, "w") as f:
        f.write("# Sample natural product molecules\n")
        for name, smiles in natural_products:
            f.write(f"{smiles} {name}\n")
    logger.info(f"Created: {np_file} ({len(natural_products)} molecules)")

    logger.info(f"Sample data created in: {output_dir}")
    return output_dir


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Upload molecular structures to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--data-dir", default="sample_data", help="Local directory with molecular structure files"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--file", help="Upload single file instead of directory")
    parser.add_argument(
        "--s3-prefix", default="molecules/", help="S3 key/prefix for uploaded files"
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Only list files without uploading"
    )
    parser.add_argument(
        "--create-samples", action="store_true", help="Create sample molecular data files"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate SMILES files before upload"
    )

    args = parser.parse_args()

    try:
        # Create sample data if requested
        if args.create_samples:
            create_sample_molecules(args.data_dir)
            return 0

        # Initialize uploader
        uploader = MolecularS3Uploader(args.bucket, args.region, args.profile)

        if args.list_only:
            # List uploaded files
            uploader.list_uploaded_files(args.s3_prefix)
        elif args.file:
            # Upload single file
            if args.validate and args.file.endswith((".smi", ".smiles")):
                valid, num_mols, errors = uploader.validate_smiles_file(args.file)
                if not valid:
                    logger.error("SMILES validation failed:")
                    for error in errors:
                        logger.error(f"  {error}")
                    return 1
                logger.info(f"SMILES file valid: {num_mols} molecules")

            s3_key = f"{args.s3_prefix.rstrip('/')}/{Path(args.file).name}"
            uploader.upload_file(args.file, s3_key)
            uploader.list_uploaded_files(args.s3_prefix)
        else:
            # Upload directory
            successful, failed = uploader.upload_directory(args.data_dir, args.s3_prefix)
            logger.info(f"\nUpload complete: {successful} successful, {failed} failed")
            uploader.list_uploaded_files(args.s3_prefix)

            sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
