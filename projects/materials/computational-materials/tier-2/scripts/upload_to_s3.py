#!/usr/bin/env python3
"""
Upload crystal structure files to S3 bucket.

This script handles uploading CIF and POSCAR files to AWS S3 for processing.
Supports resumable uploads and progress tracking.
"""

import os
import sys
import boto3
import argparse
from pathlib import Path
from tqdm import tqdm
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Uploader:
    """Upload crystal structure files to S3 with progress tracking."""

    def __init__(self, bucket_name, region='us-east-1', profile=None):
        """
        Initialize S3 uploader.

        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
            profile (str): AWS profile name
        """
        self.bucket_name = bucket_name
        self.region = region

        # Create session and S3 client
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()

        self.s3 = session.client('s3', region_name=region)

        # Verify bucket exists
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info(f"Connected to bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"Cannot access bucket: {bucket_name}")
            logger.error(f"  Error: {e}")
            raise

    def upload_file(self, file_path, s3_key, metadata=None):
        """
        Upload file to S3.

        Args:
            file_path (str): Local file path
            s3_key (str): S3 object key (path in bucket)
            metadata (dict): Optional metadata

        Returns:
            bool: Success status
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        file_size = file_path.stat().st_size
        file_size_mb = file_size / 1e6

        try:
            # Prepare metadata
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata

            # Determine content type
            suffix = file_path.suffix.lower()
            if suffix == '.cif':
                extra_args['ContentType'] = 'chemical/x-cif'
            elif suffix in ['.vasp', '.poscar', '.contcar']:
                extra_args['ContentType'] = 'text/plain'

            logger.info(f"Uploading: {file_path.name} ({file_size_mb:.2f} MB)")

            # Upload file with progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_path.name) as pbar:
                self.s3.upload_file(
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                )

            logger.info(f"Uploaded: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def upload_directory(self, local_dir, s3_prefix='structures/', pattern='*'):
        """
        Upload all matching files in directory to S3.

        Args:
            local_dir (str): Local directory path
            s3_prefix (str): S3 folder prefix
            pattern (str): File pattern to match

        Returns:
            tuple: (successful_uploads, failed_uploads)
        """
        local_dir = Path(local_dir)

        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return 0, 0

        # Find all structure files
        files = []
        for ext in ['*.cif', '*.CIF', '*.vasp', '*.VASP',
                    '*POSCAR', '*CONTCAR', '*.poscar', '*.contcar']:
            files.extend(list(local_dir.glob(ext)))
            files.extend(list(local_dir.glob(f'**/{ext}')))

        # Remove duplicates
        files = list(set(files))

        if not files:
            logger.warning(f"No structure files found in {local_dir}")
            return 0, 0

        logger.info(f"Found {len(files)} structure files to upload")

        successful = 0
        failed = 0

        for file_path in files:
            # Create S3 key maintaining directory structure
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}".replace(os.sep, '/')

            # Add metadata
            metadata = {
                'original-name': file_path.name,
                'file-type': file_path.suffix.lower(),
                'upload-source': 'tier-2-materials'
            }

            if self.upload_file(str(file_path), s3_key, metadata):
                successful += 1
            else:
                failed += 1

        return successful, failed

    def list_uploaded_files(self, prefix='structures/'):
        """List all uploaded files in S3."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                logger.info(f"No files found in s3://{self.bucket_name}/{prefix}")
                return []

            files = []
            for obj in response['Contents']:
                size_mb = obj['Size'] / 1e6
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'size_mb': size_mb,
                    'modified': obj['LastModified']
                })

            logger.info(f"\nUploaded files in {prefix}:")
            total_size = 0
            for f in files:
                logger.info(f"  {f['key']} ({f['size_mb']:.3f} MB)")
                total_size += f['size']

            logger.info(f"Total: {len(files)} files, {total_size/1e6:.3f} MB")
            return files

        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []


def create_sample_cif():
    """Create a sample CIF file for testing."""
    sample_dir = Path('sample_data/structures')
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Silicon structure
    si_cif = sample_dir / 'Si.cif'
    if not si_cif.exists():
        with open(si_cif, 'w') as f:
            f.write("""data_Si
_cell_length_a    3.867
_cell_length_b    3.867
_cell_length_c    3.867
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F d -3 m'
_symmetry_Int_Tables_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.0 0.0
Si2 Si 0.25 0.25 0.25
""")
        logger.info(f"Created sample CIF file: {si_cif}")

    return sample_dir


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Upload crystal structure files to S3'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name'
    )
    parser.add_argument(
        '--directory',
        default='sample_data/structures',
        help='Local directory with structure files'
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
        '--file',
        help='Upload single file instead of directory'
    )
    parser.add_argument(
        '--s3-prefix',
        default='structures/',
        help='S3 prefix for uploaded files'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list files without uploading'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample CIF file for testing'
    )

    args = parser.parse_args()

    try:
        # Create sample data if requested
        if args.create_sample:
            sample_dir = create_sample_cif()
            logger.info(f"Sample data created in {sample_dir}")
            if not args.file and not Path(args.directory).exists():
                args.directory = str(sample_dir)

        uploader = S3Uploader(args.bucket, args.region, args.profile)

        if args.list_only:
            uploader.list_uploaded_files(args.s3_prefix)
        elif args.file:
            # Upload single file
            s3_key = f"{args.s3_prefix}{Path(args.file).name}"
            uploader.upload_file(args.file, s3_key)
            uploader.list_uploaded_files(args.s3_prefix)
        else:
            # Upload directory
            successful, failed = uploader.upload_directory(
                args.directory,
                args.s3_prefix
            )
            logger.info(f"\nUpload complete: {successful} successful, {failed} failed")
            uploader.list_uploaded_files(args.s3_prefix)

            sys.exit(0 if failed == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
