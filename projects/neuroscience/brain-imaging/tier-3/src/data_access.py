"""
Data access module for neuroimaging datasets.

Provides classes for loading data from OpenNeuro, HCP, and other public repositories.
"""

import logging
from pathlib import Path
from typing import Optional

import boto3
import botocore
import nibabel as nib
import pandas as pd
from bids import BIDSLayout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenNeuroLoader:
    """Load neuroimaging data from OpenNeuro S3 bucket."""

    def __init__(self, bucket_name: Optional[str] = None, local_cache: str = "./data"):
        """
        Initialize OpenNeuro data loader.

        Args:
            bucket_name: Target S3 bucket for caching (if None, downloads to local only)
            local_cache: Local directory for caching data
        """
        self.source_bucket = "openneuro.org"
        self.bucket_name = bucket_name
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)

        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    def list_datasets(self, prefix: str = "ds") -> list[str]:
        """
        List available datasets on OpenNeuro.

        Args:
            prefix: Dataset prefix (default 'ds')

        Returns:
            List of dataset IDs
        """
        logger.info("Listing OpenNeuro datasets...")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.source_bucket, Delimiter="/", Prefix=prefix)

        datasets = []
        for page in pages:
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    dataset_id = prefix_obj["Prefix"].rstrip("/")
                    datasets.append(dataset_id)

        logger.info(f"Found {len(datasets)} datasets")
        return sorted(datasets)

    def get_dataset_info(self, dataset: str) -> dict:
        """
        Get dataset description and metadata.

        Args:
            dataset: Dataset ID (e.g., 'ds000030')

        Returns:
            Dictionary with dataset metadata
        """
        import json

        # Try to fetch dataset_description.json
        try:
            obj = self.s3_client.get_object(
                Bucket=self.source_bucket,
                Key=f"{dataset}/dataset_description.json",
                RequestPayer="requester",
            )
            description = json.loads(obj["Body"].read())
        except botocore.exceptions.ClientError:
            description = {}

        # Try to fetch participants.tsv
        try:
            obj = self.s3_client.get_object(
                Bucket=self.source_bucket,
                Key=f"{dataset}/participants.tsv",
                RequestPayer="requester",
            )
            participants = pd.read_csv(obj["Body"], sep="\t")
            description["n_subjects"] = len(participants)
        except botocore.exceptions.ClientError:
            participants = None

        return {"dataset_id": dataset, "description": description, "participants": participants}

    def list_subjects(self, dataset: str) -> list[str]:
        """
        List subjects in a dataset.

        Args:
            dataset: Dataset ID

        Returns:
            List of subject IDs
        """
        logger.info(f"Listing subjects in {dataset}...")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.source_bucket,
            Prefix=f"{dataset}/sub-",
            Delimiter="/",
            RequestPayer="requester",
        )

        subjects = []
        for page in pages:
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    subject_path = prefix_obj["Prefix"]
                    # Extract subject ID (e.g., 'sub-10159' from 'ds000030/sub-10159/')
                    subject_id = subject_path.split("/")[-2]
                    subjects.append(subject_id)

        logger.info(f"Found {len(subjects)} subjects")
        return sorted(subjects)

    def download_file(self, dataset: str, s3_key: str, local_path: Optional[Path] = None) -> Path:
        """
        Download a file from OpenNeuro.

        Args:
            dataset: Dataset ID
            s3_key: S3 key relative to dataset (e.g., 'sub-01/anat/sub-01_T1w.nii.gz')
            local_path: Local file path (if None, uses cache directory)

        Returns:
            Path to downloaded file
        """
        if local_path is None:
            local_path = self.local_cache / dataset / s3_key

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if local_path.exists():
            logger.info(f"File already cached: {local_path}")
            return local_path

        # Download from OpenNeuro
        full_key = f"{dataset}/{s3_key}"
        logger.info(f"Downloading {full_key}...")

        self.s3_client.download_file(
            Bucket=self.source_bucket,
            Key=full_key,
            Filename=str(local_path),
            ExtraArgs={"RequestPayer": "requester"},
        )

        logger.info(f"Downloaded to {local_path}")

        # Upload to user's bucket if specified
        if self.bucket_name:
            self.s3_client.upload_file(
                Filename=str(local_path), Bucket=self.bucket_name, Key=full_key
            )
            logger.info(f"Uploaded to s3://{self.bucket_name}/{full_key}")

        return local_path

    def download_subject(
        self, dataset: str, subject: str, modalities: Optional[list[str]] = None
    ) -> dict[str, Path]:
        """
        Download all files for a subject.

        Args:
            dataset: Dataset ID
            subject: Subject ID (e.g., 'sub-10159')
            modalities: List of modalities to download (e.g., ['anat', 'func'])
                       If None, downloads all modalities

        Returns:
            Dictionary mapping file types to paths
        """
        logger.info(f"Downloading {subject} from {dataset}...")

        # List all files for this subject
        prefix = f"{dataset}/{subject}/"
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.source_bucket, Prefix=prefix, RequestPayer="requester"
        )

        files = {}
        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                # Get modality from path
                parts = key.split("/")
                if len(parts) < 3:
                    continue

                modality = parts[2]  # e.g., 'anat', 'func'

                # Filter by modality if specified
                if modalities and modality not in modalities:
                    continue

                # Download file
                relative_key = "/".join(parts[1:])  # Remove dataset prefix
                local_path = self.download_file(dataset, relative_key)

                # Store in dictionary
                filename = parts[-1]
                files[filename] = local_path

        logger.info(f"Downloaded {len(files)} files for {subject}")
        return files

    def sync_dataset(
        self,
        dataset: str,
        max_subjects: Optional[int] = None,
        modalities: Optional[list[str]] = None,
    ):
        """
        Sync entire dataset (or subset) to local cache and S3.

        Args:
            dataset: Dataset ID
            max_subjects: Maximum number of subjects to sync (None = all)
            modalities: List of modalities to sync
        """
        subjects = self.list_subjects(dataset)

        if max_subjects:
            subjects = subjects[:max_subjects]

        logger.info(f"Syncing {len(subjects)} subjects from {dataset}...")

        for i, subject in enumerate(subjects, 1):
            logger.info(f"[{i}/{len(subjects)}] Processing {subject}...")
            self.download_subject(dataset, subject, modalities)

    def load_nifti(self, file_path: Path) -> nib.Nifti1Image:
        """
        Load NIfTI file.

        Args:
            file_path: Path to NIfTI file

        Returns:
            Nibabel NIfTI image
        """
        return nib.load(str(file_path))

    def get_bids_layout(self, dataset: str) -> BIDSLayout:
        """
        Get BIDS layout for a dataset.

        Args:
            dataset: Dataset ID

        Returns:
            BIDS layout object
        """
        dataset_path = self.local_cache / dataset
        return BIDSLayout(str(dataset_path))


class HCPLoader:
    """Load data from Human Connectome Project."""

    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize HCP data loader.

        Args:
            bucket_name: S3 bucket for caching
        """
        self.source_bucket = "hcp-openaccess"
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    def list_subjects(self) -> list[str]:
        """
        List available HCP subjects.

        Returns:
            List of subject IDs
        """
        logger.info("Listing HCP subjects...")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.source_bucket, Prefix="HCP_1200/", Delimiter="/", RequestPayer="requester"
        )

        subjects = []
        for page in pages:
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    subject_path = prefix_obj["Prefix"]
                    subject_id = subject_path.split("/")[-2]
                    # HCP subject IDs are 6-digit numbers
                    if subject_id.isdigit() and len(subject_id) == 6:
                        subjects.append(subject_id)

        logger.info(f"Found {len(subjects)} HCP subjects")
        return sorted(subjects)

    def download_structural(
        self, subject: str, local_path: Optional[Path] = None
    ) -> dict[str, Path]:
        """
        Download structural scans for an HCP subject.

        Args:
            subject: HCP subject ID
            local_path: Local directory for download

        Returns:
            Dictionary of downloaded files
        """
        files = {
            "T1w": f"HCP_1200/{subject}/T1w/T1w_acpc_dc_restore.nii.gz",
            "T2w": f"HCP_1200/{subject}/T1w/T2w_acpc_dc_restore.nii.gz",
        }

        downloaded = {}
        for name, s3_key in files.items():
            if local_path:
                local_file = local_path / f"{subject}_{name}.nii.gz"
            else:
                local_file = Path(f"./data/HCP/{subject}_{name}.nii.gz")

            local_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {name} for subject {subject}...")
            self.s3_client.download_file(
                Bucket=self.source_bucket,
                Key=s3_key,
                Filename=str(local_file),
                ExtraArgs={"RequestPayer": "requester"},
            )

            downloaded[name] = local_file

        return downloaded


def main():
    """
    Main function for data access demo.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download neuroimaging data")
    parser.add_argument("--dataset", type=str, default="ds000030", help="Dataset ID")
    parser.add_argument("--subject", type=str, help="Subject ID")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--list-subjects", action="store_true", help="List subjects in dataset")
    args = parser.parse_args()

    loader = OpenNeuroLoader()

    if args.list_datasets:
        datasets = loader.list_datasets()
        print(f"Found {len(datasets)} datasets:")
        for ds in datasets[:20]:  # Show first 20
            print(f"  {ds}")

    elif args.list_subjects:
        subjects = loader.list_subjects(args.dataset)
        print(f"Found {len(subjects)} subjects in {args.dataset}:")
        for subj in subjects[:10]:  # Show first 10
            print(f"  {subj}")

    elif args.subject:
        files = loader.download_subject(args.dataset, args.subject, modalities=["anat"])
        print(f"Downloaded {len(files)} files:")
        for filename, path in files.items():
            print(f"  {filename}: {path}")


if __name__ == "__main__":
    main()
