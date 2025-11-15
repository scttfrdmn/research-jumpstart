#!/usr/bin/env python3
"""
AWS Open Data Access for Medical Imaging

Access medical imaging datasets from AWS Open Data Registry:
- NIH Chest X-ray Dataset (ChestX-ray14)
- Cancer Imaging Archive (TCIA)
- Medical Segmentation Decathlon
- RSNA Pneumonia Detection Challenge

Usage:
    from aws_data_access import list_nih_chestxray, download_sample_images

    # List available chest X-rays
    files = list_nih_chestxray(max_results=100)

    # Download sample images
    download_sample_images(output_dir='data/chest_xrays', n_samples=10)

AWS Setup:
    # For public datasets, no credentials needed
    # For requester-pays buckets:
    aws configure
"""

import os

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

# AWS Open Data Registry S3 buckets
BUCKETS = {
    "nih_chestxray": "nih-chest-xrays",  # Public
    "tcia": "imaging.nci.nih.gov",  # TCIA Cancer Imaging
    "medical_seg": "medicalsegmentation",  # Medical Segmentation Decathlon
    "rsna_pneumonia": "rsna-pneumonia-detection-challenge",  # RSNA Challenge
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
        return boto3.client("s3", config=config)
    else:
        return boto3.client("s3")


def list_nih_chestxray(max_results=100, anonymous=True):
    """
    List NIH Chest X-ray images on AWS.

    The NIH Chest X-ray14 dataset contains 112,120 frontal-view X-ray images
    of 30,805 unique patients with 14 disease labels.

    Bucket structure:
    s3://nih-chest-xrays/png/[image_id].png

    Parameters:
    -----------
    max_results : int
        Maximum files to list
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    files : list of str
        S3 keys for image files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["nih_chestxray"]

    print(f"Listing NIH Chest X-ray images on s3://{bucket}/")
    print("Dataset: 112,120 images, 14 disease labels")

    try:
        files = []
        prefix = "png/"

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                if obj["Key"].endswith(".png"):
                    files.append(obj["Key"])

                if len(files) >= max_results:
                    break

            if len(files) >= max_results:
                break

        print(f"\nFound {len(files)} image files")
        return files

    except Exception as e:
        print(f"Error listing NIH Chest X-ray data: {e}")
        print("\nExample access:")
        print("  aws s3 ls s3://nih-chest-xrays/png/ --no-sign-request")
        return []


def download_nih_metadata(output_path="data/nih_metadata.csv", anonymous=True):
    """
    Download NIH Chest X-ray metadata CSV.

    The metadata contains:
    - Image Index (filename)
    - Finding Labels (diseases present)
    - Follow-up number
    - Patient ID
    - Patient Age
    - Patient Gender
    - View Position (PA/AP)
    - Original Image Size

    Parameters:
    -----------
    output_path : str
        Local path to save metadata
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    metadata : DataFrame
        Metadata as pandas DataFrame
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["nih_chestxray"]

    # Metadata file location
    metadata_key = "Data_Entry_2017.csv"

    print("Downloading NIH Chest X-ray metadata...")
    print(f"  From: s3://{bucket}/{metadata_key}")
    print(f"    To: {output_path}")

    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download
        s3.download_file(bucket, metadata_key, output_path)
        print("Download complete!")

        # Load and return
        df = pd.read_csv(output_path)
        print(f"\nMetadata shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns.tolist())}")

        return df

    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return None


def download_sample_images(
    output_dir="data/chest_xrays", n_samples=10, disease_filter=None, anonymous=True
):
    """
    Download sample chest X-ray images.

    Parameters:
    -----------
    output_dir : str
        Output directory for images
    n_samples : int
        Number of images to download
    disease_filter : str, optional
        Filter by disease label (e.g., 'Pneumonia', 'Effusion')
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    downloaded_files : list
        Paths to downloaded files
    """
    s3 = get_s3_client(anonymous=anonymous)
    bucket = BUCKETS["nih_chestxray"]

    print(f"Downloading {n_samples} sample chest X-ray images...")

    # First, get metadata to filter by disease if needed
    if disease_filter:
        print(f"Filtering for disease: {disease_filter}")
        metadata = download_nih_metadata(
            output_path=os.path.join(output_dir, "metadata.csv"), anonymous=anonymous
        )

        if metadata is not None:
            # Filter by disease
            filtered = metadata[metadata["Finding Labels"].str.contains(disease_filter, na=False)]
            if len(filtered) == 0:
                print(f"No images found with disease: {disease_filter}")
                return []

            # Get image filenames
            image_names = filtered["Image Index"].head(n_samples).tolist()
            image_keys = [f"png/{name}" for name in image_names]
        else:
            print("Could not load metadata, downloading random images")
            image_keys = list_nih_chestxray(max_results=n_samples, anonymous=anonymous)
    else:
        # Get random images
        image_keys = list_nih_chestxray(max_results=n_samples, anonymous=anonymous)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download images
    downloaded_files = []
    for i, key in enumerate(image_keys[:n_samples], 1):
        filename = os.path.basename(key)
        output_path = os.path.join(output_dir, filename)

        print(f"  [{i}/{n_samples}] Downloading {filename}...")

        try:
            s3.download_file(bucket, key, output_path)
            downloaded_files.append(output_path)
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nDownloaded {len(downloaded_files)} images to {output_dir}")
    return downloaded_files


def list_tcia_collections(anonymous=True):
    """
    List TCIA (The Cancer Imaging Archive) collections.

    TCIA hosts cancer imaging data organized by collections
    (disease sites, research studies).

    Note: Some TCIA data requires registration at
    https://www.cancerimagingarchive.net/

    Parameters:
    -----------
    anonymous : bool
        Use anonymous access

    Returns:
    --------
    collections : list
        Available collections
    """
    print("The Cancer Imaging Archive (TCIA)")
    print("=" * 60)
    print("\nTCIA provides cancer imaging datasets organized by collections:")
    print("  - Lung CT Screening")
    print("  - Breast MRI")
    print("  - Brain Tumor (Glioma)")
    print("  - Prostate MRI")
    print("  - And many more...")

    print("\nAccess:")
    print("  1. Register at: https://www.cancerimagingarchive.net/")
    print("  2. Browse collections: https://www.cancerimagingarchive.net/collections/")
    print("  3. Some collections available on AWS S3")

    print("\nExample collections on AWS:")
    collections = [
        "TCGA-LUAD (Lung Adenocarcinoma)",
        "TCGA-BRCA (Breast Cancer)",
        "TCGA-GBM (Glioblastoma)",
        "LIDC-IDRI (Lung Imaging Database)",
    ]

    for coll in collections:
        print(f"  - {coll}")

    return collections


def get_medical_seg_decathlon_info():
    """
    Information about Medical Segmentation Decathlon dataset.

    The Medical Segmentation Decathlon includes 10 segmentation tasks
    across different organs and modalities.
    """
    print("Medical Segmentation Decathlon")
    print("=" * 60)

    tasks = {
        "Task01_BrainTumour": {
            "modality": "MRI (4 channels)",
            "size": "484 images",
            "labels": "Edema, Non-enhancing tumor, Enhancing tumor",
        },
        "Task02_Heart": {"modality": "MRI", "size": "20 images", "labels": "Left Atrium"},
        "Task03_Liver": {"modality": "CT", "size": "131 images", "labels": "Liver, Tumor"},
        "Task04_Hippocampus": {
            "modality": "MRI",
            "size": "260 images",
            "labels": "Anterior, Posterior hippocampus",
        },
        "Task05_Prostate": {
            "modality": "MRI (2 channels)",
            "size": "32 images",
            "labels": "Peripheral zone, Transition zone",
        },
        "Task06_Lung": {"modality": "CT", "size": "64 images", "labels": "Lung cancer"},
        "Task07_Pancreas": {"modality": "CT", "size": "282 images", "labels": "Pancreas, Tumor"},
        "Task08_HepaticVessel": {"modality": "CT", "size": "303 images", "labels": "Vessel, Tumor"},
        "Task09_Spleen": {"modality": "CT", "size": "41 images", "labels": "Spleen"},
        "Task10_Colon": {"modality": "CT", "size": "126 images", "labels": "Colon cancer"},
    }

    print("\n10 Segmentation Tasks:")
    for task_name, info in tasks.items():
        print(f"\n{task_name}")
        for key, value in info.items():
            print(f"  {key:12s}: {value}")

    print("\n" + "=" * 60)
    print("\nAccess:")
    print("  Website: http://medicaldecathlon.com/")
    print("  Download: http://medicaldecathlon.com/dataaws/")
    print("  Format: NIfTI (.nii.gz)")

    return tasks


def get_bucket_info():
    """
    Print information about AWS Open Data buckets for medical imaging.
    """
    print("AWS Open Data Registry - Medical Imaging Datasets")
    print("=" * 70)

    datasets = {
        "NIH Chest X-ray14": {
            "bucket": "s3://nih-chest-xrays",
            "description": "112,120 frontal-view chest X-rays",
            "size": "~45 GB",
            "labels": "14 disease categories",
            "access": "Public, no credentials required",
            "docs": "https://registry.opendata.aws/nih-chest-xray/",
            "paper": "Wang et al., ChestX-ray8 (2017)",
        },
        "TCIA Collections": {
            "bucket": "s3://imaging.nci.nih.gov",
            "description": "Cancer imaging from clinical trials",
            "size": "Multiple TB",
            "labels": "Various cancer types and organs",
            "access": "Public (some require registration)",
            "docs": "https://www.cancerimagingarchive.net/",
            "paper": "Various publications",
        },
        "Medical Seg Decathlon": {
            "bucket": "s3://medicalsegmentation",
            "description": "10 organ segmentation tasks",
            "size": "~35 GB",
            "labels": "Organ and tumor segmentations",
            "access": "Public, no credentials required",
            "docs": "http://medicaldecathlon.com/",
            "paper": "Antonelli et al., Medical Image Analysis (2022)",
        },
    }

    for name, info in datasets.items():
        print(f"\n{name}")
        print("-" * 70)
        for key, value in info.items():
            print(f"  {key:15s}: {value}")

    print("\n" + "=" * 70)
    print("\nGetting Started:")
    print("  1. Install AWS CLI: pip install awscli boto3")
    print("  2. List data: aws s3 ls s3://nih-chest-xrays/ --no-sign-request")
    print("  3. Download: aws s3 cp s3://bucket/key local_file --no-sign-request")
    print("\nPython Access:")
    print("  from aws_data_access import download_sample_images")
    print("  download_sample_images(n_samples=10, disease_filter='Pneumonia')")


if __name__ == "__main__":
    print("AWS Open Data Access for Medical Imaging")
    print("=" * 70)

    # Show available datasets
    print("\n1. Available Datasets")
    print("-" * 70)
    get_bucket_info()

    # NIH Chest X-ray example
    print("\n\n2. Example: NIH Chest X-ray Dataset")
    print("-" * 70)

    # List some images
    print("\n2a. Listing images...")
    images = list_nih_chestxray(max_results=10)
    if images:
        print("\nSample image keys:")
        for img in images[:5]:
            print(f"  {img}")

    # Download metadata
    print("\n2b. Downloading metadata...")
    metadata = download_nih_metadata(output_path="data/nih_metadata.csv")
    if metadata is not None:
        print("\nSample metadata:")
        print(metadata.head(3))

        print("\nDisease distribution:")
        # Count diseases (split by |)
        all_diseases = []
        for labels in metadata["Finding Labels"]:
            if pd.notna(labels) and labels != "No Finding":
                all_diseases.extend(labels.split("|"))

        from collections import Counter

        disease_counts = Counter(all_diseases)
        print("Top 5 diseases:")
        for disease, count in disease_counts.most_common(5):
            print(f"  {disease:20s}: {count:5d} images")

    # TCIA info
    print("\n\n3. The Cancer Imaging Archive (TCIA)")
    print("-" * 70)
    list_tcia_collections()

    # Medical Segmentation Decathlon
    print("\n\n4. Medical Segmentation Decathlon")
    print("-" * 70)
    get_medical_seg_decathlon_info()

    print("\n" + "=" * 70)
    print("\n✓ AWS Open Data access ready")
    print("\nNext steps:")
    print("  - Run: python aws_data_access.py")
    print("  - Uncomment download functions to fetch data")
    print("  - See AWS Open Data Registry: https://registry.opendata.aws/")
