#!/usr/bin/env python3
"""
Medical Image Data Preprocessing

Utilities for loading, augmenting, and preparing medical images
for deep learning classification tasks.

Usage:
    from data_preprocessing import MedicalImageDataset, get_transforms

    train_dataset = MedicalImageDataset(
        image_paths, labels,
        transform=get_transforms('train')
    )
"""

import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for medical images.

    Supports various medical imaging formats and handles
    class imbalance through sampling strategies.
    """

    def __init__(self, image_paths, labels, transform=None, cache_images=False):
        """
        Parameters:
        -----------
        image_paths : list of str
            Paths to image files
        labels : list or array
            Class labels (integers or strings)
        transform : albumentations.Compose, optional
            Augmentation pipeline
        cache_images : bool
            Whether to cache images in memory (use for small datasets)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_images = cache_images

        # Convert string labels to integers if needed
        if isinstance(labels[0], str):
            self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            self.labels = [self.label_mapping[label] for label in labels]

        # Cache for storing images in memory
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and transform a single image-label pair."""
        # Load image (from cache or disk)
        if self.cache_images and idx in self._cache:
            image = self._cache[idx]
        else:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

            if self.cache_images:
                self._cache[idx] = image

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


def get_transforms(mode="train", image_size=224):
    """
    Get augmentation pipeline for medical images.

    Parameters:
    -----------
    mode : str
        'train', 'val', or 'test'
    image_size : int
        Target image size (square)

    Returns:
    --------
    transform : albumentations.Compose
        Augmentation pipeline
    """
    if mode == "train":
        # Training augmentations: preserve medical image characteristics
        transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                # Geometric transformations (mild)
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
                # Intensity transformations (common in medical imaging)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # Normalize and convert to tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/test: only resize and normalize
        transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    return transform


def create_data_loaders(
    train_paths, train_labels, val_paths, val_labels, batch_size=32, num_workers=4, image_size=224
):
    """
    Create train and validation data loaders.

    Parameters:
    -----------
    train_paths, train_labels : lists
        Training data paths and labels
    val_paths, val_labels : lists
        Validation data paths and labels
    batch_size : int
        Batch size for training
    num_workers : int
        Number of worker processes for data loading
    image_size : int
        Target image size

    Returns:
    --------
    train_loader, val_loader : DataLoader objects
    """
    # Create datasets
    train_dataset = MedicalImageDataset(
        train_paths, train_labels, transform=get_transforms("train", image_size)
    )

    val_dataset = MedicalImageDataset(
        val_paths, val_labels, transform=get_transforms("val", image_size)
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def load_image_metadata(csv_path, image_dir=None):
    """
    Load image metadata from CSV.

    Expected CSV format:
        filename,label,split,...
        image001.jpg,benign,train,...
        image002.jpg,malignant,val,...

    Parameters:
    -----------
    csv_path : str
        Path to metadata CSV
    image_dir : str, optional
        Directory containing images (prepended to filenames)

    Returns:
    --------
    data : dict
        Dictionary with 'train' and 'val' keys, each containing
        {'paths': [...], 'labels': [...]}
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = ["filename", "label"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must contain columns: {required}")

    # Optionally prepend image directory
    if image_dir:
        df["filename"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))

    data = {}

    # Split by train/val if split column exists
    if "split" in df.columns:
        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            if len(split_df) > 0:
                data[split] = {
                    "paths": split_df["filename"].tolist(),
                    "labels": split_df["label"].tolist(),
                }
    else:
        # Return all data
        data["all"] = {"paths": df["filename"].tolist(), "labels": df["label"].tolist()}

    return data


def balance_classes(image_paths, labels, method="undersample"):
    """
    Balance class distribution.

    Parameters:
    -----------
    image_paths : list
        Image file paths
    labels : list
        Class labels
    method : str
        'undersample' or 'oversample'

    Returns:
    --------
    balanced_paths, balanced_labels : lists
        Balanced dataset
    """
    df = pd.DataFrame({"path": image_paths, "label": labels})

    # Count samples per class
    class_counts = df["label"].value_counts()

    if method == "undersample":
        # Undersample to minority class size
        min_count = class_counts.min()
        balanced_df = (
            df.groupby("label")
            .apply(lambda x: x.sample(min_count, random_state=42))
            .reset_index(drop=True)
        )

    elif method == "oversample":
        # Oversample to majority class size
        max_count = class_counts.max()
        balanced_df = (
            df.groupby("label")
            .apply(lambda x: x.sample(max_count, replace=True, random_state=42))
            .reset_index(drop=True)
        )
    else:
        raise ValueError("method must be 'undersample' or 'oversample'")

    return balanced_df["path"].tolist(), balanced_df["label"].tolist()


def split_dataset(image_paths, labels, train_frac=0.7, val_frac=0.15, random_state=42):
    """
    Split dataset into train/val/test sets.

    Parameters:
    -----------
    image_paths : list
        Image file paths
    labels : list
        Class labels
    train_frac : float
        Fraction for training (default: 0.7)
    val_frac : float
        Fraction for validation (default: 0.15)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    splits : dict
        Dictionary with 'train', 'val', 'test' keys containing
        {'paths': [...], 'labels': [...]}
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, train_size=train_frac, stratify=labels, random_state=random_state
    )

    # Second split: val vs test
    test_frac = 1.0 - train_frac - val_frac
    val_size = val_frac / (val_frac + test_frac)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=random_state,
    )

    splits = {
        "train": {"paths": train_paths, "labels": train_labels},
        "val": {"paths": val_paths, "labels": val_labels},
        "test": {"paths": test_paths, "labels": test_labels},
    }

    return splits


def preprocess_dicom(dicom_path, output_path, window_center=40, window_width=400):
    """
    Convert DICOM to preprocessed PNG/JPG.

    Applies windowing for optimal tissue visualization.

    Parameters:
    -----------
    dicom_path : str
        Path to DICOM file
    output_path : str
        Output image path
    window_center : float
        Window center (HU)
    window_width : float
        Window width (HU)
    """
    try:
        import pydicom
    except ImportError as e:
        raise ImportError("pydicom required for DICOM processing: pip install pydicom") from e

    # Read DICOM
    dcm = pydicom.dcmread(dicom_path)
    image = dcm.pixel_array.astype(float)

    # Apply rescale slope/intercept if present
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        image = image * dcm.RescaleSlope + dcm.RescaleIntercept

    # Apply windowing
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    image = np.clip(image, img_min, img_max)

    # Normalize to 0-255
    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    # Save
    Image.fromarray(image).save(output_path)


if __name__ == "__main__":
    # Example usage
    print("Medical Image Data Preprocessing")
    print("=" * 50)

    # Generate synthetic example
    print("\n1. Creating synthetic dataset...")
    n_samples = 100
    image_paths = [f"image_{i:03d}.jpg" for i in range(n_samples)]
    labels = np.random.choice(["benign", "malignant"], n_samples).tolist()

    print(f"   {n_samples} images, {len(set(labels))} classes")

    # Split dataset
    print("\n2. Splitting into train/val/test...")
    splits = split_dataset(image_paths, labels)
    print(f"   Train: {len(splits['train']['paths'])}")
    print(f"   Val: {len(splits['val']['paths'])}")
    print(f"   Test: {len(splits['test']['paths'])}")

    # Get transforms
    print("\n3. Creating augmentation pipelines...")
    train_transform = get_transforms("train", image_size=224)
    val_transform = get_transforms("val", image_size=224)
    print(f"   Train augmentations: {len(train_transform.transforms)}")
    print(f"   Val augmentations: {len(val_transform.transforms)}")

    # Balance classes
    print("\n4. Balancing classes...")
    balanced_paths, balanced_labels = balance_classes(
        splits["train"]["paths"], splits["train"]["labels"], method="undersample"
    )
    print(f"   Original: {len(splits['train']['paths'])}")
    print(f"   Balanced: {len(balanced_paths)}")

    print("\n✓ Data preprocessing utilities ready")
