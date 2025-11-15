"""
Image preprocessing pipelines for medical imaging.

Handles normalization, augmentation, and format conversions.
"""

import numpy as np
import torchvision.transforms as transforms

# Standard preprocessing for chest X-rays
xray_train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

xray_val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def normalize_ct(
    ct_volume: np.ndarray, window_center: int = 40, window_width: int = 400
) -> np.ndarray:
    """
    Normalize CT scan using windowing.

    Args:
        ct_volume: Raw CT Hounsfield Units
        window_center: Center of window
        window_width: Width of window

    Returns:
        Normalized CT volume [0, 1]
    """
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2

    ct_normalized = np.clip(ct_volume, min_value, max_value)
    ct_normalized = (ct_normalized - min_value) / (max_value - min_value)

    return ct_normalized


def normalize_mri(mri_volume: np.ndarray) -> np.ndarray:
    """
    Normalize MRI volume using z-score normalization.

    Args:
        mri_volume: Raw MRI intensities

    Returns:
        Normalized MRI volume
    """
    mean = np.mean(mri_volume)
    std = np.std(mri_volume)

    mri_normalized = (mri_volume - mean) / std if std > 0 else mri_volume - mean

    return mri_normalized


def augment_3d_volume(volume: np.ndarray, flip: bool = True, rotate: bool = True) -> np.ndarray:
    """
    Apply data augmentation to 3D medical volume.

    Args:
        volume: Input 3D volume
        flip: Apply random flipping
        rotate: Apply random rotation

    Returns:
        Augmented volume
    """
    # Placeholder - implement actual 3D augmentation
    return volume


if __name__ == "__main__":
    print("Medical Image Preprocessing Utilities")
