"""
Data loading utilities for multi-modal medical imaging.

Supports:
- Chest X-rays (PNG format)
- CT scans (DICOM format)
- MRI volumes (NIfTI format)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Medical imaging libraries
try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import SimpleITK
except ImportError:
    SimpleITK = None


class ChestXrayDataset(Dataset):
    """
    Dataset class for NIH ChestX-ray14.

    Multi-label classification of 14 thoracic diseases.
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform=None,
        disease_classes: Optional[list[str]] = None,
    ):
        """
        Args:
            csv_file: Path to CSV with image names and labels
            img_dir: Directory with chest X-ray images
            transform: Optional transform to be applied
            disease_classes: List of disease class names
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        if disease_classes is None:
            self.disease_classes = [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia",
            ]
        else:
            self.disease_classes = disease_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.df.iloc[idx]["Image Index"]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        # Get labels (binary vector for each disease)
        labels = torch.FloatTensor([self.df.iloc[idx][disease] for disease in self.disease_classes])

        if self.transform:
            image = self.transform(image)

        return image, labels


class CTScanDataset(Dataset):
    """
    Dataset class for CT scans (LIDC-IDRI).

    3D volume processing for nodule detection.
    """

    def __init__(
        self,
        csv_file: str,
        scan_dir: str,
        transform=None,
        patch_size: tuple[int, int, int] = (64, 64, 64),
    ):
        """
        Args:
            csv_file: Path to CSV with scan IDs and labels
            scan_dir: Directory with CT DICOM series
            transform: Optional transform
            patch_size: 3D patch size for training
        """
        self.df = pd.read_csv(csv_file)
        self.scan_dir = Path(scan_dir)
        self.transform = transform
        self.patch_size = patch_size

        if pydicom is None:
            raise ImportError("pydicom required for CT scans. Install with: pip install pydicom")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scan_id = self.df.iloc[idx]["Scan ID"]
        label = torch.FloatTensor([self.df.iloc[idx]["Label"]])

        # Load DICOM series
        scan_path = self.scan_dir / scan_id
        volume = self._load_dicom_series(scan_path)

        # Extract patch
        patch = self._extract_patch(volume, self.patch_size)

        if self.transform:
            patch = self.transform(patch)

        return torch.FloatTensor(patch).unsqueeze(0), label

    def _load_dicom_series(self, series_path: Path) -> np.ndarray:
        """Load DICOM series as 3D numpy array."""
        dicom_files = sorted(series_path.glob("*.dcm"))
        slices = [pydicom.dcmread(str(f)) for f in dicom_files]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Stack into 3D volume
        volume = np.stack([s.pixel_array for s in slices])
        return volume

    def _extract_patch(self, volume: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
        """Extract random or centered patch from volume."""
        # Simple center crop for now
        d, h, w = volume.shape
        pd, ph, pw = patch_size

        start_d = (d - pd) // 2
        start_h = (h - ph) // 2
        start_w = (w - pw) // 2

        patch = volume[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]

        return patch


class MRIDataset(Dataset):
    """
    Dataset class for MRI volumes (BraTS).

    3D brain tumor segmentation with multiple modalities.
    """

    def __init__(
        self, csv_file: str, volume_dir: str, transform=None, modalities: Optional[list[str]] = None
    ):
        """
        Args:
            csv_file: Path to CSV with volume IDs and segmentation paths
            volume_dir: Directory with NIfTI volumes
            transform: Optional transform
            modalities: List of MRI modalities to load (t1, t1ce, t2, flair)
        """
        self.df = pd.read_csv(csv_file)
        self.volume_dir = Path(volume_dir)
        self.transform = transform

        if modalities is None:
            self.modalities = ["t1", "t1ce", "t2", "flair"]
        else:
            self.modalities = modalities

        if nib is None:
            raise ImportError("nibabel required for MRI. Install with: pip install nibabel")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume_id = self.df.iloc[idx]["Volume ID"]

        # Load all modalities
        volume_path = self.volume_dir / volume_id
        volumes = []
        for modality in self.modalities:
            nifti_file = volume_path / f"{volume_id}_{modality}.nii.gz"
            img = nib.load(str(nifti_file))
            volumes.append(img.get_fdata())

        # Stack modalities as channels
        volume = np.stack(volumes, axis=0)  # Shape: (4, 240, 240, 155)

        # Load segmentation mask
        seg_file = volume_path / f"{volume_id}_seg.nii.gz"
        seg = nib.load(str(seg_file))
        mask = seg.get_fdata()

        if self.transform:
            volume, mask = self.transform(volume, mask)

        return torch.FloatTensor(volume), torch.LongTensor(mask)


def load_xray(image_path: str) -> np.ndarray:
    """
    Load a single chest X-ray image.

    Args:
        image_path: Path to PNG X-ray image

    Returns:
        numpy array of shape (H, W) or (H, W, 3)
    """
    img = Image.open(image_path)
    return np.array(img)


def load_ct_scan(dicom_dir: str) -> np.ndarray:
    """
    Load CT scan from DICOM series.

    Args:
        dicom_dir: Directory containing DICOM files

    Returns:
        3D numpy array of shape (D, H, W)
    """
    if pydicom is None:
        raise ImportError("pydicom required. Install with: pip install pydicom")

    dicom_files = sorted(Path(dicom_dir).glob("*.dcm"))
    slices = [pydicom.dcmread(str(f)) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])
    return volume


def load_mri(nifti_path: str) -> np.ndarray:
    """
    Load MRI volume from NIfTI file.

    Args:
        nifti_path: Path to .nii or .nii.gz file

    Returns:
        3D or 4D numpy array
    """
    if nib is None:
        raise ImportError("nibabel required. Install with: pip install nibabel")

    img = nib.load(nifti_path)
    return img.get_fdata()


def get_data_stats(data_dir: str, modality: str) -> dict:
    """
    Calculate dataset statistics.

    Args:
        data_dir: Data directory path
        modality: 'xray', 'ct', or 'mri'

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "modality": modality,
        "num_samples": 0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
    }

    # Implementation would calculate actual statistics
    # For now, return placeholder

    return stats


if __name__ == "__main__":
    # Example usage
    print("Medical Imaging Data Utilities")
    print("=" * 50)
    print("Available dataset classes:")
    print("  - ChestXrayDataset: Multi-label X-ray classification")
    print("  - CTScanDataset: 3D CT nodule detection")
    print("  - MRIDataset: 3D brain tumor segmentation")
    print("\\nLoad functions:")
    print("  - load_xray(path)")
    print("  - load_ct_scan(path)")
    print("  - load_mri(path)")
