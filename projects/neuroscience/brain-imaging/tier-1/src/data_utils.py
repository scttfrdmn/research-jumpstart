"""
Data loading and preprocessing utilities for neuroimaging analysis.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import requests
from tqdm import tqdm


# Define base data directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# Create directories if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / 'connectivity').mkdir(exist_ok=True)
(PROCESSED_DIR / 'timeseries').mkdir(exist_ok=True)


def load_haxby_data(force_download=False):
    """
    Load Haxby 2001 dataset for quick testing.

    Parameters
    ----------
    force_download : bool
        If True, re-download even if cached

    Returns
    -------
    dict : Dataset with fmri, labels, mask
    """
    print("Loading Haxby dataset...")
    haxby_dataset = datasets.fetch_haxby()

    data = {
        'fmri': haxby_dataset.func[0],
        'labels': pd.read_csv(haxby_dataset.session_target[0], sep=' '),
        'mask': haxby_dataset.mask_vt[0],
        'dataset': haxby_dataset
    }

    print(f"✓ Loaded {len(data['labels'])} volumes")
    return data


def load_development_fmri(n_subjects=30, force_download=False):
    """
    Load development fMRI dataset from Nilearn.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to load
    force_download : bool
        If True, re-download even if cached

    Returns
    -------
    dict : Dataset with subjects, ages, phenotypes
    """
    print(f"Loading development fMRI data ({n_subjects} subjects)...")
    print("This may take several minutes on first run...")

    # Use Nilearn's development dataset
    data = datasets.fetch_development_fmri(n_subjects=n_subjects)

    print(f"✓ Loaded {len(data.func)} subjects")
    print(f"  Age range: {data.phenotypic['Child_Adult'].value_counts().to_dict()}")

    return data


def load_atlas(atlas_name='schaefer', n_rois=200):
    """
    Load brain parcellation atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name ('schaefer', 'aal', 'power')
    n_rois : int
        Number of ROIs (for Schaefer: 100, 200, 400, 600)

    Returns
    -------
    atlas_img : Nifti1Image
        Atlas parcellation
    labels : list
        ROI labels
    """
    print(f"Loading {atlas_name} atlas with {n_rois} ROIs...")

    if atlas_name.lower() == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
        atlas_img = nib.load(atlas.maps)
        labels = atlas.labels
    elif atlas_name.lower() == 'aal':
        atlas = datasets.fetch_atlas_aal()
        atlas_img = nib.load(atlas.maps)
        labels = atlas.labels
    elif atlas_name.lower() == 'power':
        atlas = datasets.fetch_coords_power_2011()
        labels = atlas.labels
        # Power atlas is coordinate-based, not volumetric
        return atlas, labels
    else:
        raise ValueError(f"Unknown atlas: {atlas_name}")

    print(f"✓ Loaded {len(labels)} regions")
    return atlas_img, labels


def extract_timeseries(fmri_file, atlas_img, labels, cache=True):
    """
    Extract ROI time series from fMRI data.

    Parameters
    ----------
    fmri_file : str or Nifti1Image
        4D fMRI data
    atlas_img : Nifti1Image
        Parcellation atlas
    labels : list
        ROI labels
    cache : bool
        If True, cache extracted time series

    Returns
    -------
    timeseries : ndarray, shape (n_timepoints, n_rois)
        Extracted time series
    """
    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0
    )

    # Extract time series
    print("Extracting time series from ROIs...")
    timeseries = masker.fit_transform(fmri_file)

    print(f"✓ Extracted time series: {timeseries.shape}")
    print(f"  {timeseries.shape[0]} time points × {timeseries.shape[1]} ROIs")

    return timeseries


def save_timeseries(timeseries, subject_id):
    """
    Save extracted time series to disk.

    Parameters
    ----------
    timeseries : ndarray
        Time series data
    subject_id : str
        Subject identifier
    """
    output_path = PROCESSED_DIR / 'timeseries' / f'{subject_id}_ts.npy'
    np.save(output_path, timeseries)
    print(f"✓ Saved time series to {output_path}")


def load_timeseries(subject_id):
    """
    Load cached time series.

    Parameters
    ----------
    subject_id : str
        Subject identifier

    Returns
    -------
    timeseries : ndarray
        Time series data
    """
    input_path = PROCESSED_DIR / 'timeseries' / f'{subject_id}_ts.npy'

    if not input_path.exists():
        raise FileNotFoundError(f"Time series not found: {input_path}")

    timeseries = np.load(input_path)
    print(f"✓ Loaded time series from cache: {timeseries.shape}")
    return timeseries


def preprocess_fmri(fmri_file, confounds=None):
    """
    Basic fMRI preprocessing.

    Parameters
    ----------
    fmri_file : str
        Path to fMRI NIfTI file
    confounds : ndarray or None
        Confound regressors

    Returns
    -------
    preprocessed : Nifti1Image
        Preprocessed fMRI data
    """
    from nilearn import image

    print("Preprocessing fMRI data...")

    # Load data
    img = nib.load(fmri_file) if isinstance(fmri_file, (str, Path)) else fmri_file

    # Smooth
    img_smooth = image.smooth_img(img, fwhm=6)

    # Standardize
    # Note: Masker handles detrending and filtering

    print("✓ Preprocessing complete")
    return img_smooth


def get_sample_subjects(dataset='haxby', n=5):
    """
    Get sample subjects for quick testing.

    Parameters
    ----------
    dataset : str
        Dataset name
    n : int
        Number of subjects

    Returns
    -------
    subjects : list
        List of subject IDs
    """
    if dataset == 'haxby':
        # Haxby only has 1 subject
        return ['subject1']
    elif dataset == 'development':
        # Development dataset has multiple subjects
        data = load_development_fmri(n_subjects=n)
        return [f'sub-{i:02d}' for i in range(n)]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def check_storage_usage():
    """
    Check disk usage of data directories.
    """
    import subprocess

    print("=== Storage Usage ===")

    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        if directory.exists():
            result = subprocess.run(
                ['du', '-sh', str(directory)],
                capture_output=True,
                text=True
            )
            print(f"{directory.name}: {result.stdout.split()[0]}")

    print(f"\nTotal Studio Lab limit: 15GB")


if __name__ == '__main__':
    # Test data loading
    print("Testing data utilities...")

    # Test Haxby dataset
    data = load_haxby_data()
    print(f"\nHaxby dataset loaded: {len(data['labels'])} samples")

    # Test atlas loading
    atlas_img, labels = load_atlas('schaefer', n_rois=200)
    print(f"\nAtlas loaded: {len(labels)} ROIs")

    # Check storage
    print("\n")
    check_storage_usage()
