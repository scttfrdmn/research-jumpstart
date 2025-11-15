#!/usr/bin/env python3
"""
Create synthetic fMRI data for testing.

This script generates sample NIfTI-format fMRI data with realistic noise and signal.
Useful for testing the Lambda preprocessing pipeline without real patient data.

Usage:
    python create_sample_fmri.py --output sample_data/sample_fmri.nii.gz
"""

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_fmri(shape=(64, 64, 32, 150), noise_level=0.1, output_path="sample_fmri.nii.gz"):
    """
    Create synthetic fMRI data.

    Args:
        shape: fMRI volume shape (x, y, z, time)
        noise_level: Gaussian noise standard deviation (relative to signal)
        output_path: Output file path

    Returns:
        None (saves to file)
    """
    logger.info("Creating synthetic fMRI data...")
    logger.info(f"  Shape: {shape}")
    logger.info(f"  Noise level: {noise_level}")

    x, y, z, t = shape

    # Create baseline signal
    logger.info("Creating baseline signal...")
    baseline = np.ones(shape) * 1000  # Typical fMRI baseline ~1000 AU

    # Add regional variation
    # Central ROI with higher signal
    cx, cy, cz = x // 2, y // 2, z // 2
    rx, _ry, _rz = 15, 15, 10

    for i in range(x):
        for j in range(y):
            for k in range(z):
                dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                if dist < rx:
                    baseline[i, j, k, :] += 100 * (1 - dist / rx)

    # Add temporal fluctuations (resting state-like oscillations)
    logger.info("Adding temporal oscillations...")
    for t_idx in range(t):
        # Low frequency oscillations
        baseline[:, :, :, t_idx] += 20 * np.sin(2 * np.pi * 0.1 * t_idx / t)
        # 0.1 Hz oscillation (typical resting state)
        baseline[:, :, :, t_idx] += 10 * np.cos(2 * np.pi * 0.05 * t_idx / t)

    # Add spatial correlations (neighboring voxels are correlated)
    logger.info("Adding spatial correlations...")
    from scipy import ndimage

    baseline_smoothed = ndimage.gaussian_filter(baseline, sigma=2)
    baseline = 0.7 * baseline_smoothed + 0.3 * baseline  # Mix smoothed and original

    # Add Gaussian noise
    logger.info(f"Adding Gaussian noise (SNR ≈ {1 / noise_level:.1f}:1)...")
    noise = np.random.normal(0, noise_level * np.mean(baseline), shape)
    fmri_data = baseline + noise

    # Ensure data is positive (BOLD signals are positive)
    fmri_data = np.maximum(fmri_data, 0)

    logger.info(f"Data range: [{np.min(fmri_data):.1f}, {np.max(fmri_data):.1f}]")
    logger.info(f"Data mean: {np.mean(fmri_data):.1f}")
    logger.info(f"Data std: {np.std(fmri_data):.1f}")

    # Create affine matrix (standard MNI affine)
    affine = np.array(
        [[-3, 0, 0, 90], [0, 3, 0, -126], [0, 0, 3, -72], [0, 0, 0, 1]], dtype=np.float32
    )

    # Create NIfTI image
    logger.info("Creating NIfTI image...")
    img = nib.Nifti1Image(fmri_data, affine)

    # Add header information
    img.header.set_data_dtype(np.float32)
    img.header["descrip"] = b"Synthetic fMRI data for testing"

    # Save file
    logger.info(f"Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(img, str(output_path))
    logger.info(f"✓ Successfully saved to {output_path}")

    # Print file info
    file_size = output_path.stat().st_size / 1e6
    logger.info(f"File size: {file_size:.2f} MB")

    return str(output_path)


def create_fmri_with_task_activation(
    shape=(64, 64, 32, 150), output_path="sample_fmri_with_task.nii.gz"
):
    """
    Create fMRI data with simulated task activation.

    Args:
        shape: fMRI volume shape
        output_path: Output file path

    Returns:
        Path to created file
    """
    logger.info("Creating fMRI data with task activation...")

    _x, _y, _z, _t = shape

    # Start with baseline
    fmri_data = np.ones(shape) * 1000

    # Add noise
    noise = np.random.normal(0, 50, shape)
    fmri_data += noise

    # Add task activation (blocks of activation)
    # Task ON for timepoints 30-60 and 90-120
    activation_regions = [
        (slice(20, 35), slice(20, 35), slice(10, 20)),  # ROI 1
        (slice(40, 55), slice(40, 55), slice(15, 25)),  # ROI 2
    ]

    task_blocks = [(30, 60), (90, 120)]

    for block_start, block_end in task_blocks:
        for region in activation_regions:
            fmri_data[(*region, slice(block_start, block_end))] += 100

    # Create affine
    affine = np.array(
        [[-3, 0, 0, 90], [0, 3, 0, -126], [0, 0, 3, -72], [0, 0, 0, 1]], dtype=np.float32
    )

    # Create and save image
    img = nib.Nifti1Image(fmri_data, affine)
    img.header.set_data_dtype(np.float32)
    img.header["descrip"] = b"Synthetic fMRI with task activation"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))

    logger.info(f"✓ Saved task fMRI to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic fMRI data for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default sample data
  python create_sample_fmri.py

  # Create with custom output path
  python create_sample_fmri.py --output sample_data/my_fmri.nii.gz

  # Create with custom dimensions
  python create_sample_fmri.py --shape 64 64 32 200

  # Create task fMRI data
  python create_sample_fmri.py --task --output sample_fmri_task.nii.gz
        """,
    )

    parser.add_argument(
        "--output",
        default="sample_fmri.nii.gz",
        help="Output file path (default: sample_fmri.nii.gz)",
    )
    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        default=[64, 64, 32, 150],
        help="fMRI shape: x y z t (default: 64 64 32 150)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Noise level relative to signal (default: 0.1)"
    )
    parser.add_argument(
        "--task", action="store_true", help="Create fMRI with simulated task activation"
    )

    args = parser.parse_args()

    # Create sample data
    if args.task:
        create_fmri_with_task_activation(tuple(args.shape), args.output)
    else:
        create_sample_fmri(tuple(args.shape), args.noise, args.output)

    logger.info("\n✓ Sample data created successfully!")
    logger.info("Ready for upload to S3:")
    logger.info(f"  python upload_to_s3.py --bucket fmri-input-myname --local-path {args.output}")


if __name__ == "__main__":
    main()
