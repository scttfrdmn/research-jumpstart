"""
Lambda function for fMRI data preprocessing.

This function performs motion correction and spatial smoothing on fMRI NIfTI files
stored in S3. It's designed to be triggered manually or through S3 events.

Author: Research Jumpstart
License: MIT
"""

import json
import logging
import os
import tempfile

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS S3 client
s3 = boto3.client("s3")

# Import neuroimaging libraries
try:
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
except ImportError as e:
    logger.warning(f"Optional dependency not available: {e}")
    raise


def load_nifti_from_s3(bucket: str, key: str, local_path: str) -> dict:
    """
    Download NIfTI file from S3 and load into memory.

    Args:
        bucket: S3 bucket name
        key: S3 object key (path)
        local_path: Local path to save file temporarily

    Returns:
        Dictionary with image data, affine, header, and file info
    """
    logger.info(f"Loading NIfTI from S3: s3://{bucket}/{key}")

    try:
        # Download from S3
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded to {local_path}")

        # Load NIfTI file
        img = nib.load(local_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        logger.info(f"Loaded NIfTI: shape={data.shape}, dtype={data.dtype}")

        return {"data": data, "affine": affine, "header": header, "original_shape": data.shape}
    except Exception as e:
        logger.error(f"Error loading NIfTI: {e!s}")
        raise


def motion_correction(data: np.ndarray, max_translation: float = 10.0) -> np.ndarray:
    """
    Apply motion correction to fMRI data using center-of-mass alignment.

    This is a simplified motion correction that aligns each volume to the mean
    using center-of-mass estimation. Production systems would use more sophisticated
    methods like FSL's MCFLIRT or SPM.

    Args:
        data: 4D fMRI data array (x, y, z, time)
        max_translation: Maximum translation allowed (voxels)

    Returns:
        Motion-corrected 4D array
    """
    logger.info(f"Starting motion correction on data shape {data.shape}")

    # Handle NaN and inf values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize intensity (zero mean, unit variance) per volume
    data_corrected = data.copy()

    # Compute mean image (reference for alignment)
    mean_img = np.mean(data, axis=3)
    mean_img = np.nan_to_num(mean_img, nan=0.0)

    # Track motion parameters for quality control
    motion_params = []

    # Apply simple motion correction to each volume
    for t in range(data.shape[3]):
        # Get current volume
        vol = data[:, :, :, t]

        # Compute center of mass
        com_vol = np.array(ndimage.center_of_mass(vol))
        com_mean = np.array(ndimage.center_of_mass(mean_img))

        # Calculate translation
        translation = com_mean - com_vol

        # Limit translation to max_translation
        translation = np.clip(translation, -max_translation, max_translation)

        # Apply translation (simplified - real implementations use interpolation)
        if np.any(translation != 0):
            # Use simple shift for demonstration
            vol_corrected = ndimage.shift(vol, translation, order=1, cval=0)
        else:
            vol_corrected = vol

        data_corrected[:, :, :, t] = vol_corrected
        motion_params.append(translation)

        if (t + 1) % 10 == 0:
            logger.info(f"Motion correction progress: {t + 1}/{data.shape[3]} volumes")

    # Log mean motion
    motion_params = np.array(motion_params)
    mean_translation = np.mean(np.abs(motion_params), axis=0)
    logger.info(
        f"Mean translation: x={mean_translation[0]:.3f}, "
        f"y={mean_translation[1]:.3f}, z={mean_translation[2]:.3f} voxels"
    )

    return data_corrected


def spatial_smoothing(data: np.ndarray, fwhm: float = 6.0) -> np.ndarray:
    """
    Apply spatial smoothing (Gaussian) to fMRI data.

    Full Width at Half Maximum (FWHM) is converted to sigma for Gaussian kernel.
    Commonly used FWHM values: 4-8mm depending on voxel size and analysis needs.

    Args:
        data: 4D fMRI data array (x, y, z, time)
        fwhm: Full Width at Half Maximum in voxels (default 6)

    Returns:
        Spatially smoothed 4D array
    """
    logger.info(f"Starting spatial smoothing with FWHM={fwhm} voxels")

    # Convert FWHM to sigma for Gaussian
    # FWHM = 2.355 * sigma
    sigma = fwhm / 2.355

    data_smoothed = data.copy()

    # Apply Gaussian smoothing to each volume
    for t in range(data.shape[3]):
        vol = data[:, :, :, t]

        # Apply 3D Gaussian filter
        vol_smoothed = ndimage.gaussian_filter(vol, sigma=sigma)
        data_smoothed[:, :, :, t] = vol_smoothed

        if (t + 1) % 10 == 0:
            logger.info(f"Smoothing progress: {t + 1}/{data.shape[3]} volumes")

    logger.info(f"Spatial smoothing complete. Sigma={sigma:.3f} voxels")

    return data_smoothed


def save_nifti_to_s3(
    data: np.ndarray, affine: np.ndarray, bucket: str, key: str, local_path: str
) -> bool:
    """
    Save NIfTI file and upload to S3.

    Args:
        data: 4D fMRI data array
        affine: Affine transformation matrix
        bucket: S3 bucket name
        key: S3 object key (path)
        local_path: Local path to save file temporarily

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving NIfTI to s3://{bucket}/{key}")

    try:
        # Create NIfTI image
        img = nib.Nifti1Image(data, affine)

        # Save locally
        nib.save(img, local_path)
        logger.info(f"Saved locally to {local_path}")

        # Upload to S3
        s3.upload_file(local_path, bucket, key)
        logger.info(f"Uploaded to S3: s3://{bucket}/{key}")

        return True
    except Exception as e:
        logger.error(f"Error saving NIfTI: {e!s}")
        return False


def lambda_handler(event, context):
    """
    Lambda handler for fMRI preprocessing.

    Expected event format:
    {
        "body": "{\"input_bucket\": \"bucket-name\", \"input_key\": \"file.nii.gz\"}"
    }

    Or for direct invocation:
    {
        "input_bucket": "bucket-name",
        "input_key": "file.nii.gz"
    }
    """
    logger.info("Lambda function started")
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Parse input parameters
        if "body" in event:
            # HTTP API call
            body = json.loads(event["body"])
            input_bucket = body.get("input_bucket")
            input_key = body.get("input_key")
        else:
            # Direct invocation
            input_bucket = event.get("input_bucket")
            input_key = event.get("input_key")

        if not input_bucket or not input_key:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": "Missing required parameters",
                        "required": ["input_bucket", "input_key"],
                    }
                ),
            }

        # Get output bucket from environment variable
        output_bucket = os.environ.get("OUTPUT_BUCKET")
        if not output_bucket:
            logger.error("OUTPUT_BUCKET environment variable not set")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "OUTPUT_BUCKET not configured"}),
            }

        logger.info(
            f"Parameters: input_bucket={input_bucket}, input_key={input_key}, output_bucket={output_bucket}"
        )

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Using temporary directory: {tmpdir}")

            # Define file paths
            input_file = os.path.join(tmpdir, "input.nii.gz")
            corrected_file = os.path.join(tmpdir, "corrected.nii.gz")
            smoothed_file = os.path.join(tmpdir, "smoothed.nii.gz")

            # Load input data from S3
            logger.info("Step 1: Loading input data")
            nifti_data = load_nifti_from_s3(input_bucket, input_key, input_file)

            # Apply motion correction
            logger.info("Step 2: Applying motion correction")
            data_corrected = motion_correction(nifti_data["data"])

            # Save motion-corrected data
            corrected_key = input_key.replace(".nii.gz", "_motion_corrected.nii.gz")
            corrected_key = corrected_key.replace(".nii", "_motion_corrected.nii")

            success = save_nifti_to_s3(
                data_corrected, nifti_data["affine"], output_bucket, corrected_key, corrected_file
            )

            if not success:
                raise Exception("Failed to save motion-corrected data")

            # Apply spatial smoothing
            logger.info("Step 3: Applying spatial smoothing")
            data_smoothed = spatial_smoothing(data_corrected)

            # Save smoothed data
            smoothed_key = input_key.replace(".nii.gz", "_smoothed.nii.gz")
            smoothed_key = smoothed_key.replace(".nii", "_smoothed.nii")

            success = save_nifti_to_s3(
                data_smoothed, nifti_data["affine"], output_bucket, smoothed_key, smoothed_file
            )

            if not success:
                raise Exception("Failed to save smoothed data")

            # Prepare response
            response = {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Processing completed successfully",
                        "input": f"s3://{input_bucket}/{input_key}",
                        "motion_corrected": f"s3://{output_bucket}/{corrected_key}",
                        "smoothed": f"s3://{output_bucket}/{smoothed_key}",
                        "input_shape": str(nifti_data["original_shape"]),
                        "processing_steps": ["motion_correction", "spatial_smoothing"],
                    }
                ),
            }

            logger.info(f"Processing complete. Response: {response}")
            return response

    except Exception as e:
        logger.error(f"Error during processing: {e!s}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": str(e), "message": "An error occurred during fMRI processing"}
            ),
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {"input_bucket": "fmri-input-test", "input_key": "sample_fmri.nii.gz"}

    # Create mock context
    class MockContext:
        def __init__(self):
            self.invoked_function_arn = (
                "arn:aws:lambda:us-east-1:123456789012:function:fmri-preprocessor"
            )
            self.aws_request_id = "test-request-id"
            self.function_version = "$LATEST"
            self.log_group_name = "/aws/lambda/fmri-preprocessor"

    context = MockContext()

    # Run handler
    result = lambda_handler(test_event, context)
    print(json.dumps(result, indent=2))
