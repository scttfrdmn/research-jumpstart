"""
Lambda function for astronomical source detection.

Performs source detection on FITS images using SEP.
Saves results as Parquet file to S3.
"""

import json
import os
import io
import logging
from datetime import datetime
import traceback

import boto3
import numpy as np
from astropy.io import fits

# Try to import SEP, fallback to simpler method if not available
try:
    import sep
    HAS_SEP = True
except ImportError:
    HAS_SEP = False
    logging.warning("SEP not available, using basic detection")

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3 = boto3.client('s3')


def simple_source_detection(data, threshold=5.0):
    """
    Simple source detection using basic thresholding.
    Fallback when SEP is not available.
    """
    logger.info("Using simple source detection (SEP not available)")

    # Calculate background
    median = np.median(data)
    std = np.std(data)

    # Find pixels above threshold
    mask = data > (median + threshold * std)

    # Label connected components
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)

    sources = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        y, x = np.where(component)

        if len(x) < 3:  # Require at least 3 pixels
            continue

        # Calculate properties
        flux = np.sum(data[component])
        peak = np.max(data[component])
        x_center = np.mean(x)
        y_center = np.mean(y)
        fwhm = np.sqrt(len(x) / np.pi) * 2  # Rough approximation

        # Estimate SNR
        snr = (peak - median) / (std if std > 0 else 1)

        sources.append({
            'x': float(x_center),
            'y': float(y_center),
            'flux': float(flux),
            'peak': float(peak),
            'fwhm': float(fwhm),
            'snr': float(snr),
            'a': float(fwhm / 2.35),  # Sigma estimate
            'b': float(fwhm / 2.35),
            'theta': 0.0
        })

    return np.array(sources) if sources else np.array([])


def sep_source_detection(data, err=None, thresh=5.0):
    """
    Source detection using SEP (Source Extraction Program).
    """
    logger.info("Using SEP source detection")

    if err is None:
        err = np.ones_like(data)

    # Extract sources
    objects = sep.extract(data, thresh=thresh, err=err, deblend_nthresh=32, deblend_cont=0.005)

    return objects


def detect_sources_in_image(image_data):
    """
    Detect sources in astronomical image.

    Args:
        image_data: 2D numpy array of image data

    Returns:
        List of detected sources with properties
    """
    logger.info(f"Image shape: {image_data.shape}")

    # Ensure float type
    image_data = image_data.astype(np.float32)

    # Calculate background
    median = np.median(image_data)
    std = np.std(image_data)
    logger.info(f"Background: median={median:.1f}, std={std:.1f}")

    # Subtract background
    data_sub = image_data - median

    # Detect sources
    if HAS_SEP:
        try:
            objects = sep_source_detection(data_sub, err=np.sqrt(np.abs(median)))
        except Exception as e:
            logger.warning(f"SEP detection failed: {e}, falling back to simple detection")
            objects = simple_source_detection(image_data, threshold=3.0)
    else:
        objects = simple_source_detection(image_data, threshold=3.0)

    # Convert to list of dicts
    sources = []
    for obj in objects:
        # Calculate SNR
        flux = float(obj['flux'])
        peak = float(obj.get('peak', obj.get('cpeak', 0)))
        snr = float(obj.get('snr', (peak - median) / (std if std > 0 else 1)))

        source = {
            'x': float(obj.get('x', obj.get('xpeak', 0))),
            'y': float(obj.get('y', obj.get('ypeak', 0))),
            'flux': flux,
            'peak': peak,
            'fwhm': float(obj.get('fwhm', 1.0)),
            'a': float(obj.get('a', 1.0)),
            'b': float(obj.get('b', 1.0)),
            'theta': float(obj.get('theta', 0.0)),
            'snr': snr
        }
        sources.append(source)

    logger.info(f"Detected {len(sources)} sources")
    return sources


def convert_to_catalog_format(sources, image_id, ra, dec, pixel_scale=0.396):
    """
    Convert source detections to catalog format.

    Args:
        sources: List of detected sources
        image_id: Identifier for the image
        ra: RA of image center (degrees)
        dec: Dec of image center (degrees)
        pixel_scale: Arcsec per pixel

    Returns:
        List of catalog entries
    """
    catalog = []

    for i, source in enumerate(sources):
        # Convert pixel coordinates to celestial coordinates
        # Simple linear transformation from image center
        dx = source['x'] - 256  # Assume 512x512 image
        dy = source['y'] - 256
        src_ra = ra + (dx * pixel_scale / 3600.0)
        src_dec = dec + (dy * pixel_scale / 3600.0)

        entry = {
            'image_id': image_id,
            'source_id': i,
            'ra': src_ra,
            'dec': src_dec,
            'x': source['x'],
            'y': source['y'],
            'flux': source['flux'],
            'flux_err': source.get('flux_err', source['flux'] * 0.1),
            'peak': source['peak'],
            'fwhm': source['fwhm'],
            'a': source['a'],
            'b': source['b'],
            'theta': source['theta'],
            'snr': source['snr'],
            'detection_time': datetime.utcnow().isoformat()
        }
        catalog.append(entry)

    return catalog


def save_catalog_parquet(catalog, output_key, bucket_catalog):
    """
    Save catalog as Parquet file to S3.

    Args:
        catalog: List of catalog entries
        output_key: S3 key for output file
        bucket_catalog: S3 bucket for catalogs
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, saving as JSON instead")
        save_catalog_json(catalog, output_key, bucket_catalog)
        return

    # Convert to DataFrame
    df = pd.DataFrame(catalog)

    # Convert to Parquet
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)

    # Upload to S3
    s3.put_object(
        Bucket=bucket_catalog,
        Key=output_key,
        Body=parquet_buffer.getvalue(),
        ContentType='application/octet-stream',
        Metadata={
            'source': 'lambda-source-detection',
            'format': 'parquet',
            'num_sources': str(len(catalog))
        }
    )

    logger.info(f"Saved catalog to s3://{bucket_catalog}/{output_key}")


def save_catalog_json(catalog, output_key, bucket_catalog):
    """
    Save catalog as JSON file to S3.

    Args:
        catalog: List of catalog entries
        output_key: S3 key for output file
        bucket_catalog: S3 bucket for catalogs
    """
    # Convert to JSON
    json_buffer = json.dumps(catalog, indent=2, default=str).encode('utf-8')

    # Upload to S3
    s3.put_object(
        Bucket=bucket_catalog,
        Key=output_key,
        Body=json_buffer,
        ContentType='application/json',
        Metadata={
            'source': 'lambda-source-detection',
            'format': 'json',
            'num_sources': str(len(catalog))
        }
    )

    logger.info(f"Saved catalog to s3://{bucket_catalog}/{output_key}")


def lambda_handler(event, context):
    """
    Lambda handler for source detection.

    Expected event format:
    {
        "bucket": "s3-bucket-name",
        "key": "path/to/image.fits"
    }
    """
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Get parameters
        bucket_raw = os.environ.get('BUCKET_RAW', event.get('bucket'))
        bucket_catalog = os.environ.get('BUCKET_CATALOG')
        s3_key = event.get('key')

        if not all([bucket_raw, bucket_catalog, s3_key]):
            raise ValueError("Missing required parameters: bucket, bucket_catalog, or key")

        logger.info(f"Processing: s3://{bucket_raw}/{s3_key}")

        # Download FITS file from S3
        logger.info("Downloading FITS file...")
        response = s3.get_object(Bucket=bucket_raw, Key=s3_key)
        fits_data = response['Body'].read()

        # Parse FITS
        logger.info("Parsing FITS...")
        hdul = fits.open(io.BytesIO(fits_data))
        image_data = hdul[0].data
        header = hdul[0].header
        hdul.close()

        if image_data is None:
            raise ValueError("No image data in FITS file")

        # Get image metadata
        filter_band = header.get('FILTER', 'unknown')
        ra = float(header.get('RA', 0.0))
        dec = float(header.get('DEC', 0.0))
        image_id = os.path.splitext(os.path.basename(s3_key))[0]

        logger.info(f"Image: {image_id}, Filter: {filter_band}, RA: {ra}, Dec: {dec}")

        # Detect sources
        logger.info("Detecting sources...")
        sources = detect_sources_in_image(image_data)

        # Convert to catalog format
        catalog = convert_to_catalog_format(sources, image_id, ra, dec)

        # Save catalog
        output_key = f"sources/{image_id}_sources.json"
        save_catalog_json(catalog, output_key, bucket_catalog)

        # Return success
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Source detection successful',
                'image_id': image_id,
                'num_sources': len(catalog),
                'output_key': output_key,
                'output_bucket': bucket_catalog
            })
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }
